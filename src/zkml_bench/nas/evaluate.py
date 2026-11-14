"""Fixed-budget evaluation harness for ZK-NAS candidates."""
from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..metrics.zk_proxy import _suppress_onnx_noise, compute_proxy
from ..training.trainer import Trainer, get_device

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    candidate_name: str
    top1: float            # 0-100
    top5: float            # 0-100
    constraints: int
    nonlinear_ops: int
    mult_depth: int
    n_params: int
    train_time_s: float
    proxy_method: str
    error: Optional[str] = None

    @property
    def failed(self) -> bool:
        return self.error is not None

    def summary(self) -> str:
        if self.failed:
            return f"{self.candidate_name}: FAILED — {self.error}"
        return (
            f"{self.candidate_name}: "
            f"top1={self.top1:.2f}% "
            f"constraints={self.constraints:,} "
            f"nonlinear_ops={self.nonlinear_ops} "
            f"mult_depth={self.mult_depth} "
            f"params={self.n_params:,}"
        )


def _extract_model_factory(code: str) -> tuple:
    """Execute agent code and return (get_model_fn, candidate_name, use_poly_init).

    Raises ValueError with a clear message if code is malformed.
    """
    namespace: dict = {"torch": torch, "nn": nn}
    try:
        exec(compile(code, "<agent_candidate>", "exec"), namespace)  # noqa: S102
    except Exception as e:
        raise ValueError(f"Code exec failed: {e}") from e

    get_model = namespace.get("get_model")
    if get_model is None or not callable(get_model):
        raise ValueError("Code must define a callable `get_model(num_classes, input_shape)`")

    candidate_name = str(namespace.get("CANDIDATE_NAME", "unnamed"))
    use_poly_init = bool(namespace.get("USE_POLY_INIT", False))
    return get_model, candidate_name, use_poly_init


def _apply_poly_init(model: nn.Module) -> None:
    """LeCun normal init for polynomial activation models."""
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            fan_in = m.weight.shape[1]
            nn.init.normal_(m.weight, 0.0, 1.0 / (fan_in ** 0.5))
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _apply_zero_init_residual(model: nn.Module) -> None:
    """Zero-initialize the last BN in residual blocks tagged with _is_last_bn.

    Causes blocks to start as identity at init, improving convergence by ~0.2-0.3%.
    Agents opt-in by setting `bn._is_last_bn = True` on the final BN in each block.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and getattr(m, "_is_last_bn", False):
            nn.init.zeros_(m.weight)


def _get_patches(x: torch.Tensor, patch_shape: tuple) -> torch.Tensor:
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2, h, 1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w).float()


def _apply_whitening_conv(model: nn.Module, train_images: torch.Tensor) -> None:
    """ZCA-initialize Conv2d layers tagged with _is_whitening_conv = True.

    Initializes filters as scaled eigenvectors of patch covariance, then freezes weights.
    Agents opt-in by setting `conv._is_whitening_conv = True` on the first conv.
    Requires the layer to have out_channels = 2 * (C * kH * kW).
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and getattr(m, "_is_whitening_conv", False):
            kH, kW = m.weight.shape[2], m.weight.shape[3]
            patches = _get_patches(train_images, (kH, kW))
            n = patches.shape[0]
            patches_flat = patches.view(n, -1)
            cov = (patches_flat.T @ patches_flat) / n
            eigenvalues, eigenvectors = torch.linalg.eigh(cov, UPLO="U")
            eigenvalues = eigenvalues.flip(0).view(-1, 1, 1, 1)
            eigenvectors = eigenvectors.T.reshape(-1, m.in_channels, kH, kW).flip(0)
            eps = 5e-4
            scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
            m.weight.data[:] = torch.cat((scaled, -scaled))
            m.weight.requires_grad = False  # freeze whitening layer


def evaluate_candidate(
    code: str,
    dataset_name: str,
    epochs: int,
    batch_size: int = 512,
    lr: float = 0.1,
    device_str: str = "auto",
    seed: int = 42,
    data_dir: str = "~/.zkml_bench/data",
) -> EvalResult:
    """Train a candidate model for `epochs` and return accuracy + ZK proxy metrics."""
    from ..datasets import get_dataset

    # --- Parse agent code ---
    try:
        get_model_fn, candidate_name, use_poly_init = _extract_model_factory(code)
    except ValueError as e:
        return EvalResult(
            candidate_name="parse_error",
            top1=0.0, top5=0.0, constraints=0,
            nonlinear_ops=0, mult_depth=0, n_params=0,
            train_time_s=0.0, proxy_method="none",
            error=str(e),
        )

    # --- Load dataset ---
    try:
        dataset_obj = get_dataset(dataset_name, data_dir=data_dir)
        train_loader = dataset_obj.train_loader(batch_size=batch_size, seed=seed)
        test_loader = dataset_obj.test_loader(batch_size=256)
        sample_input = dataset_obj.sample_input(batch_size=1)
    except Exception as e:
        return EvalResult(
            candidate_name=candidate_name,
            top1=0.0, top5=0.0, constraints=0,
            nonlinear_ops=0, mult_depth=0, n_params=0,
            train_time_s=0.0, proxy_method="none",
            error=f"Dataset error: {e}",
        )

    # --- Instantiate model ---
    try:
        model = get_model_fn(
            num_classes=dataset_obj.num_classes,
            input_shape=dataset_obj.input_shape,
        )
        if not isinstance(model, nn.Module):
            raise TypeError(f"get_model must return nn.Module, got {type(model)}")
        model.eval()
    except Exception as e:
        return EvalResult(
            candidate_name=candidate_name,
            top1=0.0, top5=0.0, constraints=0,
            nonlinear_ops=0, mult_depth=0, n_params=0,
            train_time_s=0.0, proxy_method="none",
            error=f"Model instantiation failed: {e}",
        )

    # --- ZK proxy (before training — architecture only) ---
    try:
        with _suppress_onnx_noise():
            proxy = compute_proxy(model, sample_input)
        constraints = proxy.constraint_estimate
        nonlinear_ops = proxy.nonlinear_op_count
        mult_depth = proxy.multiplicative_depth if proxy.multiplicative_depth is not None else 0
        n_params = proxy.param_count
        proxy_method = proxy.proxy_method
    except Exception as e:
        return EvalResult(
            candidate_name=candidate_name,
            top1=0.0, top5=0.0, constraints=0,
            nonlinear_ops=0, mult_depth=0, n_params=0,
            train_time_s=0.0, proxy_method="none",
            error=f"ZK proxy failed: {e}",
        )

    # --- Apply architecture-specific inits ---
    _apply_zero_init_residual(model)
    # Whitening init: use first batch of training images as calibration data
    _apply_whitening_conv(model, next(iter(train_loader))[0])
    if use_poly_init:
        _apply_poly_init(model)

    # --- Train ---
    device = get_device(device_str)
    grad_clip = 1.0 if use_poly_init else None
    warmup = 5 if use_poly_init else 0

    trainer = Trainer(
        model=model,
        model_name=candidate_name,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        grad_clip_norm=grad_clip,
        warmup_epochs=warmup,
        lr_schedule="cosine",
        label_smoothing=0.1,
        use_amp=torch.cuda.is_available(),
        use_mixup=True,
        mixup_alpha=0.4,
        use_swa=True,
        swa_start_frac=0.75,
        use_tta=True,
    )

    try:
        t0 = time.perf_counter()
        training_result, top1, top5 = trainer.train(train_loader, test_loader, seed=seed)
        train_time = time.perf_counter() - t0
    except Exception as e:
        return EvalResult(
            candidate_name=candidate_name,
            top1=0.0, top5=0.0,
            constraints=constraints,
            nonlinear_ops=nonlinear_ops,
            mult_depth=mult_depth,
            n_params=n_params,
            train_time_s=0.0,
            proxy_method=proxy_method,
            error=f"Training crashed: {e}\n{traceback.format_exc()[-500:]}",
        )

    if training_result.training_failed:
        return EvalResult(
            candidate_name=candidate_name,
            top1=0.0, top5=0.0,
            constraints=constraints,
            nonlinear_ops=nonlinear_ops,
            mult_depth=mult_depth,
            n_params=n_params,
            train_time_s=train_time,
            proxy_method=proxy_method,
            error=f"Training diverged: {training_result.failure_reason}",
        )

    return EvalResult(
        candidate_name=candidate_name,
        top1=top1,
        top5=top5,
        constraints=constraints,
        nonlinear_ops=nonlinear_ops,
        mult_depth=mult_depth,
        n_params=n_params,
        train_time_s=train_time,
        proxy_method=proxy_method,
        error=None,
    )
