from __future__ import annotations
import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..exceptions import TrainingDivergenceError, WeightLoadError
from ..metrics.accuracy import compute_accuracy
from .agc import agc_clip_gradients

logger = logging.getLogger(__name__)

@dataclass
class TrainingResult:
    epochs_run: int
    final_train_loss: float
    train_time_s: float
    eval_time_s: float
    training_failed: bool
    failure_reason: Optional[str]

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def _is_poly_model(model_name: str) -> bool:
    return model_name.endswith("_poly")

def get_device(device_str: str) -> torch.device:
    """Auto-detect or use specified device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)

class WarmupLRScheduler:
    """Linear LR warmup from lr_start to lr_target over warmup_epochs."""
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int,
                 lr_target: float, lr_start: Optional[float] = None):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.lr_target = lr_target
        self.lr_start = lr_start if lr_start is not None else lr_target * 0.01
        self.current_epoch = 0

    def step(self) -> None:
        if self.current_epoch < self.warmup_epochs:
            progress = self.current_epoch / max(self.warmup_epochs, 1)
            lr = self.lr_start + (self.lr_target - self.lr_start) * progress
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr
        self.current_epoch += 1

    def is_warmup(self) -> bool:
        return self.current_epoch < self.warmup_epochs


def _mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float, device: torch.device):
    """Returns (mixed_x, y_a, y_b, lam)."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def _cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float, device: torch.device):
    """CutMix: paste a random rectangle from a shuffled image. Returns (mixed_x, y_a, y_b, lam)."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    B, C, H, W = x.shape
    idx = torch.randperm(B, device=device)

    cut_ratio = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = max(0, cy - cut_h // 2), min(H, cy + cut_h // 2)
    x1, x2 = max(0, cx - cut_w // 2), min(W, cx + cut_w // 2)

    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)  # actual lambda after clipping
    return mixed, y, y[idx], lam


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        device: torch.device,
        epochs: int,
        batch_size: int,
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        lr_schedule: Optional[str] = "multistep",  # "multistep" | "cosine" | None
        lr_milestones: Optional[list[int]] = None,  # default [100, 150] for multistep
        lr_gamma: float = 0.1,
        label_smoothing: float = 0.0,
        grad_clip_norm: Optional[float] = None,  # None = auto-select
        warmup_epochs: int = 0,  # 0 = no warmup
        use_amp: bool = False,       # Automatic Mixed Precision (requires CUDA)
        use_mixup: bool = False,     # Randomly apply Mixup or CutMix per batch
        mixup_alpha: float = 0.4,    # Mixup alpha; CutMix uses alpha=1.0
        use_swa: bool = False,       # Stochastic Weight Averaging
        swa_start_frac: float = 0.75,  # start SWA at this fraction of training
        use_tta: bool = False,       # Test-time augmentation (orig + hflip average)
        use_agc: bool = False,       # Adaptive Gradient Clipping (NFNet, arXiv:2102.06171)
        agc_clipping: float = 0.01,  # AGC clipping factor (NFNet default)
        log_dir: Optional[str] = None,
    ):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.lr_milestones = lr_milestones or [100, 150]
        self.lr_gamma = lr_gamma
        self.label_smoothing = label_smoothing
        self.use_amp = use_amp and torch.cuda.is_available()
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.use_swa = use_swa
        self.swa_start_frac = swa_start_frac
        self.use_tta = use_tta
        self.use_agc = use_agc
        self.agc_clipping = agc_clipping
        self.log_dir = log_dir

        # Auto-configure poly model settings
        self.is_poly = _is_poly_model(model_name)
        if self.is_poly:
            self.grad_clip_norm = grad_clip_norm if grad_clip_norm is not None else 1.0
            self.warmup_epochs = warmup_epochs if warmup_epochs > 0 else 5
        else:
            self.grad_clip_norm = grad_clip_norm
            self.warmup_epochs = warmup_epochs

        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # LR schedule (applied AFTER warmup, BEFORE swa_start_epoch)
        if lr_schedule == "multistep":
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=self.lr_milestones, gamma=lr_gamma
            )
        elif lr_schedule == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=lr * 1e-3
            )
        else:
            self.lr_scheduler = None

        self.warmup_scheduler = None
        if self.warmup_epochs > 0:
            self.warmup_scheduler = WarmupLRScheduler(
                self.optimizer, self.warmup_epochs, lr
            )

        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

    def _train_epoch(self, loader: DataLoader) -> tuple[float, float]:
        """Returns (avg_loss, grad_norm)."""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        total_grad_norm = 0.0
        n_batches = 0

        for inputs, targets in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            # Randomly choose Mixup or CutMix per batch
            if self.use_mixup:
                if random.random() < 0.5:
                    inputs, targets_a, targets_b, lam = _mixup_data(
                        inputs, targets, self.mixup_alpha, self.device
                    )
                else:
                    inputs, targets_a, targets_b, lam = _cutmix_data(
                        inputs, targets, 1.0, self.device
                    )

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(inputs)
                    if self.use_mixup:
                        loss = lam * self.criterion(outputs, targets_a) + \
                               (1 - lam) * self.criterion(outputs, targets_b)
                    else:
                        loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                if self.use_mixup:
                    loss = lam * self.criterion(outputs, targets_a) + \
                           (1 - lam) * self.criterion(outputs, targets_b)
                else:
                    loss = self.criterion(outputs, targets)

            # Check for NaN
            if torch.isnan(loss):
                raise TrainingDivergenceError(
                    f"NaN loss at batch. Model: {self.model_name}"
                )

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.use_agc:
                    agc_clip_gradients(
                        self.model.parameters(), clipping=self.agc_clipping
                    )
                    grad_norm = 0.0
                elif self.grad_clip_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.grad_clip_norm
                    )
                else:
                    grad_norm = 0.0
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.use_agc:
                    agc_clip_gradients(
                        self.model.parameters(), clipping=self.agc_clipping
                    )
                    grad_norm = 0.0
                elif self.grad_clip_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.grad_clip_norm
                    )
                else:
                    grad_norm = 0.0
                self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            total_grad_norm += float(grad_norm)
            n_batches += 1

        return total_loss / total_samples, total_grad_norm / max(n_batches, 1)

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        seed: int,
    ) -> tuple[TrainingResult, float, float]:
        """
        Train model. Returns (TrainingResult, top1_accuracy, top5_accuracy).
        """
        set_seed(seed)
        self.model.to(self.device)

        # SWA setup (after model is on device)
        swa_model = None
        swa_scheduler = None
        swa_start_epoch = self.epochs  # default: never start
        if self.use_swa and self.epochs >= 10:
            swa_start_epoch = max(1, int(self.epochs * self.swa_start_frac))
            swa_model = torch.optim.swa_utils.AveragedModel(self.model)
            swa_lr = self.lr * 0.05
            swa_scheduler = torch.optim.swa_utils.SWALR(
                self.optimizer, swa_lr=swa_lr, anneal_epochs=5
            )
            logger.info("SWA enabled: start_epoch=%d, swa_lr=%.4f", swa_start_epoch, swa_lr)

        epoch_losses = []
        start_time = time.perf_counter()

        try:
            for epoch in range(self.epochs):
                # Warmup
                if self.warmup_scheduler and self.warmup_scheduler.is_warmup():
                    self.warmup_scheduler.step()

                avg_loss, grad_norm = self._train_epoch(train_loader)
                epoch_losses.append(avg_loss)

                # LR schedule: SWA overrides main scheduler after swa_start_epoch
                in_warmup = self.warmup_scheduler and self.warmup_scheduler.is_warmup()
                if swa_model is not None and epoch >= swa_start_epoch:
                    swa_model.update_parameters(self.model)
                    swa_scheduler.step()
                elif self.lr_scheduler and not in_warmup:
                    self.lr_scheduler.step()

                if self.is_poly:
                    logger.info(
                        "Epoch %d/%d | loss=%.4f | grad_norm=%.4f",
                        epoch + 1, self.epochs, avg_loss, grad_norm,
                    )
                else:
                    logger.info(
                        "Epoch %d/%d | loss=%.4f", epoch + 1, self.epochs, avg_loss
                    )

                # Exploding loss check
                if epoch >= 5 and epoch_losses[0] > 0:
                    baseline = epoch_losses[0]
                    recent_5 = epoch_losses[-5:]
                    if all(loss > 10 * baseline for loss in recent_5):
                        raise TrainingDivergenceError(
                            f"Exploding loss: {recent_5} all > 10 × baseline {baseline:.4f}"
                        )

        except TrainingDivergenceError:
            train_time = time.perf_counter() - start_time
            return (
                TrainingResult(
                    epochs_run=len(epoch_losses),
                    final_train_loss=epoch_losses[-1] if epoch_losses else float("nan"),
                    train_time_s=train_time,
                    eval_time_s=0.0,
                    training_failed=True,
                    failure_reason="divergence",
                ),
                0.0, 0.0,
            )

        train_time = time.perf_counter() - start_time

        # SWA finalization: update BN running stats with training data
        eval_model = self.model
        if swa_model is not None:
            logger.info("Running SWA BN update pass...")
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=self.device)
            eval_model = swa_model

        # Evaluate
        t0 = time.perf_counter()
        eval_model.eval()
        acc = compute_accuracy(eval_model, test_loader, self.device, top_k=(1, 5), tta=self.use_tta)
        eval_time = time.perf_counter() - t0

        return (
            TrainingResult(
                epochs_run=self.epochs,
                final_train_loss=epoch_losses[-1] if epoch_losses else 0.0,
                train_time_s=train_time,
                eval_time_s=eval_time,
                training_failed=False,
                failure_reason=None,
            ),
            acc.top1,
            acc.top5,
        )

    def load_weights(self, weights_path: str) -> None:
        """Load pre-trained weights with shape validation."""
        state = torch.load(weights_path, weights_only=True, map_location=self.device)
        model_state = self.model.state_dict()

        missing = set(model_state.keys()) - set(state.keys())
        extra = set(state.keys()) - set(model_state.keys())
        if missing or extra:
            raise WeightLoadError(
                f"State dict mismatch for {self.model_name}. "
                f"Missing keys: {missing}. Extra keys: {extra}. "
                f"Path: {weights_path}"
            )

        mismatches = []
        for k in model_state:
            if model_state[k].shape != state[k].shape:
                mismatches.append(f"{k}: expected {model_state[k].shape}, got {state[k].shape}")
        if mismatches:
            raise WeightLoadError(
                f"Shape mismatch for {self.model_name}: {mismatches[:5]}"
            )

        self.model.load_state_dict(state)
        logger.info(f"Loaded weights from {weights_path}")
