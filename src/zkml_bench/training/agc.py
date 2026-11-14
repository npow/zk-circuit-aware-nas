"""Adaptive Gradient Clipping (AGC) from NFNet (Brock et al., 2021).

Reference: arXiv:2102.06171 — "High-Performance Large-Scale Image Recognition
Without Normalization", Section 4.1.

Algorithm (per parameter p with gradient g):
    param_norm = ||p||_F + eps
    grad_norm  = ||g||_F
    unit_norm  = grad_norm / param_norm
    if unit_norm > clipping:
        g ← g * clipping * param_norm / grad_norm

This clips the *unit-wise* gradient norm rather than the global gradient norm,
making it robust to BN-free architectures where per-layer gradient magnitudes
can vary by orders of magnitude.

Default clipping=0.01 matches the NFNet paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _unitwise_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Compute the Frobenius norm of a parameter tensor (scalar result)."""
    if tensor.ndim <= 1:
        # bias / 1-D params: treat as a vector
        return tensor.norm(2.0)
    else:
        # conv / linear weight: Frobenius norm over all dimensions
        return tensor.norm(2.0)


def agc_clip_gradients(
    parameters,
    clipping: float = 0.01,
    eps: float = 1e-3,
) -> None:
    """Apply Adaptive Gradient Clipping in-place to a sequence of parameters.

    Parameters
    ----------
    parameters:
        Iterable of ``torch.nn.Parameter`` (same interface as
        ``torch.nn.utils.clip_grad_norm_``).
    clipping:
        Maximum allowed ratio of ||g|| / (||p|| + eps). Default 0.01 (NFNet).
    eps:
        Small constant to prevent division by zero and keep the clip active
        even when weights are very small. Default 1e-3.
    """
    params = [p for p in parameters if p.grad is not None]
    for p in params:
        p_norm = _unitwise_norm(p.detach()).clamp_(min=eps)
        g_norm = _unitwise_norm(p.grad.detach())

        # Compute the maximum allowed gradient norm for this parameter
        max_norm = clipping * p_norm

        # Only clip if the gradient norm exceeds the threshold
        if g_norm > max_norm:
            # Scale gradient down: g ← g * (clipping * p_norm) / g_norm
            clip_coef = max_norm / g_norm.clamp(min=1e-6)
            p.grad.detach().mul_(clip_coef)


def has_batch_norm(model: nn.Module) -> bool:
    """Return True if the model contains any BatchNorm layer."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                nn.SyncBatchNorm, nn.LazyBatchNorm1d,
                                nn.LazyBatchNorm2d, nn.LazyBatchNorm3d)):
            return True
    return False


def is_bn_free_by_name(model_name: str) -> bool:
    """Return True for known BN-free model names (fast path without model inspection)."""
    _BN_FREE_NAMES = {
        "vgg11_small",
        "resnet20_relu_bn_free",
        "lenet5_poly",
        "resnet20_poly",
        "mlp_poly",  # no BN, but already has static clip; AGC is safe here too
    }
    return model_name in _BN_FREE_NAMES


def should_use_agc(model: nn.Module, model_name: str) -> bool:
    """Return True if AGC should be used for this model.

    Decision logic (two fast-path checks, structural fallback):
    1. Known BN-free name → True
    2. Model has BatchNorm layers → False  (BN already stabilises gradients)
    3. Structural check: no BN found → True
    """
    if is_bn_free_by_name(model_name):
        return True
    if has_batch_norm(model):
        return False
    # Structural fallback: treat any BN-free model as needing AGC
    return True
