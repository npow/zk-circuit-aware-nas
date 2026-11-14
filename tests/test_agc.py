"""Tests for Adaptive Gradient Clipping (AGC) — agc.py."""

from __future__ import annotations

import torch
import torch.nn as nn

from zkml_bench.training.agc import (
    agc_clip_gradients,
    has_batch_norm,
    should_use_agc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_param_with_grad(shape, grad_scale=1.0):
    """Return a Parameter whose .grad is a random tensor scaled by grad_scale."""
    p = nn.Parameter(torch.randn(*shape))
    p.grad = torch.randn(*shape) * grad_scale
    return p


# ---------------------------------------------------------------------------
# agc_clip_gradients
# ---------------------------------------------------------------------------


def test_agc_no_clip_when_gradient_small():
    """When gradient norm << clipping * param_norm, grad should be unchanged."""
    p = nn.Parameter(torch.ones(4, 4))   # ||p|| = 4.0
    # grad with very small norm
    p.grad = torch.full((4, 4), 1e-6)
    grad_before = p.grad.clone()
    agc_clip_gradients([p], clipping=0.01)
    assert torch.allclose(p.grad, grad_before, atol=1e-8), "Grad should be unchanged"


def test_agc_clips_large_gradient():
    """When gradient norm >> clipping * param_norm, grad norm should be reduced."""
    p = nn.Parameter(torch.ones(4, 4))   # ||p|| ~ 4.0
    # Make grad very large → unit norm >> 0.01
    p.grad = torch.full((4, 4), 10.0)
    grad_before_norm = p.grad.norm().item()
    agc_clip_gradients([p], clipping=0.01, eps=1e-3)
    grad_after_norm = p.grad.norm().item()
    assert grad_after_norm < grad_before_norm, "Large gradient should be clipped"


def test_agc_clipped_norm_close_to_target():
    """After clipping, the unit-norm of the gradient should be <= clipping + small eps."""
    p = nn.Parameter(torch.ones(8, 8))
    p.grad = torch.full((8, 8), 5.0)  # very large
    clipping = 0.01
    eps = 1e-3
    agc_clip_gradients([p], clipping=clipping, eps=eps)
    p_norm = p.detach().norm().item() + eps
    g_norm = p.grad.norm().item()
    unit_norm = g_norm / p_norm
    # After clipping the unit norm should be <= clipping (within fp tolerance)
    assert unit_norm <= clipping * 1.01, f"unit_norm={unit_norm:.6f} > clipping={clipping}"


def test_agc_skips_params_without_grad():
    """Parameters without .grad set should not cause errors."""
    p_no_grad = nn.Parameter(torch.ones(4))
    p_no_grad.grad = None
    # Should complete without raising
    agc_clip_gradients([p_no_grad], clipping=0.01)


def test_agc_handles_1d_bias():
    """1-D bias parameters should be handled without errors."""
    p = nn.Parameter(torch.ones(16))
    p.grad = torch.full((16,), 100.0)
    agc_clip_gradients([p], clipping=0.01)
    # Just verify it didn't raise and gradient was clipped
    assert p.grad.norm().item() < 100.0 * 16 ** 0.5


def test_agc_empty_param_list():
    """Empty parameter list should be a no-op."""
    agc_clip_gradients([], clipping=0.01)  # must not raise


def test_agc_multiple_params():
    """Each parameter is clipped independently."""
    p1 = nn.Parameter(torch.ones(4, 4))
    p1.grad = torch.full((4, 4), 100.0)  # will be clipped
    p2 = nn.Parameter(torch.ones(4, 4))
    p2.grad = torch.full((4, 4), 1e-8)   # will NOT be clipped

    g2_before = p2.grad.clone()
    agc_clip_gradients([p1, p2], clipping=0.01)

    assert p1.grad.norm().item() < 100.0 * 4, "p1 grad should be clipped"
    assert torch.allclose(p2.grad, g2_before, atol=1e-12), "p2 grad should be unchanged"


# ---------------------------------------------------------------------------
# has_batch_norm
# ---------------------------------------------------------------------------


def test_has_batch_norm_true():
    model = nn.Sequential(nn.Linear(10, 10), nn.BatchNorm1d(10))
    assert has_batch_norm(model) is True


def test_has_batch_norm_2d():
    model = nn.Sequential(nn.Conv2d(3, 16, 3), nn.BatchNorm2d(16))
    assert has_batch_norm(model) is True


def test_has_batch_norm_false():
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
    assert has_batch_norm(model) is False


def test_has_batch_norm_nested():
    """BN inside a nested module should still be detected."""
    inner = nn.Sequential(nn.BatchNorm2d(8))
    outer = nn.Sequential(nn.Conv2d(3, 8, 3), inner)
    assert has_batch_norm(outer) is True


# ---------------------------------------------------------------------------
# should_use_agc
# ---------------------------------------------------------------------------


def test_should_use_agc_known_bn_free():
    model = nn.Sequential(nn.Linear(10, 10))  # structural: no BN
    assert should_use_agc(model, "vgg11_small") is True
    assert should_use_agc(model, "resnet20_relu_bn_free") is True
    assert should_use_agc(model, "lenet5_poly") is True


def test_should_use_agc_bn_model_by_name():
    """A model with BN should return False even if name is not recognised."""
    model = nn.Sequential(nn.Linear(10, 10), nn.BatchNorm1d(10))
    # Name is not in BN-free list, and model HAS BatchNorm
    assert should_use_agc(model, "resnet20") is False


def test_should_use_agc_structural_fallback():
    """Unknown name + no BN layers → should return True (safe fallback)."""
    model = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
    assert should_use_agc(model, "some_unknown_bn_free_model") is True


# ---------------------------------------------------------------------------
# Integration: AGC inside Trainer
# ---------------------------------------------------------------------------


def test_trainer_agc_flag():
    """Trainer stores use_agc and agc_clipping correctly."""
    from zkml_bench.training.trainer import Trainer

    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    trainer = Trainer(
        model, "vgg11_small", torch.device("cpu"),
        epochs=1, batch_size=32, lr=0.01,
        use_agc=True, agc_clipping=0.01,
    )
    assert trainer.use_agc is True
    assert trainer.agc_clipping == 0.01


def test_trainer_agc_runs_without_error():
    """Trainer with use_agc=True should complete a short training run."""
    from torch.utils.data import DataLoader, TensorDataset
    from zkml_bench.training.trainer import Trainer

    torch.manual_seed(0)
    x = torch.randn(64, 1, 28, 28)
    y = torch.randint(0, 10, (64,))
    loader = DataLoader(TensorDataset(x, y), batch_size=32)

    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    trainer = Trainer(
        model, "vgg11_small", torch.device("cpu"),
        epochs=2, batch_size=32, lr=0.01,
        use_agc=True, agc_clipping=0.01,
    )
    result, top1, top5 = trainer.train(loader, loader, seed=42)
    assert not result.training_failed
    assert result.epochs_run == 2
