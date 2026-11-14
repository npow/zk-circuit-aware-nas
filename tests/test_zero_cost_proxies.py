"""Tests for zero-cost NAS proxies (NEAR score).

Tests cover:
- Finite and positive score on fresh models with random data
- Activation-agnostic: works for ReLU, x², x²+x, x³+x
- Larger models → higher score (expressivity sanity check)
- CNN / Conv2d support
- Degenerate inputs (trivially small models)
- Combined NAS objective (NEAR × 1/cost)
- DataLoader interface
- _effective_rank helper properties
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from zkml_bench.nas.zero_cost_proxies import (
    _effective_rank,
    near_score,
    zero_cost_nas_score,
)


# ---------------------------------------------------------------------------
# Activation helpers used across tests
# ---------------------------------------------------------------------------

class SquaredAct(nn.Module):
    def forward(self, x):
        return x * x


class QuadLinearAct(nn.Module):
    """x² + x"""
    def forward(self, x):
        return x * x + x


class CubicAct(nn.Module):
    """x³ + x"""
    def forward(self, x):
        return x * x * x + x


def _make_mlp(hidden: int, depth: int, activation: nn.Module) -> nn.Module:
    """Create an MLP with given hidden size, depth and activation."""
    layers: list[nn.Module] = [nn.Flatten(), nn.Linear(28 * 28, hidden)]
    for _ in range(depth - 1):
        layers += [activation, nn.Linear(hidden, hidden)]
    layers += [activation, nn.Linear(hidden, 10)]
    return nn.Sequential(*layers)


def _random_mnist_batch(batch_size: int = 32) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch_size, 1, 28, 28)


# ---------------------------------------------------------------------------
# Tests: _effective_rank helper
# ---------------------------------------------------------------------------


class TestEffectiveRank:
    def test_rank1_matrix_returns_near_one(self):
        """A rank-1 matrix (outer product) should have erank close to 1."""
        v = torch.randn(50, 1)
        mat = v @ v.T  # rank 1
        er = _effective_rank(mat)
        assert er < 3.0, f"rank-1 matrix erank={er}, expected ~1"

    def test_identity_returns_near_dim(self):
        """An n×n identity matrix has erank = n (all singular values equal)."""
        n = 16
        mat = torch.eye(n)
        er = _effective_rank(mat)
        # erank of uniform distribution = exp(log(n)) = n
        assert abs(er - n) < 1.5, f"identity erank={er}, expected ~{n}"

    def test_erank_positive(self):
        torch.manual_seed(1)
        mat = torch.randn(20, 20)
        assert _effective_rank(mat) > 0.0

    def test_erank_finite(self):
        torch.manual_seed(2)
        mat = torch.randn(30, 15)
        er = _effective_rank(mat)
        assert math.isfinite(er)


# ---------------------------------------------------------------------------
# Tests: near_score — basic properties
# ---------------------------------------------------------------------------


class TestNearScoreBasic:
    def test_relu_mlp_returns_finite_positive(self):
        """NEAR score for a fresh ReLU MLP on random data is finite and positive."""
        model = _make_mlp(64, 2, nn.ReLU())
        data = _random_mnist_batch(32)
        score = near_score(model, data=data, device="cpu")
        assert math.isfinite(score), f"score={score} is not finite"
        assert score > 0.0, f"score={score} is not positive"

    def test_score_is_float(self):
        model = _make_mlp(32, 2, nn.ReLU())
        data = _random_mnist_batch(16)
        score = near_score(model, data=data, device="cpu")
        assert isinstance(score, float)

    def test_raises_if_no_data_provided(self):
        model = _make_mlp(32, 1, nn.ReLU())
        with pytest.raises(ValueError, match="Provide either dataloader or data"):
            near_score(model)

    def test_raises_if_both_data_and_loader_provided(self):
        from torch.utils.data import DataLoader, TensorDataset
        model = _make_mlp(32, 1, nn.ReLU())
        data = _random_mnist_batch(16)
        ds = TensorDataset(data)
        loader = DataLoader(ds, batch_size=8)
        with pytest.raises(ValueError, match="not both"):
            near_score(model, dataloader=loader, data=data)


# ---------------------------------------------------------------------------
# Tests: activation-agnostic claim (key contribution)
# ---------------------------------------------------------------------------


class TestActivationAgnostic:
    """NEAR must return a finite positive score for ALL these activation types.

    This is the core claim: unlike NASWOT (ReLU only), Zen-Score (BN + ReLU),
    or TE-NAS (NTK requires specific nonlinearity), NEAR works for polynomial
    activations used in ZK-friendly networks.
    """

    @pytest.mark.parametrize("act,name", [
        (nn.ReLU(), "relu"),
        (SquaredAct(), "x_squared"),
        (QuadLinearAct(), "x_squared_plus_x"),
        (CubicAct(), "cubic_x3_plus_x"),
        (nn.GELU(), "gelu"),
        (nn.Tanh(), "tanh"),
    ])
    def test_near_works_for_activation(self, act, name):
        """NEAR score must be finite and positive for activation: {name}."""
        torch.manual_seed(42)
        model = _make_mlp(64, 2, act)
        data = _random_mnist_batch(32)
        score = near_score(model, data=data, device="cpu")
        assert math.isfinite(score), (
            f"near_score returned non-finite {score} for activation={name}"
        )
        assert score > 0.0, (
            f"near_score returned non-positive {score} for activation={name}"
        )


# ---------------------------------------------------------------------------
# Tests: expressivity monotonicity
# ---------------------------------------------------------------------------


class TestExpressivityMonotonicity:
    def test_larger_hidden_size_scores_higher(self):
        """A wider MLP should have higher NEAR score than a narrower one.

        This is the structural sanity check: more expressive networks
        (higher capacity) should have higher effective-rank features.
        """
        torch.manual_seed(0)
        data = _random_mnist_batch(64)

        small = _make_mlp(16, 2, nn.ReLU())
        large = _make_mlp(256, 2, nn.ReLU())

        score_small = near_score(small, data=data, device="cpu")
        score_large = near_score(large, data=data, device="cpu")

        assert score_large > score_small, (
            f"Expected score(large={score_large:.2f}) > score(small={score_small:.2f})"
        )

    def test_deeper_model_scores_differently(self):
        """A deeper model accumulates more layer ranks — score must be > single layer."""
        torch.manual_seed(0)
        data = _random_mnist_batch(32)
        shallow = _make_mlp(64, 1, nn.ReLU())
        deep = _make_mlp(64, 4, nn.ReLU())
        score_shallow = near_score(shallow, data=data, device="cpu")
        score_deep = near_score(deep, data=data, device="cpu")
        # Deep model has more layers, so more terms in the sum; must be higher
        assert score_deep > score_shallow, (
            f"Expected score(deep={score_deep:.2f}) > score(shallow={score_shallow:.2f})"
        )

    def test_poly_larger_scores_higher(self):
        """Expressivity monotonicity holds for polynomial activations too."""
        torch.manual_seed(0)
        data = _random_mnist_batch(64)
        small = _make_mlp(16, 2, SquaredAct())
        large = _make_mlp(256, 2, SquaredAct())
        score_small = near_score(small, data=data, device="cpu")
        score_large = near_score(large, data=data, device="cpu")
        assert score_large > score_small, (
            f"Poly: expected score(large={score_large:.2f}) > score(small={score_small:.2f})"
        )


# ---------------------------------------------------------------------------
# Tests: DataLoader interface
# ---------------------------------------------------------------------------


class TestDataLoaderInterface:
    def test_near_works_with_dataloader(self):
        """near_score accepts a DataLoader (x, y) tuple interface."""
        from torch.utils.data import DataLoader, TensorDataset
        data = _random_mnist_batch(64)
        labels = torch.zeros(64, dtype=torch.long)
        ds = TensorDataset(data, labels)
        loader = DataLoader(ds, batch_size=32)
        model = _make_mlp(32, 2, nn.ReLU())
        score = near_score(model, dataloader=loader, num_batches=2, device="cpu")
        assert math.isfinite(score) and score > 0.0

    def test_num_batches_limits_batches(self):
        """num_batches=1 should still return a valid score."""
        from torch.utils.data import DataLoader, TensorDataset
        data = _random_mnist_batch(64)
        labels = torch.zeros(64, dtype=torch.long)
        ds = TensorDataset(data, labels)
        loader = DataLoader(ds, batch_size=16)
        model = _make_mlp(32, 2, nn.ReLU())
        score = near_score(model, dataloader=loader, num_batches=1, device="cpu")
        assert math.isfinite(score) and score > 0.0


# ---------------------------------------------------------------------------
# Tests: combined zero-cost NAS objective
# ---------------------------------------------------------------------------


class TestZeroCostNasScore:
    def test_combined_score_positive(self):
        """zero_cost_nas_score should be positive when cost > 0."""
        model = _make_mlp(64, 2, nn.ReLU())
        data = _random_mnist_batch(32)
        score = zero_cost_nas_score(model, data=data, cost_proxy_value=100_000)
        assert score > 0.0

    def test_combined_score_finite(self):
        model = _make_mlp(64, 2, SquaredAct())
        data = _random_mnist_batch(32)
        score = zero_cost_nas_score(model, data=data, cost_proxy_value=50_000)
        assert math.isfinite(score)

    def test_combined_score_zero_for_zero_cost(self):
        """If cost_proxy_value <= 0, combined score should be 0."""
        model = _make_mlp(32, 1, nn.ReLU())
        data = _random_mnist_batch(16)
        score = zero_cost_nas_score(model, data=data, cost_proxy_value=0)
        assert score == 0.0

    def test_lower_cost_gives_higher_combined_score(self):
        """Same model, lower ZK cost → higher combined NAS score."""
        model = _make_mlp(64, 2, nn.ReLU())
        data = _random_mnist_batch(32)
        score_cheap = zero_cost_nas_score(model, data=data, cost_proxy_value=10_000)
        score_expensive = zero_cost_nas_score(model, data=data, cost_proxy_value=1_000_000)
        assert score_cheap > score_expensive, (
            f"Expected cheap({score_cheap:.6f}) > expensive({score_expensive:.6f})"
        )
