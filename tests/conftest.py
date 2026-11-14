"""Shared pytest fixtures for zkml-bench tests."""

import tempfile

import pytest
import torch
import torch.nn as nn


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def simple_mlp():
    """A simple 2-layer MLP for testing."""

    class SimpleMLP(nn.Module):
        def __init__(self, num_classes=10, input_shape=(1, 28, 28)):
            super().__init__()
            in_features = 1
            for s in input_shape:
                in_features *= s
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):
            return self.net(x)

    return SimpleMLP()


@pytest.fixture
def simple_poly_mlp():
    """A simple 2-layer poly MLP (x²) for testing."""

    class PolyMLP(nn.Module):
        def __init__(self, num_classes=10, input_shape=(1, 28, 28)):
            super().__init__()
            in_features = 1
            for s in input_shape:
                in_features *= s
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(in_features, 64)
            self.fc2 = nn.Linear(64, num_classes)

        def forward(self, x):
            x = self.flatten(x)
            x = self.fc1(x)
            x = x * x  # x² polynomial activation
            return self.fc2(x)

    return PolyMLP()


@pytest.fixture
def mnist_sample():
    """Deterministic MNIST-shaped input tensor (B=1, C=1, H=28, W=28)."""
    torch.manual_seed(0)
    return torch.randn(1, 1, 28, 28)


@pytest.fixture
def cifar_sample():
    """Deterministic CIFAR-shaped input tensor (B=1, C=3, H=32, W=32)."""
    torch.manual_seed(0)
    return torch.randn(1, 3, 32, 32)
