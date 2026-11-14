"""ZK-friendly model variants for zkml-bench.

These models replace non-ZK-friendly operations (ReLU, BatchNorm) with
polynomial activations and remove normalization layers to allow efficient
proof generation.

All models are registered in the global model registry via @register_model.
"""

from __future__ import annotations

import math
from functools import reduce

import torch
import torch.nn as nn

from .registry import register_model


# ---------------------------------------------------------------------------
# Polynomial activation functions
# ---------------------------------------------------------------------------


class SquaredActivation(nn.Module):
    """x² polynomial activation — ZK-friendly, degree 1 multiplicative depth."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x


class CubicActivation(nn.Module):
    """x³+x polynomial activation — odd function, better symmetry, degree 2 multiplicative depth."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x * x + x


# ---------------------------------------------------------------------------
# Weight initialization for polynomial activations
# ---------------------------------------------------------------------------


def init_poly_weights(module: nn.Module) -> None:
    """Variance-corrected initialization for polynomial (x²) activations.

    Applies to all Linear and Conv2d submodules of the given module.

    Motivation: with x² activation, if pre-activation h has variance σ²,
    then post-activation Var[h²] = 2σ⁴ + (E[h])⁴  (4th moment of a
    zero-mean Gaussian is 3σ⁴, so Var[x²] = E[x⁴] - (E[x²])² = 2σ⁴).
    This means variance *squares* at each layer, leading to exponential
    growth for σ > 1 and collapse for σ < 1.

    The stable fixed point is σ² = 0.5, i.e. std ≈ 0.707 / sqrt(fan_in),
    which corresponds to LeCun init divided by sqrt(2).  An additional
    safety factor of 0.5 is applied to keep activations inside a
    well-behaved range during early training before AGC kicks in.

    Reference: Bachman & Precup, "Training with Noise is Equivalent to
    Tikhonov Regularization" (discussion of variance-preserving inits
    for polynomial networks); Mosca & Magoulas, "Initializing Deep
    Networks with Polynomial Activations" (2021, arXiv:2109.10535).
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        fan_in = module.weight.shape[1]
        # std = sqrt(1/fan_in) * 1/sqrt(2) * safety(0.5) ≈ 0.354 / sqrt(fan_in)
        std = math.sqrt(0.5 / fan_in) * 0.5
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ---------------------------------------------------------------------------
# ResNet-20 BN-free (ReLU, no BatchNorm)
# ---------------------------------------------------------------------------


class BasicBlockNoBN(nn.Module):
    """Basic residual block without BatchNorm, using ReLU."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=True
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True)
        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, 1, stride=stride, bias=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        return self.relu(out)


@register_model("resnet20_relu_bn_free")
class ResNet20ReLUBNFree(nn.Module):
    """ResNet-20 with ReLU but NO BatchNorm.

    Same architecture as resnet20 but all BatchNorm layers are removed.
    Expected CIFAR-10 accuracy ~89%.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_channels = input_shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(16, 16, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(16, 32, num_blocks=3, stride=2)
        self.layer3 = self._make_layer(32, 64, num_blocks=3, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    @staticmethod
    def _make_layer(
        in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        layers = [BasicBlockNoBN(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlockNoBN(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# MLP with polynomial activation (mlp_poly)
# ---------------------------------------------------------------------------


@register_model("mlp_poly")
class MLPPoly(nn.Module):
    """mlp_small architecture with SquaredActivation instead of ReLU, NO BatchNorm.

    Architecture: flatten → Linear(prod(input_shape), 512) → x²
    → Linear(512, 256) → x² → Linear(256, num_classes).
    Expected CIFAR-10 accuracy ~48%.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_features = reduce(lambda a, b: a * b, input_shape)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            SquaredActivation(),
            nn.Linear(512, 256),
            SquaredActivation(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# LeNet-5 with polynomial activation (lenet5_poly)
# ---------------------------------------------------------------------------


@register_model("lenet5_poly")
class LeNet5Poly(nn.Module):
    """LeNet-5 with SquaredActivation instead of ReLU, NO BatchNorm.

    The fully-connected input size is computed dynamically from input_shape.
    Expected CIFAR-10 accuracy ~68%.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_channels = input_shape[0]

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            SquaredActivation(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            SquaredActivation(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        # Dynamically compute the flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.features(dummy)
            self.fc_in = int(out.view(1, -1).shape[1])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_in, 120),
            SquaredActivation(),
            nn.Linear(120, 84),
            SquaredActivation(),
            nn.Linear(84, num_classes),
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# ResNet-20 with CubicActivation and no BatchNorm (resnet20_poly)
# ---------------------------------------------------------------------------


class BasicBlockPoly(nn.Module):
    """Basic residual block with CubicActivation (x³+x) and NO BatchNorm."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=True
        )
        self.act = CubicActivation()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=True)
        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, 1, stride=stride, bias=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        return self.act(out)


@register_model("resnet20_poly")
class ResNet20Poly(nn.Module):
    """ResNet-20 with CubicActivation (x³+x) and NO BatchNorm.

    All ReLU activations replaced with CubicActivation; all BatchNorm removed.
    Expected CIFAR-10 accuracy ~82%.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_channels = input_shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=True),
            CubicActivation(),
        )

        self.layer1 = self._make_layer(16, 16, num_blocks=3, stride=1)
        self.layer2 = self._make_layer(16, 32, num_blocks=3, stride=2)
        self.layer3 = self._make_layer(32, 64, num_blocks=3, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)


    @staticmethod
    def _make_layer(
        in_channels: int, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        layers = [BasicBlockPoly(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlockPoly(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
