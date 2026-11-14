"""Standard baseline models for zkml-bench.

All models are registered in the global model registry via @register_model.
"""

from __future__ import annotations

from functools import reduce

import torch
import torch.nn as nn

from .registry import register_model


# ---------------------------------------------------------------------------
# MLP baselines
# ---------------------------------------------------------------------------


@register_model("mlp_small")
class MLPSmall(nn.Module):
    """3-layer MLP: flatten → 512 → ReLU → 256 → ReLU → num_classes.

    Expected CIFAR-10 accuracy ~55%, params ~1.7M.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_features = reduce(lambda a, b: a * b, input_shape)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_model("mlp_large")
class MLPLarge(nn.Module):
    """4-layer MLP: flatten → 1024 → ReLU → 512 → ReLU → 256 → ReLU → num_classes.

    Expected CIFAR-10 accuracy ~60%, params ~4.4M.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_features = reduce(lambda a, b: a * b, input_shape)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# LeNet-5
# ---------------------------------------------------------------------------


@register_model("lenet5")
class LeNet5(nn.Module):
    """LeNet-5 with ReLU activations (not tanh/sigmoid like the original).

    Architecture:
        Conv2d(in_channels, 6, 5) → ReLU → AvgPool2d(2, 2)
        → Conv2d(6, 16, 5) → ReLU → AvgPool2d(2, 2)
        → Flatten → Linear(16*H*W, 120) → ReLU
        → Linear(120, 84) → ReLU → Linear(84, num_classes)

    The fully-connected input size is computed dynamically from input_shape.
    Expected CIFAR-10 accuracy ~75%, params ~62K.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_channels = input_shape[0]

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
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
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# ResNet-20
# ---------------------------------------------------------------------------


class BasicBlock(nn.Module):
    """Basic residual block with optional projection shortcut."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


def _make_layer(
    block_cls,
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    stride: int,
    **block_kwargs,
) -> nn.Sequential:
    """Build a sequential stage of residual blocks."""
    layers = [block_cls(in_channels, out_channels, stride=stride, **block_kwargs)]
    for _ in range(1, num_blocks):
        layers.append(block_cls(out_channels, out_channels, stride=1, **block_kwargs))
    return nn.Sequential(*layers)


@register_model("resnet20")
class ResNet20(nn.Module):
    """ResNet-20 (He et al. 2016) with ReLU + BatchNorm.

    3 stages × 3 blocks each, channels [16, 32, 64].
    Expected CIFAR-10 accuracy ~91%, params ~270K.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_channels = input_shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.layer1 = _make_layer(BasicBlock, 16, 16, num_blocks=3, stride=1)
        self.layer2 = _make_layer(BasicBlock, 16, 32, num_blocks=3, stride=2)
        self.layer3 = _make_layer(BasicBlock, 32, 64, num_blocks=3, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# VGG-11 small
# ---------------------------------------------------------------------------


@register_model("vgg11_small")
class VGG11Small(nn.Module):
    """VGG-11 with channels halved and no BatchNorm.

    Original VGG-11 channels: [64, 128, 256, 256, 512, 512, 512, 512].
    Halved: [32, 64, 128, 128, 256, 256, 256, 256].
    VGG-11 stage layout (conv layers per stage): 1-1-2-2-2, MaxPool after each stage.
    FC input size is computed dynamically from input_shape.
    Expected CIFAR-10 accuracy ~87%, params ~4.5M.
    """

    # (out_channels, 'M' for MaxPool) — VGG-11 config with halved channels
    _CFG = [32, "M", 64, "M", 128, 128, "M", 256, 256, "M", 256, 256, "M"]

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_channels = input_shape[0]
        self.features = self._make_layers(in_channels)

        # Dynamically compute flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.features(dummy)
            self.fc_in = int(out.view(1, -1).shape[1])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_in, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def _make_layers(self, in_channels: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        current_in = in_channels
        for v in self._CFG:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                assert isinstance(v, int)
                layers += [
                    nn.Conv2d(current_in, v, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ]
                current_in = v
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
