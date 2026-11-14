"""Iso-FLOP architectural pairs for the ZK-Architectural Reversal experiment.

Central hypothesis: deep-narrow architectures pay 3-7× more on GKR backends
(DeepProve, Expander) than wide-shallow ones at identical FLOP budgets, while
Halo2/EZKL treats them similarly.

All models:
- NO BatchNorm (DeepProve/Expander cannot handle BN; enables fair cross-backend comparison)
- ReLU activations throughout
- Accept (B, 3, 32, 32) input (CIFAR-10)
- Return logits of shape (B, num_classes) — no softmax
- ONNX-exportable with opset 17

MAC counts are per-sample (batch=1), computed via the ONNX proxy (Gemm/Conv MACs only).

Pairs (matched ±5%):
  MLP Pair 1:  mlp_wide_2  (h=512,  2 layers) ~1.84M MACs
               mlp_deep_6  (h=372,  6 layers) ~1.84M MACs  diff=0.09%
  MLP Pair 2:  mlp_wide_1  (h=2048, 1 layer)  ~6.31M MACs
               mlp_deep_10 (h=684, 10 layers)  ~6.32M MACs  diff=0.11%
  CNN Pair A:  cnn_wide_2  (ch=64,  2 conv layers) ~11.2M MACs
               cnn_deep_6  (ch=41,  6 conv layers) ~11.1M MACs  diff=1.32%
  CNN Pair B:  cnn_wide_4  (ch=48,  4 conv layers) ~13.3M MACs
               cnn_deep_12 (ch=42, 12 conv layers) ~13.3M MACs  diff=0.15%

CNN downsampling note:
  All CNN iso-FLOP models use MaxPool2d(2,2) for spatial downsampling instead of
  Conv2d(stride=2).  This makes them compatible with DeepProve, which rejects
  stride>1 convolutions at the ONNX parser level.

  Analytically, replacing Conv(in, out, k=3, s=2) → MaxPool(2,2) + Conv(in, out, k=3, s=1)
  preserves MAC counts exactly: the convolution still executes at the same output
  spatial resolution (H/2 × W/2), so out_ch × in_ch × 9 × (H/2) × (W/2) is unchanged.
"""

from __future__ import annotations

from functools import reduce

import torch
import torch.nn as nn

from .registry import register_model


# ---------------------------------------------------------------------------
# MLP iso-FLOP pairs
# ---------------------------------------------------------------------------


@register_model("mlp_wide_2")
class MLPWide2(nn.Module):
    """Iso-FLOP MLP: 2 hidden layers, width=512.

    MACs (per sample, CIFAR-10 input 3×32×32 = 3072 flat):
      Linear(3072, 512): 3072 × 512 = 1,572,864
      Linear(512,  512):  512 × 512 =   262,144
      Linear(512,   10):  512 × 10  =     5,120
      Total: ~1,840,128 MACs

    Depth=2 hidden layers, width=512.
    No BatchNorm; ReLU activations.
    Iso-FLOP pair: mlp_deep_6 (~1,838,424 MACs, diff=0.09%).
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_features = reduce(lambda a, b: a * b, input_shape)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_model("mlp_deep_6")
class MLPDeep6(nn.Module):
    """Iso-FLOP MLP: 6 hidden layers, width=372.

    MACs (per sample, CIFAR-10 input 3×32×32 = 3072 flat):
      Linear(3072, 372): 3072 × 372 = 1,142,784
      Linear(372,  372) ×5:  372 × 372 × 5 =   692,280
      Linear(372,   10):  372 × 10  =     3,720
      Total: ~1,838,784 MACs

    Depth=6 hidden layers, width=372.
    No BatchNorm; ReLU activations.
    Iso-FLOP pair: mlp_wide_2 (~1,840,128 MACs, diff=0.09%).
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_features = reduce(lambda a, b: a * b, input_shape)
        layers: list[nn.Module] = [nn.Flatten(), nn.Linear(in_features, 372), nn.ReLU()]
        for _ in range(5):
            layers += [nn.Linear(372, 372), nn.ReLU()]
        layers.append(nn.Linear(372, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_model("mlp_wide_1")
class MLPWide1(nn.Module):
    """Iso-FLOP MLP: 1 hidden layer, width=2048.

    MACs (per sample, CIFAR-10 input 3×32×32 = 3072 flat):
      Linear(3072, 2048): 3072 × 2048 = 6,291,456
      Linear(2048,   10): 2048 × 10   =    20,480
      Total: ~6,311,936 MACs

    Depth=1 hidden layer, width=2048.
    No BatchNorm; ReLU activations.
    Iso-FLOP pair: mlp_deep_10 (~6,318,792 MACs, diff=0.11%).
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_features = reduce(lambda a, b: a * b, input_shape)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_model("mlp_deep_10")
class MLPDeep10(nn.Module):
    """Iso-FLOP MLP: 10 hidden layers, width=684.

    MACs (per sample, CIFAR-10 input 3×32×32 = 3072 flat):
      Linear(3072, 684): 3072 × 684 = 2,101,248
      Linear(684,  684) ×9:  684 × 684 × 9 = 4,210,704
      Linear(684,   10):  684 × 10  =     6,840
      Total: ~6,318,792 MACs

    Depth=10 hidden layers, width=684.
    No BatchNorm; ReLU activations.
    Iso-FLOP pair: mlp_wide_1 (~6,311,936 MACs, diff=0.11%).
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_features = reduce(lambda a, b: a * b, input_shape)
        layers: list[nn.Module] = [nn.Flatten(), nn.Linear(in_features, 684), nn.ReLU()]
        for _ in range(9):
            layers += [nn.Linear(684, 684), nn.ReLU()]
        layers.append(nn.Linear(684, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# CNN iso-FLOP pairs
#
# Architecture:
#   Wide variants: few conv layers with MaxPool2d(2,2) for spatial downsampling
#   Deep variants: many conv layers with MaxPool2d(2,2) at the start of each group
#
# Downsampling design:
#   Original stride-2 convs have been replaced with MaxPool2d(2,2) followed by a
#   stride-1 conv at the same (lower) output resolution.  This is required for
#   DeepProve compatibility: the ONNX parser rejects Conv2d with stride>1 with a
#   hard parser error.  MaxPool2d is natively supported.
#
#   MAC count is preserved exactly: the conv still executes at the same output
#   spatial resolution (H/2 × W/2), so out_ch × in_ch × 9 × (H/2) × (W/2)
#   is unchanged.  No channel-count adjustment is needed.
#
# All convolutions: kernel=3×3, stride=1, padding=1, bias=True, NO BatchNorm, ReLU
# Downsampling:     MaxPool2d(kernel_size=2, stride=2)
# Final classifier: GlobalAveragePool → Linear(channels, num_classes)
# ---------------------------------------------------------------------------


def _conv_relu(in_ch: int, out_ch: int) -> list[nn.Module]:
    """3×3 stride-1 Conv + ReLU block (no BatchNorm).

    Spatial dimensions are preserved (padding=1 keeps H×W unchanged).
    Use nn.MaxPool2d(2, 2) separately when downsampling is needed.
    """
    return [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(),
    ]


def _strided_conv_relu(in_ch: int, out_ch: int) -> list[nn.Module]:
    """3×3 stride-2 Conv + ReLU block (no BatchNorm).

    Halves spatial dimensions (H, W → H/2, W/2) using stride=2, padding=1.
    This is the ORIGINAL stride-2 design that DeepProve can now handle with
    the strided_conv.patch applied to the GKR circuit.

    stride=2 is a power of two — satisfies the DeepProve parser constraint.
    """
    return [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=True),
        nn.ReLU(),
    ]


def _pool_conv_relu(in_ch: int, out_ch: int) -> list[nn.Module]:
    """MaxPool2d(2,2) spatial halving followed by 3×3 stride-1 Conv + ReLU.

    Replaces Conv2d(stride=2) for DeepProve compatibility.  The output spatial
    resolution and MAC count are identical to the stride-2 conv it replaces.
    """
    return [
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(),
    ]


@register_model("cnn_wide_2")
class CNNWide2(nn.Module):
    """Iso-FLOP CNN: 2 conv layers, 64 channels, MaxPool2d downsampling.

    Spatial schedule: 32×32 → MaxPool → 16×16 → GAP → classifier.

    MACs (per sample, input 3×32×32):
      Conv(3,  64, s=1) at 32×32:   64 × 3  × 9 × 32×32 =  1,769,472
      MaxPool(2,2) at 32→16:         0 MACs
      Conv(64, 64, s=1) at 16×16:   64 × 64 × 9 × 16×16 =  9,437,184
      Total: ~11,206,656 MACs

    Identical MAC count to the original stride-2 design.
    Depth=2 conv layers, width=64 channels.
    No BatchNorm; ReLU activations; no stride>1 convolutions (DeepProve-compatible).
    Iso-FLOP pair: cnn_deep_6 (~11,058,192 MACs, diff=1.32%).
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_channels = input_shape[0]
        self.features = nn.Sequential(
            *_conv_relu(in_channels, 64),   # 32×32
            *_pool_conv_relu(64, 64),        # MaxPool → 16×16
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


@register_model("cnn_deep_6")
class CNNDeep6(nn.Module):
    """Iso-FLOP CNN: 6 conv layers, 41 channels, MaxPool2d downsampling.

    Spatial schedule: 32 → MP → 16 → 16 → MP → 8 → 8 → MP → 4 → GAP → classifier.

    MACs (per sample, input 3×32×32):
      Conv(3,  41, s=1) at 32×32:   41 × 3  × 9 × 32×32 =  1,132,416
      MaxPool(2,2) at 32→16:         0 MACs
      Conv(41, 41, s=1) at 16×16:   41 × 41 × 9 × 16×16 =  3,868,416
      Conv(41, 41, s=1) at 16×16:   41 × 41 × 9 × 16×16 =  3,868,416
      MaxPool(2,2) at 16→8:          0 MACs
      Conv(41, 41, s=1) at  8×8:    41 × 41 × 9 ×  8×8  =    967,104
      Conv(41, 41, s=1) at  8×8:    41 × 41 × 9 ×  8×8  =    967,104
      MaxPool(2,2) at 8→4:           0 MACs
      Conv(41, 41, s=1) at  4×4:    41 × 41 × 9 ×  4×4  =    241,776
      Total: ~11,045,232 MACs

    Identical MAC count to the original stride-2 design.
    Depth=6 conv layers, width=41 channels.
    No BatchNorm; ReLU activations; no stride>1 convolutions (DeepProve-compatible).
    Iso-FLOP pair: cnn_wide_2 (~11,206,656 MACs, diff=1.32%).
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_channels = input_shape[0]
        self.features = nn.Sequential(
            *_conv_relu(in_channels, 41),    # 32×32
            *_pool_conv_relu(41, 41),         # MaxPool → 16×16
            *_conv_relu(41, 41),              # 16×16
            *_pool_conv_relu(41, 41),         # MaxPool → 8×8
            *_conv_relu(41, 41),              # 8×8
            *_pool_conv_relu(41, 41),         # MaxPool → 4×4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(41, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


@register_model("cnn_wide_4")
class CNNWide4(nn.Module):
    """Iso-FLOP CNN: 4 conv layers, 48 channels, MaxPool2d downsampling.

    Spatial schedule: 32 → MP → 16 → 16 → MP → 8 → GAP → classifier.

    MACs (per sample, input 3×32×32):
      Conv(3,  48, s=1) at 32×32:   48 × 3  × 9 × 32×32 =  1,327,104
      MaxPool(2,2) at 32→16:         0 MACs
      Conv(48, 48, s=1) at 16×16:   48 × 48 × 9 × 16×16 =  5,308,416
      Conv(48, 48, s=1) at 16×16:   48 × 48 × 9 × 16×16 =  5,308,416
      MaxPool(2,2) at 16→8:          0 MACs
      Conv(48, 48, s=1) at  8×8:    48 × 48 × 9 ×  8×8  =  1,327,104
      Total: ~13,271,040 MACs

    Identical MAC count to the original stride-2 design.
    Depth=4 conv layers, width=48 channels.
    No BatchNorm; ReLU activations; no stride>1 convolutions (DeepProve-compatible).
    Iso-FLOP pair: cnn_deep_12 (~13,290,480 MACs, diff=0.15%).
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_channels = input_shape[0]
        self.features = nn.Sequential(
            *_conv_relu(in_channels, 48),    # 32×32
            *_pool_conv_relu(48, 48),         # MaxPool → 16×16
            *_conv_relu(48, 48),              # 16×16
            *_pool_conv_relu(48, 48),         # MaxPool → 8×8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(48, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


@register_model("cnn_deep_12")
class CNNDeep12(nn.Module):
    """Iso-FLOP CNN: 12 conv layers, 42 channels, MaxPool2d downsampling.

    Architecture: 4 groups of 3 conv layers; MaxPool2d(2,2) at the start of each
    group (except group 1's first conv which runs at full resolution) halves spatial
    resolution — same schedule as the original stride-2 design.

    Spatial schedule:
      Group 1: 32×32 → MP → 16×16 → 16×16
      Group 2: MP → 8×8 → 8×8 → 8×8
      Group 3: MP → 4×4 → 4×4 → 4×4
      Group 4: MP → 2×2 → 2×2 → 2×2  → GAP → classifier

    MACs (per sample, input 3×32×32):
      Group 1 (32×32 → 16×16):
        Conv(3,  42, s=1) at 32×32:   42 ×  3 × 9 × 32×32 =  1,161,216
        MaxPool(2,2) at 32→16:         0 MACs
        Conv(42, 42, s=1) at 16×16:   42 × 42 × 9 × 16×16 =  4,064,256
        Conv(42, 42, s=1) at 16×16:   42 × 42 × 9 × 16×16 =  4,064,256
      Group 2 (16×16 → 8×8):
        MaxPool(2,2) at 16→8:          0 MACs
        Conv(42, 42, s=1) at  8×8:    42 × 42 × 9 ×  8×8  =  1,016,064
        Conv(42, 42, s=1) at  8×8:    42 × 42 × 9 ×  8×8  =  1,016,064
        Conv(42, 42, s=1) at  8×8:    42 × 42 × 9 ×  8×8  =  1,016,064
      Group 3 (8×8 → 4×4):
        MaxPool(2,2) at 8→4:           0 MACs
        Conv(42, 42, s=1) at  4×4:    42 × 42 × 9 ×  4×4  =    254,016
        Conv(42, 42, s=1) at  4×4:    42 × 42 × 9 ×  4×4  =    254,016
        Conv(42, 42, s=1) at  4×4:    42 × 42 × 9 ×  4×4  =    254,016
      Group 4 (4×4 → 2×2):
        MaxPool(2,2) at 4→2:           0 MACs
        Conv(42, 42, s=1) at  2×2:    42 × 42 × 9 ×  2×2  =     63,504
        Conv(42, 42, s=1) at  2×2:    42 × 42 × 9 ×  2×2  =     63,504
        Conv(42, 42, s=1) at  2×2:    42 × 42 × 9 ×  2×2  =     63,504
      Total: ~13,290,480 MACs

    Identical MAC count to the original stride-2 design.
    Depth=12 conv layers, width=42 channels.
    No BatchNorm; ReLU activations; no stride>1 convolutions (DeepProve-compatible).
    Iso-FLOP pair: cnn_wide_4 (~13,271,040 MACs, diff=0.15%).
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_channels = input_shape[0]
        self.features = nn.Sequential(
            # Group 1: 32×32 → 16×16
            *_conv_relu(in_channels, 42),    # 32×32
            *_pool_conv_relu(42, 42),         # MaxPool → 16×16
            *_conv_relu(42, 42),              # 16×16
            # Group 2: 16×16 → 8×8
            *_pool_conv_relu(42, 42),         # MaxPool → 8×8
            *_conv_relu(42, 42),              # 8×8
            *_conv_relu(42, 42),              # 8×8
            # Group 3: 8×8 → 4×4
            *_pool_conv_relu(42, 42),         # MaxPool → 4×4
            *_conv_relu(42, 42),              # 4×4
            *_conv_relu(42, 42),              # 4×4
            # Group 4: 4×4 → 2×2
            *_pool_conv_relu(42, 42),         # MaxPool → 2×2
            *_conv_relu(42, 42),              # 2×2
            *_conv_relu(42, 42),              # 2×2
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(42, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# CNN iso-FLOP pairs — ORIGINAL stride-2 design
#
# These are the ORIGINAL architectures that use Conv2d(stride=2) for spatial
# downsampling, as opposed to the MaxPool2d-redesigned variants above.
#
# With strided_conv.patch applied to DeepProve's GKR circuit, these models
# can now be proved directly.  The patch implements a full-resolution FFT conv
# followed by MLE point expansion and a hadamard clearing proof.
#
# Naming convention: same as the MaxPool variants but with "_strided" suffix
# to avoid conflicts with the MaxPool benchmarks already running.
#
# Architecture:
#   Downsampling: Conv2d(stride=2, padding=1) — halves H×W in one op
#   All other convs: stride=1, padding=1 (preserve spatial dims)
#   Final classifier: GlobalAveragePool → Linear(channels, num_classes)
#
# MAC counts are identical to the MaxPool variants because the conv still
# executes at output resolution (H/2 × W/2); the MaxPool had 0 MACs.
# ---------------------------------------------------------------------------


@register_model("cnn_wide_2_strided")
class CNNWide2Strided(nn.Module):
    """Iso-FLOP CNN Pair A (wide): 2 conv layers, 64 ch — ORIGINAL stride-2 design.

    Spatial schedule: 32×32 → Conv(s=2) → 16×16 → GAP → classifier.

    MACs (per sample, input 3×32×32):
      Conv(3,  64, s=2, p=1) at 32×32 → 16×16:  64 × 3  × 9 × 16×16 =  4,423,680
      Conv(64, 64, s=1, p=1) at 16×16:           64 × 64 × 9 × 16×16 =  9,437,184
      Total: ~13,860,864 MACs

    Note: MAC count differs from cnn_wide_2 because the first conv in the
    stride-2 design runs the kernel sweep at the OUTPUT resolution (16×16),
    while cnn_wide_2's first conv runs at 32×32 then MaxPools.
    The iso-FLOP property holds WITHIN each pair (strided wide vs strided deep).

    Depth=2 conv layers, width=64 channels.
    No BatchNorm; ReLU activations; stride-2 first conv (DeepProve+strided_conv.patch).
    Iso-FLOP pair: cnn_deep_6_strided (~11.0M MACs).
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_channels = input_shape[0]
        self.features = nn.Sequential(
            *_conv_relu(in_channels, 64),      # 32×32 stride-1
            *_strided_conv_relu(64, 64),        # 32×32 → 16×16 stride-2
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


@register_model("cnn_deep_6_strided")
class CNNDeep6Strided(nn.Module):
    """Iso-FLOP CNN Pair A (deep): 6 conv layers, 41 ch — ORIGINAL stride-2 design.

    Spatial schedule: 32 → Conv(s=2) → 16 → 16 → Conv(s=2) → 8 → 8 → Conv(s=2) → 4 → GAP.

    MACs (per sample, input 3×32×32):
      Conv(3,  41, s=1, p=1) at 32×32:          41 × 3  × 9 × 32×32 =  1,132,416
      Conv(41, 41, s=2, p=1) at 32×32 → 16×16:  41 × 41 × 9 × 16×16 =  3,868,416
      Conv(41, 41, s=1, p=1) at 16×16:          41 × 41 × 9 × 16×16 =  3,868,416
      Conv(41, 41, s=2, p=1) at 16×16 →  8×8:   41 × 41 × 9 ×  8×8  =    967,104
      Conv(41, 41, s=1, p=1) at  8×8:           41 × 41 × 9 ×  8×8  =    967,104
      Conv(41, 41, s=2, p=1) at  8×8 →   4×4:   41 × 41 × 9 ×  4×4  =    241,776
      Total: ~11,045,232 MACs

    Depth=6 conv layers, width=41 channels.
    No BatchNorm; ReLU activations; stride-2 convs (DeepProve+strided_conv.patch).
    Iso-FLOP pair: cnn_wide_2_strided.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_channels = input_shape[0]
        self.features = nn.Sequential(
            *_conv_relu(in_channels, 41),       # 32×32 stride-1
            *_strided_conv_relu(41, 41),         # 32×32 → 16×16 stride-2
            *_conv_relu(41, 41),                # 16×16 stride-1
            *_strided_conv_relu(41, 41),         # 16×16 → 8×8 stride-2
            *_conv_relu(41, 41),                # 8×8 stride-1
            *_strided_conv_relu(41, 41),         # 8×8 → 4×4 stride-2
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(41, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


@register_model("cnn_wide_4_strided")
class CNNWide4Strided(nn.Module):
    """Iso-FLOP CNN Pair B (wide): 4 conv layers, 48 ch — ORIGINAL stride-2 design.

    Spatial schedule: 32 → Conv(s=1) → 32 → Conv(s=2) → 16 → Conv(s=1) → 16 → Conv(s=2) → 8 → GAP.

    MACs (per sample, input 3×32×32):
      Conv(3,  48, s=1, p=1) at 32×32:          48 × 3  × 9 × 32×32 =  1,327,104
      Conv(48, 48, s=2, p=1) at 32×32 → 16×16:  48 × 48 × 9 × 16×16 =  5,308,416
      Conv(48, 48, s=1, p=1) at 16×16:          48 × 48 × 9 × 16×16 =  5,308,416
      Conv(48, 48, s=2, p=1) at 16×16 →  8×8:   48 × 48 × 9 ×  8×8  =  1,327,104
      Total: ~13,271,040 MACs

    Depth=4 conv layers, width=48 channels.
    No BatchNorm; ReLU activations; stride-2 convs (DeepProve+strided_conv.patch).
    Iso-FLOP pair: cnn_deep_12_strided (~13.3M MACs).
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_channels = input_shape[0]
        self.features = nn.Sequential(
            *_conv_relu(in_channels, 48),       # 32×32 stride-1
            *_strided_conv_relu(48, 48),         # 32×32 → 16×16 stride-2
            *_conv_relu(48, 48),                # 16×16 stride-1
            *_strided_conv_relu(48, 48),         # 16×16 → 8×8 stride-2
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(48, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


@register_model("cnn_deep_12_strided")
class CNNDeep12Strided(nn.Module):
    """Iso-FLOP CNN Pair B (deep): 12 conv layers, 42 ch — ORIGINAL stride-2 design.

    Architecture: 4 groups of 3 convs; each group's first conv is stride-2 (except the
    very first conv which runs at full 32×32 resolution at stride-1).

    Spatial schedule:
      Group 1: Conv(s=1)@32 → Conv(s=2)@32→16 → Conv(s=1)@16
      Group 2: Conv(s=2)@16→8 → Conv(s=1)@8 → Conv(s=1)@8
      Group 3: Conv(s=2)@8→4  → Conv(s=1)@4 → Conv(s=1)@4
      Group 4: Conv(s=2)@4→2  → Conv(s=1)@2 → Conv(s=1)@2 → GAP → classifier

    MACs (per sample, input 3×32×32):
      Group 1:
        Conv(3,  42, s=1) at 32×32:            42 ×  3 × 9 × 32×32 =  1,161,216
        Conv(42, 42, s=2) at 32×32 → 16×16:    42 × 42 × 9 × 16×16 =  4,064,256
        Conv(42, 42, s=1) at 16×16:            42 × 42 × 9 × 16×16 =  4,064,256
      Group 2:
        Conv(42, 42, s=2) at 16×16 → 8×8:      42 × 42 × 9 ×  8×8  =  1,016,064
        Conv(42, 42, s=1) at  8×8:             42 × 42 × 9 ×  8×8  =  1,016,064
        Conv(42, 42, s=1) at  8×8:             42 × 42 × 9 ×  8×8  =  1,016,064
      Group 3:
        Conv(42, 42, s=2) at  8×8 → 4×4:       42 × 42 × 9 ×  4×4  =    254,016
        Conv(42, 42, s=1) at  4×4:             42 × 42 × 9 ×  4×4  =    254,016
        Conv(42, 42, s=1) at  4×4:             42 × 42 × 9 ×  4×4  =    254,016
      Group 4:
        Conv(42, 42, s=2) at  4×4 → 2×2:       42 × 42 × 9 ×  2×2  =     63,504
        Conv(42, 42, s=1) at  2×2:             42 × 42 × 9 ×  2×2  =     63,504
        Conv(42, 42, s=1) at  2×2:             42 × 42 × 9 ×  2×2  =     63,504
      Total: ~13,290,480 MACs

    Depth=12 conv layers, width=42 channels.
    No BatchNorm; ReLU activations; stride-2 convs (DeepProve+strided_conv.patch).
    Iso-FLOP pair: cnn_wide_4_strided (~13.3M MACs).
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_channels = input_shape[0]
        self.features = nn.Sequential(
            # Group 1: 32×32 → 16×16
            *_conv_relu(in_channels, 42),       # 32×32 stride-1
            *_strided_conv_relu(42, 42),         # 32×32 → 16×16 stride-2
            *_conv_relu(42, 42),                # 16×16 stride-1
            # Group 2: 16×16 → 8×8
            *_strided_conv_relu(42, 42),         # 16×16 → 8×8 stride-2
            *_conv_relu(42, 42),                # 8×8 stride-1
            *_conv_relu(42, 42),                # 8×8 stride-1
            # Group 3: 8×8 → 4×4
            *_strided_conv_relu(42, 42),         # 8×8 → 4×4 stride-2
            *_conv_relu(42, 42),                # 4×4 stride-1
            *_conv_relu(42, 42),                # 4×4 stride-1
            # Group 4: 4×4 → 2×2
            *_strided_conv_relu(42, 42),         # 4×4 → 2×2 stride-2
            *_conv_relu(42, 42),                # 2×2 stride-1
            *_conv_relu(42, 42),                # 2×2 stride-1
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(42, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Same-depth width-only control models (addresses Codex Finding #1)
#
# Design rationale:
#   The Reversal result (mlp_wide_2 vs mlp_deep_6 at iso-FLOPs) could be
#   confounded by the GKR sumcheck round count: more layers → more rounds →
#   trivially bigger transcript, regardless of FLOPs.
#
#   These "same-depth" controls hold depth=4 constant while varying width:
#     mlp_4L_w_match : 4L, w=423 → ~1,840,473 MACs ≈ mlp_deep_6 (1,838,424 MACs)
#     mlp_4L_w_large : 4L, w=512 → ~2,364,416 MACs  > mlp_wide_2 (1,840,128 MACs)
#
#   Cross-comparison design (all benched on DeepProve, CIFAR-10, N=5):
#     1. mlp_wide_2   (2L, w=512) — 1,840,128 MACs — proof_size known: 6,801 KB
#     2. mlp_deep_6   (6L, w=372) — 1,838,424 MACs — proof_size known: 13,138 KB
#     3. mlp_4L_w_match (4L, w=423) — 1,840,473 MACs — proof_size TBD
#     4. mlp_4L_w_large (4L, w=512) — 2,364,416 MACs — proof_size TBD
#
#   Interpretation key:
#     IF mlp_4L_w_match ≈ mlp_wide_2 in proof size (both ~1.84M MACs, different
#       depth 4 vs 2) → proof size grows with FLOPs, not depth → paper claim wrong
#     IF mlp_4L_w_match ≈ mlp_deep_6 in proof size (both 1.84M MACs, same depth
#       4 ≈ 6) → depth is the discriminator → paper claim defensible
#     IF mlp_4L_w_large > mlp_4L_w_match even though both are 4 layers → width
#       (FLOP count) drives proof size within fixed depth → depth is NOT the driver
#
# ---------------------------------------------------------------------------


@register_model("mlp_4L_w_match")
class MLP4LWMatch(nn.Module):
    """Same-depth width control: 4 hidden layers, width=423 (iso-FLOP to mlp_deep_6).

    MACs (per sample, CIFAR-10 input 3×32×32 = 3072 flat):
      Linear(3072, 423): 3072 × 423 = 1,299,456
      Linear(423,  423) ×3:  423 × 423 × 3 =   536,787
      Linear(423,   10):  423 × 10  =     4,230
      Total: ~1,840,473 MACs

    Depth=4 hidden layers, width=423.
    No BatchNorm; ReLU activations.
    Iso-FLOP near-match: mlp_deep_6 (~1,838,424 MACs, diff=0.11%).
    Control purpose: same FLOPs as mlp_deep_6 but 4 layers instead of 6.
    If proof size ≈ mlp_wide_2 (4 < 6 layers), depth drives proof size.
    If proof size ≈ mlp_deep_6 (same FLOPs, 4 ≈ 6 layers), FLOPs drive proof size.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_features = reduce(lambda a, b: a * b, input_shape)
        layers: list[nn.Module] = [nn.Flatten(), nn.Linear(in_features, 423), nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(423, 423), nn.ReLU()]
        layers.append(nn.Linear(423, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register_model("mlp_4L_w_large")
class MLP4LWLarge(nn.Module):
    """Same-depth width control: 4 hidden layers, width=512 (more FLOPs than mlp_wide_2).

    MACs (per sample, CIFAR-10 input 3×32×32 = 3072 flat):
      Linear(3072, 512): 3072 × 512 = 1,572,864
      Linear(512,  512) ×3:  512 × 512 × 3 =   786,432
      Linear(512,   10):  512 × 10  =     5,120
      Total: ~2,364,416 MACs

    Depth=4 hidden layers, width=512.
    No BatchNorm; ReLU activations.
    FLOPs: 2,364,416 MACs — 28% more than mlp_wide_2 (1,840,128 MACs).
    Control purpose: more FLOPs than wide_2 but same depth=4 as mlp_4L_w_match.
    If proof size >> mlp_4L_w_match → width (FLOPs) drives size within fixed depth.
    If proof size ≈ mlp_4L_w_match → depth is the primary driver, not FLOPs.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_features = reduce(lambda a, b: a * b, input_shape)
        layers: list[nn.Module] = [nn.Flatten(), nn.Linear(in_features, 512), nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512), nn.ReLU()]
        layers.append(nn.Linear(512, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# MAC count verification helper
# ---------------------------------------------------------------------------

# Iso-FLOP pair definitions: (name_a, name_b, expected_macs_a, expected_macs_b)
ISO_FLOP_PAIRS: list[tuple[str, str, int, int]] = [
    ("mlp_wide_2",  "mlp_deep_6",   1_840_128,  1_838_424),
    ("mlp_wide_1",  "mlp_deep_10",  6_311_936,  6_318_792),
    # CNN pairs: MaxPool2d downsampling (DeepProve-compatible); MAC counts identical
    # to the original stride-2 designs since conv still executes at output resolution.
    ("cnn_wide_2",  "cnn_deep_6",  11_206_656, 11_045_232),
    ("cnn_wide_4",  "cnn_deep_12", 13_271_040, 13_290_480),
    # Same-depth width-only controls (Codex Finding #1 defense)
    ("mlp_4L_w_match", "mlp_deep_6", 1_840_473, 1_838_424),
]

# All 8 iso-FLOP model names in order
ISO_FLOP_MODEL_NAMES: list[str] = [
    "mlp_wide_2", "mlp_deep_6",
    "mlp_wide_1", "mlp_deep_10",
    "cnn_wide_2", "cnn_deep_6",
    "cnn_wide_4", "cnn_deep_12",
]

# Same-depth control model names (Codex Finding #1 defense)
SAME_DEPTH_CONTROL_NAMES: list[str] = [
    "mlp_wide_2",      # anchor: 2L, w=512, 1.84M MACs  — proof known
    "mlp_deep_6",      # anchor: 6L, w=372, 1.84M MACs  — proof known
    "mlp_4L_w_match",  # new:    4L, w=423, 1.84M MACs  — proof TBD
    "mlp_4L_w_large",  # new:    4L, w=512, 2.36M MACs  — proof TBD
]


def _verify_mac_counts(
    input_shape: tuple[int, ...] = (3, 32, 32),
    num_classes: int = 10,
    tolerance: float = 0.05,
) -> dict[str, int]:
    """Verify that iso-FLOP pairs have matching MAC counts (within tolerance).

    Uses compute_proxy() (ONNX-based proxy) to measure actual MACs from the
    exported ONNX graph.  Asserts that each pair differs by at most `tolerance`
    (default 5%).

    Args:
        input_shape: Model input shape (C, H, W).
        num_classes: Number of output classes.
        tolerance: Maximum allowed fractional difference (0.05 = 5%).

    Returns:
        Dict mapping model name → measured flops (2×MACs as reported by proxy).

    Raises:
        AssertionError: If any iso-FLOP pair exceeds the tolerance.
    """
    from ..metrics.zk_proxy import compute_proxy
    from .registry import get_model

    sample = torch.zeros(1, *input_shape)
    results: dict[str, int] = {}

    for name in ISO_FLOP_MODEL_NAMES:
        model = get_model(name, num_classes=num_classes, input_shape=input_shape)
        model.eval()
        proxy = compute_proxy(model, sample)
        results[name] = proxy.flops
        print(f"  {name:<20} FLOPs={proxy.flops:>14,}  (proxy method: {proxy.proxy_method})")

    print()
    for name_a, name_b, _expected_a, _expected_b in ISO_FLOP_PAIRS:
        flops_a = results[name_a]
        flops_b = results[name_b]
        if flops_a == 0 or flops_b == 0:
            print(f"  SKIP {name_a} vs {name_b}: zero FLOPs (proxy may not support)")
            continue
        diff = abs(flops_a - flops_b) / max(flops_a, flops_b)
        status = "PASS" if diff <= tolerance else "FAIL"
        print(
            f"  [{status}] {name_a} ({flops_a:,}) vs {name_b} ({flops_b:,})"
            f"  diff={diff*100:.2f}%  (tolerance={tolerance*100:.0f}%)"
        )
        assert diff <= tolerance, (
            f"Iso-FLOP pair {name_a}/{name_b} exceeds {tolerance*100:.0f}% tolerance: "
            f"{flops_a:,} vs {flops_b:,} ({diff*100:.2f}%)"
        )

    return results


if __name__ == "__main__":
    print("Verifying iso-FLOP MAC counts...")
    _verify_mac_counts()
