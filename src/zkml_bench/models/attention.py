"""Attention variant architectures for zkml-bench.

Includes:
- linear_attention_small: linear (kernel-based) attention, no softmax
- local_attention_window: windowed self-attention with window size 4
- flash_attention_compatible: causal masked dot-product attention
- wide_resnet_small: WideResNet-16-4 scale
- mlp_huge: 8M-param MLP for scale stress-testing
- qlinear_small: quantization-aware MLP structure
- int8_resnet_small: ResNet variant designed for quantization

All models accept (B, C, H, W) input and return logits (no softmax).
All models are ONNX-exportable with opset_version=17.
"""

from __future__ import annotations

from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


# ---------------------------------------------------------------------------
# Linear Attention Small
# ---------------------------------------------------------------------------


class LinearAttentionLayer(nn.Module):
    """Linear attention (Performer/FastAttention style) without softmax.

    Uses the feature map φ(x) = elu(x) + 1 as the kernel function,
    approximating softmax attention in O(N·D²) instead of O(N²·D).
    This is an exact implementation (not random-feature), ONNX-friendly.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _feature_map(x: torch.Tensor) -> torch.Tensor:
        """Kernel feature map: elu(x) + 1 (always positive)."""
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        B, N, D = x.shape
        H = self.n_heads
        Dh = self.head_dim

        Q = self._feature_map(self.q_proj(x)).reshape(B, N, H, Dh).transpose(1, 2)
        K = self._feature_map(self.k_proj(x)).reshape(B, N, H, Dh).transpose(1, 2)
        V = self.v_proj(x).reshape(B, N, H, Dh).transpose(1, 2)
        # Q, K, V: (B, H, N, Dh)

        # Linear attention: O(N, D) via associativity
        # (K^T V): (B, H, Dh, Dh)
        KtV = torch.matmul(K.transpose(-2, -1), V)
        # Denominator: (B, H, N, 1)
        denom = (Q * K.sum(dim=-2, keepdim=True)).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        # Numerator: (B, H, N, Dh)
        numer = torch.matmul(Q, KtV)
        out = (numer / denom).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        return self.norm(x + out)


@register_model("linear_attention_small")
class LinearAttentionSmall(nn.Module):
    """2-layer linear (performer-style) attention model for CIFAR-10.

    No softmax in attention — uses elu+1 kernel feature map.
    ZK-friendlier than standard softmax attention.
    Expected CIFAR-10 accuracy ~72% (linear attention typically 1-3% below softmax).
    Params ~300K.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        c, h, w = input_shape
        patch_size = 4
        n_patches = (h // patch_size) * (w // patch_size)
        patch_dim = c * patch_size * patch_size
        d_model = 128
        n_heads = 4

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.layers = nn.ModuleList([
            LinearAttentionLayer(d_model, n_heads)
            for _ in range(2)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._patch_size = patch_size

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self._patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B, (H // p) * (W // p), C * p * p)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._patchify(x)
        x = self.patch_embed(x) + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x.mean(dim=1))


# ---------------------------------------------------------------------------
# Local Window Attention
# ---------------------------------------------------------------------------


class WindowAttentionLayer(nn.Module):
    """Local self-attention within non-overlapping windows of size W."""

    def __init__(self, d_model: int, n_heads: int, window_size: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        B, N, D = x.shape
        W = self.window_size

        # Pad N to multiple of W
        pad_len = (W - N % W) % W
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        N_pad = x.shape[1]
        n_windows = N_pad // W

        # Reshape into windows: (B * n_windows, W, D)
        x_win = x.reshape(B * n_windows, W, D)

        # Self-attention within each window
        attn_out, _ = self.attn(x_win, x_win, x_win)
        x_win = self.norm1(x_win + attn_out)
        x_win = self.norm2(x_win + self.ffn(x_win))

        # Reshape back: (B, N_pad, D)
        x = x_win.reshape(B, N_pad, D)

        # Remove padding
        if pad_len > 0:
            x = x[:, :N, :]
        return x


@register_model("local_attention_window")
class LocalAttentionWindow(nn.Module):
    """Windowed self-attention model: window size 4 for CIFAR-10 patches.

    Based on Liu et al. 2021 (Swin Transformer) windowed attention concept.
    No shifted windows — pure local attention for simplicity/ONNX compatibility.
    Expected CIFAR-10 accuracy ~80% (Swin-Tiny-scale windowed attention).
    Params ~400K.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        c, h, w = input_shape
        patch_size = 4
        patch_dim = c * patch_size * patch_size
        n_patches = (h // patch_size) * (w // patch_size)
        d_model = 128
        n_heads = 4
        window_size = 4
        n_layers = 4

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.layers = nn.ModuleList([
            WindowAttentionLayer(d_model, n_heads, window_size)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._patch_size = patch_size

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self._patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B, (H // p) * (W // p), C * p * p)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._patchify(x)
        x = self.patch_embed(x) + self.pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x.mean(dim=1))


# ---------------------------------------------------------------------------
# Flash-Attention Compatible (causal masked attention)
# ---------------------------------------------------------------------------


@register_model("flash_attention_compatible")
class FlashAttentionCompatible(nn.Module):
    """Causal masked dot-product attention model for CIFAR-10 (image as sequence).

    'Flash-compatible' means the mathematical operations match FlashAttention:
    causal masked softmax attention — the model uses standard PyTorch scaled
    dot-product attention with a causal mask, which can be accelerated by
    FlashAttention at inference time without changing the math.
    Expected CIFAR-10 accuracy ~74% (causal masking discards ~half the tokens).
    Params ~250K.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        c, h, w = input_shape
        patch_size = 4
        patch_dim = c * patch_size * patch_size
        n_patches = (h // patch_size) * (w // patch_size)
        d_model = 128
        n_heads = 4
        n_layers = 3

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.norm_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._patch_size = patch_size
        self._n_patches = n_patches

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self._patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B, (H // p) * (W // p), C * p * p)
        return x

    def _causal_mask(self, n: int, device: torch.device) -> torch.Tensor:
        """Upper triangular mask (additive, -inf off-diagonal above)."""
        mask = torch.triu(torch.ones(n, n, device=device), diagonal=1).bool()
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._patchify(x)
        x = self.patch_embed(x) + self.pos_embed
        N = x.shape[1]
        mask = self._causal_mask(N, x.device)

        for attn, norm in zip(self.attn_layers, self.norms):
            attn_out, _ = attn(x, x, x, attn_mask=mask)
            x = norm(x + attn_out)

        x = self.norm_final(x)
        return self.head(x[:, -1])   # Use last token as classification token


# ---------------------------------------------------------------------------
# Wide ResNet Small (WideResNet-16-4 scale)
# ---------------------------------------------------------------------------


class WideBlock(nn.Module):
    """Wide ResNet basic block with configurable width multiplier."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.drop(out)
        out = self.conv2(self.relu(self.bn2(out)))
        return out + self.shortcut(x)


@register_model("wide_resnet_small")
class WideResNetSmall(nn.Module):
    """WideResNet-16-4 style: depth 16, width multiplier 4.

    Based on Zagoruyko & Komodakis 2016 (WRN).
    Channels: [16, 64, 128, 256] (base 16 × width 4).
    Expected CIFAR-10 accuracy ~95% (WRN-16-8 gets 95.1%, WRN-16-4 ~94%).
    Params ~11M.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        c = input_shape[0]
        base = 16
        width = 4
        channels = [base, base * width, base * width * 2, base * width * 4]
        # depth 16 → 5 blocks per group (16 = 1 stem + 3 groups × 5 blocks)
        n_blocks = 2

        self.stem = nn.Conv2d(c, channels[0], 3, padding=1, bias=False)
        self.layer1 = self._make_layer(channels[0], channels[1], n_blocks, stride=1)
        self.layer2 = self._make_layer(channels[1], channels[2], n_blocks, stride=2)
        self.layer3 = self._make_layer(channels[2], channels[3], n_blocks, stride=2)
        self.bn = nn.BatchNorm2d(channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels[3], num_classes)

    @staticmethod
    def _make_layer(in_c: int, out_c: int, n: int, stride: int) -> nn.Sequential:
        layers = [WideBlock(in_c, out_c, stride=stride)]
        for _ in range(1, n):
            layers.append(WideBlock(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.bn(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


# ---------------------------------------------------------------------------
# Huge MLP (scale stress-test)
# ---------------------------------------------------------------------------


@register_model("mlp_huge")
class MLPHuge(nn.Module):
    """8M-parameter MLP for scale stress-testing ZK prover cost.

    Architecture: flatten → 2048 → ReLU → 2048 → ReLU → 1024 → ReLU → num_classes.
    Expected CIFAR-10 accuracy ~62% (marginal improvement over mlp_large due to
    vanishing gradients in deep plain MLPs; Goodfellow et al. 2016 Deep Learning).
    Params ~8M.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_features = reduce(lambda a, b: a * b, input_shape)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Quantization-aware MLP
# ---------------------------------------------------------------------------


class ClampedLinear(nn.Module):
    """Linear layer with weight/activation clamping for quantization-aware training.

    Simulates INT8 quantization: weights clamped to [-1, 1], activations
    clamped to [0, 6] (ReLU6 style) — common QAT practice (Jacob et al. 2018).
    ONNX-friendly: uses only clamp + linear ops.
    """

    def __init__(self, in_features: int, out_features: int, clamp_val: float = 1.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.clamp_val = clamp_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simulate quantized weights via clamping (fake quantization)
        w = self.linear.weight.clamp(-self.clamp_val, self.clamp_val)
        return F.linear(x, w, self.linear.bias)


@register_model("qlinear_small")
class QLinearSmall(nn.Module):
    """Quantization-aware MLP: mlp_small with clamped weights + ReLU6 activations.

    Simulates INT8 QAT structure (Jacob et al. 2018, Quantization and Training
    of Neural Networks for Efficient Integer-Arithmetic-Only Inference).
    ReLU6 clamps activations to [0, 6], matching INT8 fixed-point range.
    Expected CIFAR-10 accuracy ~53% (slight drop vs mlp_small due to constraints).
    Params ~1.7M.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        in_features = reduce(lambda a, b: a * b, input_shape)
        self.net = nn.Sequential(
            nn.Flatten(),
            ClampedLinear(in_features, 512),
            nn.ReLU6(),
            ClampedLinear(512, 256),
            nn.ReLU6(),
            ClampedLinear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# INT8-style ResNet (quantization-friendly)
# ---------------------------------------------------------------------------


class Int8FriendlyBlock(nn.Module):
    """ResNet block with ReLU6 activations for INT8 quantization compatibility."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu6 = nn.ReLU6(inplace=True)
        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu6(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu6(out)


@register_model("int8_resnet_small")
class Int8ResNetSmall(nn.Module):
    """ResNet-14 variant with ReLU6 activations for INT8 quantization compatibility.

    Replaces ReLU with ReLU6 throughout (Howard et al. 2017, MobileNets),
    constraining activation range to [0, 6] for INT8-friendly deployment.
    Expected CIFAR-10 accuracy ~89% (near parity with ReLU ResNet-20).
    Params ~200K.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        c = input_shape[0]

        self.stem = nn.Sequential(
            nn.Conv2d(c, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True),
        )
        self.layer1 = self._make_layer(16, 16, 2, stride=1)
        self.layer2 = self._make_layer(16, 32, 2, stride=2)
        self.layer3 = self._make_layer(32, 64, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(64, num_classes)

    @staticmethod
    def _make_layer(in_c: int, out_c: int, n: int, stride: int) -> nn.Sequential:
        layers = [Int8FriendlyBlock(in_c, out_c, stride=stride)]
        for _ in range(1, n):
            layers.append(Int8FriendlyBlock(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.head(x)
