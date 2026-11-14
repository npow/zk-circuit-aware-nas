"""Transformer and attention-based model architectures for zkml-bench.

Includes:
- tiny_transformer: small 4-layer transformer encoder
- small_bert: 6-layer BERT-style encoder
- mlp_mixer_tiny: ImageNet-style MLP-Mixer
- vit_tiny: Vision Transformer Tiny (ViT-Ti/16-style)
- attention_only_2head: minimal 2-head self-attention block

All models accept (B, C, H, W) input for CIFAR-10 and return logits (no softmax).
All models are ONNX-exportable with opset_version=17.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .registry import register_model


# ---------------------------------------------------------------------------
# Tiny Transformer
# ---------------------------------------------------------------------------


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer: MHSA → Add+Norm → FFN → Add+Norm."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention + residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        # FFN + residual
        x = self.norm2(x + self.ffn(x))
        return x


@register_model("tiny_transformer")
class TinyTransformer(nn.Module):
    """4-layer transformer encoder for CIFAR-10, 64-dim, 4 heads.

    Patch size 4×4 → 64 patches of dim 48 → projected to 64.
    Expected CIFAR-10 accuracy ~75-78% (Dosovitskiy et al. 2021 ViT scaling).
    Params ~200K.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        c, h, w = input_shape
        patch_size = 4
        # Number of patches
        n_patches = (h // patch_size) * (w // patch_size)
        patch_dim = c * patch_size * patch_size

        d_model = 64
        n_heads = 4
        n_layers = 4
        ffn_dim = 256

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, ffn_dim)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._patch_size = patch_size
        self._h = h
        self._w = w
        self._c = c

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (B, C, H, W) → (B, N, patch_dim)."""
        B, C, H, W = x.shape
        p = self._patch_size
        # Unfold into patches
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B, (H // p) * (W // p), C * p * p)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self._patchify(x)
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        cls_out = x[:, 0]
        return self.head(cls_out)


# ---------------------------------------------------------------------------
# Small BERT-style encoder
# ---------------------------------------------------------------------------


@register_model("small_bert")
class SmallBERT(nn.Module):
    """6-layer BERT-style encoder: 256-dim, 8 heads, patch-based image input.

    Patch size 4×4 on CIFAR-10 → 64 patches → 256-dim.
    Expected CIFAR-10 accuracy ~82% (Dosovitskiy et al. 2021, small-scale ViT).
    Params ~5M.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        c, h, w = input_shape
        patch_size = 4
        n_patches = (h // patch_size) * (w // patch_size)
        patch_dim = c * patch_size * patch_size

        d_model = 256
        n_heads = 8
        n_layers = 6
        ffn_dim = 1024

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, ffn_dim)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._patch_size = patch_size
        self._h = h
        self._w = w
        self._c = c

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        p = self._patch_size
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.reshape(B, (H // p) * (W // p), C * p * p)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self._patchify(x)
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.head(x[:, 0])


# ---------------------------------------------------------------------------
# MLP-Mixer Tiny
# ---------------------------------------------------------------------------


class MixerLayer(nn.Module):
    """MLP-Mixer layer: token-mixing MLP + channel-mixing MLP."""

    def __init__(self, n_patches: int, d_model: int, token_mlp_dim: int, channel_mlp_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Token-mixing: operates on transposed patches (D, N) → (D, N)
        self.token_mix = nn.Sequential(
            nn.Linear(n_patches, token_mlp_dim),
            nn.GELU(),
            nn.Linear(token_mlp_dim, n_patches),
        )
        # Channel-mixing: operates on (N, D) → (N, D)
        self.channel_mix = nn.Sequential(
            nn.Linear(d_model, channel_mlp_dim),
            nn.GELU(),
            nn.Linear(channel_mlp_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        y = self.norm1(x)
        y = y.transpose(1, 2)          # (B, D, N)
        y = self.token_mix(y)
        y = y.transpose(1, 2)          # (B, N, D)
        x = x + y
        x = x + self.channel_mix(self.norm2(x))
        return x


@register_model("mlp_mixer_tiny")
class MLPMixerTiny(nn.Module):
    """MLP-Mixer Tiny: 8 layers, patch size 4, 128-dim for CIFAR-10.

    Based on Tolstikhin et al. 2021 (MLP-Mixer).
    Expected CIFAR-10 accuracy ~87% with proper training (Tolstikhin et al. 2021).
    Params ~200K.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        c, h, w = input_shape
        patch_size = 4
        n_patches = (h // patch_size) * (w // patch_size)
        patch_dim = c * patch_size * patch_size
        d_model = 128
        n_layers = 8
        token_mlp_dim = 64
        channel_mlp_dim = 256

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.layers = nn.ModuleList([
            MixerLayer(n_patches, d_model, token_mlp_dim, channel_mlp_dim)
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
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)


# ---------------------------------------------------------------------------
# ViT Tiny
# ---------------------------------------------------------------------------


@register_model("vit_tiny")
class ViTTiny(nn.Module):
    """Vision Transformer Tiny (ViT-Ti style): 12 layers, 192-dim, 3 heads.

    Based on Touvron et al. 2020 (DeiT) ViT-Tiny variant.
    Patch size 4×4 on CIFAR-10 (64 patches).
    Expected CIFAR-10 accuracy ~90% (Lee et al. 2021, ViT for CIFAR).
    Params ~5.7M.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        c, h, w = input_shape
        patch_size = 4
        n_patches = (h // patch_size) * (w // patch_size)
        patch_dim = c * patch_size * patch_size

        d_model = 192
        n_heads = 3
        n_layers = 12
        ffn_dim = 768

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, ffn_dim)
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
        B = x.shape[0]
        x = self._patchify(x)
        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        return self.head(x[:, 0])


# ---------------------------------------------------------------------------
# Attention-only minimal 2-head block
# ---------------------------------------------------------------------------


@register_model("attention_only_2head")
class AttentionOnly2Head(nn.Module):
    """Minimal 2-head self-attention block for ZK cost characterization.

    Single self-attention layer (no FFN, no positional encoding), 2 heads.
    Global average pool over token dimension → linear classifier.
    Expected CIFAR-10 accuracy ~50-55% (extremely shallow, no training assumed).
    Params ~20K.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        c, h, w = input_shape
        patch_size = 4
        patch_dim = c * patch_size * patch_size
        d_model = 64

        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
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
        x = self.patch_embed(x)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        x = x.mean(dim=1)
        return self.head(x)
