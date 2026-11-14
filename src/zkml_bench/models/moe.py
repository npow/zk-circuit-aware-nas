"""Mixture-of-Experts (MoE) model architectures for zkml-bench.

Includes:
- moe_tiny: 4-expert MoE MLP with gating network
- moe_resnet_block: ResNet-style CNN with MoE FFN replacing the FC head

All models accept (B, C, H, W) input and return logits (no softmax).
All models are ONNX-exportable with opset_version=17.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import register_model


# ---------------------------------------------------------------------------
# MoE Expert layer
# ---------------------------------------------------------------------------


class ExpertMLP(nn.Module):
    """Single expert: a 2-layer MLP."""

    def __init__(self, in_features: int, hidden: int, out_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TopKMoE(nn.Module):
    """Mixture-of-Experts layer: soft routing over K experts.

    Uses soft (weighted sum) routing rather than hard top-K to remain
    ONNX-exportable (no dynamic indexing).  All K experts run and are
    combined by gating weights — this avoids control-flow ops.
    """

    def __init__(self, in_features: int, out_features: int, n_experts: int, expert_hidden: int) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.gate = nn.Linear(in_features, n_experts)
        self.experts = nn.ModuleList([
            ExpertMLP(in_features, expert_hidden, out_features)
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_features)
        # Gate produces soft weights over experts
        gate_logits = self.gate(x)                          # (B, n_experts)
        gate_weights = F.softmax(gate_logits, dim=-1)        # (B, n_experts)

        # Run all experts and stack: (B, n_experts, out_features)
        expert_outs = torch.stack([e(x) for e in self.experts], dim=1)

        # Weighted sum: (B, out_features)
        # gate_weights: (B, n_experts) → (B, n_experts, 1)
        out = (gate_weights.unsqueeze(-1) * expert_outs).sum(dim=1)
        return out


# ---------------------------------------------------------------------------
# MoE Tiny
# ---------------------------------------------------------------------------


@register_model("moe_tiny")
class MoETiny(nn.Module):
    """Tiny MoE MLP: Conv stem → MoE layer (4 experts) → classifier.

    Architecture:
        Conv stem (3→32, 3×3) → GlobalAvgPool → MoE(32→128, 4 experts)
        → ReLU → Linear(128, num_classes)

    Expected CIFAR-10 accuracy ~65-70% (similar depth to lenet5 with wider FFN).
    Params ~1M.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        c = input_shape[0]
        stem_channels = 32
        moe_in = stem_channels
        moe_out = 128
        n_experts = 4
        expert_hidden = 64

        self.stem = nn.Sequential(
            nn.Conv2d(c, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, stem_channels, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.moe = TopKMoE(moe_in, moe_out, n_experts=n_experts, expert_hidden=expert_hidden)
        self.act = nn.ReLU()
        self.head = nn.Linear(moe_out, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.moe(x)
        x = self.act(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# MoE ResNet block
# ---------------------------------------------------------------------------


class ResBlockMoE(nn.Module):
    """ResNet basic block with MoE FFN instead of the second conv."""

    def __init__(self, channels: int, n_experts: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        # MoE operates on flattened spatial; we apply it pointwise
        self.moe = TopKMoE(channels, channels, n_experts=n_experts, expert_hidden=channels * 2)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))

        # Apply MoE per spatial position
        B, C, H, W = out.shape
        # (B, C, H, W) → (B*H*W, C)
        out_flat = out.permute(0, 2, 3, 1).reshape(B * H * W, C)
        out_flat = self.moe(out_flat)
        # (B*H*W, C) → (B, C, H, W)
        out = out_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

        out = self.bn2(out)
        out = out + identity
        return self.relu(out)


@register_model("moe_resnet_block")
class MoEResNetBlock(nn.Module):
    """Small CNN using ResNet blocks with MoE FFN replacing the second conv.

    Architecture: stem → 2 standard ResBlocks → 2 MoE ResBlocks → GAP → FC.
    Expected CIFAR-10 accuracy ~80% (ResNet scale with MoE capacity boost).
    Params ~600K.
    """

    def __init__(self, num_classes: int, input_shape: tuple[int, ...]) -> None:
        super().__init__()
        c = input_shape[0]
        channels = 32

        self.stem = nn.Sequential(
            nn.Conv2d(c, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # Standard residual blocks
        self.layer1 = nn.Sequential(
            *[_BasicBlockBN(channels, channels) for _ in range(2)]
        )
        # MoE residual blocks
        self.layer2 = nn.Sequential(
            *[ResBlockMoE(channels, n_experts=4) for _ in range(2)]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.head(x)


class _BasicBlockBN(nn.Module):
    """Standard ResNet basic block (same-channel, no stride)."""

    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)
