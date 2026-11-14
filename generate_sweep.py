"""Systematic Architecture Sweep Generator
==========================================
Generates 400+ architectures spanning the ZK-ML design space without requiring
LLM calls. Covers activation type × normalization × architecture family ×
width/depth variations.

This gives the statistical power (N≥400) needed for credible Spearman ρ
analysis in the proxy correlation study.

Usage:
    python generate_sweep.py [--epochs 30] [--out results/sweep/]
    python generate_sweep.py --dry-run   # show configs without training
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, "src")


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

class Square(nn.Module):
    def forward(self, x): return x ** 2

class CubicPlusLinear(nn.Module):
    """x³ + x — signed polynomial, approximates GELU behavior."""
    def forward(self, x): return x ** 3 + x

class SquarePlusLinear(nn.Module):
    """x² + x — hybrid, asymmetric polynomial."""
    def forward(self, x): return x ** 2 + x


ACTIVATIONS = {
    "relu":      lambda: nn.ReLU(),
    "x2":        lambda: Square(),
    "x3px":      lambda: CubicPlusLinear(),
    "x2px":      lambda: SquarePlusLinear(),
}

NORMALIZATIONS = {
    "none":       lambda c: nn.Identity(),
    "layernorm":  lambda c: nn.LayerNorm(c),  # applied after flatten for MLPs
    "groupnorm":  lambda c: nn.GroupNorm(min(8, c), c) if c % min(8, c) == 0 else nn.Identity(),
}


# ---------------------------------------------------------------------------
# Architecture families
# ---------------------------------------------------------------------------

def make_mlp(act_name: str, norm_name: str, hidden_dims: list[int],
             num_classes: int = 10, in_pixels: int = 3 * 32 * 32) -> nn.Module:
    """Simple MLP with configurable hidden dims."""
    act_fn = ACTIVATIONS[act_name]
    layers = [nn.Flatten()]
    in_dim = in_pixels
    for h in hidden_dims:
        layers.append(nn.Linear(in_dim, h))
        if norm_name == "layernorm":
            layers.append(nn.LayerNorm(h))
        elif norm_name == "groupnorm":
            pass  # GroupNorm doesn't apply to linear layers
        layers.append(act_fn())
        in_dim = h
    layers.append(nn.Linear(in_dim, num_classes))
    return nn.Sequential(*layers)


def make_cnn(act_name: str, norm_name: str, channels: list[int],
             depthwise: bool = False, num_classes: int = 10) -> nn.Module:
    """Simple CNN with configurable channels."""
    act_fn = ACTIVATIONS[act_name]

    class ConvBlock(nn.Module):
        def __init__(self, in_c, out_c, dws=False):
            super().__init__()
            if dws:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c, bias=False),
                    nn.Conv2d(in_c, out_c, 1, bias=False),
                )
            else:
                self.conv = nn.Conv2d(in_c, out_c, 3, padding=1, bias=False)

            self.norm = nn.Identity()
            if norm_name == "layernorm":
                self.norm = None  # applied per-sample, handled in forward
            elif norm_name == "groupnorm":
                g = min(8, out_c)
                if out_c % g == 0:
                    self.norm = nn.GroupNorm(g, out_c)
            self.act = act_fn()
            self._out_c = out_c

        def forward(self, x):
            x = self.conv(x)
            if self.norm is None:
                # LayerNorm on spatial feature map
                B, C, H, W = x.shape
                x = nn.functional.layer_norm(x, [C, H, W])
            else:
                x = self.norm(x)
            return self.act(x)

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            blocks = []
            in_c = 3
            for i, out_c in enumerate(channels):
                blocks.append(ConvBlock(in_c, out_c, dws=depthwise))
                if i < len(channels) - 1:
                    blocks.append(nn.MaxPool2d(2))
                in_c = out_c
            self.features = nn.Sequential(*blocks)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(channels[-1], num_classes)

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x)
            x = x.flatten(1)
            return self.classifier(x)

    return SimpleCNN()


def make_resnet_style(act_name: str, norm_name: str, base_c: int, n_blocks: int,
                      depthwise: bool = False, num_classes: int = 10) -> nn.Module:
    """ResNet-style with skip connections and configurable activation."""
    act_fn = ACTIVATIONS[act_name]

    def norm_layer(c):
        if norm_name == "none":
            return nn.Identity()
        elif norm_name == "layernorm":
            return None  # handled specially
        elif norm_name == "groupnorm":
            g = min(8, c)
            return nn.GroupNorm(g, c) if c % g == 0 else nn.Identity()
        return nn.Identity()

    class ResBlock(nn.Module):
        def __init__(self, c, dws=False):
            super().__init__()
            if dws:
                self.conv1 = nn.Sequential(
                    nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
                    nn.Conv2d(c, c, 1, bias=False),
                )
                self.conv2 = nn.Sequential(
                    nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
                    nn.Conv2d(c, c, 1, bias=False),
                )
            else:
                self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
                self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
            nl = norm_layer(c)
            self.norm1 = nl if nl is not None else nn.Identity()
            nl2 = norm_layer(c)
            self.norm2 = nl2 if nl2 is not None else nn.Identity()
            self.act = act_fn()
            self._use_ln = (norm_name == "layernorm")
            self._c = c

        def _norm(self, x, norm_mod):
            if self._use_ln:
                B, C, H, W = x.shape
                return nn.functional.layer_norm(x, [C, H, W])
            return norm_mod(x)

        def forward(self, x):
            out = self.conv1(x)
            out = self._norm(out, self.norm1)
            out = self.act(out)
            out = self.conv2(out)
            out = self._norm(out, self.norm2)
            return self.act(out + x)

    class ResNetStyle(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Conv2d(3, base_c, 3, padding=1, bias=False)
            self.blocks = nn.Sequential(*[
                ResBlock(base_c, dws=depthwise) for _ in range(n_blocks)
            ])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(base_c, num_classes)

        def forward(self, x):
            x = self.stem(x)
            x = self.blocks(x)
            x = self.pool(x)
            x = x.flatten(1)
            return self.classifier(x)

    return ResNetStyle()


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

def make_sweep_configs(num_classes: int = 10) -> list[dict]:
    """Generate all architecture configurations."""
    configs = []

    # 1. MLPs
    for act in ACTIVATIONS:
        for norm in ["none", "layernorm"]:
            for hidden in [[256, 128], [512, 256], [256, 128, 64],
                           [512, 256, 128], [1024, 512], [128, 64, 32]]:
                configs.append({
                    "family": "mlp",
                    "act": act, "norm": norm,
                    "hidden_dims": hidden,
                    "name": f"mlp_{act}_{norm}_{'x'.join(map(str, hidden))}",
                    "num_classes": num_classes,
                })

    # 2. Standard CNNs (different channel widths)
    for act in ACTIVATIONS:
        for norm in NORMALIZATIONS:
            for channels in [
                [16, 32, 64], [32, 64, 128], [16, 32, 64, 128],
                [32, 64], [64, 128, 256], [16, 32]
            ]:
                configs.append({
                    "family": "cnn",
                    "act": act, "norm": norm,
                    "channels": channels, "depthwise": False,
                    "name": f"cnn_{act}_{norm}_{'_'.join(map(str, channels))}",
                    "num_classes": num_classes,
                })

    # 3. DWS CNNs
    for act in ACTIVATIONS:
        for norm in NORMALIZATIONS:
            for channels in [[32, 64, 128], [64, 128, 256], [16, 32, 64],
                             [32, 64, 128, 256], [32, 64]]:
                configs.append({
                    "family": "cnn",
                    "act": act, "norm": norm,
                    "channels": channels, "depthwise": True,
                    "name": f"dws_{act}_{norm}_{'_'.join(map(str, channels))}",
                    "num_classes": num_classes,
                })

    # 4. ResNet-style
    for act in ACTIVATIONS:
        for norm in NORMALIZATIONS:
            for base_c, n_blocks in [(16, 2), (32, 2), (32, 4), (64, 2),
                                      (16, 4), (64, 4), (32, 6), (16, 6),
                                      (48, 2), (48, 4), (24, 4)]:
                for dws in [False, True]:
                    dws_tag = "dws_" if dws else ""
                    configs.append({
                        "family": "resnet",
                        "act": act, "norm": norm,
                        "base_c": base_c, "n_blocks": n_blocks, "depthwise": dws,
                        "name": f"{dws_tag}resnet_{act}_{norm}_c{base_c}_b{n_blocks}",
                        "num_classes": num_classes,
                    })

    # 5. Extra MLP variants for polynomial activation coverage
    for act in ["x2", "x3px", "x2px"]:
        for norm in ["none", "layernorm"]:
            for hidden in [[1024, 512, 256], [512, 128], [256, 256, 256],
                           [128, 128, 128, 128], [2048, 512]]:
                configs.append({
                    "family": "mlp",
                    "act": act, "norm": norm,
                    "hidden_dims": hidden,
                    "name": f"mlp_deep_{act}_{norm}_{'x'.join(map(str, hidden))}",
                    "num_classes": num_classes,
                })

    log.info("Generated %d architecture configurations", len(configs))
    return configs


def build_model(cfg: dict) -> nn.Module:
    """Build model from config."""
    fam = cfg["family"]
    act = cfg["act"]
    norm = cfg["norm"]
    nc = cfg.get("num_classes", 10)
    # Infer input pixels from dataset or explicit field
    dataset = cfg.get("dataset", "cifar10")
    in_pixels = 3 * 16 * 16 if dataset == "imagenet16" else 3 * 32 * 32
    if fam == "mlp":
        return make_mlp(act, norm, cfg["hidden_dims"], nc, in_pixels=in_pixels)
    elif fam == "cnn":
        return make_cnn(act, norm, cfg["channels"], cfg.get("depthwise", False), nc)
    elif fam == "resnet":
        return make_resnet_style(act, norm, cfg["base_c"], cfg["n_blocks"],
                                  cfg.get("depthwise", False), nc)
    raise ValueError(f"Unknown family: {fam}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(model: nn.Module, epochs: int, device: str,
              dataset: str = "cifar10") -> tuple[float, float, float]:
    """Train model for N epochs on CIFAR-10/100. Returns (top1, top5, train_time_s)."""
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if dataset == "imagenet16":
        from imagenet16 import ImageNet16
        trainset = ImageNet16("/tmp/imagenet16/ImageNet16", train=True, num_classes=120)
        testset  = ImageNet16("/tmp/imagenet16/ImageNet16", train=False, num_classes=120)
    else:
        DS = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
        trainset = DS(root="/tmp/data", train=True, download=True, transform=transform_train)
        testset = DS(root="/tmp/data", train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    # Polynomial activations (x², x³+x) without normalization can produce
    # exploding gradients under SGD lr=0.1. Detect and apply safe settings.
    has_relu = any(isinstance(m, nn.ReLU) for m in model.modules())
    module_names = {type(m).__name__ for m in model.modules()}
    is_poly = bool(module_names & {"Square", "CubicPlusLinear", "SquarePlusLinear"})
    has_norm = any(isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm))
                   for m in model.modules())
    # Use lower LR and gradient clipping for polynomial-activation, norm-free models
    use_safe_lr = is_poly and not has_norm
    lr = 0.01 if use_safe_lr else 0.1
    clip_grad = 1.0 if use_safe_lr else None

    dev = torch.device(device)
    model = model.to(dev)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for x, y in trainloader:
            x, y = x.to(dev), y.to(dev)
            optimizer.zero_grad()
            try:
                out = model(x)
                loss = criterion(out, y)
                if torch.isnan(loss) or torch.isinf(loss):
                    break  # skip rest of epoch if loss explodes
                loss.backward()
                if clip_grad is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
            except RuntimeError:
                break  # catch NaN/Inf in forward pass
        scheduler.step()

    # Evaluate
    model.eval()
    correct1 = correct5 = total = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(dev), y.to(dev)
            out = model(x)
            _, pred5 = out.topk(5, dim=1)
            correct1 += (pred5[:, 0] == y).sum().item()
            correct5 += (pred5 == y.unsqueeze(1)).any(dim=1).sum().item()
            total += y.size(0)

    train_time = time.perf_counter() - t0
    return 100 * correct1 / total, 100 * correct5 / total, train_time


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "imagenet16"])
    parser.add_argument("--out", default="results/sweep")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start", type=int, default=0, help="Start at config index N")
    parser.add_argument("--end", type=int, default=None, help="End at config index N")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s", device)

    num_classes = 100 if args.dataset == "cifar100" else (120 if args.dataset == "imagenet16" else 10)
    configs = make_sweep_configs(num_classes)
    # Inject dataset into every config so build_model can infer input shape
    for cfg in configs:
        cfg["dataset"] = args.dataset

    if args.end is not None:
        configs = configs[args.start:args.end]
    elif args.start > 0:
        configs = configs[args.start:]

    log.info("Will train %d architectures", len(configs))

    if args.dry_run:
        for i, cfg in enumerate(configs[:20]):
            try:
                model = build_model(cfg)
                n_params = sum(p.numel() for p in model.parameters())
                log.info("[%d] %s — %d params", i, cfg["name"], n_params)
            except Exception as e:
                log.warning("[%d] %s — BUILD FAILED: %s", i, cfg["name"], e)
        log.info("Dry run complete (%d configs total)", len(configs))
        return 0

    os.makedirs(args.out, exist_ok=True)
    results_path = os.path.join(args.out, f"{args.dataset}_sweep.json")

    # Load existing results to skip already-done configs
    done = set()
    results = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        done = {r["name"] for r in results if "top1" in r}
        log.info("Loaded %d existing results, skipping", len(done))

    from zkml_bench.metrics.zk_proxy import compute_proxy, _suppress_onnx_noise

    for i, cfg in enumerate(configs):
        if cfg["name"] in done:
            continue

        log.info("[%d/%d] %s", i + 1, len(configs), cfg["name"])
        try:
            model = build_model(cfg)
            n_params = sum(p.numel() for p in model.parameters())
        except Exception as e:
            log.warning("Build failed: %s", e)
            results.append({"name": cfg["name"], "error": str(e), **cfg})
            continue

        # ZK proxy at initialization (before training)
        zk_constraints = None
        img_size = 16 if args.dataset == "imagenet16" else 32
        try:
            sample = torch.randn(1, 3, img_size, img_size)
            with _suppress_onnx_noise():
                proxy = compute_proxy(model, sample)
            zk_constraints = proxy.constraint_estimate
        except Exception as e:
            log.debug("ZK proxy failed: %s", e)

        # Train
        try:
            top1, top5, train_time = train_one(model, args.epochs, device, args.dataset)
            log.info("  → top1=%.1f%%, constraints=%s", top1,
                     f"{zk_constraints:,}" if zk_constraints else "N/A")

            rec = {
                **cfg,
                "top1": top1,
                "top5": top5,
                "n_params": n_params,
                "constraints": zk_constraints,
                "train_time_s": train_time,
                "epochs": args.epochs,
                "dataset": args.dataset,
                "error": None,
            }
            results.append(rec)
            done.add(cfg["name"])

            # Save incrementally
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            log.warning("Training failed: %s\n%s", e, traceback.format_exc()[-300:])
            results.append({"name": cfg["name"], "error": str(e), **cfg})

    log.info("Done. %d architectures with results.", sum(1 for r in results if "top1" in r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
