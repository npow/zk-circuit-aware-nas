"""GPU-resident reimplementation of the zkml-bench architecture sweep.

Same architectures (generate_sweep.make_sweep_configs) and the same training recipe as
generate_sweep.train_one (SGD momentum 0.9, lr 0.1 or 0.01 with grad-clip 1.0 for
norm-free polynomial nets, weight decay 5e-4, cosine schedule, batch 128, 30 epochs,
random-crop(pad 4) + horizontal flip, per-channel normalization).  The only change is
that the dataset lives on the GPU and augmentation is done there, which removes the CPU
data-loader bottleneck.  Writes one JSON line per (architecture, seed) with test top-1/5,
the zero-cost proxy scores at initialization, and the structural ZK proxy; seed-0 trained
weights are saved for later ezkl calibration on trained networks.

Usage:
  python sweep_gpu.py --dataset cifar10 --seed 0 --shard 0 --nshards 6 --out results/sweep_gpu
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

sys.path.insert(0, "src")
sys.path.insert(0, ".")

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)


def load_data(dataset: str, dev: torch.device):
    DS = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    tr = DS(root="/tmp/data", train=True, download=True)
    te = DS(root="/tmp/data", train=False, download=True)

    def prep(ds):
        x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float().div(255.0)
        x = (x - MEAN) / STD
        return x.to(dev), torch.tensor(ds.targets).to(dev)

    return prep(tr), prep(te)


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random crop (padding 4) and horizontal flip, per sample, on the GPU."""
    B, C, H, W = x.shape
    dev = x.device
    xp = torch.nn.functional.pad(x, (4, 4, 4, 4))  # zero padding in normalized space
    dy = torch.randint(0, 9, (B,), device=dev, generator=gen)
    dx = torch.randint(0, 9, (B,), device=dev, generator=gen)
    ar = torch.arange(H, device=dev)
    rows = (dy[:, None] + ar)[:, None, :, None].expand(B, C, H, W + 8)
    tmp = xp.gather(2, rows)
    cols = (dx[:, None] + ar)[:, None, None, :].expand(B, C, H, W)
    out = tmp.gather(3, cols)
    flip = torch.rand(B, device=dev, generator=gen) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


def train_eval(model, trainset, testset, epochs, dev, seed):
    xtr, ytr = trainset
    xte, yte = testset
    names = {type(m).__name__ for m in model.modules()}
    is_poly = bool(names & {"Square", "CubicPlusLinear", "SquarePlusLinear"})
    has_norm = any(isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)) for m in model.modules())
    safe = is_poly and not has_norm
    lr, clip = (0.01, 1.0) if safe else (0.1, None)
    model = model.to(dev)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    n = xtr.shape[0]
    diverged = False
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=dev, generator=gen)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            x = augment(xtr[idx], gen); y = ytr[idx]
            opt.zero_grad(set_to_none=True)
            try:
                loss = crit(model(x), y)
                if torch.isnan(loss) or torch.isinf(loss):
                    diverged = True
                    break
                loss.backward()
                if clip is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()
            except RuntimeError:
                diverged = True
                break
        sched.step()
    model.eval()
    c1 = c5 = 0
    with torch.no_grad():
        for i in range(0, xte.shape[0], 1000):
            out = model(xte[i:i + 1000]); y = yte[i:i + 1000]
            top5 = out.topk(5, dim=1).indices
            c1 += (top5[:, 0] == y).sum().item()
            c5 += (top5 == y[:, None]).any(1).sum().item()
    return 100 * c1 / xte.shape[0], 100 * c5 / xte.shape[0], time.perf_counter() - t0, diverged, lr


def init_proxies(model, input_shape):
    from zkml_bench.nas.zero_cost_proxies import (near_score, synflow_score, zico_score, meco_score,
                                                  zenas_score, naswot_score, naswot_sign_score)
    from zkml_bench.metrics.zk_proxy import compute_proxy, _suppress_onnx_noise

    r = {}
    model = model.cpu().eval()
    dummy = torch.randn(32, *input_shape)
    for name, fn, kw in [("near", near_score, {"data": dummy}), ("synflow", synflow_score, {"input_shape": input_shape}),
                         ("zico", zico_score, {"data": dummy}), ("meco", meco_score, {"input_shape": input_shape}),
                         ("zenas", zenas_score, {"input_shape": input_shape}),
                         ("naswot", naswot_score, {"input_shape": input_shape}),
                         ("naswot_sign", naswot_sign_score, {"input_shape": input_shape})]:
        try:
            # score functions may mutate/require grad state; use a copy for safety
            import copy
            v = fn(copy.deepcopy(model), **kw, device="cpu")
            r[name] = None if (isinstance(v, float) and (math.isnan(v) or v == float("-inf"))) else float(v)
            if name in ("naswot", "naswot_sign"):
                r[name + "_neg_inf"] = bool(v == float("-inf"))
        except Exception as e:
            r[name] = None
            r[name + "_error"] = str(e)[:120]
    try:
        with _suppress_onnx_noise():
            p = compute_proxy(copy.deepcopy(model), torch.randn(1, *input_shape))
        r["zk_constraints"] = float(p.constraint_estimate)
        r["zk_constraints_snapped"] = float(p.constraint_estimate_snapped)
        r["zk_logrows_est"] = int(math.log2(p.constraint_estimate_snapped))
    except Exception as e:
        r["zk_error"] = str(e)[:120]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default="results/sweep_gpu")
    ap.add_argument("--weights", default=None, help="dir to save seed-0 trained weights")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="comma-separated architecture names")
    ap.add_argument("--no-proxies", action="store_true", help="skip init-time proxy scoring (accuracy labels only)")
    a = ap.parse_args()

    from generate_sweep import make_sweep_configs, build_model
    dev = torch.device("cuda")
    ncls = 100 if a.dataset == "cifar100" else 10
    cfgs = make_sweep_configs(ncls)
    for c in cfgs:
        c["dataset"] = a.dataset
    cfgs = [(i, c) for i, c in enumerate(cfgs) if i % a.nshards == a.shard]
    if a.only:
        keep = set(a.only.split(","))
        cfgs = [(i, c) for i, c in enumerate(make_sweep_configs(ncls)) if c["name"] in keep]
        for _, c in cfgs:
            c["dataset"] = a.dataset
    if a.limit:
        cfgs = cfgs[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.dataset}_seed{a.seed}_shard{a.shard}of{a.nshards}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)["name"] for l in open(path) if l.strip()}
    trainset, testset = load_data(a.dataset, dev)
    for i, c in cfgs:
        if c["name"] in done:
            continue
        rec = {k: c[k] for k in c}
        rec.update({"index": i, "seed": a.seed, "epochs": a.epochs})
        try:
            torch.manual_seed(1000003 * a.seed + i)
            model = build_model(c)
            rec["params"] = sum(p.numel() for p in model.parameters())
            if not a.no_proxies:
                rec.update(init_proxies(model, (3, 32, 32)))
            torch.manual_seed(1000003 * a.seed + i)
            model = build_model(c)  # same init as scored above
            top1, top5, tt, div, lr = train_eval(model, trainset, testset, a.epochs, dev, 1000003 * a.seed + i)
            rec.update({"top1": top1, "top5": top5, "train_time_s": tt, "diverged": div, "lr": lr})
            if a.weights and a.seed == 0:
                os.makedirs(a.weights, exist_ok=True)
                torch.save(model.cpu().state_dict(), os.path.join(a.weights, f"{a.dataset}_{c['name']}.pt"))
        except BaseException as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(i, c["name"], rec.get("top1"), rec.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
