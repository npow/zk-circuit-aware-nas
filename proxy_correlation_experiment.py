"""Proxy Correlation Experiment
================================
Runs all zero-cost proxies on every trained architecture in the NAS logs,
then measures Spearman rank correlation against:
  (a) 30-epoch top-1 accuracy
  (b) ZK circuit constraint count (ezkl cost proxy)

This is the core empirical contribution for the "ZK-Circuit-Aware NAS" paper.

Usage:
    python proxy_correlation_experiment.py [--dataset cifar10|cifar100|both]
    python proxy_correlation_experiment.py --out results/proxy_correlations.json
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import traceback
import warnings

import torch
import torch.nn as nn
from scipy import stats

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load architectures from NAS logs
# ---------------------------------------------------------------------------

def load_architectures(paths: list[str]) -> list[dict]:
    """Load all architectures with code + accuracy + constraints from NAS logs."""
    archs = []
    for path in paths:
        if not os.path.exists(path):
            log.warning("NAS log not found: %s", path)
            continue
        with open(path) as f:
            entries = json.load(f)
        dataset = "cifar10" if "cifar10" in path else "cifar100"
        for e in entries:
            if not (e.get("top1") and e.get("constraints") and e.get("code")):
                continue
            if e.get("error"):
                continue
            archs.append({
                "dataset": dataset,
                "name": e.get("candidate_name", f"arch_{len(archs)}"),
                "top1": float(e["top1"]),
                "constraints": int(e["constraints"]),
                "n_params": e.get("n_params", 0),
                "nonlinear_ops": e.get("nonlinear_ops", 0),
                "accepted": e.get("accepted", False),
                "code": e["code"],
            })
    log.info("Loaded %d architectures total", len(archs))
    return archs


def instantiate_model(code: str, num_classes: int = 10) -> nn.Module | None:
    """Exec the stored model code and return the model instance."""
    ns = {"torch": torch, "nn": nn}
    try:
        exec(compile(code, "<nas_arch>", "exec"), ns)  # noqa: S102
        get_model = ns.get("get_model")
        if get_model is None:
            return None
        return get_model(num_classes=num_classes, input_shape=(3, 32, 32))
    except Exception as e:
        log.debug("Failed to instantiate model: %s", e)
        return None


# ---------------------------------------------------------------------------
# Proxy computation
# ---------------------------------------------------------------------------

from zkml_bench.nas.zero_cost_proxies import (
    near_score,
    synflow_score,
    zico_score,
    meco_score,
    zenas_score,
    naswot_score,
)
from zkml_bench.metrics.zk_proxy import compute_proxy, _suppress_onnx_noise


PROXIES = {
    "near": None,
    "synflow": None,
    "zico": None,
    "meco": None,
    "zenas": None,
    "naswot": None,      # expected to fail for x² architectures
    "zk_proxy": None,    # structural ZK cost proxy
    "n_params": None,
    "nonlinear_ops": None,
}

INPUT_SHAPE = (3, 32, 32)
DUMMY_INPUT = torch.randn(32, *INPUT_SHAPE)  # 32-sample batch for proxies needing data


def compute_all_proxies(arch: dict) -> dict:
    """Compute all proxies for one architecture. Returns dict of scores."""
    num_classes = 100 if arch["dataset"] == "cifar100" else 10
    model = instantiate_model(arch["code"], num_classes=num_classes)
    if model is None:
        return {}

    results = {
        "name": arch["name"],
        "dataset": arch["dataset"],
        "top1": arch["top1"],
        "constraints": arch["constraints"],
        "n_params": arch.get("n_params", 0),
        "nonlinear_ops": arch.get("nonlinear_ops", 0),
        "accepted": arch.get("accepted", False),
    }

    model.eval()

    # 1. NEAR
    try:
        results["near"] = near_score(model, data=DUMMY_INPUT, device="cpu")
    except Exception as e:
        log.debug("NEAR failed for %s: %s", arch["name"], e)
        results["near"] = None

    # 2. Synflow (data-free, activation-agnostic)
    try:
        results["synflow"] = synflow_score(model, INPUT_SHAPE, device="cpu")
    except Exception as e:
        log.debug("Synflow failed for %s: %s", arch["name"], e)
        results["synflow"] = None

    # 3. ZiCo
    try:
        results["zico"] = zico_score(model, data=DUMMY_INPUT, device="cpu")
    except Exception as e:
        log.debug("ZiCo failed for %s: %s", arch["name"], e)
        results["zico"] = None

    # 4. MeCo (single forward pass, one random input)
    try:
        results["meco"] = meco_score(model, INPUT_SHAPE, device="cpu")
    except Exception as e:
        log.debug("MeCo failed for %s: %s", arch["name"], e)
        results["meco"] = None

    # 5. ZenNAS (Gaussian perturbation sensitivity, BN-free, data-free)
    try:
        results["zenas"] = zenas_score(model, INPUT_SHAPE, device="cpu")
    except Exception as e:
        log.debug("ZenNAS failed for %s: %s", arch["name"], e)
        results["zenas"] = None

    # 6. NASWOT (binary ReLU patterns — expected to return -inf for x² architectures)
    try:
        score = naswot_score(model, INPUT_SHAPE, device="cpu")
        results["naswot"] = score
        results["naswot_degenerate"] = (score == float("-inf"))
    except Exception as e:
        log.debug("NASWOT failed for %s: %s", arch["name"], e)
        results["naswot"] = None
        results["naswot_degenerate"] = None

    # Detect activation family by inspecting module types in the built model.
    # Do NOT use code-string matching: the LLM-generated REASONING field
    # mentions "polynomial" as a concept even in ReLU architectures.
    module_names = {type(m).__name__ for m in model.modules()}
    has_relu = "ReLU" in module_names or "LeakyReLU" in module_names
    has_poly_module = bool(module_names & {"Square", "CubicPlusLinear", "SquarePlusLinear"})
    has_gelu = "GELU" in module_names
    has_silu = "SiLU" in module_names

    if has_relu and not has_poly_module:
        results["activation_family"] = "relu"
    elif has_poly_module and not has_relu:
        results["activation_family"] = "polynomial"
    elif has_gelu and not has_relu and not has_poly_module:
        results["activation_family"] = "gelu"
    elif has_silu and not has_relu and not has_poly_module:
        results["activation_family"] = "silu"
    elif has_relu and has_poly_module:
        results["activation_family"] = "mixed_relu_poly"
    else:
        results["activation_family"] = "unknown"

    # 7. Structural ZK cost proxy (from zk_proxy.py)
    try:
        sample = torch.randn(1, *INPUT_SHAPE)
        with _suppress_onnx_noise():
            proxy_result = compute_proxy(model, sample)
        results["zk_proxy"] = proxy_result.constraint_estimate
        results["zk_proxy_snapped"] = proxy_result.constraint_estimate_snapped
        results["zk_nonlinear"] = proxy_result.nonlinear_op_count
    except Exception as e:
        log.debug("ZK proxy failed for %s: %s", arch["name"], e)
        results["zk_proxy"] = None

    return results


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def spearman_with_ci(x: list, y: list, n_bootstrap: int = 2000) -> dict:
    """Spearman ρ with p-value and 95% bootstrap CI."""
    import numpy as np
    x, y = np.array(x), np.array(y)
    rho, pval = stats.spearmanr(x, y)

    # Bootstrap CI
    rhos = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(x), len(x))
        try:
            r, _ = stats.spearmanr(x[idx], y[idx])
            if not math.isnan(r):
                rhos.append(r)
        except Exception:
            pass
    ci_lo = float(np.percentile(rhos, 2.5)) if rhos else float("nan")
    ci_hi = float(np.percentile(rhos, 97.5)) if rhos else float("nan")

    return {
        "rho": float(rho),
        "pval": float(pval),
        "ci_95": [float(ci_lo), float(ci_hi)],
        "n": int(len(x)),
        "significant": bool(pval < 0.05),
    }


def run_correlation_analysis(results: list[dict]) -> dict:
    """Compute Spearman correlations of each proxy against accuracy and ZK cost."""
    import numpy as np

    proxy_names = ["near", "synflow", "zico", "meco", "zenas", "naswot", "zk_proxy", "n_params", "nonlinear_ops"]
    targets = {
        "accuracy": "top1",
        "zk_cost": "constraints",
    }

    # Separate by dataset
    datasets = {"combined": results}
    for ds in ["cifar10", "cifar100"]:
        subset = [r for r in results if r.get("dataset") == ds]
        if subset:
            datasets[ds] = subset

    analysis = {}
    for ds_name, subset in datasets.items():
        analysis[ds_name] = {}
        for proxy in proxy_names:
            analysis[ds_name][proxy] = {}
            for target_name, target_key in targets.items():
                # Filter to rows where both proxy and target are available
                pairs = [
                    (r[proxy], r[target_key])
                    for r in subset
                    if r.get(proxy) is not None
                    and r.get(target_key) is not None
                    and not math.isnan(float(r[proxy]))
                    and not math.isnan(float(r[target_key]))
                ]
                if len(pairs) < 5:
                    analysis[ds_name][proxy][target_name] = {"n": len(pairs), "rho": None}
                    continue
                xs, ys = zip(*pairs)
                corr = spearman_with_ci(list(xs), list(ys))
                analysis[ds_name][proxy][target_name] = corr

    return analysis


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_table(analysis: dict, dataset: str = "combined"):
    """Print a formatted correlation table."""
    if dataset not in analysis:
        return
    data = analysis[dataset]
    proxy_names = ["near", "synflow", "zico", "meco", "zenas", "naswot", "zk_proxy", "n_params", "nonlinear_ops"]
    print(f"\n{'='*80}")
    print(f"Spearman Rank Correlations — {dataset.upper()}")
    print(f"{'='*80}")
    print(f"{'Proxy':<16} {'ρ(accuracy)':>12} {'p-val':>8} {'95% CI':>18}  |  {'ρ(ZK cost)':>12} {'p-val':>8} {'95% CI':>18}")
    print("-"*80)
    for proxy in proxy_names:
        if proxy not in data:
            continue
        acc = data[proxy].get("accuracy", {})
        zk = data[proxy].get("zk_cost", {})

        def fmt(d):
            if not d or d.get("rho") is None:
                return f"{'N/A':>12} {'':>8} {'':>18}"
            rho = d["rho"]
            pval = d["pval"]
            ci = d.get("ci_95", [float("nan"), float("nan")])
            sig = "*" if d.get("significant") else " "
            ci_str = f"[{ci[0]:+.3f},{ci[1]:+.3f}]"
            return f"{rho:>+11.3f}{sig} {pval:>8.3f} {ci_str:>18}"

        print(f"{proxy:<16} {fmt(acc)}  |  {fmt(zk)}")
    print()

    # Summary: best proxy for each target
    def best_proxy(target):
        best, best_rho = None, -999
        for p in proxy_names:
            r = data.get(p, {}).get(target, {}).get("rho")
            if r is not None and not math.isnan(r) and r > best_rho:
                best_rho = r
                best = p
        return best, best_rho

    acc_best, acc_rho = best_proxy("accuracy")
    zk_best, zk_rho = best_proxy("zk_cost")
    print(f"Best accuracy proxy:  {acc_best} (ρ={acc_rho:+.3f})")
    print(f"Best ZK cost proxy:   {zk_best} (ρ={zk_rho:+.3f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="both", choices=["cifar10", "cifar100", "both"])
    parser.add_argument("--out", default="results/proxy_correlations.json")
    args = parser.parse_args()

    paths = []
    if args.dataset in ("cifar10", "both"):
        paths.append("results/cifar10_nas/nas_log.json")
    if args.dataset in ("cifar100", "both"):
        paths.append("results/cifar100_nas/nas_log.json")

    archs = load_architectures(paths)
    if not archs:
        log.error("No architectures loaded. Exiting.")
        sys.exit(1)

    log.info("Computing proxies for %d architectures...", len(archs))
    proxy_results = []
    for i, arch in enumerate(archs):
        log.info("  [%d/%d] %s (dataset=%s, top1=%.1f, constraints=%d)",
                 i + 1, len(archs), arch["name"], arch["dataset"],
                 arch["top1"], arch["constraints"])
        try:
            r = compute_all_proxies(arch)
            if r:
                proxy_results.append(r)
        except Exception as e:
            log.warning("  Failed: %s\n%s", e, traceback.format_exc()[-300:])

    log.info("Proxy computation done: %d/%d succeeded", len(proxy_results), len(archs))

    # Correlation analysis
    analysis = run_correlation_analysis(proxy_results)

    # Print tables
    for ds in ["combined", "cifar10", "cifar100"]:
        print_table(analysis, ds)

    # NASWOT degeneracy analysis — key finding for the paper
    print("\n=== NASWOT Degeneracy Analysis (Key Paper Finding) ===")
    poly_archs = [r for r in proxy_results if r.get("activation_family") == "polynomial"]
    relu_archs = [r for r in proxy_results if r.get("activation_family") == "relu"]
    poly_degen = sum(1 for r in poly_archs if r.get("naswot_degenerate") is True)
    relu_degen = sum(1 for r in relu_archs if r.get("naswot_degenerate") is True)
    print(f"Polynomial architectures: {len(poly_archs)} total, "
          f"{poly_degen} NASWOT-degenerate ({100*poly_degen/max(len(poly_archs),1):.0f}%)")
    print(f"ReLU architectures:       {len(relu_archs)} total, "
          f"{relu_degen} NASWOT-degenerate ({100*relu_degen/max(len(relu_archs),1):.0f}%)")

    # Proxy performance split by activation family
    print("\n=== Spearman ρ(accuracy) by Activation Family ===")
    for proxy in ["near", "synflow", "zenas", "zico", "naswot"]:
        for family, subset in [("poly", poly_archs), ("relu", relu_archs)]:
            pairs = [(r[proxy], r["top1"]) for r in subset
                     if r.get(proxy) is not None
                     and r[proxy] != float("-inf")
                     and not math.isnan(float(r[proxy]))]
            if len(pairs) >= 4:
                xs, ys = zip(*pairs)
                rho, pval = stats.spearmanr(xs, ys)
                print(f"  {proxy:<12} [{family}] n={len(pairs):2d}  ρ={rho:+.3f}  p={pval:.3f}"
                      f"  {'*' if pval < 0.05 else ''}")
            else:
                print(f"  {proxy:<12} [{family}] n={len(pairs):2d}  (insufficient data)")

    # Save — convert numpy types to Python natives for JSON serialization
    import numpy as np

    def _to_native(obj):
        if isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    class _NpEncoder(json.JSONEncoder):
        def default(self, o):
            native = _to_native(o)
            if native is not o:
                return native
            return super().default(o)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    output = {
        "n_architectures": len(proxy_results),
        "proxy_results": proxy_results,
        "correlations": analysis,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2, cls=_NpEncoder)
    log.info("Results saved to %s", args.out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
