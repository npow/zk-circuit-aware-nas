"""Proxy correlation experiment on the systematic sweep dataset.

Uses build_model() from generate_sweep.py to reconstruct architectures
(no stored code string needed — uses config params directly).
"""
import json, math, sys, os, logging, warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
sys.path.insert(0, "src")

import torch, torch.nn as nn
import numpy as np
from scipy import stats

from generate_sweep import build_model
from zkml_bench.nas.zero_cost_proxies import (
    near_score, synflow_score, zico_score, meco_score, zenas_score, naswot_score
)
from zkml_bench.metrics.zk_proxy import compute_proxy, _suppress_onnx_noise

INPUT_SHAPE = (3, 32, 32)
DUMMY = torch.randn(32, *INPUT_SHAPE)


def compute_proxies(cfg: dict) -> dict:
    try:
        model = build_model(cfg)
    except Exception as e:
        return {"error": str(e)}

    model.eval()
    r = {}

    for name, fn, kwargs in [
        ("near",    near_score,    {"data": DUMMY}),
        ("synflow", synflow_score, {"input_shape": INPUT_SHAPE}),
        ("zico",    zico_score,    {"data": DUMMY}),
        ("meco",    meco_score,    {"input_shape": INPUT_SHAPE}),
        ("zenas",   zenas_score,   {"input_shape": INPUT_SHAPE}),
        ("naswot",  naswot_score,  {"input_shape": INPUT_SHAPE}),
    ]:
        try:
            score = fn(model, **kwargs, device="cpu")
            r[name] = None if score == float("-inf") or (isinstance(score, float) and math.isnan(score)) else score
            if name == "naswot":
                r["naswot_degenerate"] = (score == float("-inf"))
        except Exception as e:
            r[name] = None

    # Structural ZK proxy
    try:
        sample = torch.randn(1, *INPUT_SHAPE)
        with _suppress_onnx_noise():
            p = compute_proxy(model, sample)
        r["zk_proxy"] = p.constraint_estimate
    except Exception:
        r["zk_proxy"] = None

    return r


def spearman_ci(x, y, n_boot=1000):
    rho, pval = stats.spearmanr(x, y)
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        try:
            r, _ = stats.spearmanr(np.array(x)[idx], np.array(y)[idx])
            if not math.isnan(r):
                boots.append(r)
        except: pass
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else [float("nan")]*2
    return {"rho": float(rho), "pval": float(pval), "ci_95": ci, "n": len(x), "significant": bool(pval < 0.05)}


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", default="results/sweep/cifar10_sweep.json")
    p.add_argument("--out-prefix", default="results/proxy_sweep")
    args = p.parse_args()

    with open(args.sweep) as f:
        sweep = json.load(f)
    archs = [r for r in sweep if r.get("top1") is not None and not r.get("error")]
    log.info("Loaded %d trained architectures from %s", len(archs), args.sweep)

    # Compute proxies
    results = []
    for i, cfg in enumerate(archs):
        if i % 20 == 0:
            log.info("[%d/%d] %s", i+1, len(archs), cfg.get("name","?"))
        proxies = compute_proxies(cfg)
        results.append({**cfg, **proxies})

    # Save raw results
    os.makedirs("results", exist_ok=True)
    with open("results/proxy_sweep_raw.json", "w") as f:
        json.dump(results, f, indent=2)

    # Correlation analysis
    proxy_names = ["near", "synflow", "zico", "meco", "zenas", "naswot", "zk_proxy"]
    targets = {"accuracy": "top1", "zk_cost": "constraints"}
    families = {"all": results}
    for fam in ["relu", "x2", "x3px", "x2px"]:
        sub = [r for r in results if r.get("act") == fam]
        if sub: families[fam] = sub

    print("\n" + "="*85)
    print(f"PROXY CORRELATIONS — SYSTEMATIC SWEEP (n={len(results)})")
    print("="*85)

    correlations = {}
    for fam_name, subset in families.items():
        print(f"\n--- {fam_name.upper()} (n={len(subset)}) ---")
        print(f"{'Proxy':<12} {'ρ(accuracy)':>12} {'p':>7} {'CI':>18}  |  {'ρ(ZK cost)':>12} {'p':>7}")
        print("-"*75)
        fam_corrs = {}
        for proxy in proxy_names:
            row = {}
            for tname, tkey in targets.items():
                pairs = [(r[proxy], r[tkey]) for r in subset
                         if r.get(proxy) is not None and r.get(tkey) is not None]
                if len(pairs) >= 5:
                    xs, ys = zip(*pairs)
                    row[tname] = spearman_ci(list(xs), list(ys))
                else:
                    row[tname] = {"n": len(pairs), "rho": None}
            fam_corrs[proxy] = row

            def fmt(d):
                if not d or d.get("rho") is None: return f"{'N/A':>12} {'':>7} {'':>18}"
                sig = "*" if d.get("significant") else " "
                ci = d.get("ci_95", [float("nan")]*2)
                return f"{d['rho']:>+11.3f}{sig} {d['pval']:>7.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]"

            if fam_name == "all":
                print(f"{proxy:<12} {fmt(row['accuracy'])}  |  {fmt(row['zk_cost'])}")

        correlations[fam_name] = fam_corrs

    # NASWOT degeneracy by activation family
    print("\n=== NASWOT DEGENERACY BY ACTIVATION ===")
    for fam in ["relu", "x2", "x3px", "x2px"]:
        sub = [r for r in results if r.get("act") == fam]
        degen = sum(1 for r in sub if r.get("naswot_degenerate") is True)
        finite = sum(1 for r in sub if r.get("naswot") is not None)
        print(f"  {fam:<6}: {degen}/{len(sub)} degenerate ({100*degen/max(len(sub),1):.0f}%)  finite={finite}")

    # Save full analysis
    with open("results/proxy_sweep_correlations.json", "w") as f:
        json.dump({"n": len(results), "correlations": correlations}, f, indent=2, default=str)
    log.info("Saved → results/proxy_sweep_correlations.json")


if __name__ == "__main__":
    main()
