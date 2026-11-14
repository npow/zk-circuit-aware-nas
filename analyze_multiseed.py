"""Summarize the 26-architecture, 3-seed, 30- vs 100-epoch validation (results/multiseed/e{30,100}/*.jsonl).
Writes results/multiseed_summary.json."""
import collections, glob, json, statistics
import numpy as np
from scipy import stats

runs = collections.defaultdict(dict)  # (name, epochs) -> {seed: rec}
for ep in (30, 100):
    for f in glob.glob(f"results/multiseed/e{ep}/cifar10_seed*_shard*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if "top1" in r:
                runs[(r["name"], ep)][r["seed"]] = r
names = sorted({n for n, _ in runs})
out = {"n_archs": len(names), "seeds": sorted({s for v in runs.values() for s in v}), "per_arch": {}}
a30, a100 = [], []
pair30, pair100, seedwise = [], [], collections.defaultdict(lambda: ([], []))
failed = []
for n in names:
    e = {}
    for ep in (30, 100):
        recs = runs.get((n, ep), {})
        v = [r["top1"] for r in recs.values()]
        e[ep] = {"n": len(v), "mean": float(np.mean(v)) if v else None, "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                 "diverged": sum(bool(r.get("diverged")) for r in recs.values())}
    out["per_arch"][n] = e
    if e[30]["mean"] is not None and e[100]["mean"] is not None:
        a30.append(e[30]["mean"]); a100.append(e[100]["mean"])
    for s in out["seeds"]:
        r30, r100 = runs.get((n, 30), {}).get(s), runs.get((n, 100), {}).get(s)
        if r30 and r100:
            pair30.append(r30["top1"]); pair100.append(r100["top1"]); seedwise[s][0].append(r30["top1"]); seedwise[s][1].append(r100["top1"])
            if r30.get("diverged") or r100.get("diverged"):
                failed.append((n, s))
out["arch_mean_spearman"] = float(stats.spearmanr(a30, a100)[0]); out["n_arch_mean_pairs"] = len(a30)
out["paired_run_spearman"] = float(stats.spearmanr(pair30, pair100)[0]); out["n_paired_runs"] = len(pair30)
out["seedwise_spearman"] = {str(s): float(stats.spearmanr(*v)[0]) for s, v in seedwise.items()}
out["mean_gain_points"] = float(np.mean(np.array(pair100) - np.array(pair30)))
out["runs_done"] = sum(len(v) for v in runs.values()); out["runs_planned"] = len(names) * 2 * len(out["seeds"])
out["runs_diverged"] = sum(bool(r.get("diverged")) for v in runs.values() for r in v.values())
out["max_sd_30"] = max((e[30]["sd"] or 0) for e in out["per_arch"].values())
json.dump(out, open("results/multiseed_summary.json", "w"), indent=1)
print({k: v for k, v in out.items() if k != "per_arch"})
