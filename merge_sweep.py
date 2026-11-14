"""Merge parallel sweep worker outputs into a single file."""
import json, glob, sys, os

worker_files = sorted(glob.glob("results/sweep_local/worker_*/cifar10_sweep.json"))
all_results = []
seen = set()
for f in worker_files:
    with open(f) as fh:
        results = json.load(fh)
    for r in results:
        key = r.get("name","")
        if key not in seen:
            seen.add(key)
            all_results.append(r)

os.makedirs("results/sweep", exist_ok=True)
out = "results/sweep/cifar10_sweep.json"
with open(out, "w") as f:
    json.dump(all_results, f, indent=2)

trained = sum(1 for r in all_results if r.get("top1") and not r.get("error"))
print(f"Merged {len(all_results)} total ({trained} trained) → {out}")

# Also print activation family breakdown
acts = {}
for r in all_results:
    if r.get("top1"):
        a = r.get("act", "unknown")
        acts[a] = acts.get(a, 0) + 1
print("By activation:", acts)
