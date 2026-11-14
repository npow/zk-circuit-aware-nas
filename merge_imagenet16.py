"""Merge ImageNet-16-120 sweep results and run proxy analysis."""
import json, glob, os, subprocess, sys

# Merge
all_results = []; seen = set()
for f in sorted(glob.glob("results/sweep_imagenet16/worker_*/imagenet16_sweep.json")):
    with open(f) as fh: d = json.load(fh)
    for r in d:
        if r.get("top1") is not None and r["name"] not in seen:
            seen.add(r["name"]); all_results.append(r)

os.makedirs("results/sweep_imagenet16", exist_ok=True)
with open("results/sweep_imagenet16/imagenet16_sweep.json", "w") as f:
    json.dump(all_results, f, indent=2)

acts = {}
for r in all_results: acts[r.get("act","?")] = acts.get(r.get("act","?"),0)+1
print(f"Merged {len(all_results)} ImageNet-16-120 architectures: {acts}")

# Run proxy analysis if enough data
if len(all_results) >= 50:
    print("Running proxy analysis...")
    subprocess.run([sys.executable, "proxy_sweep_experiment.py",
                   "--sweep", "results/sweep_imagenet16/imagenet16_sweep.json",
                   "--out-prefix", "results/proxy_imagenet16"])
