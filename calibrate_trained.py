"""ezkl calibration of TRAINED sweep networks on REAL CIFAR-10 inputs.

For every architecture with saved seed-0 weights: load weights, extract ONNX-graph features,
export ONNX (legacy exporter, no clamping), run ezkl gen_settings + calibrate_settings
(target="resources") using real CIFAR-10 test images as calibration data, and record the
calibrated logrows and ezkl's num_rows / total_assignments.  No proofs are generated.
Each architecture runs in a subprocess with a timeout; failures are recorded, not dropped.

Usage: python calibrate_trained.py --weights DIR --out results/trained_calibration.jsonl [--workers 8]
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "src"); sys.path.insert(0, ".")
MEAN = (0.4914, 0.4822, 0.4465); STD = (0.2023, 0.1994, 0.2010)


def worker(name: str, weights: str, n_cal: int) -> None:
    import numpy as np, torch, torchvision
    res = {"name": name, "error": None, "n_cal": n_cal}
    try:
        import ezkl
        from generate_sweep import make_sweep_configs, build_model
        from zkml_bench.metrics.zk_features import extract_features
        from zkml_bench.metrics.zk_proxy import compute_proxy
        cfg = next(c for c in make_sweep_configs(10) if c["name"] == name); cfg["dataset"] = "cifar10"
        model = build_model(cfg)
        model.load_state_dict(torch.load(os.path.join(weights, f"cifar10_{name}.pt"), map_location="cpu"))
        model.eval()
        te = torchvision.datasets.CIFAR10("/tmp/data", train=False, download=False)
        x = torch.from_numpy(te.data[:n_cal]).permute(0, 3, 1, 2).float().div(255.0)
        x = (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
        sample = x[:1]
        p = compute_proxy(model, sample)
        res["proxy_constraints"] = int(p.constraint_estimate); res["proxy_snapped"] = int(p.constraint_estimate_snapped)
        res["proxy_logrows"] = int(np.log2(p.constraint_estimate_snapped))
        res["features"] = extract_features(model, sample)
        d = tempfile.mkdtemp(); onnx_path = os.path.join(d, "m.onnx")
        torch.onnx.export(model, (sample,), onnx_path, opset_version=17, input_names=["input"],
                          output_names=["output"], do_constant_folding=True, dynamo=False)
        settings = os.path.join(d, "settings.json")
        ra = ezkl.PyRunArgs(); ra.input_visibility = "private"; ra.output_visibility = "public"; ra.param_visibility = "fixed"
        assert ezkl.gen_settings(onnx_path, settings, py_run_args=ra)
        cal = os.path.join(d, "cal.json")
        json.dump({"input_data": [x[i].flatten().tolist() for i in range(n_cal)]}, open(cal, "w"))
        assert ezkl.calibrate_settings(cal, onnx_path, settings, target="resources")
        s = json.load(open(settings))
        res.update({"logrows": s["run_args"]["logrows"], "num_rows": s.get("num_rows"),
                    "total_assignments": s.get("total_assignments"), "input_scale": s["run_args"].get("input_scale"),
                    "lookup_range": s["run_args"].get("lookup_range")})
    except BaseException as e:
        res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    print("RESULT " + json.dumps(res))


def run_one(name, weights, n_cal, timeout):
    try:
        r = subprocess.run([sys.executable, __file__, "--worker", name, "--weights", weights, "--ncal", str(n_cal)],
                           capture_output=True, text=True, timeout=timeout)
        for l in r.stdout.splitlines():
            if l.startswith("RESULT "): return json.loads(l[7:])
        return {"name": name, "error": "no result: " + r.stderr[-200:]}
    except subprocess.TimeoutExpired:
        return {"name": name, "error": f"timeout>{timeout}s"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--out", default="results/trained_calibration.jsonl")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--ncal", type=int, default=20); ap.add_argument("--worker"); ap.add_argument("--only")
    ap.add_argument("--sweep", default=None, help="sweep dir: order smallest-estimated-first and apply the skip rules")
    ap.add_argument("--max-est", type=int, default=25, help="skip networks whose proxy-estimated logrows exceed this")
    ap.add_argument("--skipped-out", default=None, help="write skipped networks and reasons here")
    a = ap.parse_args()
    if a.worker:
        worker(a.worker, a.weights, a.ncal); return
    names = sorted(f[len("cifar10_"):-3] for f in os.listdir(a.weights) if f.startswith("cifar10_") and f.endswith(".pt"))
    if a.only: names = [n for n in names if n in set(a.only.split(","))]
    skipped = []
    if a.sweep:
        import glob
        info = {}
        for f in glob.glob(os.path.join(a.sweep, "cifar10_seed0_shard*.jsonl")):
            for l in open(f):
                r = json.loads(l); info[r["name"]] = r
        keep = []
        for n in names:
            r = info.get(n, {})
            if r.get("diverged") or r.get("top1", 100) < 15:
                skipped.append({"name": n, "reason": "seed-0 training diverged or ended at chance level (top1<15)"})
            elif r.get("zk_logrows_est", 0) > a.max_est:
                skipped.append({"name": n, "reason": f"proxy-estimated logrows {r.get('zk_logrows_est')} > {a.max_est}"})
            else:
                keep.append(n)
        names = sorted(keep, key=lambda n: (info.get(n, {}).get("zk_logrows_est", 99), n))
        if a.skipped_out:
            json.dump(skipped, open(a.skipped_out, "w"), indent=1)
    done = set()
    if os.path.exists(a.out): done = {json.loads(l)["name"] for l in open(a.out) if l.strip()}
    todo = [n for n in names if n not in done]
    print(len(todo), "to calibrate;", len(done), "done", flush=True)
    with ThreadPoolExecutor(a.workers) as ex, open(a.out, "a") as f:
        for r in ex.map(lambda n: run_one(n, a.weights, a.ncal, a.timeout), todo):
            f.write(json.dumps(r) + "\n"); f.flush(); print(r["name"], r.get("logrows"), r.get("error") or "", flush=True)


if __name__ == "__main__":
    main()
