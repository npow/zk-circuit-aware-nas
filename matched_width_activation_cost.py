"""Width-matched activation cost in ezkl.

For identical MLP and CNN skeletons (same widths, no normalization), vary only the
activation (relu, x2, x2px, x3px) and record ezkl's calibrated logrows and
row/assignment counts from settings.json after gen_settings + calibrate_settings
(target="resources").  No proofs are generated.  Weights are random (seed 0), so
calibration ranges depend on the untrained network.

Usage: python matched_width_activation_cost.py [--seed N] --out results/matched_width_activation.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "src")
sys.path.insert(0, ".")

ACTS = ["relu", "x2", "x2px", "x3px"]
SKELETONS = [
    {"family": "mlp", "hidden_dims": [128, 64, 32], "tag": "mlp_128x64x32"},
    {"family": "mlp", "hidden_dims": [256, 128], "tag": "mlp_256x128"},
    {"family": "mlp", "hidden_dims": [512, 256], "tag": "mlp_512x256"},
    {"family": "cnn", "channels": [16, 32], "depthwise": False, "tag": "cnn_16_32"},
    {"family": "cnn", "channels": [16, 32, 64], "depthwise": False, "tag": "cnn_16_32_64"},
]


def worker(cfg_json: str) -> None:
    import torch

    cfg = json.loads(cfg_json)
    res = {"name": cfg["name"], "act": cfg["act"], "skeleton": cfg["tag"], "error": None}
    try:
        torch.manual_seed(int(cfg.get("seed", 0)))
        res["seed"] = int(cfg.get("seed", 0))
        from generate_sweep import build_model
        import ezkl

        model = build_model(cfg).eval()
        m = model
        res["clamp"] = cfg.get("clamp", "per_activation") if cfg["act"] != "relu" else "n/a"
        if cfg["act"] != "relu" and cfg.get("clamp", "per_activation") != "none":
            from zkml_bench.provers.ezkl import _wrap_poly_model_for_export

            m = _wrap_poly_model_for_export(model, clip_value=10.0)
        x = torch.randn(1, 3, 32, 32)
        d = tempfile.mkdtemp()
        onnx_path = os.path.join(d, "m.onnx")
        torch.onnx.export(m, (x,), onnx_path, opset_version=17, input_names=["input"],
                          output_names=["output"], do_constant_folding=True, dynamo=False)
        settings = os.path.join(d, "settings.json")
        ra = ezkl.PyRunArgs()
        ra.input_visibility = "private"
        ra.output_visibility = "public"
        ra.param_visibility = "fixed"
        assert ezkl.gen_settings(onnx_path, settings, py_run_args=ra)
        n_cal = 40
        data = torch.randn(n_cal, 3, 32, 32)
        cal = os.path.join(d, "cal.json")
        json.dump({"input_data": [data[i].flatten().tolist() for i in range(n_cal)]}, open(cal, "w"))
        assert ezkl.calibrate_settings(cal, onnx_path, settings, target="resources")
        s = json.load(open(settings))
        res["logrows"] = s["run_args"]["logrows"]
        res["num_rows"] = s.get("num_rows")
        res["total_assignments"] = s.get("total_assignments")
        res["total_const_size"] = s.get("total_const_size")
        res["input_scale"] = s["run_args"].get("input_scale")
        res["lookup_range"] = s["run_args"].get("lookup_range")
    except BaseException as e:
        res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    print("RESULT " + json.dumps(res))


def run_one(cfg, timeout):
    try:
        r = subprocess.run([sys.executable, __file__, "--worker", json.dumps(cfg)],
                           capture_output=True, text=True, timeout=timeout)
        for line in r.stdout.splitlines():
            if line.startswith("RESULT "):
                return json.loads(line[7:])
        return {"name": cfg["name"], "act": cfg["act"], "skeleton": cfg["tag"], "error": "no result: " + r.stderr[-200:]}
    except subprocess.TimeoutExpired:
        return {"name": cfg["name"], "act": cfg["act"], "skeleton": cfg["tag"], "error": f"timeout>{timeout}s"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/matched_width_activation.json")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--worker")
    ap.add_argument("--seed", type=int, default=0, help="seed for weights and calibration data")
    ap.add_argument("--clamp", default="per_activation", choices=["per_activation", "none"],
                    help="polynomial export: per-activation clamp wrapper (default) or no wrapper")
    a = ap.parse_args()
    if a.worker:
        worker(a.worker)
        return
    cfgs = []
    for sk in SKELETONS:
        for act in ACTS:
            c = dict(sk)
            c.update({"act": act, "norm": "none", "num_classes": 10, "clamp": a.clamp, "seed": a.seed,
                      "name": f"{sk['tag']}_{act}"})
            if a.clamp == "none" and act == "relu":
                continue  # ReLU arm is identical with and without the wrapper
            cfgs.append(c)
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(lambda c: run_one(c, a.timeout), cfgs))
    json.dump(results, open(a.out, "w"), indent=1)
    print(sum(r.get("logrows") is not None for r in results), "/", len(results), "ok")


if __name__ == "__main__":
    main()
