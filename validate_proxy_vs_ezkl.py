"""Validate the structural ZK proxy against ezkl-calibrated logrows.

For each architecture in a fixed, seeded subset of the zkml-bench sweep
(CIFAR-10 input shape), this script
  1. builds the model with random (untrained) weights, seed 0,
  2. computes the structural proxy (estimate and snapped logrows),
  3. exports ONNX and runs ezkl gen_settings + calibrate_settings
     (target="resources"), and records the logrows ezkl chooses.
No proofs are generated.  Weights are untrained, so ezkl's range-dependent
calibration may differ from that of trained networks; this is a limitation.

Usage:
    python validate_proxy_vs_ezkl.py --out results/proxy_vs_ezkl.json [--n 48] [--workers 4]
Each model is run in a subprocess with a timeout; failures are recorded, not dropped.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "src")
sys.path.insert(0, ".")


def select_configs(n: int, seed: int = 0) -> list[dict]:
    from generate_sweep import make_sweep_configs

    cfgs = make_sweep_configs()
    # Exclude the largest ResNet-style models so ezkl calibration stays tractable.
    pool = [c for c in cfgs if c["family"] != "resnet"]
    rng = random.Random(seed)
    by_act: dict[str, list[dict]] = {}
    for c in pool:
        by_act.setdefault(c["act"], []).append(c)
    per = max(1, n // len(by_act))
    out = []
    for act in sorted(by_act):
        items = by_act[act]
        rng.shuffle(items)
        out.extend(items[:per])
    # Always include the architectures reported in the paper's ezkl table.
    must = ["mlp_x2px_none_128x64x32", "mlp_x2px_layernorm_128x64x32",
            "dws_relu_none_16_32_64", "dws_relu_none_32_64_128",
            "dws_relu_none_32_64_128_256", "cnn_relu_none_16_32_64",
            "cnn_relu_none_16_32_64_128", "cnn_relu_none_64_128_256",
            "mlp_relu_none_1024x512", "cnn_x3px_layernorm_32_64_128",
            "dws_x2px_groupnorm_32_64"]
    have = {c["name"] for c in out}
    byname = {c["name"]: c for c in cfgs}
    for nm in must:
        if nm in byname and nm not in have:
            out.append(byname[nm])
    return out


def worker(cfg_json: str) -> None:
    """Run one architecture; print a JSON result line."""
    import torch

    cfg = json.loads(cfg_json)
    res = {"name": cfg["name"], "act": cfg["act"], "family": cfg["family"], "error": None}
    try:
        torch.manual_seed(0)
        from generate_sweep import build_model
        from zkml_bench.metrics.zk_proxy import compute_proxy

        model = build_model(cfg).eval()
        x = torch.randn(1, 3, 32, 32)
        p = compute_proxy(model, x)
        res["proxy_estimate"] = int(p.constraint_estimate)
        res["proxy_snapped"] = int(p.constraint_estimate_snapped)
        res["proxy_logrows"] = int(math.log2(p.constraint_estimate_snapped))

        import ezkl

        m = model
        if cfg["act"] != "relu":
            from zkml_bench.provers.ezkl import _wrap_poly_model_for_export

            m = _wrap_poly_model_for_export(model, clip_value=10.0)
            # The wrapper only matches classes with "activation" in the name; the sweep's
            # Square/CubicPlusLinear/SquarePlusLinear do not match, so it returns the model unchanged.
            res["clamp_applied"] = m is not model
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
        n_cal = 40 if cfg["act"] != "relu" else 20
        cal = os.path.join(d, "cal.json")
        data = torch.randn(n_cal, 3, 32, 32)
        json.dump({"input_data": [data[i].flatten().tolist() for i in range(n_cal)]}, open(cal, "w"))
        ok = ezkl.calibrate_settings(cal, onnx_path, settings, target="resources")
        if not ok:
            raise RuntimeError("calibrate_settings returned False")
        s = json.load(open(settings))
        res["ezkl_logrows"] = s["run_args"]["logrows"]
        res["ezkl_scale"] = s["run_args"].get("input_scale")
    except BaseException as e:  # includes pyo3 panics
        res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    print("RESULT " + json.dumps(res))


def run_one(cfg: dict, timeout: int) -> dict:
    try:
        r = subprocess.run([sys.executable, __file__, "--worker", json.dumps(cfg)],
                           capture_output=True, text=True, timeout=timeout)
        for line in r.stdout.splitlines():
            if line.startswith("RESULT "):
                return json.loads(line[7:])
        return {"name": cfg["name"], "act": cfg["act"], "error": "no result: " + r.stderr[-200:]}
    except subprocess.TimeoutExpired:
        return {"name": cfg["name"], "act": cfg["act"], "error": f"timeout>{timeout}s"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/proxy_vs_ezkl.json")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--worker")
    a = ap.parse_args()
    if a.worker:
        worker(a.worker)
        return
    cfgs = select_configs(a.n)
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(lambda c: run_one(c, a.timeout), cfgs))
    json.dump(results, open(a.out, "w"), indent=1)
    ok = [r for r in results if r.get("ezkl_logrows") is not None]
    print(f"{len(ok)}/{len(results)} architectures calibrated; failures recorded in {a.out}")


if __name__ == "__main__":
    main()
