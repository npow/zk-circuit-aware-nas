"""ezkl timing and quantized-accuracy measurements on TRAINED networks with REAL CIFAR-10 inputs.

For each architecture: load seed-0 trained weights, export ONNX, gen_settings + calibrate_settings
(20 real test images), compile, local SRS, setup (pk/vk), then
  - mock prover x --mock-repeats
  - real prove x --prove-repeats (skipped when logrows > --max-prove-logrows), verify, proof size
  - quantized-circuit accuracy: witness outputs vs labels and vs the float PyTorch model on
    --acc-images real test images.
Records ezkl version, host CPU/RAM and the load average before/after every architecture,
because the host is shared.  One architecture at a time, no concurrency.

Usage: python time_ezkl.py --weights DIR --out results/ezkl_timing_trained.jsonl [--names a,b,c]
"""
from __future__ import annotations

import argparse, json, os, platform, subprocess, sys, tempfile, time

sys.path.insert(0, "src"); sys.path.insert(0, ".")
MEAN = (0.4914, 0.4822, 0.4465); STD = (0.2023, 0.1994, 0.2010)

DEFAULT = ["mlp_x2px_none_128x64x32", "mlp_x2px_layernorm_128x64x32", "dws_relu_none_16_32_64",
           "dws_relu_none_32_64_128", "dws_relu_none_32_64_128_256", "cnn_relu_none_16_32_64",
           "cnn_relu_none_16_32_64_128", "cnn_relu_none_64_128_256"]


def host():
    cpu = ""
    try:
        cpu = [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except Exception:
        pass
    mem = ""
    try:
        mem = [l for l in open("/proc/meminfo") if l.startswith("MemTotal")][0].split(":")[1].strip()
    except Exception:
        pass
    import ezkl, torch
    return {"cpu": cpu, "logical_cores": os.cpu_count(), "mem_total": mem, "platform": platform.platform(),
            "python": sys.version.split()[0], "torch": torch.__version__, "ezkl": __import__("importlib.metadata").metadata.version("ezkl")}


def run(name, a):
    import numpy as np, torch, torchvision, ezkl
    from generate_sweep import make_sweep_configs, build_model
    res = {"name": name, "error": None, "load_before": os.getloadavg()}
    d = tempfile.mkdtemp()
    try:
        cfg = next(c for c in make_sweep_configs(10) if c["name"] == name); cfg["dataset"] = "cifar10"
        model = build_model(cfg)
        model.load_state_dict(torch.load(os.path.join(a.weights, f"cifar10_{name}.pt"), map_location="cpu")); model.eval()
        te = torchvision.datasets.CIFAR10("/tmp/data", train=False, download=False)
        n_img = max(a.acc_images, 20)
        x = torch.from_numpy(te.data[:n_img]).permute(0, 3, 1, 2).float().div(255.0)
        x = (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
        y = np.array(te.targets[:n_img])
        onnx_path = os.path.join(d, "m.onnx")
        torch.onnx.export(model, (x[:1],), onnx_path, opset_version=17, input_names=["input"], output_names=["output"],
                          do_constant_folding=True, dynamo=False)
        settings = os.path.join(d, "settings.json")
        ra = ezkl.PyRunArgs(); ra.input_visibility = "private"; ra.output_visibility = "public"; ra.param_visibility = "fixed"
        assert ezkl.gen_settings(onnx_path, settings, py_run_args=ra)
        cal = os.path.join(d, "cal.json")
        json.dump({"input_data": [x[i].flatten().tolist() for i in range(20)]}, open(cal, "w"))
        assert ezkl.calibrate_settings(cal, onnx_path, settings, target="resources")
        s = json.load(open(settings))
        res["logrows"] = s["run_args"]["logrows"]; res["total_assignments"] = s.get("total_assignments")
        res["num_rows"] = s.get("num_rows")
        compiled = os.path.join(d, "m.compiled"); assert ezkl.compile_circuit(onnx_path, compiled, settings)
        srs = os.path.join(d, "srs"); t = time.perf_counter(); ezkl.gen_srs(srs, res["logrows"]); res["srs_s"] = time.perf_counter() - t
        # witness for image 0
        inp = os.path.join(d, "in0.json"); json.dump({"input_data": [x[0].flatten().tolist()]}, open(inp, "w"))
        wit = os.path.join(d, "w.json")
        t = time.perf_counter(); assert ezkl.gen_witness(inp, compiled, wit); res["gen_witness_s"] = time.perf_counter() - t
        res["mock_s"] = []
        for _ in range(a.mock_repeats):
            t = time.perf_counter(); ok = ezkl.mock(wit, compiled); res["mock_s"].append(time.perf_counter() - t); assert ok
        res["mock_repeats"] = a.mock_repeats
        if res["logrows"] <= a.max_prove_logrows:
            vk, pk = os.path.join(d, "vk"), os.path.join(d, "pk")
            t = time.perf_counter(); assert ezkl.setup(compiled, vk, pk, srs); res["setup_s"] = time.perf_counter() - t
            res["prove_s"], res["verify_s"] = [], []
            n_prove = a.prove_repeats_large if res["logrows"] >= a.large_logrows else a.prove_repeats
            for _ in range(n_prove):
                proof = os.path.join(d, "proof.json")
                t = time.perf_counter(); assert ezkl.prove(wit, compiled, pk, proof, "single", srs); res["prove_s"].append(time.perf_counter() - t)
                t = time.perf_counter(); ok = ezkl.verify(proof, settings, vk, srs); res["verify_s"].append(time.perf_counter() - t); assert ok
            res["proof_bytes"] = os.path.getsize(proof)
        else:
            res["prove_skipped"] = f"logrows {res['logrows']} > {a.max_prove_logrows}"
        # quantized accuracy on real images
        agree = correct_q = correct_f = 0
        with torch.no_grad():
            fl = model(x[:a.acc_images]).argmax(1).numpy()
        for i in range(a.acc_images):
            inp = os.path.join(d, f"in{i}.json"); json.dump({"input_data": [x[i].flatten().tolist()]}, open(inp, "w"))
            ezkl.gen_witness(inp, compiled, wit)
            w = json.load(open(wit))
            out = np.array(w["pretty_elements"]["rescaled_outputs"][0], dtype=float).flatten()
            q = int(out.argmax()); correct_q += int(q == y[i]); correct_f += int(fl[i] == y[i]); agree += int(q == fl[i])
        res["quantized_acc"] = {"n": a.acc_images, "float_correct": correct_f, "quantized_correct": correct_q, "agree_with_float": agree}
    except BaseException as e:
        res["error"] = f"{type(e).__name__}: {str(e)[:300]}"
    res["load_after"] = os.getloadavg()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True); ap.add_argument("--out", default="results/ezkl_timing_trained.jsonl")
    ap.add_argument("--names", default=",".join(DEFAULT)); ap.add_argument("--mock-repeats", type=int, default=3)
    ap.add_argument("--prove-repeats", type=int, default=3); ap.add_argument("--max-prove-logrows", type=int, default=22)
    ap.add_argument("--prove-repeats-large", type=int, default=1, help="prove repeats when logrows >= --large-logrows")
    ap.add_argument("--large-logrows", type=int, default=20)
    ap.add_argument("--acc-images", type=int, default=100)
    a = ap.parse_args()
    h = host(); print(json.dumps(h), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps({"host": h, "args": vars(a)}) + "\n")
        for n in a.names.split(","):
            r = run(n, a); f.write(json.dumps(r) + "\n"); f.flush()
            print(n, r.get("logrows"), r.get("mock_s"), r.get("prove_s"), r.get("error") or "", flush=True)


if __name__ == "__main__":
    main()
