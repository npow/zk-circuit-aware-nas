"""Measure actual proof size and verification time for Pareto architectures.
Adds the missing piece: prover_time, proof_size_bytes, verifier_time.
"""
import json, sys, os, math, time, tempfile, logging, warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)
sys.path.insert(0, "src")
import torch
from generate_sweep import build_model

# Focus on the fast polynomial architectures (logrows=18) + one comparison point
TARGETS = {
    "mlp_x2px_none_128x64x32":      {"logrows": 18, "top1": 51.1, "pareto": True},
    "mlp_x2px_layernorm_128x64x32":  {"logrows": 18, "top1": 52.0, "pareto": True},
    "dws_relu_none_16_32_64":         {"logrows": 21, "top1": 62.9, "pareto": True},
}

with open("results/sweep/cifar10_sweep.json") as f:
    sweep = {r["name"]: r for r in json.load(f) if r.get("top1")}

results = []
for name, meta in TARGETS.items():
    if name not in sweep:
        log.warning("Not found: %s", name)
        continue
    rec = sweep[name]
    log.info("Running full prove+verify for %s (logrows=%d)", name, meta["logrows"])
    try:
        import ezkl
        model = build_model(rec); model.eval()
        tmpdir = tempfile.mkdtemp()
        def p(f): return os.path.join(tmpdir, f)

        # ONNX export
        x = torch.randn(1, 3, 32, 32)
        torch.onnx.export(model, (x,), p("model.onnx"), opset_version=17,
                         input_names=["input"], output_names=["output"],
                         do_constant_folding=True, dynamo=False)

        # Settings + calibrate
        run_args = ezkl.PyRunArgs()
        run_args.input_visibility = "private"
        run_args.output_visibility = "public"
        run_args.param_visibility = "fixed"
        ezkl.gen_settings(p("model.onnx"), p("settings.json"), py_run_args=run_args)
        inputs = torch.randn(20, 3, 32, 32)
        cal = {"input_data": [inputs[i].flatten().tolist() for i in range(20)]}
        with open(p("cal.json"), "w") as f: json.dump(cal, f)
        ezkl.calibrate_settings(p("cal.json"), p("model.onnx"), p("settings.json"), target="resources")

        # Compile + SRS + keys
        ezkl.compile_circuit(p("model.onnx"), p("model.compiled"), p("settings.json"))
        ezkl.gen_srs(p("srs.srs"), logrows=meta["logrows"])
        ezkl.setup(p("model.compiled"), p("vk.key"), p("pk.key"), p("srs.srs"))

        # Witness
        inp = {"input_data": [torch.randn(1, 3, 32, 32).flatten().tolist()]}
        with open(p("input.json"), "w") as f: json.dump(inp, f)
        ezkl.gen_witness(p("input.json"), p("model.compiled"), p("witness.json"))

        # PROVE (full proof)
        t0 = time.perf_counter()
        ezkl.prove(p("witness.json"), p("model.compiled"), p("pk.key"),
                   p("proof.json"), p("srs.srs"))
        prove_time = time.perf_counter() - t0

        # Proof size
        proof_size = os.path.getsize(p("proof.json"))

        # VERIFY
        t0 = time.perf_counter()
        verify_ok = ezkl.verify(p("proof.json"), p("settings.json"),
                                p("vk.key"), p("srs.srs"))
        verify_time = time.perf_counter() - t0

        r = {
            "name": name, "top1": rec["top1"], "logrows": meta["logrows"],
            "pareto": meta["pareto"],
            "prove_time_s": round(prove_time, 2),
            "proof_size_bytes": proof_size,
            "proof_size_kb": round(proof_size / 1024, 1),
            "verify_time_s": round(verify_time, 3),
            "verify_ok": verify_ok,
        }
        results.append(r)
        log.info("  prove=%.1fs  proof=%.1fKB  verify=%.3fs", prove_time, proof_size/1024, verify_time)

    except Exception as e:
        log.warning("Failed %s: %s", name, e)
        results.append({"name": name, "error": str(e)[:80]})

with open("results/prove_verify_times.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== Proof Size and Verification Time ===")
print(f"{'Architecture':<42} {'top1':>6} {'lr':>4} {'prove_s':>8} {'proof_KB':>9} {'verify_s':>9}")
print("-"*85)
for r in results:
    if r.get("error"): print(f"{r['name']:<42} ERROR: {r['error']}")
    else:
        print(f"{r['name']:<42} {r.get('top1',0):>6.1f} {r.get('logrows','?'):>4} "
              f"{r.get('prove_time_s','?'):>8} {r.get('proof_size_kb','?'):>9.1f} "
              f"{r.get('verify_time_s','?'):>9.3f}s")
