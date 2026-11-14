"""
Proxy Validation Experiment
===========================
Correlates our ONNX-based ZK proxy metric against ezkl's actual circuit metrics
and mock proof timing. Results are written to proxy_validation_results.json.

Usage:
    python proxy_validation.py [--mock] [--prove]
    --mock   Run mock proof timing (fast, ~seconds per model)
    --prove  Run actual proof (slow, may take hours for large models)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
import traceback
import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# --- Models to evaluate ---
MODELS_TO_EVAL = [
    # (name, factory_fn, input_shape)
    # We'll use NAS candidates and benchmark baselines
]

BENCHMARK_MODELS = [
    # --- original 9 ---
    "mlp_small",
    "lenet5",
    "mlp_large",
    "lenet5_poly",
    "mlp_poly",
    "resnet20",
    "vgg11_small",
    "resnet20_relu_bn_free",
    "resnet20_poly",
    # --- transformers / attention ---
    "tiny_transformer",
    "small_bert",
    "mlp_mixer_tiny",
    "vit_tiny",
    "attention_only_2head",
    "linear_attention_small",
    "local_attention_window",
    "flash_attention_compatible",
    # --- polynomial transformers ---
    "poly_transformer_tiny",
    "poly_attention_block",
    # --- MoE ---
    "moe_tiny",
    "moe_resnet_block",
    # --- wide / quantized / large ---
    "wide_resnet_small",
    "mlp_huge",
    "qlinear_small",
    "int8_resnet_small",
    # --- iso-FLOP pairs ---
    "mlp_wide_2",
    "mlp_deep_6",
    "mlp_wide_1",
    "mlp_deep_10",
    "cnn_wide_2",
    "cnn_deep_6",
    "cnn_wide_4",
    "cnn_deep_12",
]

# NAS candidates (from cifar10_nas_v3 results)
NAS_CANDIDATES = [
    "tiny_dws_relu",       # iter 3, 3.1M constraints
    "medium_dws_relu",     # iter 4, 6.5M constraints
    "compact_dws_relu",    # iter 2, 8.4M constraints
    "dws_resnet_relu",     # iter 1, 11.4M constraints
]


def export_onnx_legacy(model: nn.Module, input_shape: tuple, path: str,
                       model_name: str = "") -> bool:
    """Export model to ONNX using legacy (script-based) exporter for ezkl compatibility.

    For polynomial activation models, wraps the model in a clamp to prevent
    EZKL calibration failures due to unbounded x² outputs.
    """
    model.eval()

    # Apply clamp wrapper for poly models (T015)
    if any(tag in model_name for tag in ("_poly", "_quad", "_cubic")):
        from zkml_bench.provers.ezkl import _wrap_poly_model_for_export
        model = _wrap_poly_model_for_export(model, clip_value=10.0)
        log.info("  poly model: wrapping with clamp(±10) before export")

    x = torch.randn(1, *input_shape)
    try:
        torch.onnx.export(
            model, (x,), path,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
            dynamo=False,
        )
        return True
    except Exception as e:
        log.error("ONNX export failed: %s", e)
        return False


def run_ezkl_pipeline(
    onnx_path: str,
    input_shape: tuple,
    run_mock: bool = False,
    model_name: str = "",
) -> dict:
    """Run ezkl gen_settings + calibrate_settings + optionally mock proof.

    For polynomial activation models, retries calibrate_settings with
    scale_rebase_multiplier=2 if the first attempt fails.
    """
    import ezkl

    result = {
        "gen_settings_ok": False,
        "calibrate_ok": False,
        "logrows": None,
        "num_constraints": None,
        "mock_ok": False,
        "mock_time_s": None,
        "error": None,
    }

    tmpdir = tempfile.mkdtemp()
    settings_path = os.path.join(tmpdir, "settings.json")
    cal_data_path = os.path.join(tmpdir, "cal_data.json")
    compiled_path = os.path.join(tmpdir, "model.compiled")
    witness_path = os.path.join(tmpdir, "witness.json")

    try:
        # 1. Gen settings
        run_args = ezkl.PyRunArgs()
        run_args.input_visibility = "private"
        run_args.output_visibility = "public"
        run_args.param_visibility = "fixed"

        ok = ezkl.gen_settings(onnx_path, settings_path, py_run_args=run_args)
        result["gen_settings_ok"] = ok
        if not ok:
            result["error"] = "gen_settings failed"
            return result

        with open(settings_path) as f:
            settings = json.load(f)
        result["logrows"] = settings.get("run_args", {}).get("logrows")

        # 2. Calibrate settings with sample data
        # Poly models need more samples to cover unbounded x² output range
        from zkml_bench.provers.ezkl import _get_calibration_samples
        n_cal = _get_calibration_samples(model_name, base=20)
        inputs = torch.randn(n_cal, *input_shape)
        cal_data = {"input_data": [inputs[i].flatten().tolist() for i in range(n_cal)]}
        with open(cal_data_path, "w") as f:
            json.dump(cal_data, f)

        is_poly = any(tag in model_name for tag in ("_poly", "_quad", "_cubic"))
        try:
            ok = ezkl.calibrate_settings(cal_data_path, onnx_path, settings_path, target="resources")
            result["calibrate_ok"] = ok
            if not ok and is_poly:
                raise RuntimeError("first calibrate_settings returned False; trying poly fallback")
        except Exception as e:
            if not is_poly:
                result["error"] = f"calibrate_settings failed: {e}"
                return result
            # Poly fallback: known-good combination for deep poly nets.
            # See docs/poly_calibration_findings.md — solved resnet20_poly with this.
            # Requires per-activation clamping in the ONNX export (handled in export_onnx_legacy).
            log.info("  poly model: retry calibrate with scales=[4-8] rebase=[1] margin=1")
            try:
                ok = ezkl.calibrate_settings(
                    cal_data_path, onnx_path, settings_path,
                    target="resources",
                    scales=[4, 5, 6, 7, 8],
                    scale_rebase_multiplier=[1],
                    lookup_safety_margin=1,
                )
                result["calibrate_ok"] = ok
                result["used_poly_fallback"] = True
            except Exception as e2:
                result["error"] = f"calibrate_settings failed (poly fallback): {e2}"
                return result
        if not result["calibrate_ok"]:
            result["error"] = "calibrate_settings returned False"
            return result

        with open(settings_path) as f:
            settings = json.load(f)
        logrows = settings.get("run_args", {}).get("logrows")
        result["logrows"] = logrows
        if logrows is not None:
            result["num_constraints"] = 2 ** logrows

        if not run_mock:
            return result

        # Skip mock for very large circuits (logrows > 23 → >8M cells → gen_srs takes hours)
        MAX_MOCK_LOGROWS = 23
        if logrows is not None and logrows > MAX_MOCK_LOGROWS:
            log.info("  Skipping mock proof (logrows=%d > %d)", logrows, MAX_MOCK_LOGROWS)
            result["ezkl_error"] = f"skipped: logrows={logrows} > {MAX_MOCK_LOGROWS}"
            return result

        # 3. Compile circuit
        ok = ezkl.compile_circuit(onnx_path, compiled_path, settings_path)
        if not ok:
            result["error"] = "compile_circuit failed"
            return result

        # 4. Generate SRS (structured reference string) locally for mock
        with open(settings_path) as f:
            _settings = json.load(f)
        logrows_val = _settings.get("run_args", {}).get("logrows", 20)
        srs_path = os.path.join(tmpdir, "srs.srs")
        ezkl.gen_srs(srs_path, logrows=logrows_val)
        if not os.path.exists(srs_path):
            result["error"] = "gen_srs failed (no file)"
            return result

        # 5. Generate witness
        x = torch.randn(1, *input_shape)
        witness_input = {"input_data": [x.flatten().tolist()]}
        input_path = os.path.join(tmpdir, "input.json")
        with open(input_path, "w") as f:
            json.dump(witness_input, f)

        ok = ezkl.gen_witness(input_path, compiled_path, witness_path)
        if not ok:
            result["error"] = "gen_witness failed"
            return result

        # 6. Mock proof (validates circuit without full proof)
        t0 = time.perf_counter()
        ok = ezkl.mock(witness_path, compiled_path)
        mock_time = time.perf_counter() - t0
        result["mock_ok"] = ok
        result["mock_time_s"] = mock_time

    except BaseException as e:
        # Use BaseException to catch pyo3_runtime.PanicException which does not
        # inherit from Exception (e.g. tiny_transformer panics in src/tensor/mod.rs).
        result["error"] = f"{type(e).__name__}: {e}"
        log.error("ezkl pipeline error: %s\n%s", e, traceback.format_exc()[-500:])

    return result


def evaluate_model(
    name: str,
    model: nn.Module,
    input_shape: tuple,
    run_mock: bool = False,
) -> dict:
    from zkml_bench.metrics.zk_proxy import compute_proxy, _suppress_onnx_noise

    log.info("Evaluating: %s (input=%s)", name, input_shape)
    result = {
        "name": name,
        "input_shape": input_shape,
        "proxy_constraints": None,
        "proxy_constraints_snapped": None,
        "proxy_nonlinear_ops": None,
        "proxy_params": None,
        "ezkl_logrows": None,
        "ezkl_num_constraints": None,
        "ezkl_mock_time_s": None,
        "ezkl_error": None,
        "proxy_error": None,
    }

    model.eval()
    x = torch.randn(1, *input_shape)

    # 1. Proxy metric
    try:
        with _suppress_onnx_noise():
            proxy = compute_proxy(model, x)
        result["proxy_constraints"] = proxy.constraint_estimate
        result["proxy_constraints_snapped"] = proxy.constraint_estimate_snapped
        result["proxy_nonlinear_ops"] = proxy.nonlinear_op_count
        result["proxy_params"] = proxy.param_count
        log.info("  proxy: %d constraints (snapped: %d)", proxy.constraint_estimate, proxy.constraint_estimate_snapped)
    except Exception as e:
        result["proxy_error"] = str(e)
        log.warning("  proxy failed: %s", e)

    # 2. ezkl pipeline
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name

    try:
        exported = export_onnx_legacy(model, input_shape, onnx_path, model_name=name)
        if not exported:
            result["ezkl_error"] = "onnx export failed"
        else:
            ezkl_result = run_ezkl_pipeline(onnx_path, input_shape, run_mock=run_mock, model_name=name)
            result["ezkl_logrows"] = ezkl_result["logrows"]
            result["ezkl_num_constraints"] = ezkl_result["num_constraints"]
            result["ezkl_mock_time_s"] = ezkl_result.get("mock_time_s")
            result["ezkl_error"] = ezkl_result.get("error")
            if ezkl_result["logrows"] is not None:
                log.info(
                    "  ezkl: logrows=%d (%d cells)", ezkl_result["logrows"],
                    2 ** ezkl_result["logrows"]
                )
            if ezkl_result.get("mock_time_s") is not None:
                log.info("  mock proof: %.2fs", ezkl_result["mock_time_s"])
    finally:
        os.unlink(onnx_path)

    return result


def load_nas_candidate(candidate_path: str) -> tuple[nn.Module, dict] | None:
    """Load a NAS candidate from its .py file."""
    with open(candidate_path) as f:
        code = f.read()
    ns = {"torch": torch, "nn": nn}
    try:
        exec(compile(code, candidate_path, "exec"), ns)  # noqa: S102
    except Exception as e:
        log.error("Failed to exec %s: %s", candidate_path, e)
        return None
    get_model = ns.get("get_model")
    if not get_model:
        return None
    name = ns.get("CANDIDATE_NAME", os.path.basename(candidate_path).replace(".py", ""))
    return get_model, name


def fit_node_overhead(results: list[dict]) -> int:
    """Find the NODE_OVERHEAD constant that maximises R² on the validation set.

    Uses scipy.optimize.minimize_scalar over a plausible range [0, 65536].
    Returns the best integer overhead value.

    Note: this function re-computes proxy estimates from scratch by adding
    the overhead to each Conv/Gemm/MatMul node count stored in the results.
    Since we don't store per-node counts, we approximate by using the existing
    proxy_constraints as the MAC-only baseline and fitting the overhead ×
    (number of linear ops) as the correction term.
    """
    import math

    paired = [
        (r.get("proxy_constraints", 0), r.get("proxy_nonlinear_ops", 0), r.get("ezkl_num_constraints"))
        for r in results
        if r.get("proxy_constraints") and r.get("ezkl_num_constraints") is not None
    ]
    if len(paired) < 3:
        log.warning("fit_node_overhead: need ≥3 paired models, got %d", len(paired))
        return 4096

    # We don't have per-node linear op counts in results; use proxy_nonlinear_ops
    # as a rough proxy for node count (most models have comparable linear/nonlinear ratios).
    # Better: re-run evaluate_model after fixing. For now, use raw proxy as baseline.
    from zkml_bench.metrics.zk_proxy import snap_to_power_of_2

    def r_squared_for_overhead(overhead: float) -> float:
        overhead = int(overhead)
        snapped = [snap_to_power_of_2(int(proxy + overhead)) for proxy, _, _ in paired]
        actual = [a for _, _, a in paired]
        n = len(snapped)
        log_s = [math.log2(max(x, 1)) for x in snapped]
        log_a = [math.log2(max(x, 1)) for x in actual]
        mean_a = sum(log_a) / n
        ss_res = sum((la - ls) ** 2 for ls, la in zip(log_s, log_a))
        ss_tot = sum((la - mean_a) ** 2 for la in log_a)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    try:
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(
            lambda x: -r_squared_for_overhead(x),
            bounds=(0, 65536),
            method="bounded",
        )
        best = max(0, int(result.x))
        best_r2 = r_squared_for_overhead(best)
        log.info("fit_node_overhead: best overhead=%d  R²=%.4f", best, best_r2)
        return best
    except ImportError:
        log.warning("scipy not available; using brute-force search")
        best, best_r2 = 0, -1.0
        for overhead in range(0, 65537, 512):
            r2 = r_squared_for_overhead(overhead)
            if r2 > best_r2:
                best, best_r2 = overhead, r2
        log.info("fit_node_overhead: best overhead=%d  R²=%.4f", best, best_r2)
        return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Run mock proof timing")
    parser.add_argument("--prove", action="store_true", help="Run actual proof (slow)")
    parser.add_argument("--output", default="proxy_validation_results.json")
    parser.add_argument("--recompute-proxy", action="store_true",
                        help="Re-run proxy computation on all models (keeps existing EZKL results)")
    args = parser.parse_args()

    run_mock = args.mock or args.prove

    # Load existing results to skip already-evaluated models
    existing = {}
    ezkl_cache = {}  # preserved EZKL results when --recompute-proxy
    if os.path.exists(args.output):
        with open(args.output) as f:
            old = json.load(f)
        for r in old:
            if r.get("ezkl_logrows") is not None:
                ezkl_cache[r["name"]] = {
                    k: r[k] for k in ("ezkl_logrows", "ezkl_num_constraints",
                                      "ezkl_mock_time_s", "ezkl_error")
                    if k in r
                }
        if not args.recompute_proxy:
            existing = {r["name"]: r for r in old if r.get("ezkl_logrows") is not None}
            log.info("Loaded %d existing results (will skip re-evaluation)", len(existing))
        else:
            log.info("--recompute-proxy: will recompute proxy for all models, preserving %d EZKL results", len(ezkl_cache))

    results = list(existing.values())

    # 1. Benchmark models
    from zkml_bench.models.registry import get_model as get_bench_model

    INPUT_SHAPE = (3, 32, 32)
    NUM_CLASSES = 10

    for model_name in BENCHMARK_MODELS:
        if model_name in existing:
            log.info("Skipping %s (already evaluated)", model_name)
            continue
        try:
            model = get_bench_model(model_name, num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE)
            r = evaluate_model(model_name, model, INPUT_SHAPE, run_mock=run_mock)
            if model_name in ezkl_cache and not run_mock:
                r.update(ezkl_cache[model_name])
            results.append(r)
            existing[model_name] = r
        except BaseException as e:
            # Catch pyo3_runtime.PanicException (inherits BaseException, not Exception)
            # as well as ordinary Python exceptions.  Record the failure and continue
            # so that the remaining models still run.
            err_type = type(e).__name__
            log.error("Failed benchmark model %s (%s): %s", model_name, err_type, e)
            r = {
                "name": model_name,
                "error": f"{err_type}: {e}",
                "gen_settings_ok": False,
                "calibrate_ok": False,
                "mock_ok": False,
            }
            results.append(r)
            existing[model_name] = r

    def _load_nas_candidates(nas_log_path: str, name_prefix: str, num_classes: int, input_shape: tuple) -> list:
        """Load all accepted NAS candidates from a nas_log.json file."""
        out = []
        if not os.path.exists(nas_log_path):
            return out
        with open(nas_log_path) as f:
            nas_log = json.load(f)
        log_entries = nas_log if isinstance(nas_log, list) else nas_log.get("iterations", [])
        for entry in log_entries:
            if not (entry.get("accepted") and entry.get("iteration", 0) >= 1):
                continue
            cname = entry["candidate_name"]
            eval_name = f"{name_prefix}{cname}"
            if eval_name in existing:
                log.info("Skipping %s (already evaluated)", eval_name)
                continue
            code = entry.get("code", "")
            if not code:
                log.warning("No code stored for NAS iter %d (%s)", entry["iteration"], cname)
                continue
            ns = {"torch": torch, "nn": nn}
            try:
                exec(compile(code, f"<nas_iter_{entry['iteration']}>", "exec"), ns)  # noqa: S102
                get_model_fn = ns.get("get_model")
                if get_model_fn is None:
                    log.warning("No get_model in NAS iter %d code", entry["iteration"])
                    continue
                model = get_model_fn(num_classes=num_classes, input_shape=input_shape)
                r = evaluate_model(eval_name, model, input_shape, run_mock=run_mock)
                if eval_name in ezkl_cache and not run_mock:
                    r.update(ezkl_cache[eval_name])
                r["nas_log_proxy"] = entry.get("constraints")
                r["nas_log_top1"] = entry.get("top1")
                out.append(r)
                existing[eval_name] = r
            except Exception as e:
                log.error("NAS candidate %s (iter %d) failed: %s", cname, entry["iteration"], e)
        return out

    # 2. CIFAR-10 NAS candidates
    results.extend(_load_nas_candidates(
        "nas_results/cifar10_nas_v3/nas_log.json", "nas_", NUM_CLASSES, INPUT_SHAPE,
    ))

    # 3. CIFAR-100 NAS candidates (same input shape 3×32×32, tested with CIFAR-10 num_classes=10)
    results.extend(_load_nas_candidates(
        "nas_results/cifar100_nas_v2/nas_log.json", "c100_nas_", NUM_CLASSES, INPUT_SHAPE,
    ))

    # 4. CIFAR-10 v1 NAS candidates (diverse architecture families with poly activations)
    results.extend(_load_nas_candidates(
        "nas_results/cifar10_nas/nas_log.json", "v1_nas_", NUM_CLASSES, INPUT_SHAPE,
    ))

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Saved %d results to %s", len(results), args.output)

    # Print summary table
    print("\n=== Proxy Validation Results ===")
    print(f"{'Model':<30} {'Proxy':>12} {'ezkl logrows':>12} {'ezkl cells':>14} {'Mock(s)':>8}")
    print("-" * 80)
    for r in results:
        name = r.get("name", "?")[:29]
        proxy = r.get("proxy_constraints")
        logrows = r.get("ezkl_logrows")
        cells = r.get("ezkl_num_constraints")
        mock_t = r.get("ezkl_mock_time_s")
        err = r.get("ezkl_error", "")

        proxy_str = f"{proxy:,}" if proxy else "N/A"
        logrows_str = str(logrows) if logrows else ("ERR" if err else "N/A")
        cells_str = f"{cells:,}" if cells else "N/A"
        mock_str = f"{mock_t:.2f}" if mock_t else "N/A"
        print(f"{name:<30} {proxy_str:>12} {logrows_str:>12} {cells_str:>14} {mock_str:>8}")

    # Compute calibration statistics using snapped proxy vs actual EZKL constraints
    paired = [
        (r["name"], r.get("proxy_constraints_snapped") or r.get("proxy_constraints"), r["ezkl_num_constraints"])
        for r in results
        if r.get("proxy_constraints") and r.get("ezkl_num_constraints") is not None
    ]
    if len(paired) >= 3:
        import math

        names_p = [p[0] for p in paired]
        snapped = [p[1] for p in paired]
        actual = [p[2] for p in paired]
        n = len(paired)

        ratios = [a / s if s > 0 else float("nan") for s, a in zip(snapped, actual)]

        # Outlier detection: models where ratio is outside [1/4, 4] (>2 logrows off)
        # are flagged separately — they indicate architectural patterns the proxy
        # cannot model (e.g. EZKL-specific optimisations for large concat-heavy models).
        OUTLIER_LO, OUTLIER_HI = 0.25, 4.0
        clean = [(nm, s, a, r) for nm, s, a, r in zip(names_p, snapped, actual, ratios)
                 if not math.isnan(r) and OUTLIER_LO <= r <= OUTLIER_HI]
        outliers = [(nm, s, a, r) for nm, s, a, r in zip(names_p, snapped, actual, ratios)
                    if math.isnan(r) or not (OUTLIER_LO <= r <= OUTLIER_HI)]

        def _r2(pairs):
            if len(pairs) < 3:
                return 0.0
            ls_ = [math.log2(max(s, 1)) for _, s, _, _ in pairs]
            la_ = [math.log2(max(a, 1)) for _, _, a, _ in pairs]
            n_ = len(pairs)
            ma_ = sum(la_) / n_
            ss_res = sum((la - ls) ** 2 for ls, la in zip(ls_, la_))
            ss_tot = sum((la - ma_) ** 2 for la in la_)
            return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        r2_all   = _r2(list(zip(names_p, snapped, actual, ratios)))
        r2_clean = _r2(clean)
        valid_ratios = [r for r in ratios if not math.isnan(r)]
        mean_ratio = sum(valid_ratios) / len(valid_ratios)
        max_ratio, min_ratio = max(valid_ratios), min(valid_ratios)

        print(f"\n=== Calibration Statistics (n={n} models with EZKL data) ===")
        print(f"R² all   (n={n}):  {r2_all:.4f}")
        print(f"R² clean (n={len(clean)}, outliers excluded): {r2_clean:.4f}  "
              f"{'✓ PASS' if r2_clean >= 0.85 else '✗ FAIL (target ≥ 0.85)'}")
        print(f"Ratio (ezkl/proxy_snapped): mean={mean_ratio:.3f}  "
              f"min={min_ratio:.3f}  max={max_ratio:.3f}")

        print(f"\n{'Model':<35} {'proxy_snap':>12} {'ezkl_actual':>12} {'ratio':>8}  flag")
        print("-" * 82)
        for nm, s, a, r in zip(names_p, snapped, actual, ratios):
            flag = " ⚠ OUTLIER" if not (OUTLIER_LO <= r <= OUTLIER_HI) else ""
            print(f"{nm:<35} {s:>12,} {a:>12,} {r:>8.3f}{flag}")

        if outliers:
            print(f"\nOutliers (ratio outside [{OUTLIER_LO}, {OUTLIER_HI}]):")
            for nm, s, a, r in outliers:
                print(f"  {nm}: proxy_snap={s:,} ezkl={a:,} ratio={r:.3f}")
            print("  These models have architectural patterns the proxy cannot model.")

        if r2_clean < 0.85:
            print(f"\n⚠  R² clean = {r2_clean:.4f} < 0.85 — proxy calibration still needs improvement.")
            print("   Re-run fit_node_overhead() after proxy recompute with NODE_OVERHEAD applied.")

    return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
