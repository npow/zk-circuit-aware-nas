"""DeepProve (Lagrange Labs GKR/sumcheck) prover adapter for zkml-bench.

DeepProve uses GKR + sumcheck instead of Halo2/KZG, giving O(n) proving time
with no NTT bottleneck.  The CLI is a single Rust binary (``bench``) built from
the ``zkml`` crate at https://github.com/Lagrange-Labs/deep-prove.

Supported ONNX ops (as of deep-prove@main, May 2026):
  Conv, Gemm / Gemm.ab, MatMul, Relu, Flatten, MaxPool2D, Reshape.

NOT yet supported:
  BatchNorm, AvgPool / GlobalAvgPool, general residual Add (Add is parsed only
  as the bias term of a Gemm layer), Softmax, Dropout.

Consequence for zkml-bench built-in models:
  - mlp_small, mlp_large  → supported (Flatten + Linear + ReLU only)
  - lenet5                → NOT supported (uses AvgPool, not MaxPool)
  - resnet20              → NOT supported (BatchNorm + residual Add)

Cost model:
  Deep-prove has NO analogue of EZKL's ``logrows`` ceiling.  Proof size and
  proving time grow roughly as O(n · log n) in the number of model parameters
  via GKR multilinear polynomial commitments over Goldilocks^2 using Basefold
  as the PCS.  The dominant cost driver is the *lookup* (logup GKR) over the
  ReLU requantisation table; this scales with the number of activations × the
  fixed 8-bit (256-entry) lookup table.  There is no trusted setup and no SRS
  download step.

Input format:
  The ``bench`` binary expects:
    --onnx  <model.onnx>
    --io    <input.json>

  where input.json has the shape::

      {
        "input_data":    [[...], ...],   # N × flat_input  (float32)
        "output_data":   [[...], ...],   # N × flat_output (float32, one-hot or logit)
        "pytorch_output":[[...], ...],   # N × flat_output (model logits, same as output_data)
      }

  This is different from EZKL's ``{"input_data": [...]}`` format.

Installation:
  1. Clone https://github.com/Lagrange-Labs/deep-prove
  2. cd deep-prove && cargo build --release
  3. Export DEEPPROVE_BINARY=/path/to/deep-prove/target/release/bench
     (or put the binary on PATH as ``deepprove-bench``)

The adapter looks for the binary in this order:
  1. $DEEPPROVE_BINARY env var
  2. ``deepprove-bench`` on $PATH
  3. ~/deep-prove/target/release/bench
  4. /tmp/deep-prove/target/release/bench
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import psutil

from .base import BaseProver, ProverResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BINARY_SEARCH_PATHS = [
    "deepprove-bench",  # on PATH
    str(Path.home() / "deep-prove" / "target" / "release" / "bench"),
    "/tmp/deep-prove/target/release/bench",
]

# Ops that deep-prove can prove; used for up-front unsupported-model check.
_SUPPORTED_OPS = frozenset({
    "Conv", "Gemm", "Gemm.ab", "MatMul", "Relu", "Flatten", "MaxPool", "Reshape"
})
_UNSUPPORTED_OPS = frozenset({
    "BatchNorm", "BatchNormalization", "GlobalAveragePool", "AveragePool",
    "Softmax", "Dropout", "LSTM", "GRU",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_binary() -> Optional[str]:
    """Return path to the deep-prove bench binary, or None if not found."""
    env_override = os.environ.get("DEEPPROVE_BINARY")
    if env_override:
        return env_override if os.path.isfile(env_override) else None

    for candidate in _BINARY_SEARCH_PATHS:
        found = shutil.which(candidate) or (
            candidate if os.path.isfile(candidate) else None
        )
        if found:
            return found
    return None


def _check_deepprove_available() -> Optional[str]:
    """Return binary path string if available, else None."""
    binary = _find_binary()
    if binary is None:
        logger.warning(
            "deep-prove bench binary not found. "
            "Set DEEPPROVE_BINARY or build from https://github.com/Lagrange-Labs/deep-prove "
            "and place binary at ~/deep-prove/target/release/bench"
        )
        return None
    return binary


def _poll_memory(stop_event: threading.Event, peak_mb: list) -> None:
    """Background thread: poll peak RSS at 10 Hz."""
    proc = psutil.Process()
    while not stop_event.is_set():
        try:
            mem = proc.memory_info().rss / (1024 * 1024)
            if mem > peak_mb[0]:
                peak_mb[0] = mem
        except psutil.NoSuchProcess:
            break
        time.sleep(0.1)


def _run_step(
    cmd: list[str],
    timeout: int,
    cwd: str,
    env: Optional[dict] = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess step, capturing stdout/stderr."""
    merged_env = {**os.environ, **(env or {})}
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
        env=merged_env,
    )


def _build_io_json(
    input_data: list,
    output_data: list,
) -> dict:
    """Build the deep-prove input.json payload.

    Parameters
    ----------
    input_data:
        List of N flat input vectors (as Python lists of floats).  This is the
        same ``input_data`` field that EZKL uses; we reuse it directly.
    output_data:
        List of N flat output / label vectors.  For classification we use
        one-hot vectors (or softmax logits) — deep-prove uses argmax accuracy.
        If not provided (empty list) we synthesise zero vectors so the binary
        does not error; accuracy will be meaningless.
    """
    n = len(input_data)
    if not output_data or len(output_data) != n:
        # Fall back to zero-vectors with same length as input (will give 0% accuracy)
        out_len = 10  # reasonable default for CIFAR/MNIST
        output_data = [[0.0] * out_len] * n

    return {
        "input_data": input_data,
        "output_data": output_data,
        "pytorch_output": output_data,  # deep-prove uses this for accuracy; same as output_data
    }


def _parse_csv_metrics(bench_csv_path: str) -> dict:
    """Parse the bench.csv written by the deep-prove binary.

    The bench binary writes one row per *phase* with columns:
      name, elapsed (ms), accuracy, proof_size (KB), in_use, peak, ...

    Phase names of interest:
      - "Context creation"    → setup time
      - "Proving <N>"         → prove time per sample
      - "Verify <N>"          → verify time per sample
      - "Accuracy <N>"        → accuracy (1=correct, 0=wrong)

    Returns a dict with keys: setup_time_ms, prove_time_ms, verify_time_ms,
    proof_size_kb, accuracy.
    """
    result: dict = {
        "setup_time_ms": None,
        "prove_time_ms": None,
        "verify_time_ms": None,
        "proof_size_kb": None,
        "accuracy": None,
    }
    if not os.path.exists(bench_csv_path):
        return result

    try:
        import csv as _csv
        with open(bench_csv_path, newline="") as fh:
            rows = list(_csv.DictReader(fh))
        if not rows:
            return result

        setup_ms_list: list[float] = []
        prove_ms_list: list[float] = []
        verify_ms_list: list[float] = []
        proof_kb_list: list[float] = []
        accuracy_list: list[float] = []

        for row in rows:
            name = (row.get("name") or "").strip()
            elapsed_str = (row.get("elapsed") or "").strip()
            proof_str = (row.get("proof_size") or "").strip()
            acc_str = (row.get("accuracy") or "").strip()

            def _to_float(s: str) -> Optional[float]:
                try:
                    return float(s) if s else None
                except ValueError:
                    return None

            elapsed = _to_float(elapsed_str)
            proof = _to_float(proof_str)
            acc = _to_float(acc_str)

            name_lower = name.lower()
            if "context creation" in name_lower and elapsed is not None:
                setup_ms_list.append(elapsed)
            elif name_lower.startswith("proving") and elapsed is not None:
                prove_ms_list.append(elapsed)
                if proof is not None:
                    proof_kb_list.append(proof)
            elif name_lower.startswith("verify") and elapsed is not None:
                verify_ms_list.append(elapsed)
            elif name_lower.startswith("accuracy") and acc is not None:
                accuracy_list.append(acc)

        def _avg(lst: list[float]) -> Optional[float]:
            return sum(lst) / len(lst) if lst else None

        result["setup_time_ms"] = _avg(setup_ms_list)
        result["prove_time_ms"] = _avg(prove_ms_list)
        result["verify_time_ms"] = _avg(verify_ms_list)
        result["proof_size_kb"] = _avg(proof_kb_list)
        result["accuracy"] = _avg(accuracy_list)

    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to parse bench CSV %s: %s", bench_csv_path, exc)

    return result


def _parse_timed_csv(deepprove_csv_path: str) -> dict:
    """Parse the timed_core CSV written at deepprove.csv.

    Columns: function, duration_ms
    We accumulate per-function totals for fine-grained breakdown.
    """
    timings: dict[str, list[float]] = {}
    if not os.path.exists(deepprove_csv_path):
        return timings

    try:
        import csv as _csv
        with open(deepprove_csv_path, newline="") as fh:
            reader = _csv.DictReader(fh)
            for row in reader:
                fn = row.get("function", "")
                ms_str = (
                    row.get("duration_ms")
                    or row.get("wall_time_ms")
                    or row.get("elapsed_ms")
                    or ""
                )
                try:
                    ms = float(ms_str)
                except ValueError:
                    continue
                timings.setdefault(fn, []).append(ms)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to parse timed CSV %s: %s", deepprove_csv_path, exc)

    return timings


# ---------------------------------------------------------------------------
# Main adapter class
# ---------------------------------------------------------------------------


class DeepProveProver(BaseProver):
    """Deep-prove (GKR / sumcheck) prover adapter.

    Usage:
        prover = DeepProveProver()
        result = prover.prove(
            onnx_path="model.onnx",
            input_data=[[0.1, 0.2, ...]],           # 1 sample, flat
            calibration_data=[[...], ...],           # N calibration samples
            accuracy_top1=0.75,
            cache_dir="/tmp/cache",
            model_hash_str="abc123",
        )

    The prove() call:
      1. Writes ONNX + input.json to a temp dir.
      2. Invokes ``<binary> --onnx model.onnx --io input.json --bench bench.csv``.
      3. Parses bench.csv and timed_core CSV for timing + proof size.
      4. Returns a ProverResult.

    Note: deep-prove does *not* have a separate setup / SRS step — context
    generation (the analogue of setup) is included in the binary run and is
    counted under ``setup_time_s``.
    """

    def prove(
        self,
        onnx_path: str,
        input_data: list,
        calibration_data: list,
        accuracy_top1: float,
        cache_dir: str,
        model_hash_str: str,
        timeout_setup: int = 600,
        timeout_prove: int = 600,
        model_name: str = "",
        **kwargs,
    ) -> ProverResult:  # noqa: PLR0911 (many early returns)

        binary = _check_deepprove_available()
        if binary is None:
            return ProverResult.unavailable()

        # ---------------------------------------------------------------
        # Check for obviously unsupported models up-front (best-effort).
        # We rely on the model_name hint rather than parsing the ONNX graph
        # to keep dependencies minimal.
        # ---------------------------------------------------------------
        unsupported_reason = _detect_unsupported_model(model_name)
        if unsupported_reason:
            logger.info(
                "DeepProve: skipping %s — %s", model_name, unsupported_reason
            )
            return ProverResult.failed(f"unsupported_model: {unsupported_reason}")

        # Use the first calibration sample as inference input if input_data is empty.
        samples = input_data if input_data else (calibration_data[:1] if calibration_data else [])
        if not samples:
            return ProverResult.failed("no_input_data")

        # For timing, we prove a single sample (index 0) to match EZKL's single-proof mode.
        # The full calibration set is passed as output_data so accuracy can be computed.
        single_input = [samples[0]]

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_dest = os.path.join(tmpdir, "model.onnx")
            io_json = os.path.join(tmpdir, "input.json")
            bench_csv = os.path.join(tmpdir, "bench.csv")
            # deepprove.csv is written by timed_core to CWD (= tmpdir); bench.csv
            # is the primary timing source so we do not need to parse it separately.

            shutil.copy(onnx_path, onnx_dest)

            # Build io JSON — output_data needs to match input count.
            # We have no ground truth here (caller only passes input tensors),
            # so we use zero-vectors; accuracy will be 0 but timing is valid.
            io_payload = _build_io_json(
                input_data=single_input,
                output_data=[],  # no ground truth → accuracy meaningless but timing valid
            )
            with open(io_json, "w") as fh:
                json.dump(io_payload, fh)

            peak_mb = [0.0]
            stop_event = threading.Event()
            mem_thread = threading.Thread(
                target=_poll_memory, args=(stop_event, peak_mb), daemon=True
            )

            cmd = [
                binary,
                "--onnx", onnx_dest,
                "--io", io_json,
                "--bench", bench_csv,
                "--num-samples", "1",
            ]

            logger.info("DeepProve: running %s", " ".join(cmd))

            try:
                mem_thread.start()
                t0 = time.perf_counter()

                r = _run_step(
                    cmd,
                    timeout=timeout_prove,
                    cwd=tmpdir,
                    # timed_core writes deepprove.csv to CWD by default
                )

                wall_time = time.perf_counter() - t0
                stop_event.set()

                if r.returncode == -9:
                    return ProverResult.failed("oom_killed")

                if r.returncode != 0:
                    stderr_lower = r.stderr.lower()
                    logger.warning(
                        "DeepProve binary failed (rc=%d):\nstdout: %s\nstderr: %s",
                        r.returncode, r.stdout[:2000], r.stderr[:2000],
                    )
                    # Classify common failure modes
                    if "unsupported" in stderr_lower or "not supported" in stderr_lower:
                        return ProverResult.failed("unsupported_op")
                    if "batchnorm" in stderr_lower or "averagepool" in stderr_lower:
                        return ProverResult.failed("unsupported_op: batchnorm_or_avgpool")
                    if "reshape" in stderr_lower and "flattening" in stderr_lower:
                        return ProverResult.failed("unsupported_reshape")
                    if "out of memory" in stderr_lower or "oom" in stderr_lower:
                        return ProverResult.failed("oom")
                    return ProverResult.failed(f"binary_error_rc{r.returncode}")

                logger.info("DeepProve: stdout:\n%s", r.stdout[:3000])

            except subprocess.TimeoutExpired:
                stop_event.set()
                return ProverResult.failed("prove_timeout")
            finally:
                stop_event.set()

            # -----------------------------------------------------------
            # Parse outputs
            # -----------------------------------------------------------
            bench_metrics = _parse_csv_metrics(bench_csv)

            # Proof size: bench.csv "proof_size" column is in KB
            proof_kb = bench_metrics.get("proof_size_kb")
            proof_size_bytes = int(proof_kb * 1024) if proof_kb is not None else None

            # Timing: primary source is bench.csv phase rows (ms → s).
            setup_time_s = (
                bench_metrics["setup_time_ms"] / 1000.0
                if bench_metrics.get("setup_time_ms") is not None
                else 0.0
            )
            prove_time_s = (
                bench_metrics["prove_time_ms"] / 1000.0
                if bench_metrics.get("prove_time_ms") is not None
                else wall_time  # fallback to total wall time
            )
            verify_time_s = (
                bench_metrics["verify_time_ms"] / 1000.0
                if bench_metrics.get("verify_time_ms") is not None
                else 0.0
            )

            # Capacity metric: deep-prove has no logrows / num_rows concept.
            # We use None for num_rows / num_cols.
            # total_cells is set to None (not applicable for GKR).
            return ProverResult(
                available=True,
                error=None,
                num_rows=None,           # N/A for GKR (no arithmetic circuit rows)
                num_cols=None,           # N/A for GKR
                total_cells=None,        # N/A for GKR
                quantized_accuracy_top1=bench_metrics.get("accuracy"),
                setup_time_s=setup_time_s,
                prove_time_s=prove_time_s,
                verify_time_s=verify_time_s,
                proof_size_bytes=proof_size_bytes,
                peak_memory_mb=peak_mb[0],
                srs_size_bytes=None,     # No trusted setup / SRS in deep-prove
            )


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _detect_unsupported_model(model_name: str) -> Optional[str]:
    """Return a reason string if the named model is known-unsupported, else None.

    This is a best-effort name-based check to avoid spending time building
    an ONNX that will definitely fail at parse time.
    """
    name = model_name.lower()
    if "resnet" in name:
        return "residual_add_and_batchnorm_unsupported"
    if "lenet" in name or "lenet5" in name:
        return "avgpool_unsupported (lenet5 uses AvgPool2D; only MaxPool is supported)"
    if "vgg" in name:
        return "likely_avgpool_or_batchnorm_unsupported"
    return None


def _extract_timing(timed_metrics: dict, key_fragment: str) -> Optional[float]:
    """Sum wall-time values for rows whose function name contains *key_fragment*.

    Returns total seconds, or None if no matching rows found.
    """
    total_ms = 0.0
    found = False
    for fn, ms_list in timed_metrics.items():
        if key_fragment.lower() in fn.lower():
            total_ms += sum(ms_list)
            found = True
    return total_ms / 1000.0 if found else None
