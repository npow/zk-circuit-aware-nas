"""Expander (Polyhedra Network GKR/sumcheck) prover adapter for zkml-bench.

Architecture overview
---------------------
Expander is a **circuit-level** GKR prover, not an ML prover.  The proving
pipeline is:

  ONNX model
      │
      ▼  (this adapter, Python)
  weights.json  ←─ quantized int32 weights + one input sample
      │
      ▼  (expander-mlp-gen Rust binary, built from ExpanderCompilerCollection)
  circuit.txt + witness.txt   ←─ layered GKR circuit + witness assignment
      │
      ▼  (expander-exec binary from Expander repo)
  proof.bin
      │
      ▼  (expander-exec verify)
  verified ✓

Supported model topology
------------------------
Only **linear-only MLP** models are supported in this initial integration:
  Flatten → (Linear → [ReLU]) * N → Linear

ReLU is supported via a lookup-table-based circuit (bit-decomposition of the
quantized activation), but it roughly doubles the circuit depth.  BatchNorm,
Conv2d, AvgPool, and residual Add are NOT supported.

The adapter automatically detects unsupported models from their ONNX graph
and returns ProverResult.failed("unsupported_model: <reason>").

Cost model (GKR/sumcheck)
-------------------------
Expander uses the GKR protocol over the BN254 scalar field.  The dominant
cost drivers are:

1. **Circuit depth (number of layers)**: each layer requires one round of the
   sumcheck protocol.  For an MLP with L linear layers + ReLU, the layered
   GKR circuit has O(L) GKR layers.  More depth → more sumcheck rounds.

2. **Layer width (log2 of max gate count per layer)**: the prover runs
   multi-linear sumcheck over polynomials of degree proportional to 2^(log2_W)
   where W is the number of gates in the widest layer.  This is the GKR
   analogue of EZKL's ``logrows``.  We report it as ``capacity_metric``
   (``log2_gates``).

3. **Polynomial commitment scheme (PCS)**: with ``Raw`` (no PCS) the proof is
   O(n) in witness size.  With ``Orion`` or ``Hyrax`` the proof is smaller but
   proving is slower.

Unlike EZKL (Halo2/KZG), there is **no trusted setup / SRS download** step.
``setup_time_s`` reflects the circuit compilation + witness generation step,
not a cryptographic setup.

Unlike DeepProve, Expander requires an **explicit circuit compile step** that
writes circuit.txt; the resulting file is cached by model_hash_str so repeated
proofs reuse it.

Binary discovery
----------------
The adapter searches for two binaries:

expander-exec
  1. $EXPANDER_EXEC env var
  2. ``expander-exec`` on $PATH
  3. /tmp/Expander/target/release/expander-exec
  4. ~/expander/target/release/expander-exec

expander-mlp-gen  (Rust circuit/witness generator built from this repo)
  1. $EXPANDER_MLP_GEN env var
  2. ``expander-mlp-gen`` on $PATH
  3. /tmp/ExpanderCompilerCollection/target/release/expander-mlp-gen
  4. ~/expander-mlp-gen

Build instructions
------------------
  # Expander prover binary
  cd /tmp/Expander
  RUSTFLAGS="-C target-cpu=native" cargo build --release -p bin --bin expander-exec

  # MLP circuit generator
  cd /tmp/ExpanderCompilerCollection
  RUSTFLAGS="-C target-cpu=native" cargo build --release --bin expander-mlp-gen

See src/zkml_bench/provers/expander_mlp_gen/ for the circuit generator source.
"""

from __future__ import annotations

import json
import logging
import math
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

_EXEC_SEARCH_PATHS = [
    "expander-exec",
    "/tmp/Expander/target/release/expander-exec",
    str(Path.home() / "expander" / "target" / "release" / "expander-exec"),
]

_GEN_SEARCH_PATHS = [
    "expander-mlp-gen",
    "/tmp/ExpanderCompilerCollection/target/release/expander-mlp-gen",
    str(Path.home() / "bin" / "expander-mlp-gen"),
]

# Quantization scale: map float32 activations to int32 field elements.
# Using scale=2^16 gives ~16-bit precision while staying well within BN254.
_QUANT_SCALE = 2 ** 14  # conservative; avoids overflow in deep MatMul chains

# ---------------------------------------------------------------------------
# Circuit size guard
#
# The ECC compiler (`expander-mlp-gen --compile-circuit`) materialises a
# symbolic IR whose memory cost is driven by the product of inter-layer
# constraint propagation, not just total gate count.  Empirically:
#
#   mlp_deep_6  MNIST: total=570K, non-first=277K → peak RSS ~18 GB, ~38s
#   mlp_wide_2  MNIST: total=669K, non-first=267K → peak RSS ~23 GB, ~52s
#   mlp_wide_1  MNIST: total=1.63M, non-first=20K → peak RSS  ~4 GB, ~19s
#   mlp_deep_10 MNIST: total=1.47M, non-first=935K → peak RSS ~71 GB, ~155s
#   mlp_wide_2  CIFAR: total=1.84M, non-first=267K → peak RSS ~70 GB, ~180s
#
# Key insight (pre-patch): memory scaled with non-first-layer gate count (the
# intermediate symbolic expressions) AND first-layer size.  We guarded on two
# thresholds to avoid silent OOM on G4DN_METAL containers (<64 GB available).
#
# Post-patch (scripts/expander_patches/ecc_memory_reduction.patch):
#   The ECC frontend now emits ONE LinComb instruction per output neuron instead
#   of O(2×n_in) Mul+Add instructions, eliminating the expression-expansion
#   blowup.  Measured peak RSS on mlp_deep_10 MNIST (1.47M gates): 1.1 GB
#   (was ~71 GB before the patch).  Guards are now raised to 20M (effectively
#   disabled for any model we bench) but retained for forward-compatibility.
#
# Override via env vars:
#   EXPANDER_MAX_TOTAL_GATES (default 20_000_000; set to 0 to disable)
#   EXPANDER_MAX_FIRST_LAYER_GATES (default 0 = disabled)
# ---------------------------------------------------------------------------
_DEFAULT_MAX_TOTAL_GATES = 20_000_000  # raised post-patch; was 1_000_000
_MAX_TOTAL_GATES = int(os.environ.get("EXPANDER_MAX_TOTAL_GATES", str(_DEFAULT_MAX_TOTAL_GATES)))

# First-layer gate guard: disabled by default post-patch (set >0 to re-enable).
_DEFAULT_MAX_FIRST_LAYER_GATES = 0  # disabled post-patch; was 1_400_000
_MAX_FIRST_LAYER_GATES = int(
    os.environ.get("EXPANDER_MAX_FIRST_LAYER_GATES", str(_DEFAULT_MAX_FIRST_LAYER_GATES))
)

# ---------------------------------------------------------------------------
# Binary discovery
# ---------------------------------------------------------------------------


def _find_binary(env_var: str, search_paths: list[str]) -> Optional[str]:
    """Return path to binary from env var or search list, or None."""
    env_path = os.environ.get(env_var)
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path
    for p in search_paths:
        resolved = shutil.which(p) or (p if os.path.isfile(p) else None)
        if resolved and os.access(resolved, os.X_OK):
            return resolved
    return None


def _check_expander_available() -> Optional[str]:
    """Return path to expander-exec binary, or None if unavailable."""
    return _find_binary("EXPANDER_EXEC", _EXEC_SEARCH_PATHS)


def _check_mlp_gen_available() -> Optional[str]:
    """Return path to expander-mlp-gen binary, or None if unavailable."""
    return _find_binary("EXPANDER_MLP_GEN", _GEN_SEARCH_PATHS)


# ---------------------------------------------------------------------------
# Memory polling
# ---------------------------------------------------------------------------


def _poll_memory(stop_event: threading.Event, peak_mb: list) -> None:
    """Background thread to poll peak RSS memory at 10 Hz."""
    proc = psutil.Process()
    while not stop_event.is_set():
        try:
            mem = proc.memory_info().rss / (1024 * 1024)
            if mem > peak_mb[0]:
                peak_mb[0] = mem
        except psutil.NoSuchProcess:
            break
        time.sleep(0.1)


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _run_step(
    cmd: list[str],
    timeout: int,
    cwd: str,
) -> subprocess.CompletedProcess:
    """Run a subprocess step with timeout; capture stdout + stderr loudly.

    On failure the caller already logs rc + last 2000 bytes of stdout/stderr.
    This wrapper additionally logs a compact summary line at WARNING level so
    it is visible even when the full output is trimmed by a log aggregator.
    The timeout fires subprocess.TimeoutExpired which is caught by the caller.
    """
    t0 = time.perf_counter()
    binary_name = os.path.basename(cmd[0]) if cmd else "?"
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired as exc:
        wall = time.perf_counter() - t0
        logger.error(
            "ExpanderProver: TIMEOUT after %.1fs (limit=%ds): %s — "
            "last stdout: %r  last stderr: %r",
            wall,
            timeout,
            binary_name,
            (exc.stdout or b"")[-500:] if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")[-500:],
            (exc.stderr or b"")[-500:] if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")[-500:],
        )
        raise

    wall = time.perf_counter() - t0
    if result.returncode != 0:
        logger.error(
            "ExpanderProver: FAILED rc=%d wall=%.1fs: %s\n"
            "  last 500B stdout: %r\n"
            "  last 500B stderr: %r",
            result.returncode,
            wall,
            binary_name,
            result.stdout[-500:] if result.stdout else "",
            result.stderr[-500:] if result.stderr else "",
        )
    return result


# ---------------------------------------------------------------------------
# ONNX inspection helpers
# ---------------------------------------------------------------------------

_UNSUPPORTED_OPS = frozenset({
    "BatchNormalization",
    "GlobalAveragePool",
    "AveragePool",
    "LSTM",
    "GRU",
    "ConvTranspose",
    "InstanceNormalization",
    "GroupNormalization",
    "LayerNormalization",
})

_LINEAR_OPS = frozenset({
    "Flatten",
    "Gemm",
    "MatMul",
    "Relu",
    "Reshape",
    "Transpose",
    "Add",
    "Constant",
    "Shape",
    "Gather",
    "Unsqueeze",
    "Squeeze",
})


def _inspect_onnx(onnx_path: str) -> tuple[bool, str, list[dict]]:
    """Parse ONNX graph, return (ok, reason, layers) where layers is MLP spec.

    Returns (False, reason, []) if the model cannot be compiled to Expander.
    Returns (True, "", layers) on success where layers is a list like:
        [{"type": "linear", "in": 784, "out": 512, "has_bias": True}, ...]

    Only supports: Flatten → (Gemm/MatMul → [Relu]) * N → Gemm/MatMul
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        return False, "onnx_not_installed", []

    try:
        model = onnx.load(onnx_path)
    except Exception as e:
        return False, f"onnx_load_error: {e}", []

    # Check for unsupported ops
    graph = model.graph
    op_types = {n.op_type for n in graph.node}

    bad_ops = op_types & _UNSUPPORTED_OPS
    if bad_ops:
        return False, f"unsupported_ops: {','.join(sorted(bad_ops))}", []

    # Check for Conv (need to allow only linear-only nets)
    if "Conv" in op_types:
        return False, "conv_unsupported: expander_mlp_gen_only_supports_linear_ops", []

    # Build weight index for Gemm inspection
    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    layers = []
    for node in graph.node:
        if node.op_type in ("Gemm", "MatMul"):
            # Get weight shape to infer in/out sizes
            w_name = node.input[1] if len(node.input) > 1 else None
            w_arr = initializers.get(w_name) if w_name else None
            if w_arr is not None:
                # Gemm: weight is (out_features, in_features) or (in, out) depending on transB
                transB = 1  # default Gemm transB=1
                for attr in node.attribute:
                    if attr.name == "transB":
                        transB = attr.i
                if node.op_type == "Gemm" and transB == 1:
                    out_f, in_f = w_arr.shape
                else:
                    in_f, out_f = w_arr.shape[:2]
                has_bias = len(node.input) > 2 and node.input[2] in initializers
                layers.append({
                    "type": "linear",
                    "in": int(in_f),
                    "out": int(out_f),
                    "has_bias": has_bias,
                })

    if not layers:
        return False, "no_linear_layers_found", []

    return True, "", layers


def _extract_weights_from_onnx(onnx_path: str) -> Optional[dict]:
    """Extract quantized weights and biases from ONNX for the MLP circuit.

    Returns a dict with:
        {
          "scale": int,
          "layers": [
            {"weight": [[int, ...], ...], "bias": [int, ...] or null},
            ...
          ]
        }

    Weight quantization: w_q = round(w_f * scale), clamped to int32.
    """
    try:
        import onnx
        import numpy as np
        from onnx import numpy_helper
    except ImportError:
        return None

    try:
        model = onnx.load(onnx_path)
    except Exception:
        return None

    graph = model.graph
    initializers = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    scale = _QUANT_SCALE
    result_layers = []

    for node in graph.node:
        if node.op_type in ("Gemm", "MatMul"):
            w_name = node.input[1] if len(node.input) > 1 else None
            w_arr = initializers.get(w_name) if w_name else None
            if w_arr is None:
                continue

            # Normalize weight to (out_features, in_features)
            transB = 1
            for attr in node.attribute:
                if attr.name == "transB":
                    transB = attr.i
            if node.op_type == "Gemm" and transB == 1:
                # w_arr shape is (out, in)
                w_mat = w_arr
            else:
                # w_arr shape is (in, out), transpose
                w_mat = w_arr.T

            w_quantized = np.clip(
                np.round(w_mat.astype(np.float64) * scale).astype(np.int64),
                -(2 ** 30), 2 ** 30
            ).tolist()

            bias_quantized = None
            if len(node.input) > 2:
                b_name = node.input[2]
                b_arr = initializers.get(b_name)
                if b_arr is not None:
                    bias_quantized = np.clip(
                        np.round(b_arr.astype(np.float64) * scale).astype(np.int64),
                        -(2 ** 30), 2 ** 30,
                    ).tolist()

            result_layers.append({"weight": w_quantized, "bias": bias_quantized})

    if not result_layers:
        return None

    return {"scale": scale, "layers": result_layers}


def _quantize_input(input_flat: list, scale: int) -> list[int]:
    """Quantize a flat float input to int for the circuit witness."""
    import numpy as np
    arr = np.array(input_flat, dtype=np.float64)
    return np.clip(np.round(arr * scale).astype(np.int64), -(2 ** 30), 2 ** 30).tolist()


# ---------------------------------------------------------------------------
# Circuit capacity estimation (before compile)
# ---------------------------------------------------------------------------


def _estimate_log2_gates(layers: list[dict]) -> int:
    """Estimate log2 of max gate count across GKR layers.

    For a Gemm layer with in_features × out_features, the number of
    multiplication gates is in_features × out_features.  GKR layers are
    padded to the next power of two.
    """
    max_gates = 1
    for layer in layers:
        gates = layer.get("in", 1) * layer.get("out", 1)
        max_gates = max(max_gates, gates)
    return max(1, math.ceil(math.log2(max_gates)))


def _estimate_total_gates(layers: list[dict], max_layers: int = 3) -> int:
    """Estimate total gate count across the first max_layers GKR layers.

    This mirrors the MAX_LAYERS_TO_PROVE clamping in expander-mlp-gen so we
    only count the layers that will actually be compiled into the circuit.
    """
    total = 0
    for layer in layers[:max_layers]:
        total += layer.get("in", 1) * layer.get("out", 1)
    return total


def _estimate_first_layer_gates(layers: list[dict]) -> int:
    """Estimate gate count for the first linear layer only."""
    if not layers:
        return 0
    return layers[0].get("in", 1) * layers[0].get("out", 1)


def _estimate_heavy_layer_count(layers: list[dict], max_layers: int = 3, threshold: int = 200_000) -> int:
    """Count how many of the first max_layers layers have gate count > threshold.

    The ECC compiler's memory cost explodes when multiple layers each have
    large gate counts (>~200K), creating deep dependency chains in the IR.
    A single large layer (e.g. 784→2048) is fast; three large layers are not.
    """
    count = 0
    for layer in layers[:max_layers]:
        if layer.get("in", 1) * layer.get("out", 1) > threshold:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Main prover class
# ---------------------------------------------------------------------------


class ExpanderProver(BaseProver):
    """Expander (Polyhedra Network GKR/sumcheck) proof system integration.

    This is a **circuit-level** adapter.  The adapter:
      1. Inspects the ONNX model to check support and extract weights.
      2. Writes a ``model_spec.json`` with quantized weights + one input sample.
      3. Invokes ``expander-mlp-gen`` to compile the MLP to circuit.txt +
         witness.txt using the ExpanderCompilerCollection Rust frontend.
      4. Invokes ``expander-exec prove`` to generate a GKR proof.
      5. Invokes ``expander-exec verify`` to verify the proof.

    circuit.txt is cached by model_hash_str to avoid recompilation.
    witness.txt is regenerated per prove() call (it contains the input).
    """

    def prove(
        self,
        onnx_path: str,
        input_data: list,
        calibration_data: list,
        accuracy_top1: float,
        cache_dir: str,
        model_hash: str,
        timeout_setup: int = 600,
        timeout_prove: int = 300,
        model_name: str = "",
        **kwargs,
    ) -> ProverResult:

        # ----------------------------------------------------------------
        # 1. Check binary availability
        # ----------------------------------------------------------------
        exec_bin = _check_expander_available()
        if exec_bin is None:
            logger.info(
                "ExpanderProver: expander-exec not found; "
                "set $EXPANDER_EXEC or build /tmp/Expander with cargo"
            )
            return ProverResult.unavailable()

        gen_bin = _check_mlp_gen_available()
        if gen_bin is None:
            logger.info(
                "ExpanderProver: expander-mlp-gen not found; "
                "set $EXPANDER_MLP_GEN or build from "
                "src/zkml_bench/provers/expander_mlp_gen/"
            )
            return ProverResult.unavailable()

        # ----------------------------------------------------------------
        # 2. Inspect ONNX — early-exit for unsupported models
        # ----------------------------------------------------------------
        ok, reason, layers = _inspect_onnx(onnx_path)
        if not ok:
            logger.info("ExpanderProver: skipping %s — %s", model_name, reason)
            return ProverResult.failed(f"unsupported_model: {reason}")

        log2_gates = _estimate_log2_gates(layers)

        # ----------------------------------------------------------------
        # 2b. Circuit size guard — fail fast before OOM-killing the container
        #
        # The ECC symbolic compiler's peak RAM scales with both total gate
        # count and the depth of inter-layer dependency propagation.
        # Two complementary guards (both can be disabled by setting to 0):
        #
        #   EXPANDER_MAX_TOTAL_GATES (default 1M):
        #     Catches deep models with many mid-size layers.
        #     mlp_deep_10 MNIST: 1.47M total gates → 71 GB RSS → OOM.
        #
        #   EXPANDER_MAX_FIRST_LAYER_GATES (default 1.4M):
        #     Catches wide-input models (CIFAR-10 3072-dim first layer).
        #     mlp_wide_2 CIFAR: first layer = 3072×512 = 1.57M → OOM.
        #
        # mlp_wide_1 MNIST (784→2048→10, first_layer=1.6M, total=1.63M)
        # compiled in 18.7s / 4 GB because the second layer is tiny (20K).
        # It exceeds EXPANDER_MAX_FIRST_LAYER_GATES but NOT in a problematic
        # way — however, to be conservative on containers we skip it too.
        # ----------------------------------------------------------------
        layer_shapes_str = str([(lyr.get("in"), lyr.get("out")) for lyr in layers[:3]])
        if _MAX_TOTAL_GATES > 0:
            total_gates = _estimate_total_gates(layers, max_layers=3)
            # The ECC compiler's peak RSS scales roughly with total gates WHEN
            # multiple layers are large.  A single large layer (e.g. mlp_wide_1
            # MNIST: 784→2048→10 = 1.63M gates but only one 1.6M layer and one
            # 20K layer) is handled efficiently by the compiler (~4 GB peak).
            # Three large layers (e.g. mlp_deep_10 MNIST: 784→684, 684→684,
            # 684→684 = 1.47M gates, each layer >200K) → ~71 GB peak.
            # We apply the gate cap only when there are ≥2 heavy layers.
            heavy_layers = _estimate_heavy_layer_count(layers, max_layers=3, threshold=200_000)
            if total_gates > _MAX_TOTAL_GATES and heavy_layers >= 2:
                skip_msg = (
                    f"circuit_too_large: total_gates={total_gates:,} > "
                    f"EXPANDER_MAX_TOTAL_GATES={_MAX_TOTAL_GATES:,} with "
                    f"{heavy_layers} layers each >200K gates. "
                    f"Peak RSS empirically ~{total_gates * 48 // (1024**3):.0f} GB+, "
                    f"exceeding container memory budget. "
                    f"Layer shapes: {layer_shapes_str}. "
                    f"Use DATASET=mnist or set EXPANDER_MAX_TOTAL_GATES=0 to override."
                )
                logger.warning("ExpanderProver: SKIPPING %s — %s", model_name, skip_msg)
                return ProverResult.failed(skip_msg)

        if _MAX_FIRST_LAYER_GATES > 0:
            first_layer_gates = _estimate_first_layer_gates(layers)
            if first_layer_gates > _MAX_FIRST_LAYER_GATES:
                skip_msg = (
                    f"circuit_too_large: first_layer_gates={first_layer_gates:,} > "
                    f"EXPANDER_MAX_FIRST_LAYER_GATES={_MAX_FIRST_LAYER_GATES:,}. "
                    f"Large first-layer gate counts (high-dim inputs like CIFAR-10 3072-dim) "
                    f"cause peak RSS ~{first_layer_gates * 48 // (1024**3):.0f} GB+. "
                    f"Layer shapes: {layer_shapes_str}. "
                    f"Use DATASET=mnist or set EXPANDER_MAX_FIRST_LAYER_GATES=0 to override."
                )
                logger.warning("ExpanderProver: SKIPPING %s — %s", model_name, skip_msg)
                return ProverResult.failed(skip_msg)

        # ----------------------------------------------------------------
        # 3. Extract quantized weights
        # ----------------------------------------------------------------
        weight_spec = _extract_weights_from_onnx(onnx_path)
        if weight_spec is None:
            return ProverResult.failed("weight_extraction_failed")

        scale = weight_spec["scale"]

        # Get input sample (first calibration sample if input_data is empty)
        samples = input_data if input_data else (calibration_data[:1] if calibration_data else [])
        if not samples:
            return ProverResult.failed("no_input_data")

        # Flatten the first sample
        import numpy as np
        raw_sample = samples[0]
        if isinstance(raw_sample, list):
            flat = []
            def _flatten(x):
                if isinstance(x, list):
                    for item in x:
                        _flatten(item)
                else:
                    flat.append(float(x))
            _flatten(raw_sample)
        else:
            flat = list(np.array(raw_sample).flatten().tolist())

        quantized_input = _quantize_input(flat, scale)
        weight_spec["input"] = quantized_input

        # ----------------------------------------------------------------
        # 4. Set up cache and working directories
        # ----------------------------------------------------------------
        cache_path = Path(cache_dir) / model_hash
        cache_path.mkdir(parents=True, exist_ok=True)
        circuit_file = str(cache_path / "circuit.txt")
        witness_solver_file = str(cache_path / "witness_solver.txt")

        peak_mb = [0.0]
        stop_event = threading.Event()
        mem_thread = threading.Thread(
            target=_poll_memory, args=(stop_event, peak_mb), daemon=True
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            spec_json = os.path.join(tmpdir, "model_spec.json")
            witness_file = os.path.join(tmpdir, "witness.txt")
            proof_file = os.path.join(tmpdir, "proof.bin")

            with open(spec_json, "w") as f:
                json.dump(weight_spec, f)

            mem_thread.start()

            try:
                # --------------------------------------------------------
                # 5. Compile circuit (cached) + generate witness
                # --------------------------------------------------------
                t_setup_start = time.perf_counter()

                circuit_needs_compile = not os.path.isfile(circuit_file)

                gen_cmd = [
                    gen_bin,
                    "--spec", spec_json,
                    "--output-circuit", circuit_file,
                    "--output-witness", witness_file,
                    "--output-witness-solver", witness_solver_file,
                ]
                if circuit_needs_compile:
                    gen_cmd.append("--compile-circuit")

                logger.info("ExpanderProver: circuit gen: %s", " ".join(gen_cmd))
                r = _run_step(gen_cmd, timeout=timeout_setup, cwd=tmpdir)

                if r.returncode != 0:
                    stop_event.set()
                    logger.warning(
                        "ExpanderProver: expander-mlp-gen failed (rc=%d):\n%s\n%s",
                        r.returncode, r.stdout[:2000], r.stderr[:2000],
                    )
                    if "unsupported" in r.stderr.lower():
                        return ProverResult.failed("unsupported_model_at_compile")
                    if "out of memory" in r.stderr.lower():
                        return ProverResult.failed("oom_compile")
                    return ProverResult.failed(f"circuit_gen_failed_rc{r.returncode}")

                setup_time = time.perf_counter() - t_setup_start

                # Parse num_layers_proven and total_params from gen stdout
                num_layers_proven = None
                total_params_proven = None
                for line in r.stdout.splitlines():
                    line = line.strip()
                    if line.startswith("num_layers_proven:"):
                        try:
                            num_layers_proven = int(line.split(":", 1)[1].strip())
                        except (ValueError, IndexError):
                            pass
                    elif line.startswith("Total params (weight elements):"):
                        try:
                            total_params_proven = int(line.split(":", 1)[1].strip().split()[0])
                        except (ValueError, IndexError):
                            pass

                logger.info(
                    "ExpanderProver: circuit/witness ready (%.1fs, compile=%s, layers=%s, params=%s)",
                    setup_time, circuit_needs_compile, num_layers_proven, total_params_proven,
                )

                # --------------------------------------------------------
                # 6. Prove
                # --------------------------------------------------------
                t_prove_start = time.perf_counter()
                prove_cmd = [
                    exec_bin,
                    "-f", "SHA256",
                    "-p", "Raw",
                    "prove",
                    "-c", circuit_file,
                    "-w", witness_file,
                    "-o", proof_file,
                ]
                logger.info("ExpanderProver: proving: %s", " ".join(prove_cmd))
                r = _run_step(prove_cmd, timeout=timeout_prove, cwd=tmpdir)
                prove_time = time.perf_counter() - t_prove_start

                if r.returncode == -9:
                    stop_event.set()
                    return ProverResult.failed("oom_killed")
                if r.returncode != 0:
                    stop_event.set()
                    logger.warning(
                        "ExpanderProver: prove failed (rc=%d):\n%s\n%s",
                        r.returncode, r.stdout[:2000], r.stderr[:2000],
                    )
                    return ProverResult.failed(f"prove_failed_rc{r.returncode}")

                logger.info(
                    "ExpanderProver: prove done in %.1fs, stdout: %s",
                    prove_time, r.stdout[:500],
                )

                # --------------------------------------------------------
                # 7. Verify
                # --------------------------------------------------------
                t_verify_start = time.perf_counter()
                verify_cmd = [
                    exec_bin,
                    "-f", "SHA256",
                    "-p", "Raw",
                    "verify",
                    "-c", circuit_file,
                    "-w", witness_file,
                    "-i", proof_file,
                ]
                r = _run_step(verify_cmd, timeout=120, cwd=tmpdir)
                verify_time = time.perf_counter() - t_verify_start

                if r.returncode != 0:
                    stop_event.set()
                    logger.warning(
                        "ExpanderProver: verify failed (rc=%d):\n%s\n%s",
                        r.returncode, r.stdout[:2000], r.stderr[:2000],
                    )
                    return ProverResult.failed("verify_failed")

                stop_event.set()

                # --------------------------------------------------------
                # 8. Collect metrics
                # --------------------------------------------------------
                proof_size_bytes = (
                    os.path.getsize(proof_file) if os.path.isfile(proof_file) else None
                )

                return ProverResult(
                    available=True,
                    error=None,
                    # GKR has no arithmetic circuit rows/cols — use log2_gates as proxy
                    num_rows=2 ** log2_gates,
                    num_cols=num_layers_proven,      # repurposed: num linear layers proven
                    total_cells=total_params_proven,  # repurposed: total weight params proven
                    quantized_accuracy_top1=None,  # not measured here
                    setup_time_s=setup_time,
                    prove_time_s=prove_time,
                    verify_time_s=verify_time,
                    proof_size_bytes=proof_size_bytes,
                    peak_memory_mb=peak_mb[0],
                    srs_size_bytes=None,  # no trusted setup
                )

            except subprocess.TimeoutExpired as exc:
                stop_event.set()
                subcommand = (
                    exc.cmd[1] if isinstance(exc.cmd, (list, tuple)) and len(exc.cmd) > 1 else ""
                )
                if "prove" in subcommand or "expander-exec" in str(exc.cmd):
                    return ProverResult.failed("prove_timeout")
                return ProverResult.failed("setup_timeout")
            finally:
                stop_event.set()
