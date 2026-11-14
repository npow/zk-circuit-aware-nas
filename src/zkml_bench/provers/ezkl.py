from __future__ import annotations
import json
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


def _get_calibration_samples(model_name: str, base: int = 64) -> int:
    """Return calibration sample count for EZKL.

    Polynomial activation models (x², x²+x, x³+x) produce unbounded outputs
    that require more calibration samples to cover the quantization range.
    """
    if any(tag in model_name for tag in ("_poly", "_quad", "_cubic")):
        return max(base, 128)
    return base


def _wrap_poly_model_for_export(model, clip_value: float = 10.0,
                                 per_activation_clip: bool = True):
    """Return a clamp-wrapped copy of model for ONNX export.

    Polynomial activations (x²) can produce large outputs that compound through
    deep networks, causing EZKL calibration to fail (the 'escaping activation
    problem'; see Sisyphus 2021).

    Two clipping modes:
      - `per_activation_clip=True` (default): wraps EVERY polynomial-activation
        module so each one is clamp(act(x), -clip, +clip). Bounds intermediate
        activations between blocks. Required for deep poly networks like
        ResNet-20.
      - `per_activation_clip=False`: clamps only the model's final output.
        Sufficient for shallow nets (MLP, LeNet) where activations don't
        compound far.

    Both are eval-only (ONNX export path); training is unaffected.
    """
    import copy
    import torch
    import torch.nn as nn

    class _ClampedActivation(nn.Module):
        """Wraps a polynomial activation to clamp its output."""
        def __init__(self, inner: nn.Module, clip: float) -> None:
            super().__init__()
            self._inner = inner
            self._clip = clip

        def forward(self, x):
            return torch.clamp(self._inner(x), -self._clip, self._clip)

    class _OutputClampWrapper(nn.Module):
        """Clamps only the final model output."""
        def __init__(self, inner: nn.Module, clip: float) -> None:
            super().__init__()
            self._inner = inner
            self._clip = clip

        def forward(self, x):
            return torch.clamp(self._inner(x), -self._clip, self._clip)

    def _is_poly_activation(mod: nn.Module) -> bool:
        n = type(mod).__name__.lower()
        # Match SquaredActivation, CubicActivation, QuadActivation, PolyActivation, etc.
        return "activation" in n and any(
            tag in n for tag in ("squar", "cubic", "quad", "poly")
        )

    # Check if model has any poly activations
    poly_modules = [m for m in model.modules() if _is_poly_activation(m)]
    if not poly_modules:
        return model

    if not per_activation_clip:
        # Old behaviour: wrap only the output
        return _OutputClampWrapper(model, clip_value)

    # New behaviour: deep-copy the model, then replace every poly activation
    # in-place with a clamp-wrapped version
    wrapped_model = copy.deepcopy(model)

    def _replace_poly_acts(parent: nn.Module) -> int:
        n_replaced = 0
        for name, child in list(parent.named_children()):
            if _is_poly_activation(child):
                setattr(parent, name, _ClampedActivation(child, clip_value))
                n_replaced += 1
            else:
                n_replaced += _replace_poly_acts(child)
        return n_replaced

    n = _replace_poly_acts(wrapped_model)
    wrapped_model.eval()  # ensure eval mode
    if n > 0:
        import logging
        logging.getLogger(__name__).info(
            "Wrapped %d polynomial activations with clamp(±%.1f) for ONNX export",
            n, clip_value,
        )
    return wrapped_model


def _check_ezkl_available() -> Optional[str]:
    """Returns ezkl version string or None if not available."""
    try:
        result = subprocess.run(["ezkl", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _poll_memory(stop_event: threading.Event, peak_mb: list) -> None:
    """Background thread to poll peak RSS memory at 10Hz."""
    proc = psutil.Process()
    while not stop_event.is_set():
        try:
            mem = proc.memory_info().rss / (1024 * 1024)
            if mem > peak_mb[0]:
                peak_mb[0] = mem
        except psutil.NoSuchProcess:
            break
        time.sleep(0.1)


def _run_step(cmd: list[str], timeout: int, cwd: str) -> subprocess.CompletedProcess:
    """Run EZKL subprocess step with timeout."""
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd
    )


class EZKLProver(BaseProver):
    """EZKL proof system integration."""

    def prove(self, onnx_path, input_data, calibration_data, accuracy_top1,
              cache_dir, model_hash_str, timeout_setup=600, timeout_prove=300,
              model_name: str = "", activation_clip: Optional[float] = None) -> ProverResult:

        ezkl_version = _check_ezkl_available()
        if ezkl_version is None:
            return ProverResult.unavailable()

        cache_path = Path(cache_dir) / model_hash_str
        cache_path.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write input files
            input_json = os.path.join(tmpdir, "input.json")
            cal_json = os.path.join(tmpdir, "cal_data.json")
            model_onnx = os.path.join(tmpdir, "model.onnx")
            settings_json = os.path.join(tmpdir, "settings.json")
            circuit_ezkl = os.path.join(tmpdir, "circuit.ezkl")
            srs_path = str(cache_path / "kzg.srs")
            vk_path = str(cache_path / "vk.key")
            pk_path = str(cache_path / "pk.key")
            witness_json = os.path.join(tmpdir, "witness.json")
            proof_json = os.path.join(tmpdir, "proof.json")

            # Copy ONNX to tmpdir
            shutil.copy(onnx_path, model_onnx)

            # Write input.json (EZKL format)
            with open(input_json, "w") as f:
                json.dump({"input_data": input_data}, f)

            # Write calibration data
            with open(cal_json, "w") as f:
                json.dump({"input_data": calibration_data}, f)

            peak_mb = [0.0]
            stop_event = threading.Event()
            mem_thread = threading.Thread(
                target=_poll_memory, args=(stop_event, peak_mb), daemon=True
            )

            try:
                # Step 3: gen-settings
                r = _run_step(
                    ["ezkl", "gen-settings", "-M", model_onnx, "-O", settings_json],
                    timeout=60,
                    cwd=tmpdir,
                )
                if r.returncode != 0:
                    return ProverResult.failed("gen_settings_failed")

                # Step 4: calibrate-settings (sample count depends on model type)
                cal_cmd = [
                    "ezkl", "calibrate-settings",
                    "-M", model_onnx,
                    "-D", cal_json,
                    "--settings-path", settings_json,
                    "--target", "resources",
                ]
                r = _run_step(cal_cmd, timeout=180, cwd=tmpdir)
                used_scale_rebase = False
                if r.returncode != 0:
                    # Fallback: expand quantization search range for poly models
                    cal_cmd_fallback = cal_cmd + ["--scale-rebase-multiplier", "2"]
                    r2 = _run_step(cal_cmd_fallback, timeout=180, cwd=tmpdir)
                    if r2.returncode != 0:
                        return ProverResult.failed("calibrate_settings_failed")
                    used_scale_rebase = True

                # Step 5: compile-circuit
                r = _run_step(
                    [
                        "ezkl", "compile-circuit",
                        "-M", model_onnx,
                        "--settings-path", settings_json,
                        "-O", circuit_ezkl,
                    ],
                    timeout=120,
                    cwd=tmpdir,
                )
                if r.returncode != 0:
                    stderr = r.stderr.lower()
                    if "unsupported" in stderr or "not supported" in stderr:
                        # Try to extract op name
                        op_name = "unknown"
                        for line in r.stderr.split("\n"):
                            if "unsupported" in line.lower():
                                op_name = line.strip()[:50]
                                break
                        return ProverResult.failed(f"unsupported_op: {op_name}")
                    return ProverResult.failed("compile_circuit_failed")

                # Step 6: Check disk space >= 10 GB
                usage = shutil.disk_usage(tmpdir)
                if usage.free < 10 * (1024 ** 3):
                    return ProverResult.failed("insufficient_disk")

                mem_thread.start()

                # Step 7: get-srs (cached by logrows)
                if not os.path.exists(srs_path):
                    r = _run_step(
                        [
                            "ezkl", "get-srs",
                            "--settings-path", settings_json,
                            "--srs-path", srs_path,
                        ],
                        timeout=300,
                        cwd=tmpdir,
                    )
                    if r.returncode != 0:
                        stop_event.set()
                        return ProverResult.failed("srs_download_failed")

                # Step 8: setup
                t0 = time.perf_counter()
                r = _run_step(
                    [
                        "ezkl", "setup",
                        "-M", circuit_ezkl,
                        "--srs-path", srs_path,
                        "--vk-path", vk_path,
                        "--pk-path", pk_path,
                    ],
                    timeout=timeout_setup,
                    cwd=tmpdir,
                )
                setup_time = time.perf_counter() - t0
                if r.returncode == -9:
                    stop_event.set()
                    return ProverResult.failed("oom_killed")
                if r.returncode != 0:
                    stop_event.set()
                    return ProverResult.failed("setup_failed")

                # Step 9: gen-witness
                r = _run_step(
                    [
                        "ezkl", "gen-witness",
                        "-M", circuit_ezkl,
                        "-D", input_json,
                        "--settings-path", settings_json,
                        "-O", witness_json,
                    ],
                    timeout=120,
                    cwd=tmpdir,
                )
                if r.returncode != 0:
                    stop_event.set()
                    return ProverResult.failed("gen_witness_failed")

                # Step 10: prove
                t0 = time.perf_counter()
                r = _run_step(
                    [
                        "ezkl", "prove",
                        "-W", witness_json,
                        "-M", circuit_ezkl,
                        "--pk-path", pk_path,
                        "-O", proof_json,
                        "--proof-type", "single",
                    ],
                    timeout=timeout_prove,
                    cwd=tmpdir,
                )
                prove_time = time.perf_counter() - t0
                if r.returncode == -9:
                    stop_event.set()
                    return ProverResult.failed("oom_killed")
                if r.returncode != 0:
                    stop_event.set()
                    return ProverResult.failed("prove_failed")

                # Step 11: verify
                t0 = time.perf_counter()
                r = _run_step(
                    [
                        "ezkl", "verify",
                        "--proof-path", proof_json,
                        "--vk-path", vk_path,
                        "--settings-path", settings_json,
                    ],
                    timeout=60,
                    cwd=tmpdir,
                )
                verify_time = time.perf_counter() - t0
                if r.returncode != 0:
                    stop_event.set()
                    return ProverResult.failed("verify_failed")

                stop_event.set()

                # Parse settings.json for rows/cols
                with open(settings_json) as f:
                    settings = json.load(f)
                logrows = settings.get("run_args", {}).get("logrows", 14)
                num_rows = 2 ** logrows
                num_cols = settings.get("num_cols", 256)

                proof_size = os.path.getsize(proof_json) if os.path.exists(proof_json) else None
                srs_size = os.path.getsize(srs_path) if os.path.exists(srs_path) else None

                return ProverResult(
                    available=True,
                    error=None,
                    num_rows=num_rows,
                    num_cols=num_cols,
                    total_cells=num_rows * num_cols,
                    quantized_accuracy_top1=None,  # TODO: measure separately
                    setup_time_s=setup_time,
                    prove_time_s=prove_time,
                    verify_time_s=verify_time,
                    proof_size_bytes=proof_size,
                    peak_memory_mb=peak_mb[0],
                    srs_size_bytes=srs_size,
                    activation_clip=activation_clip,
                    used_scale_rebase=used_scale_rebase,
                )

            except subprocess.TimeoutExpired as e:
                stop_event.set()
                # Use the specific EZKL subcommand (e.cmd[1]) for reliable matching
                subcommand = e.cmd[1] if isinstance(e.cmd, (list, tuple)) and len(e.cmd) > 1 else ""
                if subcommand == "setup":
                    return ProverResult.failed("setup_timeout")
                elif subcommand == "prove":
                    return ProverResult.failed("prove_timeout")
                return ProverResult.failed("timeout")
            finally:
                stop_event.set()
