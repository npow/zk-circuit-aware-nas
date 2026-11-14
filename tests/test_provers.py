"""Tests for prover implementations: MockProver, ProverResult, and EZKLProver."""

from unittest.mock import MagicMock, patch

import pytest

from zkml_bench.provers.base import ProverResult
from zkml_bench.provers.mock import MockProver


# ---------------------------------------------------------------------------
# ProverResult factory methods
# ---------------------------------------------------------------------------


def test_prover_result_unavailable():
    r = ProverResult.unavailable()
    assert r.available is False
    assert r.error is None
    assert r.total_cells is None
    assert r.num_rows is None
    assert r.num_cols is None
    assert r.prove_time_s is None
    assert r.verify_time_s is None
    assert r.proof_size_bytes is None
    assert r.peak_memory_mb is None


def test_prover_result_failed():
    r = ProverResult.failed("oom_killed")
    assert r.available is True
    assert r.error == "oom_killed"
    assert r.total_cells is None
    assert r.num_rows is None
    assert r.num_cols is None


def test_prover_result_failed_verify():
    r = ProverResult.failed("verify_failed")
    assert r.available is True
    assert r.error == "verify_failed"


def test_prover_result_failed_timeout():
    r = ProverResult.failed("timeout")
    assert r.available is True
    assert r.error == "timeout"


def test_prover_result_unavailable_vs_failed():
    unavail = ProverResult.unavailable()
    failed = ProverResult.failed("oom_killed")
    assert unavail.available is False
    assert failed.available is True


# ---------------------------------------------------------------------------
# MockProver
# ---------------------------------------------------------------------------


def test_mock_prover_returns_result(tmp_dir):
    prover = MockProver()
    result = prover.prove(
        onnx_path="/dev/null",
        input_data=[[0.0] * 784],
        calibration_data=[[0.0] * 784] * 64,
        accuracy_top1=90.0,
        cache_dir=tmp_dir,
        model_hash="abc123",
    )
    assert isinstance(result, ProverResult)


def test_mock_prover_available(tmp_dir):
    prover = MockProver()
    result = prover.prove(
        onnx_path="/dev/null",
        input_data=[],
        calibration_data=[],
        accuracy_top1=90.0,
        cache_dir=tmp_dir,
        model_hash="abc123",
    )
    assert result.available is True
    assert result.error is None


def test_mock_prover_num_rows(tmp_dir):
    prover = MockProver()
    result = prover.prove(
        onnx_path="/dev/null",
        input_data=[],
        calibration_data=[],
        accuracy_top1=90.0,
        cache_dir=tmp_dir,
        model_hash="abc",
    )
    assert result.num_rows == 2 ** 14


def test_mock_prover_num_cols(tmp_dir):
    prover = MockProver()
    result = prover.prove(
        onnx_path="/dev/null",
        input_data=[],
        calibration_data=[],
        accuracy_top1=90.0,
        cache_dir=tmp_dir,
        model_hash="abc",
    )
    assert result.num_cols == 256


def test_mock_prover_total_cells(tmp_dir):
    prover = MockProver()
    result = prover.prove(
        onnx_path="/dev/null",
        input_data=[],
        calibration_data=[],
        accuracy_top1=90.0,
        cache_dir=tmp_dir,
        model_hash="abc",
    )
    assert result.total_cells == 2 ** 14 * 256


def test_mock_prover_prove_time(tmp_dir):
    prover = MockProver()
    result = prover.prove(
        onnx_path="/dev/null",
        input_data=[],
        calibration_data=[],
        accuracy_top1=90.0,
        cache_dir=tmp_dir,
        model_hash="abc",
    )
    assert result.prove_time_s == 2.5


def test_mock_prover_verify_time(tmp_dir):
    prover = MockProver()
    result = prover.prove(
        onnx_path="/dev/null",
        input_data=[],
        calibration_data=[],
        accuracy_top1=90.0,
        cache_dir=tmp_dir,
        model_hash="abc",
    )
    assert result.verify_time_s == 0.1


def test_mock_prover_proof_size(tmp_dir):
    prover = MockProver()
    result = prover.prove(
        onnx_path="/dev/null",
        input_data=[],
        calibration_data=[],
        accuracy_top1=90.0,
        cache_dir=tmp_dir,
        model_hash="abc",
    )
    assert result.proof_size_bytes == 8192


def test_mock_prover_peak_memory(tmp_dir):
    prover = MockProver()
    result = prover.prove(
        onnx_path="/dev/null",
        input_data=[],
        calibration_data=[],
        accuracy_top1=90.0,
        cache_dir=tmp_dir,
        model_hash="abc",
    )
    assert result.peak_memory_mb == 512.0


def test_mock_prover_quantized_accuracy_degraded(tmp_dir):
    prover = MockProver()
    result = prover.prove(
        onnx_path="/dev/null",
        input_data=[],
        calibration_data=[],
        accuracy_top1=90.0,
        cache_dir=tmp_dir,
        model_hash="abc",
    )
    assert result.quantized_accuracy_top1 == pytest.approx(90.0 - 1.5)


def test_mock_prover_quantization_degradation_different_accuracy(tmp_dir):
    """Quantised accuracy is always accuracy_top1 - 1.5."""
    prover = MockProver()
    result = prover.prove(
        onnx_path="/dev/null",
        input_data=[],
        calibration_data=[],
        accuracy_top1=75.3,
        cache_dir=tmp_dir,
        model_hash="abc",
    )
    assert result.quantized_accuracy_top1 == pytest.approx(75.3 - 1.5)


def test_mock_prover_is_base_prover():
    from zkml_bench.provers.base import BaseProver
    assert isinstance(MockProver(), BaseProver)


# ---------------------------------------------------------------------------
# EZKLProver — mocked subprocess tests
# ---------------------------------------------------------------------------


def test_ezkl_not_installed_returns_unavailable(tmp_dir):
    """When ezkl is not in PATH, EZKLProver returns unavailable."""
    from zkml_bench.provers.ezkl import EZKLProver

    with patch("zkml_bench.provers.ezkl._check_ezkl_available", return_value=None):
        prover = EZKLProver()
        result = prover.prove(
            onnx_path="/dev/null",
            input_data=[],
            calibration_data=[],
            accuracy_top1=90.0,
            cache_dir=tmp_dir,
            model_hash_str="abc",
        )
    assert result.available is False


def test_ezkl_gen_settings_failure_returns_failed(tmp_dir):
    """gen-settings failure returns ProverResult.failed('gen_settings_failed')."""
    from zkml_bench.provers.ezkl import EZKLProver

    fail_result = MagicMock()
    fail_result.returncode = 1
    fail_result.stderr = "error in gen-settings"

    with patch("zkml_bench.provers.ezkl._check_ezkl_available", return_value="9.1.0"), \
         patch("zkml_bench.provers.ezkl._run_step", return_value=fail_result), \
         patch("shutil.copy"):
        prover = EZKLProver()
        result = prover.prove(
            onnx_path="/dev/null",
            input_data=[],
            calibration_data=[],
            accuracy_top1=90.0,
            cache_dir=tmp_dir,
            model_hash_str="testmodel",
        )
    assert result.available is True
    assert result.error == "gen_settings_failed"


def test_ezkl_insufficient_disk(tmp_dir):
    """Disk space check triggers before EZKL setup when space < 10 GB."""
    from zkml_bench.provers.ezkl import EZKLProver

    success_result = MagicMock()
    success_result.returncode = 0
    success_result.stdout = ""
    success_result.stderr = ""

    call_count = [0]

    def fake_run(cmd, **kwargs):
        call_count[0] += 1
        return success_result

    with patch("zkml_bench.provers.ezkl._check_ezkl_available", return_value="9.1.0"), \
         patch("zkml_bench.provers.ezkl._run_step", side_effect=fake_run), \
         patch("shutil.copy"), \
         patch("shutil.disk_usage") as mock_disk:
        mock_disk.return_value = MagicMock(free=5 * 1024 ** 3)  # only 5 GB
        prover = EZKLProver()
        result = prover.prove(
            onnx_path="/dev/null",
            input_data=[],
            calibration_data=[],
            accuracy_top1=90.0,
            cache_dir=tmp_dir,
            model_hash_str="testmodel",
        )
    assert result.error == "insufficient_disk"


def test_ezkl_oom_detected_at_setup(tmp_dir):
    """OOM (returncode -9) at setup step is recorded as oom_killed."""
    from zkml_bench.provers.ezkl import EZKLProver

    success_result = MagicMock()
    success_result.returncode = 0
    success_result.stdout = ""
    success_result.stderr = ""

    oom_result = MagicMock()
    oom_result.returncode = -9
    oom_result.stderr = ""

    call_count = [0]

    def fake_run(cmd, **kwargs):
        call_count[0] += 1
        # gen-settings (1), calibrate (2), compile (3) succeed; get-srs (4) succeed; setup (5) OOM
        if call_count[0] <= 4:
            return success_result
        return oom_result

    with patch("zkml_bench.provers.ezkl._check_ezkl_available", return_value="9.1.0"), \
         patch("zkml_bench.provers.ezkl._run_step", side_effect=fake_run), \
         patch("shutil.copy"), \
         patch("shutil.disk_usage") as mock_disk, \
         patch("os.path.exists", return_value=False):  # force get-srs to run
        mock_disk.return_value = MagicMock(free=20 * 1024 ** 3)  # 20 GB free
        prover = EZKLProver()
        result = prover.prove(
            onnx_path="/dev/null",
            input_data=[],
            calibration_data=[],
            accuracy_top1=90.0,
            cache_dir=tmp_dir,
            model_hash_str="testmodel",
        )
    assert result.error == "oom_killed"


def test_ezkl_timeout_returns_failed(tmp_dir):
    """subprocess.TimeoutExpired returns a timeout error code."""
    import subprocess
    from zkml_bench.provers.ezkl import EZKLProver

    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=60)

    with patch("zkml_bench.provers.ezkl._check_ezkl_available", return_value="9.1.0"), \
         patch("zkml_bench.provers.ezkl._run_step", side_effect=fake_run), \
         patch("shutil.copy"):
        prover = EZKLProver()
        result = prover.prove(
            onnx_path="/dev/null",
            input_data=[],
            calibration_data=[],
            accuracy_top1=90.0,
            cache_dir=tmp_dir,
            model_hash_str="testmodel",
        )
    assert result.available is True
    assert result.error is not None
    assert "timeout" in result.error


def test_ezkl_unsupported_op_detected(tmp_dir):
    """Unsupported op in compile-circuit is detected and reported."""
    from zkml_bench.provers.ezkl import EZKLProver

    success_result = MagicMock()
    success_result.returncode = 0
    success_result.stdout = ""
    success_result.stderr = ""

    unsupported_result = MagicMock()
    unsupported_result.returncode = 1
    unsupported_result.stderr = "Error: unsupported op: LSTM"

    call_count = [0]

    def fake_run(cmd, **kwargs):
        call_count[0] += 1
        if call_count[0] <= 2:  # gen-settings, calibrate succeed
            return success_result
        return unsupported_result  # compile-circuit fails with unsupported op

    with patch("zkml_bench.provers.ezkl._check_ezkl_available", return_value="9.1.0"), \
         patch("zkml_bench.provers.ezkl._run_step", side_effect=fake_run), \
         patch("shutil.copy"):
        prover = EZKLProver()
        result = prover.prove(
            onnx_path="/dev/null",
            input_data=[],
            calibration_data=[],
            accuracy_top1=90.0,
            cache_dir=tmp_dir,
            model_hash_str="testmodel",
        )
    assert result.available is True
    assert result.error is not None
    assert "unsupported_op" in result.error


def test_ezkl_verify_failed(tmp_dir):
    """Verification failure (non-zero from verify step) recorded as verify_failed."""
    from zkml_bench.provers.ezkl import EZKLProver

    success_result = MagicMock()
    success_result.returncode = 0
    success_result.stdout = ""
    success_result.stderr = ""

    fail_result = MagicMock()
    fail_result.returncode = 1
    fail_result.stderr = "verification failed"

    def fake_run(cmd, **kwargs):
        if "verify" in cmd:
            return fail_result
        return success_result

    import json
    fake_settings = {"run_args": {"logrows": 14}, "num_cols": 256}

    with patch("zkml_bench.provers.ezkl._check_ezkl_available", return_value="9.1.0"), \
         patch("zkml_bench.provers.ezkl._run_step", side_effect=fake_run), \
         patch("shutil.copy"), \
         patch("shutil.disk_usage") as mock_disk, \
         patch("os.path.exists", return_value=True), \
         patch("os.path.getsize", return_value=1024), \
         patch("builtins.open", MagicMock(return_value=MagicMock(
             __enter__=MagicMock(return_value=MagicMock(
                 read=MagicMock(return_value=json.dumps(fake_settings))
             )),
             __exit__=MagicMock(return_value=False)
         ))):
        mock_disk.return_value = MagicMock(free=20 * 1024 ** 3)
        prover = EZKLProver()
        result = prover.prove(
            onnx_path="/dev/null",
            input_data=[],
            calibration_data=[],
            accuracy_top1=90.0,
            cache_dir=tmp_dir,
            model_hash_str="testmodel",
        )
    assert result.error == "verify_failed"


def test_ezkl_oom_at_prove_step(tmp_dir):
    """OOM (returncode -9) at prove step is recorded as oom_killed."""
    from zkml_bench.provers.ezkl import EZKLProver

    success_result = MagicMock()
    success_result.returncode = 0
    success_result.stdout = ""
    success_result.stderr = ""

    oom_result = MagicMock()
    oom_result.returncode = -9
    oom_result.stderr = ""

    def fake_run(cmd, **kwargs):
        if "prove" in cmd:
            return oom_result
        return success_result

    with patch("zkml_bench.provers.ezkl._check_ezkl_available", return_value="9.1.0"), \
         patch("zkml_bench.provers.ezkl._run_step", side_effect=fake_run), \
         patch("shutil.copy"), \
         patch("shutil.disk_usage") as mock_disk, \
         patch("os.path.exists", return_value=True):
        mock_disk.return_value = MagicMock(free=20 * 1024 ** 3)
        prover = EZKLProver()
        result = prover.prove(
            onnx_path="/dev/null",
            input_data=[],
            calibration_data=[],
            accuracy_top1=90.0,
            cache_dir=tmp_dir,
            model_hash_str="testmodel",
        )
    assert result.error == "oom_killed"


def test_ezkl_insufficient_disk_stops_before_srs(tmp_dir):
    """Disk check triggers at Step 6; SRS download step should NOT be called."""
    from zkml_bench.provers.ezkl import EZKLProver

    success_result = MagicMock()
    success_result.returncode = 0
    success_result.stderr = ""
    call_log = []

    def fake_run(cmd, **kwargs):
        call_log.append(cmd[1] if len(cmd) > 1 else "unknown")
        return success_result

    with patch("zkml_bench.provers.ezkl._check_ezkl_available", return_value="9.1.0"), \
         patch("zkml_bench.provers.ezkl._run_step", side_effect=fake_run), \
         patch("shutil.copy"), \
         patch("shutil.disk_usage") as mock_disk:
        mock_disk.return_value = MagicMock(free=5 * 1024 ** 3)  # 5 GB only
        prover = EZKLProver()
        result = prover.prove(
            onnx_path="/dev/null",
            input_data=[],
            calibration_data=[],
            accuracy_top1=90.0,
            cache_dir=tmp_dir,
            model_hash_str="testmodel",
        )
    assert result.error == "insufficient_disk"
    # Should have called gen-settings, calibrate-settings, compile-circuit (3 steps) but NOT get-srs
    assert "get-srs" not in call_log
    assert len(call_log) == 3


# ---------------------------------------------------------------------------
# T021: polynomial EZKL helper unit tests
# ---------------------------------------------------------------------------

def test_get_calibration_samples_poly():
    from zkml_bench.provers.ezkl import _get_calibration_samples
    assert _get_calibration_samples("resnet20_poly") == 128
    assert _get_calibration_samples("lenet5_quad") == 128
    assert _get_calibration_samples("mlp_cubic") == 128


def test_get_calibration_samples_relu():
    from zkml_bench.provers.ezkl import _get_calibration_samples
    assert _get_calibration_samples("resnet20") == 64
    assert _get_calibration_samples("vgg11_small") == 64
    assert _get_calibration_samples("") == 64


def test_wrap_poly_model_clamps():
    import torch
    import torch.nn as nn
    from zkml_bench.provers.ezkl import _wrap_poly_model_for_export

    class SquaredActivation(nn.Module):
        """x² — name contains 'activation' so wrap detection fires."""
        def forward(self, x):
            return x * x

    class PolyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.act = SquaredActivation()

        def forward(self, x):
            return self.act(x)

    model = PolyModel().eval()
    wrapped = _wrap_poly_model_for_export(model, clip_value=5.0)
    x = torch.tensor([[-100.0, 0.0, 100.0]])
    out = wrapped(x)
    assert out.max().item() <= 5.0
    assert out.min().item() >= -5.0


def test_wrap_non_poly_model_unchanged():
    import torch.nn as nn
    from zkml_bench.provers.ezkl import _wrap_poly_model_for_export

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
    wrapped = _wrap_poly_model_for_export(model)
    # Non-poly model: no PolyActivation detected, returned as-is (same object)
    assert wrapped is model
