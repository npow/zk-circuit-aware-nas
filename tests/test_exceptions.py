"""Tests for the zkml_bench exception hierarchy."""

import pytest

from zkml_bench.exceptions import (
    ZKMLBenchError,
    ConfigValidationError,
    DatasetDownloadError,
    TrainingDivergenceError,
    WeightLoadError,
    OnnxExportError,
    EZKLError,
    ModelRegistrationError,
)


def test_all_exceptions_are_zkml_bench_errors():
    for exc_class in [
        ConfigValidationError,
        DatasetDownloadError,
        TrainingDivergenceError,
        WeightLoadError,
        OnnxExportError,
        EZKLError,
        ModelRegistrationError,
    ]:
        assert issubclass(exc_class, ZKMLBenchError), (
            f"{exc_class.__name__} should be a subclass of ZKMLBenchError"
        )
        assert issubclass(exc_class, Exception), (
            f"{exc_class.__name__} should be a subclass of Exception"
        )


def test_zkml_bench_error_is_exception():
    assert issubclass(ZKMLBenchError, Exception)


def test_config_validation_error_message():
    try:
        raise ConfigValidationError("epochs must be positive")
    except ConfigValidationError as e:
        assert "epochs" in str(e)


def test_ezkl_error_has_error_code():
    try:
        raise EZKLError("something failed", error_code="oom_killed")
    except EZKLError as e:
        assert e.error_code == "oom_killed"


def test_ezkl_error_default_error_code():
    try:
        raise EZKLError("something failed")
    except EZKLError as e:
        assert e.error_code == "unknown"


def test_ezkl_error_repr_contains_code():
    e = EZKLError("test msg", error_code="timeout")
    r = repr(e)
    assert "timeout" in r
    assert "EZKLError" in r


def test_ezkl_error_verify_failed_code():
    try:
        raise EZKLError("verification failed", error_code="verify_failed")
    except EZKLError as e:
        assert e.error_code == "verify_failed"


def test_weight_load_error_message():
    try:
        raise WeightLoadError("keys mismatch: missing=['fc.weight']")
    except WeightLoadError as e:
        assert "mismatch" in str(e).lower() or "missing" in str(e)


def test_training_divergence_error():
    try:
        raise TrainingDivergenceError("NaN loss detected at batch 5")
    except TrainingDivergenceError as e:
        assert "NaN" in str(e)


def test_dataset_download_error():
    try:
        raise DatasetDownloadError("Download failed, visit http://example.com manually")
    except DatasetDownloadError as e:
        assert "Download" in str(e)


def test_onnx_export_error():
    try:
        raise OnnxExportError("Cannot export model with dynamic control flow")
    except OnnxExportError as e:
        assert "export" in str(e).lower()


def test_model_registration_error():
    try:
        raise ModelRegistrationError("Model 'mlp_small' already registered")
    except ModelRegistrationError as e:
        assert "mlp_small" in str(e)


def test_exceptions_can_be_caught_as_base():
    """All derived exceptions can be caught as ZKMLBenchError."""
    for exc_class, msg in [
        (ConfigValidationError, "cfg"),
        (DatasetDownloadError, "dl"),
        (TrainingDivergenceError, "div"),
        (WeightLoadError, "wt"),
        (OnnxExportError, "onnx"),
        (ModelRegistrationError, "reg"),
    ]:
        with pytest.raises(ZKMLBenchError):
            raise exc_class(msg)


def test_ezkl_error_can_be_caught_as_base():
    with pytest.raises(ZKMLBenchError):
        raise EZKLError("ezkl broke", error_code="oom_killed")
