"""Custom exception hierarchy for zkml-bench."""

from __future__ import annotations


class ZKMLBenchError(Exception):
    """Base class for all zkml-bench exceptions."""


class ConfigValidationError(ZKMLBenchError):
    """Raised when a configuration value is invalid (e.g. negative epochs)."""


class DatasetDownloadError(ZKMLBenchError):
    """Raised when a dataset download fails.

    The exception message includes a manual download URL so users can retrieve
    the data themselves if automated download is not possible.
    """


class TrainingDivergenceError(ZKMLBenchError):
    """Raised when training diverges due to NaN loss or exploding loss."""


class WeightLoadError(ZKMLBenchError):
    """Raised when loading a state_dict fails due to key or shape mismatches.

    The exception message includes details about the mismatched keys / shapes.
    """


class OnnxExportError(ZKMLBenchError):
    """Raised when ONNX export of a model fails."""


class EZKLError(ZKMLBenchError):
    """Raised when an EZKL subprocess fails.

    Attributes
    ----------
    error_code:
        A short machine-readable code describing the failure mode.
        Common values: ``"oom_killed"``, ``"verify_failed"``,
        ``"timeout"``, ``"unknown"``.
    """

    def __init__(self, message: str, error_code: str = "unknown") -> None:
        super().__init__(message)
        self.error_code: str = error_code

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self)!r}, error_code={self.error_code!r})"


class ModelRegistrationError(ZKMLBenchError):
    """Raised when model registration fails.

    Causes include attempting to register a duplicate model name or providing
    a constructor with an incompatible signature.
    """
