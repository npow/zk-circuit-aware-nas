from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class ProverResult:
    available: bool
    error: Optional[str]  # None if success; error code string if failed
    num_rows: Optional[int]
    num_cols: Optional[int]
    total_cells: Optional[int]
    quantized_accuracy_top1: Optional[float]
    setup_time_s: Optional[float]
    prove_time_s: Optional[float]
    verify_time_s: Optional[float]
    proof_size_bytes: Optional[int]
    peak_memory_mb: Optional[float]
    srs_size_bytes: Optional[int]
    # Polynomial model calibration metadata
    activation_clip: Optional[float] = None   # clamp value applied before export
    used_scale_rebase: bool = False            # True if --scale-rebase-multiplier fallback used

    @classmethod
    def unavailable(cls) -> "ProverResult":
        return cls(
            available=False, error=None, num_rows=None, num_cols=None,
            total_cells=None, quantized_accuracy_top1=None,
            setup_time_s=None, prove_time_s=None, verify_time_s=None,
            proof_size_bytes=None, peak_memory_mb=None, srs_size_bytes=None,
        )

    @classmethod
    def failed(cls, error_code: str) -> "ProverResult":
        return cls(
            available=True, error=error_code, num_rows=None, num_cols=None,
            total_cells=None, quantized_accuracy_top1=None,
            setup_time_s=None, prove_time_s=None, verify_time_s=None,
            proof_size_bytes=None, peak_memory_mb=None, srs_size_bytes=None,
        )


class BaseProver(ABC):
    @abstractmethod
    def prove(
        self,
        onnx_path: str,
        input_data: list,  # list of input arrays for EZKL input.json format
        calibration_data: list,  # 64 samples for calibration
        accuracy_top1: float,  # floating-point accuracy (to compute quantization degradation)
        cache_dir: str,
        model_hash: str,
        timeout_setup: int = 600,
        timeout_prove: int = 300,
    ) -> ProverResult:
        ...
