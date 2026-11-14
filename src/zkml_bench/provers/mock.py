from .base import BaseProver, ProverResult


class MockProver(BaseProver):
    """Mock prover for testing. Returns realistic fake values without EZKL."""

    NUM_ROWS = 2 ** 14  # 16384
    NUM_COLS = 256

    def prove(self, onnx_path, input_data, calibration_data, accuracy_top1,
              cache_dir, model_hash, timeout_setup=600, timeout_prove=300) -> ProverResult:
        total_cells = self.NUM_ROWS * self.NUM_COLS
        return ProverResult(
            available=True,
            error=None,
            num_rows=self.NUM_ROWS,
            num_cols=self.NUM_COLS,
            total_cells=total_cells,
            quantized_accuracy_top1=accuracy_top1 - 1.5,  # simulate quantization degradation
            setup_time_s=5.0,
            prove_time_s=2.5,
            verify_time_s=0.1,
            proof_size_bytes=8192,
            peak_memory_mb=512.0,
            srs_size_bytes=1024 * 1024 * 10,  # 10 MB
        )
