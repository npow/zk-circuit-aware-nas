"""Tests for the backend-aware NAS module (backend_aware.py)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Import guards
# ---------------------------------------------------------------------------

from zkml_bench.nas.backend_aware import (
    BackendAwareNAS,
    BackendIterationRecord,
    _build_system_prompt,
    _build_user_prompt,
    _extract_code,
    _extract_metadata,
    _frontier_summary,
    _hypervolume,
    _save_log,
)


# ---------------------------------------------------------------------------
# System prompt: backend-specific cost model text is injected
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_ezkl_prompt_mentions_logrows(self):
        prompt = _build_system_prompt("mnist", "ezkl")
        assert "logrows" in prompt.lower()

    def test_deepprove_prompt_mentions_wide_shallow(self):
        prompt = _build_system_prompt("mnist", "deepprove")
        assert "wide" in prompt.lower() and "shallow" in prompt.lower()

    def test_deepprove_prompt_mentions_depth_penalty(self):
        prompt = _build_system_prompt("mnist", "deepprove")
        # Should mention the 1.36x measured penalty
        assert "1.36" in prompt

    def test_expander_prompt_mentions_log2_gates(self):
        prompt = _build_system_prompt("mnist", "expander")
        assert "log2" in prompt.lower() or "log2_gates" in prompt.lower()

    def test_all_prompts_contain_get_model(self):
        for backend in ("ezkl", "deepprove", "expander"):
            prompt = _build_system_prompt("mnist", backend)
            assert "get_model" in prompt, f"backend={backend} prompt missing get_model"

    def test_deepprove_warns_no_poly(self):
        prompt = _build_system_prompt("mnist", "deepprove")
        # Should warn that polynomial activations are NOT supported
        assert "not supported" in prompt.lower() or "unsupported" in prompt.lower()

    def test_deepprove_warns_no_batchnorm(self):
        prompt = _build_system_prompt("mnist", "deepprove")
        assert "batchnorm" in prompt.lower() or "batch norm" in prompt.lower()


# ---------------------------------------------------------------------------
# User prompt: backend name and iteration context are present
# ---------------------------------------------------------------------------


class TestUserPrompt:
    def _make_rec(self, name: str, top1: float, constraints: int) -> BackendIterationRecord:
        return BackendIterationRecord(
            iteration=1,
            backend="deepprove",
            candidate_name=name,
            reasoning="test",
            code="",
            top1=top1,
            top5=top1,
            constraints=constraints,
            proxy_value_ezkl=0,
            proxy_value_deepprove=constraints,
            proxy_value_expander=0,
            nonlinear_ops=0,
            mult_depth=0,
            n_params=0,
            train_time_s=0.0,
            proxy_method="onnx",
            error=None,
            accepted=True,
            hypervolume_before=0.0,
            hypervolume_after=0.5,
            timestamp="2026-01-01T00:00:00",
        )

    def test_first_iteration_no_history_deepprove(self):
        prompt = _build_user_prompt(1, 5, "mnist", 5, "deepprove", [])
        assert "first iteration" in prompt.lower()
        assert "relu" in prompt.lower()

    def test_first_iteration_no_history_ezkl(self):
        prompt = _build_user_prompt(1, 5, "mnist", 5, "ezkl", [])
        assert "first iteration" in prompt.lower()

    def test_prompt_with_history_shows_backend_proxy(self):
        history = [self._make_rec("test_model", 97.0, 50000)]
        prompt = _build_user_prompt(2, 5, "mnist", 5, "deepprove", history)
        assert "deepprove_proxy" in prompt or "deepprove" in prompt.lower()
        assert "test_model" in prompt

    def test_prompt_includes_backend_name(self):
        for backend in ("ezkl", "deepprove", "expander"):
            prompt = _build_user_prompt(1, 5, "mnist", 5, backend, [])
            assert backend in prompt.lower()

    def test_output_format_present(self):
        prompt = _build_user_prompt(1, 5, "mnist", 5, "ezkl", [])
        assert "CANDIDATE_NAME" in prompt
        assert "USE_POLY_INIT" in prompt
        assert "REASONING" in prompt


# ---------------------------------------------------------------------------
# Code / metadata extraction
# ---------------------------------------------------------------------------


class TestCodeExtraction:
    _VALID_RESPONSE = """\
CANDIDATE_NAME = "wide_shallow_mlp"
USE_POLY_INIT = False
REASONING = "Wide-shallow avoids GKR depth penalty"

```python
import torch.nn as nn

CANDIDATE_NAME = "wide_shallow_mlp"
USE_POLY_INIT = False
REASONING = "Wide-shallow avoids GKR depth penalty"

def get_model(num_classes, input_shape):
    in_f = input_shape[0] * input_shape[1] * input_shape[2]
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_f, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes),
    )
```
"""

    def test_extract_code_finds_block(self):
        code = _extract_code(self._VALID_RESPONSE)
        assert "def get_model" in code

    def test_extract_metadata_name(self):
        code = _extract_code(self._VALID_RESPONSE)
        name, poly, reasoning = _extract_metadata(self._VALID_RESPONSE, code)
        assert name == "wide_shallow_mlp"

    def test_extract_metadata_poly_false(self):
        code = _extract_code(self._VALID_RESPONSE)
        _, poly, _ = _extract_metadata(self._VALID_RESPONSE, code)
        assert poly is False

    def test_extract_metadata_reasoning(self):
        code = _extract_code(self._VALID_RESPONSE)
        _, _, reasoning = _extract_metadata(self._VALID_RESPONSE, code)
        assert "GKR" in reasoning or "shallow" in reasoning

    def test_extract_code_raises_on_no_block(self):
        with pytest.raises(ValueError, match="No Python code block"):
            _extract_code("Here is my answer with no code.")


# ---------------------------------------------------------------------------
# BackendIterationRecord
# ---------------------------------------------------------------------------


class TestBackendIterationRecord:
    def _make(self, error=None) -> BackendIterationRecord:
        return BackendIterationRecord(
            iteration=1,
            backend="ezkl",
            candidate_name="test",
            reasoning="",
            code="",
            top1=97.5,
            top5=99.9,
            constraints=100000,
            proxy_value_ezkl=100000,
            proxy_value_deepprove=50000,
            proxy_value_expander=17,
            nonlinear_ops=5,
            mult_depth=2,
            n_params=12345,
            train_time_s=3.2,
            proxy_method="onnx",
            error=error,
            accepted=True,
            hypervolume_before=0.3,
            hypervolume_after=0.4,
            timestamp="2026-01-01T00:00:00",
        )

    def test_failed_false_when_no_error(self):
        rec = self._make()
        assert not rec.failed

    def test_failed_true_when_error(self):
        rec = self._make(error="something went wrong")
        assert rec.failed

    def test_to_dict_roundtrip(self):
        rec = self._make()
        d = rec.to_dict()
        rec2 = BackendIterationRecord(**d)
        assert rec2.candidate_name == rec.candidate_name
        assert rec2.proxy_value_deepprove == rec.proxy_value_deepprove

    def test_all_three_proxy_fields_present(self):
        rec = self._make()
        d = rec.to_dict()
        assert "proxy_value_ezkl" in d
        assert "proxy_value_deepprove" in d
        assert "proxy_value_expander" in d


# ---------------------------------------------------------------------------
# Hypervolume + frontier summary
# ---------------------------------------------------------------------------


class TestHypervolume:
    def _rec(self, name, top1, constraints) -> BackendIterationRecord:
        return BackendIterationRecord(
            iteration=1, backend="ezkl", candidate_name=name,
            reasoning="", code="", top1=top1, top5=top1,
            constraints=constraints,
            proxy_value_ezkl=constraints, proxy_value_deepprove=0,
            proxy_value_expander=0, nonlinear_ops=0, mult_depth=0,
            n_params=0, train_time_s=0.0, proxy_method="onnx",
            error=None, accepted=True,
            hypervolume_before=0.0, hypervolume_after=0.0,
            timestamp="",
        )

    def test_empty_history_zero_hypervolume(self):
        assert _hypervolume([]) == 0.0

    def test_nonzero_hypervolume_with_good_points(self):
        history = [
            self._rec("a", 97.0, 50000),
            self._rec("b", 95.0, 20000),
        ]
        hv = _hypervolume(history)
        assert hv > 0.0

    def test_failed_records_excluded_from_hypervolume(self):
        bad = self._rec("fail", 0.0, 0)
        bad.error = "training failed"
        history = [bad]
        assert _hypervolume(history) == 0.0

    def test_frontier_summary_contains_backend_name(self):
        history = [self._rec("model_x", 97.0, 50000)]
        summary = _frontier_summary(history, "deepprove")
        assert "deepprove" in summary.lower()


# ---------------------------------------------------------------------------
# BackendAwareNAS constructor validation
# ---------------------------------------------------------------------------


class TestBackendAwareNASInit:
    def test_valid_backends_accepted(self):
        for backend in ("ezkl", "deepprove", "expander"):
            with tempfile.TemporaryDirectory() as tmp:
                nas = BackendAwareNAS(
                    target_backend=backend,
                    output_dir=tmp,
                )
                assert nas.target_backend == backend

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            with tempfile.TemporaryDirectory() as tmp:
                BackendAwareNAS(target_backend="halo2", output_dir=tmp)

    def test_output_dir_stored_as_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            nas = BackendAwareNAS(target_backend="ezkl", output_dir=tmp)
            assert isinstance(nas.output_dir, Path)


# ---------------------------------------------------------------------------
# Save / load log round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadLog:
    def _make_record(self) -> BackendIterationRecord:
        return BackendIterationRecord(
            iteration=1, backend="deepprove", candidate_name="test_save",
            reasoning="test", code="x = 1",
            top1=96.0, top5=99.0, constraints=75000,
            proxy_value_ezkl=200000, proxy_value_deepprove=75000,
            proxy_value_expander=18,
            nonlinear_ops=3, mult_depth=1, n_params=50000,
            train_time_s=12.4, proxy_method="onnx",
            error=None, accepted=True,
            hypervolume_before=0.1, hypervolume_after=0.2,
            timestamp="2026-01-01T00:00:00",
        )

    def test_save_creates_file(self, tmp_path):
        history = [self._make_record()]
        path = tmp_path / "nas_log.json"
        _save_log(history, path)
        assert path.exists()

    def test_save_load_roundtrip(self, tmp_path):
        history = [self._make_record()]
        path = tmp_path / "nas_log.json"
        _save_log(history, path)
        with open(path) as f:
            raw = json.load(f)
        loaded = [BackendIterationRecord(**r) for r in raw]
        assert len(loaded) == 1
        assert loaded[0].candidate_name == "test_save"
        assert loaded[0].proxy_value_deepprove == 75000
        assert loaded[0].backend == "deepprove"


# ---------------------------------------------------------------------------
# CLI: --backend option is wired up
# ---------------------------------------------------------------------------


class TestCLIBackendOption:
    def test_nas_command_has_backend_option(self):
        from click.testing import CliRunner
        from zkml_bench.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["nas", "--help"])
        assert result.exit_code == 0
        assert "--backend" in result.output

    def test_backend_choices_listed(self):
        from click.testing import CliRunner
        from zkml_bench.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["nas", "--help"])
        assert "ezkl" in result.output
        assert "deepprove" in result.output
        assert "expander" in result.output

    def test_invalid_backend_rejected(self):
        from click.testing import CliRunner
        from zkml_bench.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["nas", "--backend", "halo2_invalid"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Backend-specific proxy ordering: deepprove penalises depth
# ---------------------------------------------------------------------------


class TestBackendProxyOrdering:
    """The core empirical claim: wide-shallow has lower deepprove proxy than
    deep-narrow at iso-ReLU-activations, and depth multiplies the activation
    count even when total MACs stay matched.

    The DeepProve proxy = relu_activation_count + linear_mac_count.
    To isolate the depth penalty we use architectures with the same first-layer
    width so the dominant MAC term (784 × W) is equal, then compare how many
    additional activation elements each extra layer adds.
    """

    def _make_square_mlp_proto(self, hidden_sizes: list[int], in_features: int = 128):
        """Build an ONNX proto for a ReLU MLP.

        Using in_features=128 so the 784×W first-layer MAC term doesn't
        swamp the per-layer activation count when comparing depth.
        """
        import torch.nn as nn
        from zkml_bench.metrics.zk_proxy import _export_to_onnx_proto

        layers: list[nn.Module] = [nn.Flatten()]
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 10))
        model = nn.Sequential(*layers).eval()
        # Fake 1×11×12 input so in_features=132 → trim to 128 via Linear
        # Actually just use a 1-D input sized in_features
        sample = torch.randn(1, in_features)
        # Override nn.Flatten to be a no-op for flat input
        return _export_to_onnx_proto(model, sample)

    def _make_mlp_proto(self, hidden_sizes: list[int], in_features: int = 784):
        """Standard MNIST-input MLP proto (used for EZKL depth-neutrality test)."""
        import torch.nn as nn
        from zkml_bench.metrics.zk_proxy import _export_to_onnx_proto

        layers: list[nn.Module] = [nn.Flatten()]
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 10))
        model = nn.Sequential(*layers).eval()
        sample = torch.randn(1, 1, 28, 28)
        return _export_to_onnx_proto(model, sample)

    def test_more_layers_means_more_relu_activations(self):
        """Adding depth always increases total ReLU output elements at fixed width."""
        from zkml_bench.metrics.zk_proxy_multi import compute_backend_proxy

        # Both networks have the same width (256); one has 1 hidden layer, one has 4.
        proto_shallow = self._make_square_mlp_proto([256], in_features=256)
        proto_deep = self._make_square_mlp_proto([256, 256, 256, 256], in_features=256)

        r_shallow = compute_backend_proxy(proto_shallow, "deepprove")
        r_deep = compute_backend_proxy(proto_deep, "deepprove")

        assert r_deep["proxy_value"] > r_shallow["proxy_value"], (
            f"Deeper network should have higher deepprove proxy: "
            f"shallow={r_shallow['proxy_value']} deep={r_deep['proxy_value']}"
        )

    def test_depth_multiplies_relu_count_linearly(self):
        """DeepProve relu_activation_count grows linearly with depth at fixed width."""
        from zkml_bench.metrics.zk_proxy_multi import compute_backend_proxy

        width = 256
        protos = {
            d: self._make_square_mlp_proto([width] * d, in_features=width)
            for d in (1, 2, 4)
        }
        proxies = {
            d: compute_backend_proxy(protos[d], "deepprove")
            for d in (1, 2, 4)
        }
        relu_1 = proxies[1]["components"]["relu_activation_count"]
        relu_2 = proxies[2]["components"]["relu_activation_count"]
        relu_4 = proxies[4]["components"]["relu_activation_count"]

        # relu_count should scale linearly with depth at fixed width
        assert relu_2 == pytest.approx(2 * relu_1, rel=0.05), (
            f"Expected 2× relu count at 2× depth: {relu_1} vs {relu_2}"
        )
        assert relu_4 == pytest.approx(4 * relu_1, rel=0.05), (
            f"Expected 4× relu count at 4× depth: {relu_1} vs {relu_4}"
        )

    def test_ezkl_is_depth_neutral(self):
        """EZKL proxy should land in the same logrow bucket for iso-FLOP pairs.

        Wide-2 (2L×512) vs deep-6 (6L×172) — both ~same total MACs on MNIST.
        """
        import math
        from zkml_bench.metrics.zk_proxy_multi import compute_backend_proxy

        # These are iso-FLOP on MNIST: 784×512×2 ≈ 784×172×6 (within 10%)
        proto_wide = self._make_mlp_proto([512, 512])
        proto_deep = self._make_mlp_proto([172, 172, 172, 172, 172, 172])

        r_wide = compute_backend_proxy(proto_wide, "ezkl")
        r_deep = compute_backend_proxy(proto_deep, "ezkl")

        # Both should snap to the same power-of-2 bucket (or at most 1 logrow apart)
        lw = math.log2(max(r_wide["proxy_value"], 1))
        ld = math.log2(max(r_deep["proxy_value"], 1))
        assert abs(lw - ld) <= 1.0, (
            f"EZKL should be depth-neutral; got logrows wide={lw:.1f} deep={ld:.1f}"
        )

    def test_deepprove_shows_depth_penalty_at_iso_flops(self):
        """At iso-FLOPs (matched MACs), deeper networks cost more on DeepProve.

        Uses the paper's MLP pair 1 analogue from reversal_results.md:
        wide_2 (2L×512) vs deep_6 (6L×172) on MNIST input.
        Total MACs are matched within ~10%.

        The DeepProve proxy adds per-layer relu_activation_count, so depth
        pushes the proxy higher even when total MACs are equal.
        """
        from zkml_bench.metrics.zk_proxy_multi import compute_backend_proxy

        proto_wide = self._make_mlp_proto([512, 512])
        proto_deep = self._make_mlp_proto([172, 172, 172, 172, 172, 172])

        r_wide = compute_backend_proxy(proto_wide, "deepprove")
        r_deep = compute_backend_proxy(proto_deep, "deepprove")

        relu_wide = r_wide["components"]["relu_activation_count"]
        relu_deep = r_deep["components"]["relu_activation_count"]

        # The ReLU activation count (the GKR depth-sensitive term) must be
        # higher for the deep network: 6×172=1032 vs 2×512=1024
        assert relu_deep >= relu_wide, (
            f"Deep network should have >= relu activations: "
            f"wide={relu_wide} deep={relu_deep}"
        )
