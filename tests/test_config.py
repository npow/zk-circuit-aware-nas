"""Tests for BenchmarkConfig loading, validation, and override application."""

import warnings

import pytest

from zkml_bench.config import BenchmarkConfig, DEFAULT_SEED
from zkml_bench.exceptions import ConfigValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MINIMAL_VALID = {
    "datasets": [{"name": "mnist"}],
    "models": [{"name": "mlp_small"}],
}


def minimal_cfg(**overrides):
    d = dict(MINIMAL_VALID)
    d.update(overrides)
    return d


# ---------------------------------------------------------------------------
# Valid configs
# ---------------------------------------------------------------------------


def test_default_config_valid():
    """BenchmarkConfig from minimal valid dict is valid and sets default seed."""
    cfg = BenchmarkConfig.from_dict(MINIMAL_VALID)
    assert cfg.seed == DEFAULT_SEED


def test_default_training_epochs():
    cfg = BenchmarkConfig.from_dict(MINIMAL_VALID)
    assert cfg.training.epochs == 200


def test_default_device():
    cfg = BenchmarkConfig.from_dict(MINIMAL_VALID)
    assert cfg.training.device == "auto"


def test_default_zk_proxy_only():
    cfg = BenchmarkConfig.from_dict(MINIMAL_VALID)
    assert cfg.zk.proxy_only is True


def test_multiple_datasets():
    cfg = BenchmarkConfig.from_dict({
        "datasets": [{"name": "mnist"}, {"name": "cifar10"}],
        "models": [{"name": "mlp_small"}],
    })
    assert len(cfg.datasets) == 2


def test_multiple_models():
    cfg = BenchmarkConfig.from_dict({
        "datasets": [{"name": "mnist"}],
        "models": [{"name": "mlp_small"}, {"name": "resnet20"}],
    })
    assert len(cfg.models) == 2


def test_dataset_as_string():
    """Dataset entries that are plain strings should be accepted."""
    cfg = BenchmarkConfig.from_dict({
        "datasets": ["mnist"],
        "models": [{"name": "mlp_small"}],
    })
    assert cfg.datasets[0].name == "mnist"


def test_name_field_set():
    cfg = BenchmarkConfig.from_dict({
        **MINIMAL_VALID,
        "name": "my_run",
    })
    assert cfg.name == "my_run"


def test_num_seeds_default():
    cfg = BenchmarkConfig.from_dict(MINIMAL_VALID)
    assert cfg.num_seeds == 1


def test_cifar100_valid_dataset():
    cfg = BenchmarkConfig.from_dict({
        "datasets": [{"name": "cifar100"}],
        "models": [{"name": "mlp_small"}],
    })
    assert cfg.datasets[0].name == "cifar100"


# ---------------------------------------------------------------------------
# Unknown keys → UserWarning, not error
# ---------------------------------------------------------------------------


def test_unknown_top_level_key_warns():
    """Unknown top-level config keys produce UserWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BenchmarkConfig.from_dict({
            **MINIMAL_VALID,
            "unknown_section": {"foo": "bar"},
        })
    msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
    assert any("unknown" in m.lower() or "Unknown" in m for m in msgs)


def test_unknown_training_key_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BenchmarkConfig.from_dict({
            **MINIMAL_VALID,
            "training": {"epochs": 10, "nonexistent_key": 42},
        })
    msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
    assert any("nonexistent_key" in m for m in msgs)


def test_unknown_zk_key_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BenchmarkConfig.from_dict({
            **MINIMAL_VALID,
            "zk": {"proxy_only": True, "bogus_zk_key": 99},
        })
    msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
    assert any("bogus_zk_key" in m for m in msgs)


def test_unknown_model_key_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BenchmarkConfig.from_dict({
            "datasets": [{"name": "mnist"}],
            "models": [{"name": "mlp_small", "extra_field": "oops"}],
        })
    msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
    assert any("extra_field" in m for m in msgs)


def test_unknown_dataset_key_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BenchmarkConfig.from_dict({
            "datasets": [{"name": "mnist", "surprise": True}],
            "models": [{"name": "mlp_small"}],
        })
    msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
    assert any("surprise" in m for m in msgs)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_negative_epochs_raises():
    with pytest.raises(ConfigValidationError, match="epoch"):
        BenchmarkConfig.from_dict({
            **MINIMAL_VALID,
            "training": {"epochs": -5},
        })


def test_zero_epochs_raises():
    with pytest.raises(ConfigValidationError, match="epoch"):
        BenchmarkConfig.from_dict({
            **MINIMAL_VALID,
            "training": {"epochs": 0},
        })


def test_invalid_dataset_name_raises():
    with pytest.raises(ConfigValidationError, match="dataset"):
        BenchmarkConfig.from_dict({
            "datasets": [{"name": "imagenet"}],
            "models": [{"name": "mlp_small"}],
        })


def test_empty_models_raises():
    with pytest.raises(ConfigValidationError):
        BenchmarkConfig.from_dict({
            "datasets": [{"name": "mnist"}],
            "models": [],
        })


def test_empty_datasets_raises():
    with pytest.raises(ConfigValidationError):
        BenchmarkConfig.from_dict({
            "datasets": [],
            "models": [{"name": "mlp_small"}],
        })


def test_invalid_device_raises():
    with pytest.raises(ConfigValidationError, match="device"):
        BenchmarkConfig.from_dict({
            **MINIMAL_VALID,
            "training": {"device": "tpu"},
        })


def test_negative_batch_size_raises():
    with pytest.raises(ConfigValidationError, match="batch_size"):
        BenchmarkConfig.from_dict({
            **MINIMAL_VALID,
            "training": {"batch_size": -1},
        })


def test_zero_lr_raises():
    with pytest.raises(ConfigValidationError, match="lr"):
        BenchmarkConfig.from_dict({
            **MINIMAL_VALID,
            "training": {"lr": 0.0},
        })


def test_negative_lr_raises():
    with pytest.raises(ConfigValidationError, match="lr"):
        BenchmarkConfig.from_dict({
            **MINIMAL_VALID,
            "training": {"lr": -0.01},
        })


def test_zero_num_seeds_raises():
    with pytest.raises(ConfigValidationError, match="num_seeds"):
        BenchmarkConfig.from_dict({
            **MINIMAL_VALID,
            "num_seeds": 0,
        })


def test_ezkl_calibration_samples_below_16_raises():
    """calibration_samples < 16 raises ConfigValidationError."""
    with pytest.raises(ConfigValidationError, match="ezkl_calibration_samples"):
        BenchmarkConfig.from_dict({
            **MINIMAL_VALID,
            "zk": {"ezkl_calibration_samples": 8},
        })


def test_ezkl_calibration_samples_warn_below_64():
    """calibration_samples in [16, 63] warns but doesn't error."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = BenchmarkConfig.from_dict({
            **MINIMAL_VALID,
            "zk": {"ezkl_calibration_samples": 16},
        })
    assert cfg.zk.ezkl_calibration_samples == 16
    msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
    assert any("16" in m or "calibration" in m.lower() for m in msgs)


def test_ezkl_calibration_samples_64_no_warn():
    """calibration_samples == 64 (default) should not warn."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BenchmarkConfig.from_dict({
            **MINIMAL_VALID,
            "zk": {"ezkl_calibration_samples": 64},
        })
    calibration_warns = [
        x for x in w
        if issubclass(x.category, UserWarning) and "calibration" in str(x.message).lower()
    ]
    assert len(calibration_warns) == 0


def test_custom_model_missing_module_raises():
    """custom model without 'module' key should raise ConfigValidationError."""
    with pytest.raises(ConfigValidationError, match="module"):
        BenchmarkConfig.from_dict({
            "datasets": [{"name": "mnist"}],
            "models": [{"name": "my_custom", "type": "custom"}],
        })


def test_custom_model_nonimportable_module_raises():
    """custom model with unimportable module path should raise ConfigValidationError."""
    with pytest.raises(ConfigValidationError):
        BenchmarkConfig.from_dict({
            "datasets": [{"name": "mnist"}],
            "models": [{"name": "my_custom", "type": "custom", "module": "nonexistent.module.xyz"}],
        })


def test_invalid_model_type_raises():
    with pytest.raises(ConfigValidationError, match="type"):
        BenchmarkConfig.from_dict({
            "datasets": [{"name": "mnist"}],
            "models": [{"name": "mlp_small", "type": "external"}],
        })


def test_dataset_entry_not_string_or_dict_raises():
    with pytest.raises(ConfigValidationError):
        BenchmarkConfig.from_dict({
            "datasets": [42],
            "models": [{"name": "mlp_small"}],
        })


# ---------------------------------------------------------------------------
# apply_overrides
# ---------------------------------------------------------------------------


def test_apply_overrides_int():
    cfg = BenchmarkConfig.from_dict({
        **MINIMAL_VALID,
        "training": {"epochs": 100},
    })
    cfg2 = cfg.apply_overrides(["training.epochs=10"])
    assert cfg2.training.epochs == 10


def test_apply_overrides_bool_false():
    cfg = BenchmarkConfig.from_dict({
        **MINIMAL_VALID,
        "zk": {"proxy_only": True},
    })
    cfg2 = cfg.apply_overrides(["zk.proxy_only=false"])
    assert cfg2.zk.proxy_only is False


def test_apply_overrides_bool_true():
    cfg = BenchmarkConfig.from_dict({
        **MINIMAL_VALID,
        "zk": {"proxy_only": False},
    })
    cfg2 = cfg.apply_overrides(["zk.proxy_only=true"])
    assert cfg2.zk.proxy_only is True


def test_apply_overrides_float():
    cfg = BenchmarkConfig.from_dict({
        **MINIMAL_VALID,
        "training": {"lr": 0.1},
    })
    cfg2 = cfg.apply_overrides(["training.lr=0.001"])
    assert abs(cfg2.training.lr - 0.001) < 1e-9


def test_apply_overrides_string():
    cfg = BenchmarkConfig.from_dict(MINIMAL_VALID)
    cfg2 = cfg.apply_overrides(["training.device=cpu"])
    assert cfg2.training.device == "cpu"


def test_apply_overrides_returns_new_instance():
    cfg = BenchmarkConfig.from_dict(MINIMAL_VALID)
    cfg2 = cfg.apply_overrides(["training.epochs=5"])
    assert cfg is not cfg2
    assert cfg.training.epochs != cfg2.training.epochs


def test_apply_overrides_invalid_format_raises():
    cfg = BenchmarkConfig.from_dict(MINIMAL_VALID)
    with pytest.raises(ConfigValidationError, match="format"):
        cfg.apply_overrides(["training.epochs"])  # missing '='


def test_apply_multiple_overrides():
    cfg = BenchmarkConfig.from_dict(MINIMAL_VALID)
    cfg2 = cfg.apply_overrides(["training.epochs=5", "training.device=cpu"])
    assert cfg2.training.epochs == 5
    assert cfg2.training.device == "cpu"


def test_apply_overrides_seed():
    cfg = BenchmarkConfig.from_dict(MINIMAL_VALID)
    cfg2 = cfg.apply_overrides(["seed=99"])
    assert cfg2.seed == 99


# ---------------------------------------------------------------------------
# Path expansion
# ---------------------------------------------------------------------------


def test_tilde_expanded_in_data_dir():
    cfg = BenchmarkConfig.from_dict(MINIMAL_VALID)
    assert "~" not in cfg.data_dir


def test_tilde_expanded_in_ezkl_cache_dir():
    cfg = BenchmarkConfig.from_dict(MINIMAL_VALID)
    assert "~" not in cfg.zk.ezkl_cache_dir


# ---------------------------------------------------------------------------
# from_yaml
# ---------------------------------------------------------------------------


def test_from_yaml_not_found_raises(tmp_dir):
    import os
    missing = os.path.join(tmp_dir, "missing.yaml")
    with pytest.raises(FileNotFoundError):
        BenchmarkConfig.from_yaml(missing)


def test_from_yaml_valid(tmp_dir):
    import os
    import yaml as _yaml

    data = {
        "datasets": [{"name": "mnist"}],
        "models": [{"name": "mlp_small"}],
        "training": {"epochs": 3},
    }
    path = os.path.join(tmp_dir, "cfg.yaml")
    with open(path, "w") as f:
        _yaml.dump(data, f)

    cfg = BenchmarkConfig.from_yaml(path)
    assert cfg.training.epochs == 3


# ---------------------------------------------------------------------------
# Valid device values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda", "mps"])
def test_valid_device_values(device):
    cfg = BenchmarkConfig.from_dict({
        **MINIMAL_VALID,
        "training": {"device": device},
    })
    assert cfg.training.device == device


def test_benchmark_section_nesting_supported():
    """Users should be able to nest config under 'benchmark:' section per spec YAML example."""
    cfg = BenchmarkConfig.from_dict({
        "benchmark": {
            "name": "my_bench",
            "seed": 123,
            "output_dir": "/tmp/results",
        },
        "datasets": [{"name": "mnist"}],
        "models": [{"name": "mlp_small"}],
    })
    assert cfg.name == "my_bench"
    assert cfg.seed == 123
    assert cfg.output_dir == "/tmp/results"
