"""Config loading, validation, and defaults for zkml-bench."""

from __future__ import annotations

import copy
import importlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from .exceptions import ConfigValidationError

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------

DEFAULT_SEED = 42
DEFAULT_NUM_SEEDS = 1
DEFAULT_DATA_DIR = "~/.zkml_bench/data"
DEFAULT_EZKL_CACHE_DIR = "~/.zkml_bench/ezkl_cache"
DEFAULT_EZKL_TIMEOUT_SETUP = 600
DEFAULT_EZKL_TIMEOUT_PROVE = 300
DEFAULT_EZKL_CALIBRATION_SAMPLES = 64

_VALID_DATASETS = {"cifar10", "cifar100", "mnist"}
_VALID_DEVICES = {"auto", "cpu", "cuda", "mps"}
_VALID_OUTPUT_FORMATS = {"json", "csv", "plot", "markdown"}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    name: str
    type: str = "builtin"  # "builtin" | "custom"
    module: Optional[str] = None  # required when type="custom"
    weights_path: Optional[str] = None


@dataclass
class DatasetConfig:
    name: str  # cifar10 | cifar100 | mnist


@dataclass
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    device: str = "auto"
    grad_clip_norm: Optional[float] = None  # None = use default per model type


@dataclass
class ZKConfig:
    proxy_only: bool = True
    ezkl_timeout_setup: int = DEFAULT_EZKL_TIMEOUT_SETUP
    ezkl_timeout_prove: int = DEFAULT_EZKL_TIMEOUT_PROVE
    ezkl_cache_dir: str = DEFAULT_EZKL_CACHE_DIR
    ezkl_calibration_samples: int = DEFAULT_EZKL_CALIBRATION_SAMPLES


@dataclass
class OutputConfig:
    formats: list[str] = field(default_factory=lambda: ["json", "csv", "plot", "markdown"])
    plot_log_scale_x: bool = True


@dataclass
class BenchmarkConfig:
    name: str = "benchmark"
    output_dir: str = "./results"
    seed: int = DEFAULT_SEED
    num_seeds: int = DEFAULT_NUM_SEEDS
    data_dir: str = DEFAULT_DATA_DIR
    datasets: list[DatasetConfig] = field(default_factory=list)
    models: list[ModelConfig] = field(default_factory=list)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    zk: ZKConfig = field(default_factory=ZKConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "BenchmarkConfig":
        """Load a BenchmarkConfig from a YAML file at *path*."""
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Config file not found: {resolved}")
        with resolved.open("r") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BenchmarkConfig":
        """Construct a BenchmarkConfig from a plain dictionary.

        Unknown top-level and nested keys emit a :class:`UserWarning` for
        forward compatibility instead of raising an error.
        """
        d = copy.deepcopy(d)

        # Accept nested "benchmark:" section (as shown in spec YAML examples) or flat keys
        if "benchmark" in d and isinstance(d["benchmark"], dict):
            bench_section = d.pop("benchmark")
            for k, v in bench_section.items():
                d.setdefault(k, v)

        _KNOWN_TOP_LEVEL = {
            "name",
            "output_dir",
            "seed",
            "num_seeds",
            "data_dir",
            "datasets",
            "models",
            "training",
            "zk",
            "output",
        }
        for key in list(d.keys()):
            if key not in _KNOWN_TOP_LEVEL:
                warnings.warn(f"Unknown config key: {key}", UserWarning, stacklevel=2)

        # --- datasets ---
        datasets: list[DatasetConfig] = []
        for item in d.get("datasets", []):
            if isinstance(item, str):
                datasets.append(DatasetConfig(name=item))
            elif isinstance(item, dict):
                _KNOWN_DS = {"name"}
                for k in item:
                    if k not in _KNOWN_DS:
                        warnings.warn(f"Unknown config key: datasets.{k}", UserWarning, stacklevel=2)
                datasets.append(DatasetConfig(name=item["name"]))
            else:
                raise ConfigValidationError(
                    f"Each entry in 'datasets' must be a string or mapping, got {type(item)}"
                )

        # --- models ---
        models: list[ModelConfig] = []
        _KNOWN_MODEL = {"name", "type", "module", "weights_path"}
        for item in d.get("models", []):
            if not isinstance(item, dict):
                raise ConfigValidationError(
                    f"Each entry in 'models' must be a mapping, got {type(item)}"
                )
            for k in item:
                if k not in _KNOWN_MODEL:
                    warnings.warn(
                        f"Unknown config key: models.{k}", UserWarning, stacklevel=2
                    )
            models.append(
                ModelConfig(
                    name=item["name"],
                    type=item.get("type", "builtin"),
                    module=item.get("module"),
                    weights_path=item.get("weights_path"),
                )
            )

        # --- training ---
        training_dict: dict[str, Any] = d.get("training", {})
        _KNOWN_TRAINING = {
            "epochs",
            "batch_size",
            "lr",
            "momentum",
            "weight_decay",
            "device",
            "grad_clip_norm",
        }
        for k in training_dict:
            if k not in _KNOWN_TRAINING:
                warnings.warn(f"Unknown config key: training.{k}", UserWarning, stacklevel=2)
        training = TrainingConfig(
            epochs=training_dict.get("epochs", 200),
            batch_size=training_dict.get("batch_size", 128),
            lr=training_dict.get("lr", 0.1),
            momentum=training_dict.get("momentum", 0.9),
            weight_decay=training_dict.get("weight_decay", 5e-4),
            device=training_dict.get("device", "auto"),
            grad_clip_norm=training_dict.get("grad_clip_norm", None),
        )

        # --- zk ---
        zk_dict: dict[str, Any] = d.get("zk", {})
        _KNOWN_ZK = {
            "proxy_only",
            "ezkl_timeout_setup",
            "ezkl_timeout_prove",
            "ezkl_cache_dir",
            "ezkl_calibration_samples",
        }
        for k in zk_dict:
            if k not in _KNOWN_ZK:
                warnings.warn(f"Unknown config key: zk.{k}", UserWarning, stacklevel=2)
        zk = ZKConfig(
            proxy_only=zk_dict.get("proxy_only", True),
            ezkl_timeout_setup=zk_dict.get("ezkl_timeout_setup", DEFAULT_EZKL_TIMEOUT_SETUP),
            ezkl_timeout_prove=zk_dict.get("ezkl_timeout_prove", DEFAULT_EZKL_TIMEOUT_PROVE),
            ezkl_cache_dir=zk_dict.get("ezkl_cache_dir", DEFAULT_EZKL_CACHE_DIR),
            ezkl_calibration_samples=zk_dict.get(
                "ezkl_calibration_samples", DEFAULT_EZKL_CALIBRATION_SAMPLES
            ),
        )

        # --- output ---
        output_dict: dict[str, Any] = d.get("output", {})
        _KNOWN_OUTPUT = {"formats", "plot_log_scale_x"}
        for k in output_dict:
            if k not in _KNOWN_OUTPUT:
                warnings.warn(f"Unknown config key: output.{k}", UserWarning, stacklevel=2)
        output = OutputConfig(
            formats=output_dict.get("formats", ["json", "csv", "plot", "markdown"]),
            plot_log_scale_x=output_dict.get("plot_log_scale_x", True),
        )

        cfg = cls(
            name=d.get("name", "benchmark"),
            output_dir=d.get("output_dir", "./results"),
            seed=d.get("seed", DEFAULT_SEED),
            num_seeds=d.get("num_seeds", DEFAULT_NUM_SEEDS),
            data_dir=d.get("data_dir", DEFAULT_DATA_DIR),
            datasets=datasets,
            models=models,
            training=training,
            zk=zk,
            output=output,
        )
        cfg._expand_paths()
        cfg.validate()
        return cfg

    # ------------------------------------------------------------------
    # Path expansion
    # ------------------------------------------------------------------

    def _expand_paths(self) -> None:
        """Expand ~ in all path fields in-place."""
        self.data_dir = str(Path(self.data_dir).expanduser())
        self.zk.ezkl_cache_dir = str(Path(self.zk.ezkl_cache_dir).expanduser())
        self.output_dir = str(Path(self.output_dir).expanduser())

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate the configuration, raising :class:`ConfigValidationError` on failure."""
        errors: list[str] = []

        # datasets: non-empty, each name in allowed set
        if not self.datasets:
            errors.append("'datasets' must be a non-empty list.")
        for ds in self.datasets:
            if ds.name not in _VALID_DATASETS:
                errors.append(
                    f"Unknown dataset {ds.name!r}. "
                    f"Valid values: {sorted(_VALID_DATASETS)}"
                )

        # models: non-empty; custom requires module
        if not self.models:
            errors.append("'models' must be a non-empty list.")
        for m in self.models:
            if m.type not in ("builtin", "custom"):
                errors.append(
                    f"Model {m.name!r}: 'type' must be 'builtin' or 'custom', got {m.type!r}."
                )
            if m.type == "custom":
                if not m.module:
                    errors.append(
                        f"Model {m.name!r}: type='custom' requires a 'module' key."
                    )
                else:
                    # Validate that the module is importable right now.
                    try:
                        importlib.import_module(m.module)
                    except ImportError as exc:
                        errors.append(
                            f"Model {m.name!r}: custom module {m.module!r} "
                            f"is not importable: {exc}"
                        )

        # training.epochs positive integer
        if not isinstance(self.training.epochs, int) or self.training.epochs <= 0:
            errors.append(
                f"'training.epochs' must be a positive integer, got {self.training.epochs!r}."
            )

        # training.batch_size positive integer
        if not isinstance(self.training.batch_size, int) or self.training.batch_size <= 0:
            errors.append(
                f"'training.batch_size' must be a positive integer, "
                f"got {self.training.batch_size!r}."
            )

        # training.lr positive
        if not (isinstance(self.training.lr, (int, float)) and self.training.lr > 0):
            errors.append(
                f"'training.lr' must be a positive number, got {self.training.lr!r}."
            )

        # training.device in allowed set
        if self.training.device not in _VALID_DEVICES:
            errors.append(
                f"'training.device' must be one of {sorted(_VALID_DEVICES)}, "
                f"got {self.training.device!r}."
            )

        # zk.ezkl_calibration_samples >= 16; warn if < 64
        cal = self.zk.ezkl_calibration_samples
        if not isinstance(cal, int) or cal < 16:
            errors.append(
                f"'zk.ezkl_calibration_samples' must be an integer >= 16, got {cal!r}."
            )
        elif cal < 64:
            warnings.warn(
                f"'zk.ezkl_calibration_samples' is {cal}, which is below the recommended "
                "minimum of 64. Proof quality may be reduced.",
                UserWarning,
                stacklevel=3,
            )

        # benchmark.num_seeds positive integer
        if not isinstance(self.num_seeds, int) or self.num_seeds <= 0:
            errors.append(
                f"'num_seeds' must be a positive integer, got {self.num_seeds!r}."
            )

        if errors:
            raise ConfigValidationError(
                "Configuration is invalid:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    # ------------------------------------------------------------------
    # Override application
    # ------------------------------------------------------------------

    def apply_overrides(self, overrides: list[str]) -> "BenchmarkConfig":
        """Return a new :class:`BenchmarkConfig` with dot-notation overrides applied.

        Each override must be of the form ``"key.subkey=value"``.  The value is
        parsed as a YAML scalar, so ``"true"`` becomes :data:`True`, ``"10"``
        becomes ``10``, ``"0.1"`` becomes ``0.1``, etc.

        Parameters
        ----------
        overrides:
            List of ``"dotted.path=value"`` strings.

        Returns
        -------
        BenchmarkConfig
            A new config instance with the overrides applied.
        """
        # Deep-copy to a plain dict, apply changes, then reconstruct.
        raw = _config_to_dict(self)

        for override in overrides:
            if "=" not in override:
                raise ConfigValidationError(
                    f"Invalid override {override!r}: must be in 'key.path=value' format."
                )
            key_path, _, raw_value = override.partition("=")
            # Parse value as a YAML scalar
            parsed_value = yaml.safe_load(raw_value)
            _set_nested(raw, key_path.strip().split("."), parsed_value, override)

        return BenchmarkConfig.from_dict(raw)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _config_to_dict(cfg: BenchmarkConfig) -> dict[str, Any]:
    """Convert a BenchmarkConfig back to a plain dict suitable for from_dict()."""
    return {
        "name": cfg.name,
        "output_dir": cfg.output_dir,
        "seed": cfg.seed,
        "num_seeds": cfg.num_seeds,
        "data_dir": cfg.data_dir,
        "datasets": [{"name": ds.name} for ds in cfg.datasets],
        "models": [
            {
                "name": m.name,
                "type": m.type,
                **({"module": m.module} if m.module is not None else {}),
                **({"weights_path": m.weights_path} if m.weights_path is not None else {}),
            }
            for m in cfg.models
        ],
        "training": {
            "epochs": cfg.training.epochs,
            "batch_size": cfg.training.batch_size,
            "lr": cfg.training.lr,
            "momentum": cfg.training.momentum,
            "weight_decay": cfg.training.weight_decay,
            "device": cfg.training.device,
            "grad_clip_norm": cfg.training.grad_clip_norm,
        },
        "zk": {
            "proxy_only": cfg.zk.proxy_only,
            "ezkl_timeout_setup": cfg.zk.ezkl_timeout_setup,
            "ezkl_timeout_prove": cfg.zk.ezkl_timeout_prove,
            "ezkl_cache_dir": cfg.zk.ezkl_cache_dir,
            "ezkl_calibration_samples": cfg.zk.ezkl_calibration_samples,
        },
        "output": {
            "formats": list(cfg.output.formats),
            "plot_log_scale_x": cfg.output.plot_log_scale_x,
        },
    }


def _set_nested(
    d: dict[str, Any],
    keys: list[str],
    value: Any,
    original_override: str,
) -> None:
    """Set *value* at the nested location described by *keys* inside *d*.

    Traverses intermediate dicts, creating them if they do not exist.
    Raises :class:`ConfigValidationError` if an intermediate node is not a dict.
    """
    if not keys:
        raise ConfigValidationError(
            f"Invalid override {original_override!r}: empty key path."
        )

    node = d
    for key in keys[:-1]:
        if key not in node:
            node[key] = {}
        if not isinstance(node[key], dict):
            raise ConfigValidationError(
                f"Invalid override {original_override!r}: "
                f"'{key}' is not a mapping and cannot be traversed."
            )
        node = node[key]

    node[keys[-1]] = value
