"""Top-level Benchmark orchestrator for zkml-bench."""

from __future__ import annotations

import importlib
import logging
import os
import platform
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import psutil
import torch
import yaml

from .config import BenchmarkConfig
from .exceptions import TrainingDivergenceError, WeightLoadError
from .metrics.zk_proxy import _suppress_onnx_noise

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelResult:
    """All metrics for one (model, dataset) combination across seeds."""

    model_name: str
    dataset: str
    seeds_run: list[int]
    accuracy_top1_mean: float
    accuracy_top1_std: Optional[float]
    accuracy_top5_mean: float
    accuracy_top5_std: Optional[float]
    per_seed: list[dict]  # [{"seed": int, "top1": float, "top5": float}, ...]
    param_count: int
    flops: int
    model_hash_val: Optional[str]  # model hash hex string
    zk_proxy: Any  # ZKProxyResult
    ezkl: Any  # ProverResult
    training: Any  # TrainingResult (last seed's result)
    errors: list[str]


class BenchmarkResults:
    """Container for all results produced by a benchmark run."""

    def __init__(
        self,
        results: list[ModelResult],
        config: BenchmarkConfig,
    ) -> None:
        self.results = results
        self.config = config

    def __iter__(self):
        return iter(self.results)

    def pareto_frontier(
        self, cost_metric: str = "constraint_estimate"
    ) -> list[ModelResult]:
        """Return the Pareto-optimal subset (maximize accuracy, minimize cost)."""
        from .metrics.pareto import compute_pareto

        points = _results_to_model_points(self.results, cost_metric)
        pareto = compute_pareto(points, cost_metric, num_seeds=self.config.num_seeds)
        frontier_names = {p.name for p in pareto.frontier}
        return [r for r in self.results if r.model_name in frontier_names]

    def pareto_hypervolume(
        self, cost_metric: str = "constraint_estimate"
    ) -> float:
        """Return the Pareto hypervolume indicator."""
        from .metrics.pareto import compute_pareto

        points = _results_to_model_points(self.results, cost_metric)
        pareto = compute_pareto(points, cost_metric, num_seeds=self.config.num_seeds)
        return pareto.hypervolume

    def save(self, output_dir: str) -> None:
        """Write all output files into *output_dir*."""
        _write_all_outputs(self, output_dir)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _results_to_model_points(
    results: list[ModelResult], cost_metric: str
) -> list:
    """Convert ModelResult list to ModelPoint list for Pareto analysis."""
    from .metrics.pareto import ModelPoint

    points = []
    for r in results:
        if r.zk_proxy is None:
            continue
        if cost_metric == "constraint_estimate":
            cost = float(r.zk_proxy.constraint_estimate)
        elif cost_metric == "total_cells" and r.ezkl is not None:
            cost = float(r.ezkl.total_cells or r.zk_proxy.constraint_estimate)
        else:
            cost = float(r.zk_proxy.constraint_estimate)
        points.append(
            ModelPoint(
                name=r.model_name,
                accuracy_mean=r.accuracy_top1_mean,
                accuracy_std=r.accuracy_top1_std,
                cost=cost,
            )
        )
    return points


def _model_result_to_dict(r: ModelResult) -> dict:
    """Serialise a ModelResult to a JSON-safe dict."""

    def _proxy_to_dict(p) -> Optional[dict]:
        if p is None:
            return None
        return {
            "constraint_estimate": p.constraint_estimate,
            "constraint_estimate_snapped": p.constraint_estimate_snapped,
            "nonlinear_op_count": p.nonlinear_op_count,
            "linear_op_count": p.linear_op_count,
            "multiplicative_depth": p.multiplicative_depth,
            "relu_depth_incompatible": p.relu_depth_incompatible,
            "nonlinear_breakdown": p.nonlinear_breakdown,
            "param_count": p.param_count,
            "flops": p.flops,
            "proxy_method": p.proxy_method,
        }

    def _prover_to_dict(p) -> Optional[dict]:
        if p is None:
            return None
        return {
            "available": p.available,
            "error": p.error,
            "num_rows": p.num_rows,
            "num_cols": p.num_cols,
            "total_cells": p.total_cells,
            "quantized_accuracy_top1": p.quantized_accuracy_top1,
            "setup_time_s": p.setup_time_s,
            "prove_time_s": p.prove_time_s,
            "verify_time_s": p.verify_time_s,
            "proof_size_bytes": p.proof_size_bytes,
            "peak_memory_mb": p.peak_memory_mb,
            "srs_size_bytes": p.srs_size_bytes,
            "activation_clip": getattr(p, "activation_clip", None),
            "used_scale_rebase": getattr(p, "used_scale_rebase", False),
        }

    def _training_to_dict(t) -> Optional[dict]:
        if t is None:
            return None
        return {
            "epochs_run": t.epochs_run,
            "final_train_loss": t.final_train_loss,
            "train_time_s": t.train_time_s,
            "eval_time_s": t.eval_time_s,
            "training_failed": t.training_failed,
            "failure_reason": t.failure_reason,
        }

    return {
        "model_name": r.model_name,
        "dataset": r.dataset,
        "seeds_run": r.seeds_run,
        "accuracy_top1_mean": r.accuracy_top1_mean,
        "accuracy_top1_std": r.accuracy_top1_std,
        "accuracy_top5_mean": r.accuracy_top5_mean,
        "accuracy_top5_std": r.accuracy_top5_std,
        "per_seed": r.per_seed,
        "param_count": r.param_count,
        "flops": r.flops,
        "model_hash_val": r.model_hash_val,
        "zk_proxy": _proxy_to_dict(r.zk_proxy),
        "ezkl": _prover_to_dict(r.ezkl),
        "training": _training_to_dict(r.training),
        "errors": r.errors,
    }


def _write_all_outputs(bench_results: BenchmarkResults, output_dir: str) -> None:
    """Write results.json, results.csv, pareto.json, pareto_proxy.png, summary.md, config.yaml."""
    from .reporting import write_json, write_csv, plot_pareto
    from .metrics.pareto import compute_pareto

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results = bench_results.results
    config = bench_results.config

    # ---- results.json ----
    result_dicts = [_model_result_to_dict(r) for r in results]
    write_json(
        {"in_progress": False, "results": result_dicts},
        str(out / "results.json"),
    )

    # ---- results.csv ----
    if "csv" in config.output.formats:
        write_csv(result_dicts, str(out / "results.csv"))

    # ---- pareto analysis ----
    cost_metric = "constraint_estimate"
    points = _results_to_model_points(results, cost_metric)
    pareto_result = compute_pareto(points, cost_metric, num_seeds=config.num_seeds)
    frontier_names = {p.name for p in pareto_result.frontier}

    # ---- pareto.json ----
    if "json" in config.output.formats:
        pareto_data = {
            "cost_metric": cost_metric,
            "hypervolume": pareto_result.hypervolume,
            "frontier": [
                {
                    "name": p.name,
                    "accuracy_mean": p.accuracy_mean,
                    "accuracy_std": p.accuracy_std,
                    "cost": p.cost,
                }
                for p in pareto_result.frontier
            ],
            "all_models": [
                {
                    "name": p.name,
                    "accuracy_mean": p.accuracy_mean,
                    "accuracy_std": p.accuracy_std,
                    "cost": p.cost,
                    "on_frontier": p.name in frontier_names,
                }
                for p in pareto_result.all_models
            ],
        }
        write_json(pareto_data, str(out / "pareto.json"))

    # ---- pareto_proxy.png ----
    if "plot" in config.output.formats and points:
        plot_points = [
            {
                "name": p.name,
                "accuracy": p.accuracy_mean,
                "cost": p.cost,
                "on_frontier": p.name in frontier_names,
            }
            for p in pareto_result.all_models
        ]
        plot_pareto(
            all_points=plot_points,
            output_path=str(out / "pareto_proxy.png"),
            cost_metric=cost_metric,
            log_scale_x=config.output.plot_log_scale_x,
            title=f"Benchmark: {config.name}",
        )

    # ---- summary.md ----
    if "markdown" in config.output.formats:
        _write_markdown_summary(results, frontier_names, config.name, str(out / "summary.md"))

    # ---- config.yaml ----
    from .config import _config_to_dict
    config_dict = _config_to_dict(config)
    with (out / "config.yaml").open("w") as fh:
        yaml.safe_dump(config_dict, fh, default_flow_style=False)


def _write_markdown_summary(
    results: list[ModelResult],
    frontier_names: set[str],
    bench_name: str,
    path: str,
) -> None:
    """Write a markdown summary table."""
    lines = [
        f"# Benchmark: {bench_name}",
        "",
        "| Model | Dataset | Top-1 | Top-5 | Constraints | On Frontier |",
        "|-------|---------|-------|-------|-------------|-------------|",
    ]
    for r in results:
        top1 = f"{r.accuracy_top1_mean:.2f}%"
        top5 = f"{r.accuracy_top5_mean:.2f}%"
        constraints = (
            f"{r.zk_proxy.constraint_estimate:,}"
            if r.zk_proxy is not None
            else "N/A"
        )
        on_frontier = "✓" if r.model_name in frontier_names else ""
        lines.append(
            f"| {r.model_name} | {r.dataset} | {top1} | {top5} "
            f"| {constraints} | {on_frontier} |"
        )
    lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))


def _collect_environment() -> dict:
    """Collect runtime environment metadata."""
    try:
        import torchvision as _tv
        _torchvision_version = _tv.__version__
    except ImportError:
        _torchvision_version = None

    env: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "torchvision_version": _torchvision_version,
        "onnx_version": None,
        "onnx_opset_version": 17,
        "ezkl_version": None,
        "ezkl_available": False,
        "cpu_model": platform.processor(),
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "ram_gb": psutil.virtual_memory().total / (1024 ** 3),
        "cuda_version": torch.version.cuda,
        "os": platform.platform(),
    }

    try:
        import onnx
        env["onnx_version"] = onnx.__version__
    except ImportError:
        pass

    try:
        import subprocess
        result = subprocess.run(
            ["ezkl", "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            env["ezkl_version"] = result.stdout.strip()
            env["ezkl_available"] = True
    except Exception:
        pass

    return env


def _instantiate_model(
    model_config,
    num_classes: int,
    input_shape: tuple,
):
    """Instantiate a model from a ModelConfig."""
    from .models import get_model

    if model_config.type == "builtin":
        return get_model(model_config.name, num_classes=num_classes, input_shape=input_shape)
    else:
        # Custom model
        parts = model_config.module.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Custom model module {model_config.module!r} must be 'package.ClassName'"
            )
        mod = importlib.import_module(parts[0])
        cls = getattr(mod, parts[1])
        return cls(num_classes=num_classes, input_shape=input_shape)


def _is_poly_model(model_name: str) -> bool:
    return model_name.endswith("_poly")


# BN-free models that need Adaptive Gradient Clipping (AGC).
# Detection is by name; the structural fallback in agc.should_use_agc()
# covers any future BN-free models not listed here.
_BN_FREE_MODEL_NAMES: frozenset[str] = frozenset(
    {
        "vgg11_small",
        "resnet20_relu_bn_free",
        "lenet5_poly",
        "resnet20_poly",
        # mlp_poly has no BN but already converges with static clip; include
        # it here too for consistency — AGC is strictly safer.
        "mlp_poly",
    }
)


def _is_bn_free_model(model_name: str) -> bool:
    """Return True if *model_name* is a known BN-free architecture."""
    return model_name in _BN_FREE_MODEL_NAMES


def _get_training_hyperparams(
    dataset_name: str,
    model_name: str,
    config,
) -> dict:
    """Return training hyperparameters based on dataset and model type.

    Per-dataset defaults are applied when the config value equals the
    dataclass default (i.e. has not been explicitly overridden by the user).
    The config's training.epochs is always respected if explicitly set.

    BN-free models automatically receive:
    - use_agc=True  (Adaptive Gradient Clipping, NFNet arXiv:2102.06171)
    - agc_clipping=0.01  (NFNet default)
    - A longer warmup (10 epochs instead of 5) for non-poly BN-free models
    - Reduced LR (0.05) for poly BN-free models to prevent early activation explosion
    """
    is_poly = _is_poly_model(model_name)
    is_bn_free = _is_bn_free_model(model_name)

    # Detect whether the user explicitly set epochs (any integer is accepted
    # as-is; we always honour it).
    epochs = config.training.epochs

    if dataset_name == "mnist":
        default_epochs = 20 if epochs == 200 else epochs
        return {
            "epochs": default_epochs,
            "lr": 0.01,
            "weight_decay": 1e-4,
            "lr_schedule": None,
            "lr_milestones": None,
            "warmup_epochs": 0,
            "use_agc": False,
            "agc_clipping": 0.01,
        }
    else:
        # cifar10 / cifar100
        default_epochs = 200 if epochs == 200 else epochs

        if is_poly and is_bn_free:
            # e.g. lenet5_poly, resnet20_poly, mlp_poly
            # Reduced LR (0.05) prevents early activation explosion — x² activations
            # magnify pre-activation values quadratically; lr=0.1 is too aggressive
            # without BN to re-centre activations.  Reference: the "escaping activation
            # problem" (Sisyphus 2021); NFNet uses lr ≈ 0.1/2 for early training.
            # Longer warmup (10 epochs) ramps into the stable regime gradually.
            # AGC handles any per-layer gradient imbalance.
            # Allow the user to override via config if they set a non-default value.
            default_lr = 0.05 if config.training.lr == 0.1 else config.training.lr
            return {
                "epochs": default_epochs,
                "lr": default_lr,
                "weight_decay": config.training.weight_decay,
                "lr_schedule": "multistep",
                "lr_milestones": [100, 150],
                "warmup_epochs": 10,
                "use_agc": True,
                "agc_clipping": 0.01,
            }
        elif is_bn_free:
            # e.g. vgg11_small, resnet20_relu_bn_free
            # Same LR as BN models but with AGC + longer warmup for stability.
            return {
                "epochs": default_epochs,
                "lr": 0.1,
                "weight_decay": 5e-4,
                "lr_schedule": "multistep",
                "lr_milestones": [100, 150],
                "warmup_epochs": 10,
                "use_agc": True,
                "agc_clipping": 0.01,
            }
        elif is_poly:
            # poly model WITH BN (currently no such model in registry, but future-proof)
            return {
                "epochs": default_epochs,
                "lr": config.training.lr,
                "weight_decay": config.training.weight_decay,
                "lr_schedule": "multistep",
                "lr_milestones": [100, 150],
                "warmup_epochs": 5,
                "use_agc": False,
                "agc_clipping": 0.01,
            }
        else:
            # Standard BN model (resnet20, lenet5, mlp_small, mlp_large, …)
            return {
                "epochs": default_epochs,
                "lr": 0.1,
                "weight_decay": 5e-4,
                "lr_schedule": "multistep",
                "lr_milestones": [100, 150],
                "warmup_epochs": 0,
                "use_agc": False,
                "agc_clipping": 0.01,
            }


def _compute_model_hash(model, sample_input: torch.Tensor) -> Optional[str]:
    """Export model to ONNX and return its structural hash, or None on failure."""
    import tempfile
    import os

    from .metrics.zk_proxy import model_hash

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    try:
        with _suppress_onnx_noise():
            torch.onnx.export(
                model,
                sample_input,
                onnx_path,
                opset_version=17,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
            )
        return model_hash(onnx_path)
    except Exception as e:
        logger.warning("Model hash computation failed: %s", e)
        return None
    finally:
        try:
            os.unlink(onnx_path)
        except OSError:
            pass


def _run_ezkl(
    model,
    sample_input: torch.Tensor,
    calibration_data_list: list,
    accuracy_top1: float,
    hash_val: Optional[str],
    config,
) -> Any:
    """Run EZKL proving pipeline. Returns a ProverResult."""
    from .provers.ezkl import EZKLProver
    import tempfile
    import os

    prover = EZKLProver()

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name
    try:
        with _suppress_onnx_noise():
            torch.onnx.export(
                model,
                sample_input,
                onnx_path,
                opset_version=17,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
            )
        input_data = [sample_input.numpy().tolist()]
        result = prover.prove(
            onnx_path=onnx_path,
            input_data=input_data,
            calibration_data=calibration_data_list,
            accuracy_top1=accuracy_top1,
            cache_dir=config.zk.ezkl_cache_dir,
            model_hash_str=hash_val or "unknown",
            timeout_setup=config.zk.ezkl_timeout_setup,
            timeout_prove=config.zk.ezkl_timeout_prove,
        )
    finally:
        try:
            os.unlink(onnx_path)
        except OSError:
            pass

    return result


def _load_incremental_results(results_json_path: str) -> dict[tuple, dict]:
    """Load existing per-seed results from a results.json file for resume.

    Returns a dict keyed by (model_name, dataset, seed) -> seed result dict.
    """
    from .reporting import read_json

    try:
        data = read_json(results_json_path)
    except Exception:
        return {}

    completed: dict[tuple, dict] = {}
    for entry in data.get("results", []):
        model_name = entry.get("model_name")
        dataset = entry.get("dataset")
        for seed_entry in entry.get("per_seed", []):
            seed = seed_entry.get("seed")
            if model_name is not None and dataset is not None and seed is not None:
                completed[(model_name, dataset, seed)] = seed_entry
    return completed


def _write_incremental(
    output_dir: str,
    partial_results: list[dict],
    in_progress: bool,
) -> None:
    """Write a partial results.json with in_progress flag."""
    from .reporting import write_json

    write_json(
        {"in_progress": in_progress, "results": partial_results},
        str(Path(output_dir) / "results.json"),
    )


def plot_from_results(results_path: str) -> None:
    """Load an existing results.json and regenerate plots."""
    from .reporting import read_json, plot_pareto
    from .metrics.pareto import compute_pareto, ModelPoint

    results_path = str(Path(results_path).resolve())
    output_dir = str(Path(results_path).parent)

    data = read_json(results_path)
    results_list = data.get("results", [])

    if not results_list:
        print("No results found in", results_path)
        return

    cost_metric = "constraint_estimate"
    points: list[ModelPoint] = []
    for entry in results_list:
        proxy = entry.get("zk_proxy") or {}
        cost = proxy.get("constraint_estimate", 0)
        accuracy = entry.get("accuracy_top1_mean", 0.0)
        std = entry.get("accuracy_top1_std")
        points.append(
            ModelPoint(
                name=entry.get("model_name", "unknown"),
                accuracy_mean=accuracy,
                accuracy_std=std,
                cost=float(cost),
            )
        )

    pareto_result = compute_pareto(points, cost_metric)
    frontier_names = {p.name for p in pareto_result.frontier}

    plot_points = [
        {
            "name": p.name,
            "accuracy": p.accuracy_mean,
            "cost": p.cost,
            "on_frontier": p.name in frontier_names,
        }
        for p in points
    ]

    output_path = str(Path(output_dir) / "pareto_proxy.png")
    plot_pareto(
        all_points=plot_points,
        output_path=output_path,
        cost_metric=cost_metric,
        log_scale_x=True,
        title="Accuracy vs ZK Proving Cost",
    )
    print(f"Plot saved to {output_path}")
    print(f"Frontier models: {[p.name for p in pareto_result.frontier]}")
    print(f"Hypervolume:     {pareto_result.hypervolume:.4f}")


# ---------------------------------------------------------------------------
# Main Benchmark class
# ---------------------------------------------------------------------------


class Benchmark:
    """Orchestrates model training, ONNX export, ZK proving, and reporting.

    Parameters
    ----------
    config:
        A validated :class:`~zkml_bench.config.BenchmarkConfig` describing
        which models, datasets, and ZK back-ends to run.
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config
        self.env = _collect_environment()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, resume_dir: Optional[str] = None) -> BenchmarkResults:
        """Execute the full benchmark pipeline.

        Parameters
        ----------
        resume_dir:
            Path to a previous run directory to resume from. If provided,
            already-completed (model, dataset, seed) triples are skipped.

        Returns
        -------
        BenchmarkResults
            Aggregated results for all completed runs.
        """
        config = self.config

        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = str(Path(config.output_dir) / f"{config.name}_{timestamp}")
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        logs_dir = Path(run_dir) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Output directory: %s", run_dir)

        # Load previously completed results if resuming
        completed: dict[tuple, dict] = {}
        if resume_dir is not None:
            existing_json = str(Path(resume_dir) / "results.json")
            completed = _load_incremental_results(existing_json)
            logger.info("Resuming with %d completed (model, dataset, seed) triples", len(completed))

        # We collect per-(model, dataset) aggregated data here.
        # Key: (model_name, dataset_name) -> list of per-seed dicts
        per_model_dataset: dict[tuple, list[dict]] = {}
        # Flat list of finished ModelResult dicts (for incremental writes)
        incremental_dicts: list[dict] = []

        # Write initial results.json so interrupted runs have a valid file to resume from
        _write_incremental(run_dir, [], in_progress=True)

        # ------------------------------------------------------------------
        # Main loop: dataset × model × seed
        # ------------------------------------------------------------------
        from .datasets import get_dataset
        from .metrics import compute_proxy
        from .provers.base import ProverResult
        from .training import Trainer, get_device, set_seed

        device = get_device(config.training.device)
        logger.info("Using device: %s", device)

        for dataset_cfg in config.datasets:
            dataset_name = dataset_cfg.name
            logger.info("Loading dataset: %s", dataset_name)
            dataset_obj = get_dataset(dataset_name, config.data_dir)

            sample_input = dataset_obj.sample_input(1)

            for model_cfg in config.models:
                model_name = model_cfg.name
                key = (model_name, dataset_name)
                if key not in per_model_dataset:
                    per_model_dataset[key] = []

                for seed_offset in range(config.num_seeds):
                    seed = config.seed + seed_offset

                    # Resume: skip if already done
                    if (model_name, dataset_name, seed) in completed:
                        logger.info(
                            "Skipping completed: %s/%s/seed=%d", model_name, dataset_name, seed
                        )
                        per_model_dataset[key].append(
                            completed[(model_name, dataset_name, seed)]
                        )
                        continue

                    logger.info(
                        "Running: model=%s  dataset=%s  seed=%d",
                        model_name, dataset_name, seed,
                    )

                    # Mark in-progress
                    _write_incremental(run_dir, incremental_dicts, in_progress=True)

                    # Per-run file logger
                    run_log_path = str(
                        logs_dir / f"{model_name}_{dataset_name}_{seed}.log"
                    )
                    file_handler = logging.FileHandler(run_log_path)
                    file_handler.setFormatter(
                        logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
                    )
                    run_logger = logging.getLogger(
                        f"zkml_bench.run.{model_name}.{dataset_name}.{seed}"
                    )
                    run_logger.addHandler(file_handler)
                    run_logger.setLevel(logging.INFO)

                    errors: list[str] = []

                    try:
                        # Instantiate model
                        set_seed(seed)
                        model = _instantiate_model(
                            model_cfg,
                            num_classes=dataset_obj.num_classes,
                            input_shape=dataset_obj.input_shape,
                        )

                        # Apply poly init if applicable
                        if _is_poly_model(model_name):
                            from .models.zk_friendly import init_poly_weights
                            model.apply(init_poly_weights)

                        # Hyperparameters
                        hparams = _get_training_hyperparams(dataset_name, model_name, config)

                        trainer = Trainer(
                            model=model,
                            model_name=model_name,
                            device=device,
                            epochs=hparams["epochs"],
                            batch_size=config.training.batch_size,
                            lr=hparams["lr"],
                            momentum=config.training.momentum,
                            weight_decay=hparams["weight_decay"],
                            lr_schedule=hparams["lr_schedule"],
                            lr_milestones=hparams["lr_milestones"],
                            warmup_epochs=hparams["warmup_epochs"],
                            grad_clip_norm=config.training.grad_clip_norm,
                            use_agc=hparams.get("use_agc", False),
                            agc_clipping=hparams.get("agc_clipping", 0.01),
                            log_dir=str(logs_dir),
                        )

                        # Train or load weights
                        if model_cfg.weights_path:
                            run_logger.info("Loading weights from %s", model_cfg.weights_path)
                            trainer.load_weights(model_cfg.weights_path)
                            model.to(device)
                            model.eval()
                            # Evaluate accuracy from test loader
                            test_loader = dataset_obj.test_loader(config.training.batch_size)
                            from .metrics.accuracy import compute_accuracy
                            acc = compute_accuracy(model, test_loader, device, top_k=(1, 5))
                            top1_acc = acc.top1
                            top5_acc = acc.top5
                            from .training import TrainingResult
                            training_result = TrainingResult(
                                epochs_run=0,
                                final_train_loss=0.0,
                                train_time_s=0.0,
                                eval_time_s=0.0,
                                training_failed=False,
                                failure_reason=None,
                            )
                        else:
                            train_loader = dataset_obj.train_loader(
                                config.training.batch_size, seed
                            )
                            test_loader = dataset_obj.test_loader(config.training.batch_size)
                            training_result, top1_acc, top5_acc = trainer.train(
                                train_loader, test_loader, seed
                            )
                            if training_result.training_failed:
                                raise TrainingDivergenceError(
                                    f"Training diverged for {model_name}/{dataset_name}/seed={seed}: "
                                    f"{training_result.failure_reason}"
                                )

                        run_logger.info(
                            "Accuracy: top1=%.2f%% top5=%.2f%%", top1_acc, top5_acc
                        )

                        # Save trained model checkpoint for downstream EZKL evaluation
                        # (allows quantized-accuracy measurement on the same trained weights)
                        checkpoint_dir = os.path.join(run_dir, "checkpoints")
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        checkpoint_path = os.path.join(
                            checkpoint_dir, f"{model_name}_{dataset_name}_seed{seed}.pt"
                        )
                        torch.save({
                            "model_state_dict": model.state_dict(),
                            "model_name": model_name,
                            "dataset": dataset_name,
                            "seed": seed,
                            "accuracy_top1": top1_acc,
                            "accuracy_top5": top5_acc,
                            "num_classes": dataset_obj.num_classes,
                            "input_shape": list(dataset_obj.input_shape),
                        }, checkpoint_path)
                        run_logger.info("Saved checkpoint: %s", checkpoint_path)

                        # ZK proxy (always)
                        model.eval()
                        proxy_result = compute_proxy(model, sample_input.to(device))
                        run_logger.info(
                            "Proxy: constraints=%d nonlinear_ops=%d",
                            proxy_result.constraint_estimate,
                            proxy_result.nonlinear_op_count,
                        )

                        # Model hash
                        hash_val = _compute_model_hash(model.cpu(), sample_input)
                        model.to(device)

                        # EZKL (optional)
                        ezkl_result: Any
                        if not config.zk.proxy_only and self.env["ezkl_available"]:
                            run_logger.info("Running EZKL proving pipeline...")
                            cal_loader = dataset_obj.calibration_loader(
                                config.zk.ezkl_calibration_samples, seed
                            )
                            cal_tensors, _ = next(iter(cal_loader))
                            calibration_data_list = cal_tensors.numpy().tolist()
                            ezkl_result = _run_ezkl(
                                model=model.cpu(),
                                sample_input=sample_input,
                                calibration_data_list=calibration_data_list,
                                accuracy_top1=top1_acc,
                                hash_val=hash_val,
                                config=config,
                            )
                            model.to(device)
                        else:
                            from .provers.base import ProverResult
                            ezkl_result = ProverResult.unavailable()

                        seed_record = {
                            "seed": seed,
                            "top1": top1_acc,
                            "top5": top5_acc,
                        }
                        per_model_dataset[key].append(seed_record)

                        # Build and record a partial ModelResult for this seed
                        seed_results_so_far = per_model_dataset[key]
                        top1_values = [s["top1"] for s in seed_results_so_far]
                        top5_values = [s["top5"] for s in seed_results_so_far]
                        partial_result = ModelResult(
                            model_name=model_name,
                            dataset=dataset_name,
                            seeds_run=[s["seed"] for s in seed_results_so_far],
                            accuracy_top1_mean=float(np.mean(top1_values)),
                            accuracy_top1_std=float(np.std(top1_values)) if len(top1_values) > 1 else None,
                            accuracy_top5_mean=float(np.mean(top5_values)),
                            accuracy_top5_std=float(np.std(top5_values)) if len(top5_values) > 1 else None,
                            per_seed=seed_results_so_far,
                            param_count=proxy_result.param_count,
                            flops=proxy_result.flops,
                            model_hash_val=hash_val,
                            zk_proxy=proxy_result,
                            ezkl=ezkl_result,
                            training=training_result,
                            errors=errors,
                        )

                        # Update or append to incremental_dicts
                        existing_keys = {
                            (d["model_name"], d["dataset"]) for d in incremental_dicts
                        }
                        entry_as_dict = _model_result_to_dict(partial_result)
                        if (model_name, dataset_name) in existing_keys:
                            for i, d in enumerate(incremental_dicts):
                                if d["model_name"] == model_name and d["dataset"] == dataset_name:
                                    incremental_dicts[i] = entry_as_dict
                                    break
                        else:
                            incremental_dicts.append(entry_as_dict)

                        # Incremental write
                        _write_incremental(run_dir, incremental_dicts, in_progress=True)
                        run_logger.info("Seed %d complete.", seed)

                    except TrainingDivergenceError as exc:
                        msg = f"Training diverged: {exc}"
                        logger.warning(msg)
                        run_logger.warning(msg)
                        errors.append(msg)

                        seed_record = {"seed": seed, "top1": 0.0, "top5": 0.0}
                        per_model_dataset[key].append(seed_record)

                    except WeightLoadError as exc:
                        msg = f"Weight load error: {exc}"
                        logger.error(msg)
                        run_logger.error(msg)
                        errors.append(msg)

                    except Exception as exc:
                        msg = f"Unexpected error: {type(exc).__name__}: {exc}"
                        logger.exception("Unexpected error for %s/%s/seed=%d", model_name, dataset_name, seed)
                        run_logger.error(msg)
                        errors.append(msg)

                    finally:
                        file_handler.close()
                        run_logger.removeHandler(file_handler)

        # ------------------------------------------------------------------
        # Aggregate across seeds and build final ModelResult objects
        # ------------------------------------------------------------------
        final_results: list[ModelResult] = []
        for dataset_cfg in config.datasets:
            dataset_name = dataset_cfg.name
            dataset_obj = get_dataset(dataset_name, config.data_dir)
            sample_input = dataset_obj.sample_input(1)

            for model_cfg in config.models:
                model_name = model_cfg.name
                key = (model_name, dataset_name)
                seed_records = per_model_dataset.get(key, [])

                if not seed_records:
                    continue

                top1_values = [s["top1"] for s in seed_records]  # include all seeds; diverged seeds have 0.0 accuracy
                top5_values = [s["top5"] for s in seed_records]

                if not top1_values:
                    top1_mean, top1_std = 0.0, None
                    top5_mean, top5_std = 0.0, None
                elif len(top1_values) == 1:
                    top1_mean, top1_std = top1_values[0], None
                    top5_mean, top5_std = top5_values[0] if top5_values else 0.0, None
                else:
                    top1_mean = float(np.mean(top1_values))
                    top1_std = float(np.std(top1_values))
                    top5_mean = float(np.mean(top5_values)) if top5_values else 0.0
                    top5_std = float(np.std(top5_values)) if top5_values else None

                # Find a partial result in incremental_dicts for this combo
                matching_partial = next(
                    (
                        d for d in incremental_dicts
                        if d["model_name"] == model_name and d["dataset"] == dataset_name
                    ),
                    None,
                )

                if matching_partial:
                    # Rebuild a ModelResult from the serialised dict
                    # (proxy, ezkl, training come from the last successful seed)
                    from .metrics.zk_proxy import ZKProxyResult
                    from .provers.base import ProverResult
                    from .training import TrainingResult as TR

                    proxy_d = matching_partial.get("zk_proxy") or {}
                    if proxy_d:
                        from .metrics.zk_proxy import snap_to_power_of_2
                        _ce = proxy_d.get("constraint_estimate", 0)
                        proxy = ZKProxyResult(
                            constraint_estimate=_ce,
                            constraint_estimate_snapped=proxy_d.get(
                                "constraint_estimate_snapped", snap_to_power_of_2(_ce)
                            ),
                            nonlinear_op_count=proxy_d.get("nonlinear_op_count", 0),
                            linear_op_count=proxy_d.get("linear_op_count", 0),
                            multiplicative_depth=proxy_d.get("multiplicative_depth"),
                            relu_depth_incompatible=proxy_d.get("relu_depth_incompatible", False),
                            nonlinear_breakdown=proxy_d.get("nonlinear_breakdown", {}),
                            param_count=proxy_d.get("param_count", 0),
                            flops=proxy_d.get("flops", 0),
                            proxy_method=proxy_d.get("proxy_method", "unknown"),
                        )
                    else:
                        proxy = None

                    ezkl_d = matching_partial.get("ezkl") or {}
                    ezkl = (
                        ProverResult(
                            available=ezkl_d.get("available", False),
                            error=ezkl_d.get("error"),
                            num_rows=ezkl_d.get("num_rows"),
                            num_cols=ezkl_d.get("num_cols"),
                            total_cells=ezkl_d.get("total_cells"),
                            quantized_accuracy_top1=ezkl_d.get("quantized_accuracy_top1"),
                            setup_time_s=ezkl_d.get("setup_time_s"),
                            prove_time_s=ezkl_d.get("prove_time_s"),
                            verify_time_s=ezkl_d.get("verify_time_s"),
                            proof_size_bytes=ezkl_d.get("proof_size_bytes"),
                            peak_memory_mb=ezkl_d.get("peak_memory_mb"),
                            srs_size_bytes=ezkl_d.get("srs_size_bytes"),
                        )
                        if ezkl_d
                        else ProverResult.unavailable()
                    )

                    tr_d = matching_partial.get("training") or {}
                    training = (
                        TR(
                            epochs_run=tr_d.get("epochs_run", 0),
                            final_train_loss=tr_d.get("final_train_loss", 0.0),
                            train_time_s=tr_d.get("train_time_s", 0.0),
                            eval_time_s=tr_d.get("eval_time_s", 0.0),
                            training_failed=tr_d.get("training_failed", False),
                            failure_reason=tr_d.get("failure_reason"),
                        )
                        if tr_d
                        else None
                    )

                    final_result = ModelResult(
                        model_name=model_name,
                        dataset=dataset_name,
                        seeds_run=[s["seed"] for s in seed_records],
                        accuracy_top1_mean=top1_mean,
                        accuracy_top1_std=top1_std,
                        accuracy_top5_mean=top5_mean,
                        accuracy_top5_std=top5_std,
                        per_seed=seed_records,
                        param_count=matching_partial.get("param_count", 0),
                        flops=matching_partial.get("flops", 0),
                        model_hash_val=matching_partial.get("model_hash_val"),
                        zk_proxy=proxy,
                        ezkl=ezkl,
                        training=training,
                        errors=matching_partial.get("errors", []),
                    )
                    final_results.append(final_result)

        bench_results = BenchmarkResults(results=final_results, config=config)

        # Write final outputs
        _write_all_outputs(bench_results, run_dir)
        # Final results.json with in_progress=False
        _write_incremental(
            run_dir,
            [_model_result_to_dict(r) for r in final_results],
            in_progress=False,
        )

        logger.info("Benchmark complete. Results in: %s", run_dir)
        return bench_results

    def dry_run(self) -> None:
        """Validate config, try ONNX export for each model, print summary, exit 0."""
        config = self.config
        print(f"Dry run: {config.name}")
        print(f"  Models:   {len(config.models)}")
        print(f"  Datasets: {len(config.datasets)}")
        print(f"  Seeds:    {config.num_seeds}")
        print()

        from .datasets import get_dataset

        errors: list[str] = []
        total_combos = len(config.models) * len(config.datasets) * config.num_seeds

        for dataset_cfg in config.datasets:
            try:
                dataset_obj = get_dataset(dataset_cfg.name, config.data_dir)
                sample_input = dataset_obj.sample_input(1)
            except Exception as exc:
                errors.append(f"Dataset {dataset_cfg.name}: {exc}")
                continue

            for model_cfg in config.models:
                print(f"  Checking model: {model_cfg.name} / dataset: {dataset_cfg.name}")
                try:
                    model = _instantiate_model(
                        model_cfg,
                        num_classes=dataset_obj.num_classes,
                        input_shape=dataset_obj.input_shape,
                    )
                    model.eval()

                    # Try ONNX export
                    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
                        onnx_path = f.name
                    try:
                        with _suppress_onnx_noise():
                            torch.onnx.export(
                                model,
                                sample_input,
                                onnx_path,
                                opset_version=17,
                                do_constant_folding=True,
                                input_names=["input"],
                                output_names=["output"],
                            )
                        print("    ONNX export: OK")
                    finally:
                        try:
                            os.unlink(onnx_path)
                        except OSError:
                            pass

                    # Rough time estimate
                    from .metrics.zk_proxy import _count_params
                    n_params = _count_params(model)
                    hparams = _get_training_hyperparams(
                        dataset_cfg.name, model_cfg.name, config
                    )
                    epochs = hparams["epochs"]
                    est_s = config.num_seeds * epochs * (n_params / 1000) * 0.1
                    print(
                        f"    Params: {n_params:,}  Estimated time: {est_s:.0f}s "
                        f"({est_s / 60:.1f} min)"
                    )

                except Exception as exc:
                    msg = f"Model {model_cfg.name}: {exc}"
                    errors.append(msg)
                    print(f"    ERROR: {exc}")

        total_time_heuristic = 0
        for dataset_cfg in config.datasets:
            try:
                dataset_obj = get_dataset(dataset_cfg.name, config.data_dir)
            except Exception:
                continue
            for model_cfg in config.models:
                try:
                    model = _instantiate_model(
                        model_cfg,
                        num_classes=dataset_obj.num_classes,
                        input_shape=dataset_obj.input_shape,
                    )
                    from .metrics.zk_proxy import _count_params
                    n_params = _count_params(model)
                    hparams = _get_training_hyperparams(
                        dataset_cfg.name, model_cfg.name, config
                    )
                    epochs = hparams["epochs"]
                    total_time_heuristic += config.num_seeds * epochs * (n_params / 1000) * 0.1
                except Exception:
                    pass

        print()
        print(f"Total combinations: {total_combos}")
        print(
            f"Estimated total time: {total_time_heuristic:.0f}s "
            f"({total_time_heuristic / 3600:.2f}h) [rough heuristic]"
        )
        if errors:
            print()
            print(f"Errors ({len(errors)}):")
            for e in errors:
                print(f"  - {e}")
        else:
            print()
            print("All checks passed.")

    def calibrate(self) -> None:
        """Calibrate proxy metrics against EZKL (requires EZKL)."""
        if not self.env["ezkl_available"]:
            print("EZKL is not available. Install ezkl and ensure it is on PATH.")
            return

        config = self.config
        from .datasets import get_dataset

        print(f"Calibrating proxy for benchmark: {config.name}")

        for dataset_cfg in config.datasets:
            dataset_obj = get_dataset(dataset_cfg.name, config.data_dir)
            sample_input = dataset_obj.sample_input(1)

            for model_cfg in config.models:
                print(f"  Model: {model_cfg.name}  Dataset: {dataset_cfg.name}")
                try:
                    model = _instantiate_model(
                        model_cfg,
                        num_classes=dataset_obj.num_classes,
                        input_shape=dataset_obj.input_shape,
                    )
                    model.eval()

                    from .metrics import compute_proxy
                    proxy = compute_proxy(model, sample_input)
                    print(f"    Proxy constraint estimate: {proxy.constraint_estimate:,}")

                    cal_loader = dataset_obj.calibration_loader(
                        config.zk.ezkl_calibration_samples, config.seed
                    )
                    cal_tensors, _ = next(iter(cal_loader))
                    calibration_data_list = cal_tensors.numpy().tolist()

                    hash_val = _compute_model_hash(model, sample_input)
                    ezkl_result = _run_ezkl(
                        model=model,
                        sample_input=sample_input,
                        calibration_data_list=calibration_data_list,
                        accuracy_top1=0.0,
                        hash_val=hash_val,
                        config=config,
                    )

                    if ezkl_result.available and ezkl_result.error is None:
                        print(f"    EZKL total_cells: {ezkl_result.total_cells:,}")
                        ratio = (
                            ezkl_result.total_cells / proxy.constraint_estimate
                            if proxy.constraint_estimate > 0
                            else float("nan")
                        )
                        print(f"    Calibration ratio (ezkl/proxy): {ratio:.3f}")
                    else:
                        print(f"    EZKL failed: {ezkl_result.error}")

                except Exception as exc:
                    print(f"    ERROR: {exc}")
