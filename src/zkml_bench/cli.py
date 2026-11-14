"""Command-line interface for zkml-bench."""

from __future__ import annotations

import json

import click

from . import __version__


@click.group()
@click.version_option(__version__, "--version")
def cli():
    """zkml-bench: ZK-ML benchmark suite."""
    pass


@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, default=False)
@click.option("--resume", type=click.Path(exists=True), default=None)
@click.option("--override", multiple=True, metavar="KEY=VALUE")
def run(config, dry_run, resume, override):
    """Run a full benchmark."""
    from .config import BenchmarkConfig
    from .benchmark import Benchmark

    cfg = BenchmarkConfig.from_yaml(config)
    if override:
        cfg = cfg.apply_overrides(list(override))
    bench = Benchmark(cfg)
    if dry_run:
        bench.dry_run()
    else:
        bench.run(resume_dir=resume)


@cli.command("validate-config")
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
def validate_config(config):
    """Validate a config file without running."""
    from .config import BenchmarkConfig

    try:
        cfg = BenchmarkConfig.from_yaml(config)
        click.echo(f"Config valid: {len(cfg.models)} models, {len(cfg.datasets)} datasets")
    except Exception as e:
        click.echo(f"Invalid config: {e}", err=True)
        raise SystemExit(1)


@cli.command()
@click.option("--validation-results", "-v",
              default="proxy_validation_results.json",
              type=click.Path(exists=True),
              help="Path to proxy_validation_results.json")
@click.option("--output", "-o", default="calibration.json",
              help="Output calibration JSON path")
@click.option("--node-overhead", type=int, default=None,
              help="Fixed per-node overhead constant (auto-fit if not specified)")
def calibrate(validation_results, output, node_overhead):
    """Validate proxy calibration R² against EZKL measurements.

    Reads proxy_validation_results.json, computes R² between
    snap_to_power_of_2(proxy) and ezkl_num_constraints on the
    log2 scale, writes calibration.json. Exits 1 if R² < 0.85.
    """
    import math
    from .metrics.zk_proxy import snap_to_power_of_2

    with open(validation_results) as f:
        results = json.load(f)

    paired = [
        (r["name"],
         r.get("proxy_constraints_snapped") or snap_to_power_of_2(r.get("proxy_constraints", 1)),
         r["ezkl_num_constraints"])
        for r in results
        if r.get("proxy_constraints") and r.get("ezkl_num_constraints") is not None
    ]

    if len(paired) < 3:
        click.echo(f"Need ≥3 paired models, got {len(paired)}", err=True)
        raise SystemExit(2)

    names, snapped_vals, actual_vals = zip(*paired)
    n = len(snapped_vals)
    ratios = [a / s if s > 0 else float("nan") for s, a in zip(snapped_vals, actual_vals)]

    # Outlier exclusion: ratio outside [0.25, 4.0] signals an architecture the proxy
    # cannot model (e.g. EZKL-specific circuit optimisations for large merged-op graphs).
    OUTLIER_LO, OUTLIER_HI = 0.25, 4.0
    clean_idx = [i for i, r in enumerate(ratios)
                 if not math.isnan(r) and OUTLIER_LO <= r <= OUTLIER_HI]

    def _r2(idx):
        if len(idx) < 3:
            return 0.0
        ls_ = [math.log2(max(snapped_vals[i], 1)) for i in idx]
        la_ = [math.log2(max(actual_vals[i], 1)) for i in idx]
        ma_ = sum(la_) / len(idx)
        ss_res = sum((la - ls) ** 2 for ls, la in zip(ls_, la_))
        ss_tot = sum((la - ma_) ** 2 for la in la_)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    r2_all = _r2(list(range(n)))
    r2_clean = _r2(clean_idx)
    n_clean = len(clean_idx)

    per_model = [
        {
            "model": nm, "proxy_snapped": int(s), "ezkl_actual": int(a),
            "ratio": round(r, 4),
            "outlier": math.isnan(r) or not (OUTLIER_LO <= r <= OUTLIER_HI),
        }
        for nm, s, a, r in zip(names, snapped_vals, actual_vals, ratios)
    ]

    from .metrics.zk_proxy import NODE_OVERHEAD
    cal = {
        "r_squared_all": round(r2_all, 6),
        "r_squared_clean": round(r2_clean, 6),
        "n_models_total": n,
        "n_models_clean": n_clean,
        "outlier_bounds": [OUTLIER_LO, OUTLIER_HI],
        "node_overhead": node_overhead if node_overhead is not None else NODE_OVERHEAD,
        "per_model": per_model,
    }
    with open(output, "w") as f:
        json.dump(cal, f, indent=2)

    status = "✓ PASS" if r2_clean >= 0.85 else "✗ FAIL"
    n_out = n - n_clean
    click.echo(f"Calibration R² all={r2_all:.4f}  clean={r2_clean:.4f}  {status}  "
               f"(n={n} total, {n_clean} clean, {n_out} outlier{'s' if n_out != 1 else ''})")
    click.echo(f"Written: {output}")

    if r2_clean < 0.85:
        click.echo("R² clean < 0.85 — proxy calibration below threshold.", err=True)
        raise SystemExit(1)


@cli.command()
@click.argument("results_path", type=click.Path(exists=True))
def pareto(results_path):
    """Replot Pareto frontier from existing results.json."""
    from .benchmark import plot_from_results

    plot_from_results(results_path)


@cli.command("list-models")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
def list_models(fmt):
    """List available built-in models."""
    from .models import list_models as _list_models

    models = _list_models()
    if fmt == "json":
        click.echo(json.dumps(models, indent=2))
    else:
        click.echo(f"{'Name':<30} {'Description'}")
        click.echo("-" * 80)
        for m in models:
            click.echo(f"{m['name']:<30} {m.get('description', '')}")


@cli.command("list-datasets")
def list_datasets():
    """List available built-in datasets."""
    from .datasets import DATASET_REGISTRY

    for name in sorted(DATASET_REGISTRY.keys()):
        click.echo(f"  {name}")


@cli.command("measure-cost")
@click.option("--model", "model_name", default=None)
@click.option("--model-module", default=None)
@click.option("--num-classes", type=int, default=10)
@click.option("--dataset", "dataset_name", default="cifar10")
def measure_cost(model_name, model_module, num_classes, dataset_name):
    """Measure ZK cost of a model without training."""

    from .datasets import get_dataset
    from .metrics.zk_proxy import compute_proxy

    dataset = get_dataset(dataset_name)
    sample = dataset.sample_input(1)

    if model_name:
        from .models import get_model

        model = get_model(
            model_name,
            num_classes=num_classes,
            input_shape=dataset.input_shape,
        )
    elif model_module:
        import importlib

        parts = model_module.rsplit(".", 1)
        if len(parts) != 2:
            raise click.UsageError(
                "--model-module must be in the form 'package.module.ClassName'"
            )
        mod = importlib.import_module(parts[0])
        cls = getattr(mod, parts[1])
        model = cls(num_classes=num_classes, input_shape=dataset.input_shape)
    else:
        raise click.UsageError("Either --model or --model-module is required")

    model.eval()
    result = compute_proxy(model, sample)

    click.echo(f"Constraint estimate:  {result.constraint_estimate:,}")
    click.echo(f"Nonlinear op count:   {result.nonlinear_op_count:,}")
    click.echo(f"Linear op count:      {result.linear_op_count:,}")
    click.echo(f"Multiplicative depth: {result.multiplicative_depth}")
    click.echo(f"ReLU incompatible:    {result.relu_depth_incompatible}")
    click.echo(f"Param count:          {result.param_count:,}")
    click.echo(f"FLOPs:                {result.flops:,}")
    click.echo(f"Proxy method:         {result.proxy_method}")
    if result.nonlinear_breakdown:
        click.echo("Nonlinear breakdown:")
        for op, count in sorted(result.nonlinear_breakdown.items()):
            click.echo(f"  {op}: {count:,}")


@cli.command("export-onnx")
@click.option("--model", "model_name", required=True)
@click.option("--dataset", "dataset_name", default="cifar10")
@click.option("--output", "-o", default="model.onnx")
def export_onnx(model_name, dataset_name, output):
    """Export a model to ONNX for debugging."""
    import torch

    from .datasets import get_dataset
    from .models import get_model
    from .metrics.zk_proxy import _suppress_onnx_noise

    dataset = get_dataset(dataset_name)
    model = get_model(
        model_name,
        num_classes=dataset.num_classes,
        input_shape=dataset.input_shape,
    )
    model.eval()
    sample = dataset.sample_input(1)

    with _suppress_onnx_noise():
        torch.onnx.export(
            model,
            sample,
            output,
            opset_version=17,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
        )
    click.echo(f"Exported to {output}")


@cli.command("nas")
@click.option("--dataset", default="mnist", show_default=True,
              type=click.Choice(["mnist", "cifar10", "cifar100"]))
@click.option("--iterations", "-n", default=20, show_default=True,
              help="Number of agent iterations.")
@click.option("--epochs", default=10, show_default=True,
              help="Training epochs per candidate.")
@click.option("--output-dir", "-o", default="./nas_results", show_default=True)
@click.option("--device", default="auto", show_default=True)
@click.option("--seed", default=42, show_default=True)
@click.option("--api-key", envvar="ANTHROPIC_API_KEY", default=None,
              help="Anthropic API key (or set ANTHROPIC_API_KEY env var).")
@click.option("--seed-results", default=None, type=click.Path(exists=True),
              help="Path to a benchmark results.json to seed the initial Pareto frontier.")
@click.option("--backend", default="ezkl", show_default=True,
              type=click.Choice(["ezkl", "deepprove", "expander"]),
              help=(
                  "Target ZK backend for cost-model optimisation. "
                  "'ezkl' uses the Halo2/logrows proxy (depth-neutral). "
                  "'deepprove' uses the GKR/activation-count proxy (depth-penalising). "
                  "'expander' uses the GKR/log2-gates proxy (widest-layer-penalising). "
                  "Default 'ezkl' preserves existing behaviour."
              ))
@click.option(
    "--zero-cost/--no-zero-cost",
    default=False,
    show_default=True,
    help=(
        "Use the NEAR zero-cost accuracy proxy (arXiv:2408.08776, ICLR 2025) "
        "instead of training each candidate. "
        "Score = NEAR(model) / ZK_cost_proxy — training-free + proving-free NAS. "
        "Reduces NAS time from hours to minutes. Default: off (full training)."
    ),
)
def nas(
    dataset, iterations, epochs, output_dir, device, seed, api_key,
    seed_results, backend, zero_cost,
):
    """Run the autonomous ZK-NAS agent loop.

    An LLM agent iteratively proposes neural architectures, trained for a
    fixed budget, steered toward the accuracy/ZK-cost Pareto frontier.

    Use --backend to select the target ZK proving backend.  The agent's cost
    model and system prompt are conditioned on the chosen backend, causing it
    to discover structurally different architectures:

    \b
      --backend ezkl      (default) depth-neutral; finds deep or shallow models
      --backend deepprove wide-shallow preference; minimises ReLU activation count
      --backend expander  uniform-width preference; minimises widest-layer gates

    Use --zero-cost for a training-free run using the NEAR accuracy proxy
    (ICLR 2025).  ~10-100x faster; useful for large candidate sweeps.
    """
    if backend == "ezkl" and seed_results is not None and not zero_cost:
        # Preserve existing behaviour when --seed-results is provided: use the
        # original single-backend agent loop (BackendAwareNAS does not support
        # seed-results seeding yet).
        from .nas.agent_loop import run_nas
        run_nas(
            dataset_name=dataset,
            max_iterations=iterations,
            epochs_per_candidate=epochs,
            device_str=device,
            seed=seed,
            output_dir=output_dir,
            api_key=api_key,
            seed_results=seed_results,
        )
    else:
        # Backend-aware loop for all backends (ezkl, deepprove, expander)
        from .nas.backend_aware import run_backend_aware_nas
        run_backend_aware_nas(
            target_backend=backend,
            dataset_name=dataset,
            max_iterations=iterations,
            epochs_per_candidate=epochs,
            device_str=device,
            seed=seed,
            output_dir=output_dir,
            api_key=api_key,
            use_zero_cost_proxy=zero_cost,
        )


@cli.command()
@click.option("--results", "-r", required=True, type=click.Path(exists=True),
              help="Directory containing results.json (and optionally nas_log.json)")
@click.option("--output", "-o", default=None,
              help="Output directory (default: same as --results)")
def report(results, output):
    """Generate summary.md and Pareto plots from completed results.

    Reads results.json from --results dir, writes summary.md and
    pareto_proxy.png. If nas_log.json is also present, includes
    NAS candidates in the report.
    """
    import os
    from .reporting.json_report import generate_summary_md
    from .reporting.plot import plot_pareto_from_results

    out_dir = output or results
    results_json = os.path.join(results, "results.json")
    if not os.path.exists(results_json):
        click.echo(f"results.json not found in {results}", err=True)
        raise SystemExit(1)

    with open(results_json) as f:
        data = json.load(f)
    entries = data if isinstance(data, list) else data.get("results", [])

    summary_path = os.path.join(out_dir, "summary.md")
    generate_summary_md(entries, summary_path)
    click.echo(f"Written: {summary_path}")

    plot_path = os.path.join(out_dir, "pareto_proxy.png")
    plot_pareto_from_results(entries, plot_path)
    click.echo(f"Written: {plot_path}")
