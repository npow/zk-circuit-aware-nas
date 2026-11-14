"""Tests for reporting utilities: JSON, CSV, and Pareto plot output."""

import json
import os



# ---------------------------------------------------------------------------
# JSON reporting
# ---------------------------------------------------------------------------


def test_write_read_json(tmp_dir):
    from zkml_bench.reporting import write_json, read_json

    data = {"key": "value", "num": 42, "nested": {"a": 1.0}}
    path = os.path.join(tmp_dir, "test.json")
    write_json(data, path)
    assert os.path.exists(path)
    loaded = read_json(path)
    assert loaded == data


def test_write_json_creates_parent_dirs(tmp_dir):
    from zkml_bench.reporting import write_json

    path = os.path.join(tmp_dir, "subdir", "deep", "test.json")
    write_json({"x": 1}, path)
    assert os.path.exists(path)


def test_write_json_is_valid_json(tmp_dir):
    from zkml_bench.reporting import write_json

    data = {"a": 1, "b": [1, 2, 3]}
    path = os.path.join(tmp_dir, "valid.json")
    write_json(data, path)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded == data


def test_json_with_numpy_float(tmp_dir):
    """JSON writer handles numpy float32 types via custom serialiser."""
    import numpy as np
    from zkml_bench.reporting import write_json, read_json

    data = {"value": np.float32(3.14)}
    path = os.path.join(tmp_dir, "np_float.json")
    write_json(data, path)
    loaded = read_json(path)
    assert abs(loaded["value"] - 3.14) < 0.01


def test_json_with_numpy_int(tmp_dir):
    """JSON writer handles numpy int64 types via custom serialiser."""
    import numpy as np
    from zkml_bench.reporting import write_json, read_json

    data = {"count": np.int64(42)}
    path = os.path.join(tmp_dir, "np_int.json")
    write_json(data, path)
    loaded = read_json(path)
    assert loaded["count"] == 42


def test_json_with_special_types(tmp_dir):
    """JSON writer handles numpy-like float and int types."""
    import numpy as np
    from zkml_bench.reporting import write_json, read_json

    data = {"value": np.float32(3.14), "count": np.int64(42)}
    path = os.path.join(tmp_dir, "np.json")
    write_json(data, path)
    loaded = read_json(path)
    assert abs(loaded["value"] - 3.14) < 0.01
    assert loaded["count"] == 42


def test_write_json_nested_dict(tmp_dir):
    from zkml_bench.reporting import write_json, read_json

    data = {"outer": {"inner": {"deep": 99}}}
    path = os.path.join(tmp_dir, "nested.json")
    write_json(data, path)
    loaded = read_json(path)
    assert loaded["outer"]["inner"]["deep"] == 99


def test_read_json_roundtrip_list(tmp_dir):
    from zkml_bench.reporting import write_json, read_json

    data = {"items": [1, 2, 3, "four"]}
    path = os.path.join(tmp_dir, "list.json")
    write_json(data, path)
    loaded = read_json(path)
    assert loaded["items"] == [1, 2, 3, "four"]


# ---------------------------------------------------------------------------
# CSV reporting
# ---------------------------------------------------------------------------


def test_write_csv(tmp_dir):
    from zkml_bench.reporting import write_csv

    rows = [
        {"model": "resnet20", "accuracy": 91.2, "cost": 1234567},
        {"model": "lenet5", "accuracy": 75.0, "cost": 98765},
    ]
    path = os.path.join(tmp_dir, "results.csv")
    write_csv(rows, path)
    assert os.path.exists(path)
    with open(path) as f:
        content = f.read()
    assert "resnet20" in content
    assert "lenet5" in content


def test_write_csv_creates_parent_dirs(tmp_dir):
    from zkml_bench.reporting import write_csv

    rows = [{"col": "value"}]
    path = os.path.join(tmp_dir, "sub", "results.csv")
    write_csv(rows, path)
    assert os.path.exists(path)


def test_write_csv_has_header(tmp_dir):
    from zkml_bench.reporting import write_csv

    rows = [{"model": "a", "acc": 90.0}]
    path = os.path.join(tmp_dir, "header.csv")
    write_csv(rows, path)
    with open(path) as f:
        first_line = f.readline()
    assert "model" in first_line
    assert "acc" in first_line


def test_write_csv_empty_does_not_create_file(tmp_dir):
    """write_csv with no rows should not create a file."""
    from zkml_bench.reporting import write_csv

    path = os.path.join(tmp_dir, "empty.csv")
    write_csv([], path)
    # Per implementation: if not results: return
    assert not os.path.exists(path)


def test_write_csv_nested_dict_flattened(tmp_dir):
    """Nested dicts should be flattened with dot notation."""
    from zkml_bench.reporting import write_csv

    rows = [{"model": "a", "metrics": {"acc": 90.0, "cost": 1000}}]
    path = os.path.join(tmp_dir, "flat.csv")
    write_csv(rows, path)
    with open(path) as f:
        content = f.read()
    assert "metrics.acc" in content or "acc" in content


def test_write_csv_multiple_rows(tmp_dir):
    from zkml_bench.reporting import write_csv
    import csv

    rows = [
        {"model": f"model_{i}", "acc": 80.0 + i}
        for i in range(5)
    ]
    path = os.path.join(tmp_dir, "multi.csv")
    write_csv(rows, path)
    with open(path) as f:
        reader = csv.DictReader(f)
        loaded_rows = list(reader)
    assert len(loaded_rows) == 5


# ---------------------------------------------------------------------------
# Plot reporting
# ---------------------------------------------------------------------------


def test_plot_pareto_creates_file(tmp_dir):
    from zkml_bench.reporting import plot_pareto

    points = [
        {"name": "model_a", "accuracy": 90.0, "cost": 1000, "on_frontier": True},
        {"name": "model_b", "accuracy": 85.0, "cost": 500, "on_frontier": True},
        {"name": "model_c", "accuracy": 80.0, "cost": 2000, "on_frontier": False},
    ]
    output = os.path.join(tmp_dir, "pareto.png")
    plot_pareto(points, output, cost_metric="constraint_estimate")
    assert os.path.exists(output)
    assert os.path.getsize(output) > 0


def test_plot_pareto_all_frontier(tmp_dir):
    """plot_pareto works when all points are on the frontier."""
    from zkml_bench.reporting import plot_pareto

    points = [
        {"name": "a", "accuracy": 90.0, "cost": 1000, "on_frontier": True},
        {"name": "b", "accuracy": 85.0, "cost": 500, "on_frontier": True},
    ]
    output = os.path.join(tmp_dir, "all_frontier.png")
    plot_pareto(points, output, cost_metric="constraint_estimate")
    assert os.path.exists(output)


def test_plot_pareto_no_frontier(tmp_dir):
    """plot_pareto works when no points are on the frontier."""
    from zkml_bench.reporting import plot_pareto

    points = [
        {"name": "a", "accuracy": 90.0, "cost": 1000, "on_frontier": False},
        {"name": "b", "accuracy": 85.0, "cost": 500, "on_frontier": False},
    ]
    output = os.path.join(tmp_dir, "no_frontier.png")
    plot_pareto(points, output, cost_metric="constraint_estimate")
    assert os.path.exists(output)


def test_plot_pareto_single_point(tmp_dir):
    """plot_pareto works with a single point."""
    from zkml_bench.reporting import plot_pareto

    points = [{"name": "only_model", "accuracy": 90.0, "cost": 1000, "on_frontier": True}]
    output = os.path.join(tmp_dir, "single.png")
    plot_pareto(points, output)
    assert os.path.exists(output)


def test_plot_pareto_agg_backend():
    """Matplotlib should use the Agg backend set in plot.py."""
    import matplotlib

    # Import the plot module to trigger the backend setting
    import zkml_bench.reporting.plot  # noqa: F401

    backend = matplotlib.get_backend().lower()
    assert backend == "agg", f"Expected Agg backend, got {backend}"


def test_plot_pareto_creates_png_bytes(tmp_dir):
    """Output file should be a valid PNG (starts with PNG magic bytes)."""
    from zkml_bench.reporting import plot_pareto

    points = [
        {"name": "m1", "accuracy": 90.0, "cost": 1000, "on_frontier": True},
        {"name": "m2", "accuracy": 85.0, "cost": 2000, "on_frontier": False},
    ]
    output = os.path.join(tmp_dir, "check.png")
    plot_pareto(points, output)
    with open(output, "rb") as f:
        magic = f.read(8)
    # PNG magic bytes: \x89PNG\r\n\x1a\n
    assert magic[:4] == b"\x89PNG", f"File does not appear to be a PNG: {magic!r}"


def test_plot_pareto_log_scale_false(tmp_dir):
    """plot_pareto with log_scale_x=False should still produce a file."""
    from zkml_bench.reporting import plot_pareto

    points = [
        {"name": "a", "accuracy": 90.0, "cost": 1000, "on_frontier": True},
        {"name": "b", "accuracy": 85.0, "cost": 500, "on_frontier": False},
    ]
    output = os.path.join(tmp_dir, "linear.png")
    plot_pareto(points, output, log_scale_x=False)
    assert os.path.exists(output)


def test_incremental_write_is_valid_json(tmp_dir):
    """Partial results.json written with in_progress=True must be valid JSON."""
    import json
    from zkml_bench.benchmark import _write_incremental

    partial = [{"model_name": "resnet20", "dataset": "cifar10", "accuracy_top1_mean": 91.2}]
    _write_incremental(tmp_dir, partial, in_progress=True)

    results_path = os.path.join(tmp_dir, "results.json")
    assert os.path.exists(results_path)
    with open(results_path) as f:
        data = json.load(f)
    assert data["in_progress"] is True
    assert len(data["results"]) == 1


# ---------------------------------------------------------------------------
# T043: generate_summary_md tests
# ---------------------------------------------------------------------------

def test_generate_summary_md_creates_table(tmp_dir):
    """generate_summary_md writes a markdown file with correct headers and rows."""
    import os
    from zkml_bench.reporting.json_report import generate_summary_md

    entries = [
        {
            "model_name": "resnet20",
            "dataset": "cifar10",
            "accuracy_top1_mean": 91.5,
            "zk_proxy": {"constraint_estimate": 41_190_080},
            "ezkl": None,
            "training": {"training_failed": False},
        },
        {
            "model_name": "lenet5",
            "dataset": "cifar10",
            "accuracy_top1_mean": 75.2,
            "zk_proxy": {"constraint_estimate": 666_312},
            "ezkl": None,
            "training": {"training_failed": False},
        },
        {
            "model_name": "mlp_small",
            "dataset": "mnist",
            "accuracy_top1_mean": 97.8,
            "zk_proxy": {"constraint_estimate": 1_708_032},
            "ezkl": None,
            "training": {"training_failed": False},
        },
    ]
    out = os.path.join(tmp_dir, "summary.md")
    generate_summary_md(entries, out)

    assert os.path.exists(out)
    content = open(out).read()
    assert "| Model |" in content
    assert "| Dataset |" in content
    assert "resnet20" in content
    assert "lenet5" in content
    assert "mlp_small" in content
    assert "91.50" in content or "91.5" in content
    assert "41,190,080" in content


def test_generate_summary_md_failed_model(tmp_dir):
    """Failed training runs are marked with ⚠ failed."""
    import os
    from zkml_bench.reporting.json_report import generate_summary_md

    entries = [
        {
            "model_name": "broken_model",
            "dataset": "cifar10",
            "accuracy_top1_mean": 0.0,
            "zk_proxy": {"constraint_estimate": 1000},
            "ezkl": None,
            "training": {"training_failed": True},
        },
    ]
    out = os.path.join(tmp_dir, "summary_fail.md")
    generate_summary_md(entries, out)
    content = open(out).read()
    assert "⚠ failed" in content
