from __future__ import annotations
import json
from pathlib import Path
from typing import Any


def write_json(data: dict, path: str) -> None:
    """Write results dict to JSON with nice formatting."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_serializer)


def _json_serializer(obj: Any) -> Any:
    """Handle non-serializable types."""
    if hasattr(obj, "__float__"):
        return float(obj)
    if hasattr(obj, "__int__"):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def generate_summary_md(entries: list[dict], output_path: str) -> None:
    """Write a Markdown model comparison table from results entries.

    Columns: Model | Dataset | top-1 (%) | Proxy Constraints | EZKL Constraints | Notes
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Sort by dataset then by constraint_estimate ascending
    def sort_key(e):
        return (e.get("dataset", ""), e.get("zk_proxy", {}).get("constraint_estimate", 0))

    sorted_entries = sorted(entries, key=sort_key)

    lines = [
        "# zkml-bench Results Summary\n",
        "| Model | Dataset | top-1 (%) | Proxy Constraints | EZKL Constraints | Notes |",
        "|-------|---------|-----------|-------------------|-----------------|-------|",
    ]

    for e in sorted_entries:
        model = e.get("model_name", "?")
        dataset = e.get("dataset", "?")
        top1 = e.get("accuracy_top1_mean")
        top1_str = f"{top1:.2f}" if top1 is not None else "—"

        proxy_d = e.get("zk_proxy") or {}
        proxy = proxy_d.get("constraint_estimate")
        proxy_str = f"{proxy:,}" if proxy else "—"

        ezkl_d = e.get("ezkl") or {}
        ezkl = ezkl_d.get("total_cells") or ezkl_d.get("num_rows")
        ezkl_str = f"{ezkl:,}" if ezkl else "—"

        notes = ""
        if e.get("training", {}).get("training_failed"):
            notes = "⚠ failed"
        elif top1 is None:
            notes = "preliminary"

        lines.append(f"| {model} | {dataset} | {top1_str} | {proxy_str} | {ezkl_str} | {notes} |")

    lines.append("")
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
