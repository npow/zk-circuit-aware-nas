from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any


def write_csv(results: list[dict], path: str) -> None:
    """Write flat CSV from list of result dicts (one row per model/dataset/seed combo)."""
    if not results:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Flatten nested dicts to dot notation
    flat_results = [_flatten(r) for r in results]

    # Collect all keys in insertion order, deduped
    all_keys = []
    seen = set()
    for row in flat_results:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in flat_results:
            writer.writerow(row)


def _flatten(d: dict, prefix: str = "") -> dict:
    """Recursively flatten nested dict to dot-notation keys."""
    flat = {}
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten(v, full_key))
        elif isinstance(v, list):
            flat[full_key] = json_safe(v)
        else:
            flat[full_key] = v
    return flat


def json_safe(v: Any) -> str:
    try:
        return json.dumps(v)
    except Exception:
        return str(v)
