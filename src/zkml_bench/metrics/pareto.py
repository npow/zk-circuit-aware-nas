"""Pareto frontier analysis and hypervolume computation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional



@dataclass
class ModelPoint:
    name: str
    accuracy_mean: float  # top-1 percentage
    accuracy_std: Optional[float]  # None if single seed
    cost: float  # constraint_estimate or ezkl_total_cells


@dataclass
class ParetoResult:
    frontier: list[ModelPoint]  # Pareto-optimal models
    all_models: list[ModelPoint]
    hypervolume: float
    cost_metric: str


def compute_pareto(
    points: list[ModelPoint],
    cost_metric: str,
    num_seeds: int = 1,
) -> ParetoResult:
    """Compute the Pareto frontier (maximize accuracy, minimize cost).

    For num_seeds == 1: standard dominance.
    For num_seeds > 1: conservative dominance using confidence bounds.
    """
    if not points:
        return ParetoResult(
            frontier=[], all_models=[], hypervolume=0.0, cost_metric=cost_metric
        )

    frontier = []
    for i, m in enumerate(points):
        dominated = False
        for j, n in enumerate(points):
            if i == j:
                continue
            if _dominates(n, m, num_seeds):
                dominated = True
                break
        if not dominated:
            frontier.append(m)

    hv = compute_hypervolume(frontier, points)
    return ParetoResult(
        frontier=frontier,
        all_models=points,
        hypervolume=hv,
        cost_metric=cost_metric,
    )


def _dominates(a: ModelPoint, b: ModelPoint, num_seeds: int) -> bool:
    """Return True if model a dominates model b.

    Dominance means a is at least as good as b in all objectives and strictly
    better in at least one (maximize accuracy, minimize cost).

    For num_seeds > 1, conservative dominance is used:
      - a's lower accuracy bound exceeds b's upper bound AND a.cost <= b.cost, OR
      - a.accuracy_mean >= b.accuracy_mean AND a.cost < b.cost * 0.95
    """
    if num_seeds > 1 and a.accuracy_std is not None and b.accuracy_std is not None:
        # Conservative dominance
        acc_lower_a = a.accuracy_mean - a.accuracy_std
        acc_upper_b = b.accuracy_mean + b.accuracy_std
        acc_bound_dominates = acc_lower_a >= acc_upper_b and a.cost <= b.cost
        cost_buffer_dominates = (
            a.accuracy_mean >= b.accuracy_mean and a.cost < b.cost * 0.95
        )
        return acc_bound_dominates or cost_buffer_dominates
    else:
        # Standard Pareto dominance
        better_acc = a.accuracy_mean >= b.accuracy_mean
        better_cost = a.cost <= b.cost
        strictly_better = (a.accuracy_mean > b.accuracy_mean) or (a.cost < b.cost)
        return better_acc and better_cost and strictly_better


def compute_hypervolume(
    frontier: list[ModelPoint], all_models: list[ModelPoint]
) -> float:
    """Compute the 2D dominated hypervolume indicator.

    Reference point: (accuracy=0, cost=max_cost * 1.1).

    The hypervolume is the area in the (accuracy, cost) space dominated by the
    frontier and bounded by the reference point. Higher is better.

    Uses a sweep-line algorithm: sort frontier by accuracy descending, accumulate
    cost rectangles toward the reference cost.
    """
    if not frontier:
        return 0.0

    max_cost = max(m.cost for m in all_models) if all_models else 1.0
    ref_cost = max_cost * 1.1
    ref_acc = 0.0

    # Sort frontier by accuracy descending; sweep over cost dimension
    pts_by_acc = sorted(frontier, key=lambda m: m.accuracy_mean, reverse=True)

    hv = 0.0
    prev_cost = ref_cost
    for pt in pts_by_acc:
        if pt.cost <= prev_cost:
            # Rectangle: height = (accuracy - ref_acc), width = (prev_cost - pt.cost)
            hv += (pt.accuracy_mean - ref_acc) * (prev_cost - pt.cost)
            prev_cost = pt.cost

    return hv
