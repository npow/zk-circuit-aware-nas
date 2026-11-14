"""Tests for Pareto frontier analysis and hypervolume computation."""


from zkml_bench.metrics.pareto import compute_pareto, ModelPoint, compute_hypervolume


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_point(name, acc, cost, std=None):
    return ModelPoint(name=name, accuracy_mean=acc, accuracy_std=std, cost=cost)


# ---------------------------------------------------------------------------
# Single / trivial cases
# ---------------------------------------------------------------------------


def test_single_model_on_frontier():
    pts = [make_point("a", 90.0, 1000)]
    result = compute_pareto(pts, cost_metric="constraint_estimate")
    assert len(result.frontier) == 1
    assert result.frontier[0].name == "a"


def test_single_model_result_fields():
    pts = [make_point("a", 90.0, 1000)]
    result = compute_pareto(pts, cost_metric="x")
    assert result.cost_metric == "x"
    assert result.all_models == pts


def test_empty_input():
    result = compute_pareto([], cost_metric="x")
    assert result.frontier == []
    assert result.hypervolume == 0.0
    assert result.all_models == []


def test_two_identical_models_both_on_frontier():
    """Models with identical accuracy AND cost: neither dominates, both on frontier."""
    pts = [make_point("a", 90.0, 1000), make_point("b", 90.0, 1000)]
    result = compute_pareto(pts, cost_metric="x")
    assert len(result.frontier) == 2


def test_all_identical_on_frontier():
    """When all models have identical accuracy and cost, all are on the frontier."""
    pts = [make_point(f"m{i}", 90.0, 1000) for i in range(5)]
    result = compute_pareto(pts, cost_metric="x")
    assert len(result.frontier) == 5


# ---------------------------------------------------------------------------
# Dominance correctness
# ---------------------------------------------------------------------------


def test_dominated_model_not_on_frontier():
    """A clearly dominated model (worse accuracy AND higher cost) must not appear."""
    pts = [
        make_point("good", 90.0, 1000),
        make_point("bad", 80.0, 2000),  # worse accuracy AND higher cost
    ]
    result = compute_pareto(pts, cost_metric="x")
    frontier_names = {m.name for m in result.frontier}
    assert "bad" not in frontier_names
    assert "good" in frontier_names


def test_pareto_frontier_subset_of_input():
    pts = [
        make_point("a", 90.0, 1000),
        make_point("b", 85.0, 2000),  # dominated by a
        make_point("c", 92.0, 500),   # dominates a
    ]
    result = compute_pareto(pts, cost_metric="x")
    frontier_names = {m.name for m in result.frontier}
    # Frontier must be a subset of inputs
    all_names = {m.name for m in pts}
    assert frontier_names.issubset(all_names)
    # c dominates a (higher acc, lower cost): b is dominated by a
    assert "b" not in frontier_names
    assert "c" in frontier_names


def test_no_frontier_model_dominated_by_another():
    """No model on the frontier should be dominated by another frontier model."""
    pts = [
        make_point("a", 90.0, 1000),
        make_point("b", 85.0, 500),
        make_point("c", 95.0, 2000),
        make_point("d", 88.0, 1500),
    ]
    result = compute_pareto(pts, cost_metric="x")
    frontier = result.frontier
    for i, m in enumerate(frontier):
        for j, n in enumerate(frontier):
            if i == j:
                continue
            better_acc = n.accuracy_mean >= m.accuracy_mean
            better_cost = n.cost <= m.cost
            strictly_better = (n.accuracy_mean > m.accuracy_mean) or (n.cost < m.cost)
            assert not (better_acc and better_cost and strictly_better), (
                f"Frontier model {n.name} dominates {m.name}"
            )


def test_higher_acc_same_cost_dominates():
    pts = [
        make_point("better", 90.0, 1000),
        make_point("worse", 80.0, 1000),
    ]
    result = compute_pareto(pts, cost_metric="x")
    frontier_names = {m.name for m in result.frontier}
    assert "better" in frontier_names
    assert "worse" not in frontier_names


def test_same_acc_lower_cost_dominates():
    pts = [
        make_point("cheaper", 90.0, 500),
        make_point("costlier", 90.0, 2000),
    ]
    result = compute_pareto(pts, cost_metric="x")
    frontier_names = {m.name for m in result.frontier}
    assert "cheaper" in frontier_names
    assert "costlier" not in frontier_names


def test_tradeoff_both_on_frontier():
    """When model A has higher accuracy at higher cost, and B has lower accuracy at lower cost,
    both are on the Pareto frontier."""
    pts = [
        make_point("accurate", 95.0, 5000),
        make_point("cheap", 85.0, 500),
    ]
    result = compute_pareto(pts, cost_metric="x")
    frontier_names = {m.name for m in result.frontier}
    assert "accurate" in frontier_names
    assert "cheap" in frontier_names


def test_chain_dominance():
    """a dominates b, b dominates c → only a on frontier."""
    pts = [
        make_point("a", 95.0, 500),   # best of all
        make_point("b", 90.0, 1000),  # dominated by a
        make_point("c", 80.0, 2000),  # dominated by b (and a)
    ]
    result = compute_pareto(pts, cost_metric="x")
    frontier_names = {m.name for m in result.frontier}
    assert frontier_names == {"a"}


def test_multiple_frontier_models():
    """Standard tradeoff scenario: all 3 models on Pareto front."""
    pts = [
        make_point("a", 95.0, 3000),
        make_point("b", 90.0, 1500),
        make_point("c", 85.0, 500),
    ]
    result = compute_pareto(pts, cost_metric="x")
    assert len(result.frontier) == 3


# ---------------------------------------------------------------------------
# Hypervolume
# ---------------------------------------------------------------------------


def test_hypervolume_positive():
    pts = [make_point("a", 90.0, 1000), make_point("b", 85.0, 500)]
    result = compute_pareto(pts, cost_metric="x")
    assert result.hypervolume > 0


def test_hypervolume_single_point_positive():
    pts = [make_point("a", 80.0, 2000)]
    result = compute_pareto(pts, cost_metric="x")
    assert result.hypervolume > 0


def test_hypervolume_empty():
    assert compute_hypervolume([], []) == 0.0


def test_hypervolume_better_frontier_is_larger():
    """A set with higher accuracy at the same cost scale has a larger hypervolume.

    The 2D hypervolume indicator uses reference point (acc=0, cost=max_cost*1.1).
    Higher accuracy (taller rectangles) increases hypervolume for same cost scale.
    """
    # Same cost scale, better accuracy → larger rectangles in the sweep
    pts_good = [make_point("a", 90.0, 500), make_point("b", 85.0, 200)]
    pts_bad = [make_point("a", 50.0, 500), make_point("b", 45.0, 200)]

    result_good = compute_pareto(pts_good, cost_metric="x")
    result_bad = compute_pareto(pts_bad, cost_metric="x")
    assert result_good.hypervolume > result_bad.hypervolume


def test_hypervolume_direct():
    """compute_hypervolume with explicit frontier and all_models."""
    frontier = [make_point("a", 90.0, 1000)]
    all_models = [make_point("a", 90.0, 1000)]
    hv = compute_hypervolume(frontier, all_models)
    assert hv > 0


def test_all_models_field_preserves_input():
    pts = [
        make_point("a", 90.0, 1000),
        make_point("b", 80.0, 2000),
    ]
    result = compute_pareto(pts, cost_metric="y")
    assert len(result.all_models) == 2


# ---------------------------------------------------------------------------
# Multi-seed conservative dominance
# ---------------------------------------------------------------------------


def test_pareto_multi_seed_conservative():
    """Conservative dominance: clear separation → dominant model on frontier."""
    # a's lower bound (90 - 0.5 = 89.5) > b's upper bound (88 + 0.5 = 88.5)
    # AND a.cost == b.cost: acc_bound_dominates → a dominates b
    pts = [
        make_point("a", 90.0, 1000, std=0.5),  # lower bound 89.5
        make_point("b", 88.0, 1000, std=0.5),  # upper bound 88.5
    ]
    result = compute_pareto(pts, cost_metric="x", num_seeds=3)
    frontier_names = {m.name for m in result.frontier}
    assert "a" in frontier_names


def test_pareto_multi_seed_overlapping_bounds():
    """When accuracy bounds overlap significantly, dominance is harder to establish."""
    # a lower bound = 90 - 5 = 85; b upper bound = 88 + 5 = 93 → no bound dominance
    pts = [
        make_point("a", 90.0, 1000, std=5.0),
        make_point("b", 88.0, 1000, std=5.0),
    ]
    result = compute_pareto(pts, cost_metric="x", num_seeds=3)
    # With large overlapping std, neither clearly dominates via bound check
    # cost_buffer_dominates: a.cost == b.cost (not < b.cost * 0.95) → also false
    # So both should be on the frontier
    frontier_names = {m.name for m in result.frontier}
    assert "a" in frontier_names
    assert "b" in frontier_names


def test_pareto_num_seeds_1_standard_dominance():
    """num_seeds=1 uses standard Pareto dominance."""
    pts = [
        make_point("a", 90.0, 1000),
        make_point("b", 80.0, 2000),
    ]
    result = compute_pareto(pts, cost_metric="x", num_seeds=1)
    frontier_names = {m.name for m in result.frontier}
    assert "a" in frontier_names
    assert "b" not in frontier_names


def test_tie_both_on_frontier():
    """Models with identical accuracy AND cost: neither dominates, both on frontier."""
    pts = [
        make_point("a", 90.0, 1000),
        make_point("b", 90.0, 1000),
    ]
    result = compute_pareto(pts, cost_metric="x")
    assert len(result.frontier) == 2
