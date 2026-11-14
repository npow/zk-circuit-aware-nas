import matplotlib
matplotlib.use('Agg')  # Must be set before any other matplotlib imports

import matplotlib.pyplot as plt


def plot_pareto(
    all_points: list[dict],  # list of {"name": str, "accuracy": float, "cost": float, "on_frontier": bool}
    output_path: str,
    cost_metric: str = "constraint_estimate",
    log_scale_x: bool = True,
    title: str = "Accuracy vs ZK Proving Cost",
) -> None:
    """
    Scatter plot of all models (accuracy vs. log(cost)).
    Frontier models highlighted with distinct marker, connected by step line.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    frontier = [p for p in all_points if p["on_frontier"]]
    non_frontier = [p for p in all_points if not p["on_frontier"]]

    # Plot non-frontier
    if non_frontier:
        ax.scatter(
            [p["cost"] for p in non_frontier],
            [p["accuracy"] for p in non_frontier],
            color="steelblue", alpha=0.6, s=60, label="Models"
        )
        for p in non_frontier:
            ax.annotate(p["name"], (p["cost"], p["accuracy"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.7)

    # Plot frontier
    if frontier:
        frontier_sorted = sorted(frontier, key=lambda p: p["cost"])
        ax.scatter(
            [p["cost"] for p in frontier_sorted],
            [p["accuracy"] for p in frontier_sorted],
            color="crimson", marker="*", s=200, zorder=5, label="Pareto Frontier"
        )
        # Step line connecting frontier points
        xs = [p["cost"] for p in frontier_sorted]
        ys = [p["accuracy"] for p in frontier_sorted]
        ax.step(xs, ys, where="post", color="crimson", alpha=0.4, linewidth=1.5)
        for p in frontier_sorted:
            ax.annotate(p["name"], (p["cost"], p["accuracy"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8,
                        color="crimson", fontweight="bold")

    if log_scale_x and all(p["cost"] > 0 for p in all_points):
        ax.set_xscale("log")

    ax.set_xlabel(f"ZK Cost ({cost_metric})")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pareto_from_results(entries: list[dict], output_path: str) -> None:
    """Build Pareto plot directly from results.json entries.

    Renders two frontier lines when both proxy and EZKL data are available:
    - Solid line: proxy constraint_estimate
    - Dashed line: EZKL total_cells
    """
    from ..metrics.pareto import ModelPoint, compute_pareto

    proxy_points = []
    ezkl_points = []
    for e in entries:
        if e.get("training", {}).get("training_failed"):
            continue
        top1 = e.get("accuracy_top1_mean")
        if top1 is None:
            continue
        proxy_d = e.get("zk_proxy") or {}
        cost = proxy_d.get("constraint_estimate")
        if cost:
            proxy_points.append(ModelPoint(
                name=e.get("model_name", "?"),
                accuracy_mean=float(top1),
                accuracy_std=None,
                cost=float(cost),
            ))
        ezkl_d = e.get("ezkl") or {}
        ezkl_cost = ezkl_d.get("total_cells") or ezkl_d.get("num_rows")
        if ezkl_cost:
            ezkl_points.append(ModelPoint(
                name=e.get("model_name", "?"),
                accuracy_mean=float(top1),
                accuracy_std=None,
                cost=float(ezkl_cost),
            ))

    if not proxy_points:
        return

    proxy_result = compute_pareto(proxy_points, cost_metric="constraints")
    proxy_ids = {p.name for p in proxy_result.frontier}

    fig, ax = plt.subplots(figsize=(10, 6))

    # Proxy scatter
    for p in proxy_points:
        on_f = p.name in proxy_ids
        ax.scatter(p.cost, p.accuracy_mean,
                   color="crimson" if on_f else "steelblue",
                   marker="*" if on_f else "o",
                   s=200 if on_f else 60,
                   alpha=1.0 if on_f else 0.6,
                   zorder=5 if on_f else 3)
        ax.annotate(p.name, (p.cost, p.accuracy_mean),
                    textcoords="offset points", xytext=(5, 5),
                    fontsize=7 if not on_f else 8,
                    color="crimson" if on_f else "gray",
                    fontweight="bold" if on_f else "normal")

    # Proxy frontier step line
    if proxy_result.frontier:
        fpts = sorted(proxy_result.frontier, key=lambda p: p.cost)
        ax.step([p.cost for p in fpts], [p.accuracy_mean for p in fpts],
                where="post", color="crimson", linewidth=2, label="Proxy frontier")

    # EZKL frontier (dashed), if available
    if ezkl_points:
        ezkl_result = compute_pareto(ezkl_points, cost_metric="constraints")
        if ezkl_result.frontier:
            epts = sorted(ezkl_result.frontier, key=lambda p: p.cost)
            ax.step([p.cost for p in epts], [p.accuracy_mean for p in epts],
                    where="post", color="darkorange", linewidth=2,
                    linestyle="--", label="EZKL frontier")

    ax.set_xscale("log")
    ax.set_xlabel("ZK Constraint Estimate (proxy)")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Accuracy vs ZK Proving Cost — Pareto Frontier")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
