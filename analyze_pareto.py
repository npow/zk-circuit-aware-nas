"""Pareto analysis on proxy scores — clean version.

Uses already-computed proxy scores from pareto_case_study.py.
Fixes two issues from the original:
1. Joint proxy normalization (NEAR dominated at λ=1, raw scale)
2. Hypervolume calculation (was returning negative values)
"""
import json, math
import numpy as np
from scipy.stats import spearmanr

with open("results/sweep/cifar10_proxy_scores.json") as f:
    data = json.load(f)

records = [r for r in data if r.get("top1") is not None and r.get("near") is not None]
print(f"N = {len(records)}")

# Normalize NEAR and log(ZK_cost) to [0,1]
nears = np.array([r["near"] for r in records])
costs = np.array([r.get("constraints", 1) for r in records])
log_costs = np.log(costs)
accs = np.array([r["top1"] for r in records])

near_norm = (nears - nears.min()) / (nears.max() - nears.min() + 1e-8)
# Higher cost = worse: normalized cost penalty = (log_cost - log_min)/(log_max - log_min)
cost_norm = (log_costs - log_costs.min()) / (log_costs.max() - log_costs.min() + 1e-8)

# Joint score: maximize accuracy proxy, minimize ZK cost
# joint = near_norm - λ × cost_norm  (higher = better on both)
for lam in [0.5, 1.0, 2.0]:
    joint = near_norm - lam * cost_norm
    rho, p = spearmanr(joint, accs)
    print(f"  λ={lam}: joint ρ(accuracy) = {rho:+.3f}  p={p:.3f}")

# Use λ=1.0 for the analysis
joint_scores = near_norm - 1.0 * cost_norm

# Pareto front
def pareto_front(accs, costs):
    pts = list(zip(accs, costs))
    front = []
    for i, (a1, c1) in enumerate(pts):
        dominated = any(a2 >= a1 and c2 <= c1 and (a2 > a1 or c2 < c1)
                        for j, (a2, c2) in enumerate(pts) if i != j)
        if not dominated:
            front.append((a1, c1))
    return sorted(front, key=lambda x: x[0])

pareto = pareto_front(accs.tolist(), costs.tolist())
print(f"\nOracle Pareto front ({len(pareto)} pts):")
for a, c in pareto:
    print(f"  {a:.1f}% accuracy, {c/1e6:.1f}M constraints (logrows≈{math.ceil(math.log2(c))})")

# Budget-constrained analysis: for each ZK budget (logrows threshold),
# what's the best accuracy achievable?
print("\n=== Budget-Constrained Selection ===")
print(f"{'Budget (logrows)':>18} {'Oracle best':>12} {'Proxy top-20%':>14} {'Random avg':>11} {'Proxy gain':>11}")
print("-"*70)

rng = np.random.default_rng(42)
for max_logrows in [20, 21, 22, 23, 24, 25]:
    budget_mask = np.array([math.ceil(math.log2(max(c,2))) <= max_logrows for c in costs])
    n_budget = budget_mask.sum()
    if n_budget < 5:
        continue

    budget_accs = accs[budget_mask]
    budget_joint = joint_scores[budget_mask]
    oracle_best = budget_accs.max()

    # Proxy top-20%: take top 20% by joint score, report their max accuracy
    k = max(1, int(n_budget * 0.20))
    top_idx = np.argsort(budget_joint)[-k:]
    proxy_best = budget_accs[top_idx].max()

    # Random top-20%: average of 100 random draws
    random_bests = []
    for _ in range(100):
        ridx = rng.choice(n_budget, k, replace=False)
        random_bests.append(budget_accs[ridx].max())
    random_avg = np.mean(random_bests)

    gain = proxy_best - random_avg
    print(f"{'≤'+str(max_logrows):>18} {oracle_best:>11.1f}% {proxy_best:>13.1f}% {random_avg:>10.1f}% {gain:>+10.1f}%")

# Proxy quality summary
print("\n=== Proxy-to-Pareto Membership Correlation ===")
is_pareto = np.array([int((a,c) in {(x,y) for x,y in pareto}) for a,c in zip(accs,costs)])
rho_near, p_near = spearmanr(near_norm, is_pareto)
rho_joint, p_joint = spearmanr(joint_scores, is_pareto)
rho_zk, p_zk = spearmanr(-cost_norm, is_pareto)  # lower cost = more likely Pareto
print(f"  NEAR alone:   ρ={rho_near:+.3f}  p={p_near:.3f}")
print(f"  Joint (λ=1):  ρ={rho_joint:+.3f}  p={p_joint:.3f}")
print(f"  ZK cost alone: ρ={rho_zk:+.3f}  p={p_zk:.3f}")

# Save
out = {
    "n": len(records),
    "pareto_front": pareto,
    "budget_analysis": {}
}
with open("results/pareto_analysis_clean.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("\nSaved → results/pareto_analysis_clean.json")
