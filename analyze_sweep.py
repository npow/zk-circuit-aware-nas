"""Analysis of the regenerated 3-seed CIFAR-10 sweep.

Inputs : results/sweep/cifar10_seed{0,1,2}_shard*.jsonl  (from sweep_gpu.py)
         results/trained_calibration.jsonl                (from calibrate_trained.py; optional)
Outputs: results/sweep_summary.json  (every number used in the paper's sweep tables)

Sections
  A  proxy-vs-accuracy Spearman per activation family: bootstrap CI, p, Holm correction,
     partial correlation controlling for architecture type, normalization and log-params
  B  NASWOT (standard and sign-pattern variant)
  C  seed noise of the accuracy labels
  D  budget-constrained search under ezkl-calibrated budgets with a tuning/evaluation split
  E  cost-model fit (non-negative least squares) with a fit/evaluation split

Usage: python analyze_sweep.py --sweep DIR [--calib FILE] [--val73 results/proxy_vs_ezkl.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import random

import numpy as np
from scipy import stats
from scipy.optimize import nnls

PROXIES = ["near", "synflow", "zico", "meco", "zenas", "naswot", "naswot_sign"]
ACTS = ["relu", "x2", "x3px", "x2px"]


def load_sweep(d):
    arch = collections.defaultdict(lambda: {"seeds": {}})
    for f in glob.glob(f"{d}/cifar10_seed*_shard*.jsonl"):
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            a = arch[r["name"]]
            a["cfg"] = {k: r[k] for k in ("name", "family", "act", "norm") if k in r}
            a["cfg"]["depthwise"] = bool(r.get("depthwise", False))
            if "error" not in r and "top1" in r:
                a["seeds"][r["seed"]] = r
    out = {}
    for n, a in arch.items():
        if not a["seeds"]:
            continue
        c = a["cfg"]
        c["arch_type"] = ("dws_" if c["depthwise"] else "") + c["family"]
        recs = list(a["seeds"].values())
        row = dict(c)
        row["n_seeds"] = len(recs)
        row["top1_mean"] = float(np.mean([r["top1"] for r in recs]))
        row["top1_sd"] = float(np.std([r["top1"] for r in recs], ddof=1)) if len(recs) > 1 else float("nan")
        row["top1_by_seed"] = {s: r["top1"] for s, r in a["seeds"].items()}
        row["diverged_any"] = any(r.get("diverged") for r in recs)
        r0_ = a["seeds"].get(0)
        row["seed0_failed"] = bool(r0_ is not None and (r0_.get("diverged") or r0_.get("top1", 100) < 15))
        # init-time quantities (proxies, cost estimate) exist only on seed-0 records; seed >=1 runs skip them
        r0 = a["seeds"].get(0)
        row["params"] = recs[0].get("params")
        for p in PROXIES:
            row[p] = r0.get(p) if r0 is not None else None
        for p in ("naswot", "naswot_sign"):
            row[p + "_neg_inf_frac"] = float(bool(r0.get(p + "_neg_inf"))) if r0 is not None else 0.0
        row["zk_logrows_est"] = r0.get("zk_logrows_est") if r0 is not None else None
        row["zk_constraints"] = r0.get("zk_constraints") if r0 is not None else None
        out[n] = row
    return out


def rank(x):
    return stats.rankdata(x)


def boot_ci(x, y, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(x)
    v = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        r = stats.spearmanr(x[i], y[i])[0]
        if r == r:
            v.append(r)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [float("nan")] * 2


def holm(pvals):
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [None] * len(pvals)
    m = len(pvals)
    run = 0.0
    for k, i in enumerate(idx):
        run = max(run, min(1.0, (m - k) * pvals[i]))
        adj[i] = run
    return adj


def design(rows, with_params=True):
    types = sorted({r["arch_type"] for r in rows})
    norms = sorted({r["norm"] for r in rows})
    cols = []
    for t in types[1:]:
        cols.append([1.0 if r["arch_type"] == t else 0.0 for r in rows])
    for t in norms[1:]:
        cols.append([1.0 if r["norm"] == t else 0.0 for r in rows])
    if with_params:
        cols.append(list(rank([math.log(r["params"]) for r in rows])))
    Z = np.column_stack([np.ones(len(rows))] + [np.array(c) for c in cols]) if cols else np.ones((len(rows), 1))
    return Z


def partial_spearman(rows, xk, yk):
    x = rank([r[xk] for r in rows]); y = rank([r[yk] for r in rows])
    Z = design(rows)
    bx = np.linalg.lstsq(Z, x, rcond=None)[0]; by = np.linalg.lstsq(Z, y, rcond=None)[0]
    rx, ry = x - Z @ bx, y - Z @ by
    r = float(np.corrcoef(rx, ry)[0, 1])
    df = len(rows) - Z.shape[1] - 1
    t = r * math.sqrt(df / max(1e-12, 1 - r * r)) if df > 0 else float("nan")
    p = float(2 * stats.t.sf(abs(t), df)) if df > 0 else float("nan")
    return r, p


def is_failed(r):
    return bool(r["diverged_any"]) or r["top1_mean"] < 15


def section_A(arch, exclude_failed=False, normalized_only=False):
    res = {}
    tests = []
    for fam in ["all"] + ACTS:
        rows = [r for r in arch.values() if (fam == "all" or r["act"] == fam) and not (exclude_failed and is_failed(r))
                and not (normalized_only and r["norm"] == "none")]
        for p in PROXIES:
            sub = [r for r in rows if r[p] is not None]
            if len(sub) < 8:
                res[f"{fam}/{p}"] = {"n": len(sub), "note": "too few finite scores"}
                continue
            x = np.array([r[p] for r in sub]); y = np.array([r["top1_mean"] for r in sub])
            rho, pv = stats.spearmanr(x, y)
            ent = {"n": len(sub), "rho": float(rho), "p": float(pv), "ci95": boot_ci(x, y)}
            if len({r["arch_type"] for r in sub}) > 1 or len({r["norm"] for r in sub}) > 1:
                try:
                    pr, pp = partial_spearman(sub, p, "top1_mean")
                    ent["partial_rho"] = pr; ent["partial_p"] = pp
                except Exception as e:
                    ent["partial_error"] = str(e)
            res[f"{fam}/{p}"] = ent
            if fam != "all":
                tests.append(f"{fam}/{p}")
    adj = holm([res[t]["p"] for t in tests])  # Holm family: the activation-family tests shown in the tables
    for t, a in zip(tests, adj):
        res[t]["p_holm"] = a
    # zk structural proxy vs accuracy for reference
    return res


def section_B(arch):
    out = {}
    for act in ACTS:
        rows = [r for r in arch.values() if r["act"] == act]
        out[act] = {"n": len(rows),
                    "naswot_neg_inf": sum(r["naswot_neg_inf_frac"] == 1.0 for r in rows),
                    "naswot_sign_neg_inf": sum(r["naswot_sign_neg_inf_frac"] == 1.0 for r in rows)}
    return out


def section_C(arch):
    rows = [r for r in arch.values() if r["n_seeds"] >= 2]
    sd = [r["top1_sd"] for r in rows if r["top1_sd"] == r["top1_sd"]]
    out = {"n_archs_multi_seed": len(rows)}
    if sd:
        out.update({"median_sd": float(np.median(sd)), "p90_sd": float(np.percentile(sd, 90)), "max_sd": float(np.max(sd))})
    pairs = {}
    for a, b in ((0, 1), (0, 2), (1, 2)):
        xs = [(r["top1_by_seed"].get(a), r["top1_by_seed"].get(b)) for r in rows]
        xs = [t for t in xs if t[0] is not None and t[1] is not None]
        if len(xs) > 5:
            pairs[f"{a}v{b}"] = float(stats.spearmanr([t[0] for t in xs], [t[1] for t in xs])[0])
    out["seed_pair_spearman"] = pairs
    out["diverged_any"] = sum(r["diverged_any"] for r in arch.values())
    out["chance_level_lt15"] = sum(r["top1_mean"] < 15 for r in arch.values())
    return out


def depth_of(name):
    t = name.split("_")
    if "resnet" in name:
        b = [x for x in t if x.startswith("b") and x[1:].isdigit()]
        return int(b[0][1:]) if b else None
    last = t[-1]
    parts = last.replace("x", "_").split("_") if name.startswith("mlp") else [x for x in t if x.isdigit()]
    return len([x for x in parts if x.isdigit()])


def section_F(arch):
    """Zen score vs depth, size, and training failure within norm-free polynomial networks."""
    out = {}
    for label, rows in (("poly_norm_free", [r for r in arch.values() if r["act"] != "relu" and r["norm"] == "none" and r["zenas"] is not None]),
                        ("relu_norm_free", [r for r in arch.values() if r["act"] == "relu" and r["norm"] == "none" and r["zenas"] is not None])):
        if len(rows) < 8:
            continue
        z = np.array([r["zenas"] for r in rows]); acc = np.array([r["top1_mean"] for r in rows])
        d = np.array([depth_of(r["name"]) or 0 for r in rows], float); lp = np.array([math.log(r["params"]) for r in rows])
        fail = np.array([1.0 if (r["diverged_any"] or r["top1_mean"] < 15) else 0.0 for r in rows])
        ent = {"n": len(rows), "rho_zen_acc": float(stats.spearmanr(z, acc)[0]), "rho_zen_depth": float(stats.spearmanr(z, d)[0]),
               "rho_zen_logparams": float(stats.spearmanr(z, lp)[0]), "rho_depth_acc": float(stats.spearmanr(d, acc)[0]),
               "failure_rate_overall": float(fail.mean())}
        q = np.quantile(z, [0.25, 0.75])
        ent["failure_rate_top_zen_quartile"] = float(fail[z >= q[1]].mean()); ent["failure_rate_bottom_zen_quartile"] = float(fail[z <= q[0]].mean())
        # accuracy among non-failed networks only
        ok = fail == 0
        if ok.sum() > 8:
            ent["rho_zen_acc_excluding_failed"] = float(stats.spearmanr(z[ok], acc[ok])[0]); ent["n_excluding_failed"] = int(ok.sum())
        out[label] = ent
    return out


# ---------------- D: search ----------------
def zscore(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / (s if s > 0 else 1.0)


def select(pool, method, lam, K, rng):
    n = len(pool)
    K = min(K, n)
    if method == "random":
        return rng.sample(range(n), K)
    near = zscore([p["near"] for p in pool]); cost = zscore([p["log_cost"] for p in pool])
    if method == "near":
        sc = near
    elif method == "zk_cost":
        sc = -cost
    else:
        sc = near - lam * cost
    return list(np.argsort(-sc, kind="stable")[:K])


def outcome(pool, sel):
    return max(pool[i]["acc"] for i in sel)


MAX_EST = 25


def apply_skip_rule(arch, calib):
    """The rule used to decide which networks to calibrate: skip networks whose seed-0 training failed
    (diverged or top-1 < 15%) and networks whose estimated logrows exceed MAX_EST.  A few networks that the
    rule would skip were calibrated in an earlier, alphabetical pass; they are dropped here so every analysis
    uses one rule.  Returns (kept, dropped)."""
    kept, dropped = [], []
    for c in calib:
        r = arch.get(c["name"])
        bad = r is None or r["seed0_failed"] or (r.get("zk_logrows_est") or 0) > MAX_EST
        (dropped if bad else kept).append(c)
    return kept, dropped


def section_D(arch, calib, K=10, splits=300, lams=(0.0, 0.25, 0.5, 1.0, 2.0, 4.0), seed=0):
    out = {}
    cal = {c["name"]: c for c in calib if c.get("logrows") is not None and c.get("total_assignments")}
    base = []
    for n, r in arch.items():
        if n in cal and r["near"] is not None:
            base.append({"name": n, "acc": r["top1_mean"], "near": r["near"], "arch_type": r["arch_type"], "act": r["act"],
                         "logrows": cal[n]["logrows"], "log_cost": math.log2(cal[n]["total_assignments"])})
    out["n_calibrated_with_near"] = len(base)
    out["logrows_hist"] = dict(sorted(collections.Counter(b["logrows"] for b in base).items()))
    rng = random.Random(seed)
    for B in (20, 21, 22, 23):
        pool_all = [b for b in base if b["logrows"] <= B]
        ent = {"pool": len(pool_all), "pool_failed_any_seed_or_mean_lt15": sum(is_failed(arch[b["name"]]) for b in pool_all), "pool_by_act": dict(collections.Counter(b["act"] for b in pool_all)),
               "near_top_K_by_act": dict(collections.Counter(pool_all[i]["act"] for i in select(pool_all, "near", 0, K, rng)))}
        if len(pool_all) < 24:
            ent["note"] = "pool too small for a 50/50 tuning/evaluation split"
            out[f"B<={B}"] = ent
            continue
        oracle = max(b["acc"] for b in pool_all)
        ent["oracle_best_acc"] = oracle
        reg = collections.defaultdict(list)
        reg_by_lam = collections.defaultdict(list)
        chosen_lams = []
        for _ in range(splits):
            idx = list(range(len(pool_all))); rng.shuffle(idx)
            h = len(idx) // 2
            tune = [pool_all[i] for i in idx[:h]]; test = [pool_all[i] for i in idx[h:]]
            best_lam, best_val = None, -1
            for lam in lams:  # ties are broken toward the smaller lambda (lambda=0 is NEAR-only), because of the strict '>'
                v = outcome(tune, select(tune, "joint", lam, K, rng))
                if v > best_val + 1e-9:
                    best_val, best_lam = v, lam
            for lam in lams:  # untuned: regret of each fixed lambda on the evaluation half
                reg_by_lam[lam].append(max(b["acc"] for b in test) - outcome(test, select(test, "joint", lam, K, rng)))
            chosen_lams.append(best_lam)
            o_test = max(b["acc"] for b in test)
            reg["joint"].append(o_test - outcome(test, select(test, "joint", best_lam, K, rng)))
            reg["near_only"].append(o_test - outcome(test, select(test, "near", 0, K, rng)))
            reg["zk_cost_only"].append(o_test - outcome(test, select(test, "zk_cost", 0, K, rng)))
            reg["random"].append(o_test - float(np.mean([outcome(test, select(test, "random", 0, K, rng)) for _ in range(50)])))
        for m, v in reg.items():
            v = np.array(v)
            ent[m] = {"mean_regret": float(v.mean()), "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                      "zero_regret_frac": float((v < 1e-9).mean())}
        j, nr, rd = np.array(reg["joint"]), np.array(reg["near_only"]), np.array(reg["random"])
        ent["joint_vs_near"] = {"win": int((j < nr - 1e-9).sum()), "tie": int((abs(j - nr) <= 1e-9).sum()), "loss": int((j > nr + 1e-9).sum())}
        ent["joint_vs_random"] = {"win": int((j < rd - 1e-9).sum()), "tie": int((abs(j - rd) <= 1e-9).sum()), "loss": int((j > rd + 1e-9).sum())}
        ent["chosen_lambda_counts"] = dict(collections.Counter(chosen_lams))
        ent["regret_by_lambda_untuned"] = {str(l): float(np.mean(v)) for l, v in reg_by_lam.items()}
        base0 = np.array(reg_by_lam[lams[0]])
        ent["paired_vs_lambda0"] = {str(l): {"mean": float(np.mean(np.array(reg_by_lam[l]) - base0)),
                                              "ci95": [float(np.percentile(np.array(reg_by_lam[l]) - base0, 2.5)), float(np.percentile(np.array(reg_by_lam[l]) - base0, 97.5))],
                                              "better_frac": float((np.array(reg_by_lam[l]) < base0 - 1e-9).mean()), "worse_frac": float((np.array(reg_by_lam[l]) > base0 + 1e-9).mean())}
                                    for l in lams[1:]}
        ent["tie_rule"] = "ties broken toward smaller lambda; lambda=0 is NEAR-only"
        out[f"B<={B}"] = ent
    out["K"] = K; out["splits"] = splits; out["lambda_grid"] = list(lams)
    return out


def section_D2(arch, calib, K=10, boot=500, seed=0):
    """Budget filtering by ESTIMATED logrows (no ezkl run): how often does the selection violate the CALIBRATED budget?"""
    cal = {c["name"]: c for c in calib if c.get("logrows") is not None}
    base = [{"acc": r["top1_mean"], "near": r["near"], "est": r["zk_logrows_est"], "cal": cal[n]["logrows"], "log_cost": 0.0}
            for n, r in arch.items() if n in cal and r["near"] is not None and r.get("zk_logrows_est") is not None]
    rng = np.random.default_rng(seed)
    out = {"n_pool": len(base)}
    for B in (20, 21, 22, 23):
        def stats_for(pool):
            est_pool = [b for b in pool if b["est"] <= B]
            feas_oracle = max([b["acc"] for b in pool if b["cal"] <= B], default=float("nan"))
            res = {}
            if len(est_pool) >= 1:
                top = sorted(est_pool, key=lambda b: -b["near"])[:K]
                res["near_in_est_budget"] = {"violation": float(np.mean([b["cal"] > B for b in top])),
                                              "best_feasible_regret": feas_oracle - max([b["acc"] for b in top if b["cal"] <= B], default=0.0)}
                rnd = [est_pool[i] for i in rng.choice(len(est_pool), min(K, len(est_pool)), replace=False)]
                res["random_in_est_budget"] = {"violation": float(np.mean([b["cal"] > B for b in rnd])),
                                                "best_feasible_regret": feas_oracle - max([b["acc"] for b in rnd if b["cal"] <= B], default=0.0)}
            top = sorted(pool, key=lambda b: -b["near"])[:K]
            res["near_no_budget"] = {"violation": float(np.mean([b["cal"] > B for b in top])),
                                      "best_feasible_regret": feas_oracle - max([b["acc"] for b in top if b["cal"] <= B], default=0.0)}
            res["n_est_pool"] = len(est_pool); res["n_cal_feasible"] = sum(b["cal"] <= B for b in pool)
            # of networks the estimate admits, share that actually exceed the budget; of the feasible ones, share the estimate rejects
            res["est_admits_but_exceeds"] = float(np.mean([b["cal"] > B for b in est_pool])) if est_pool else float("nan")
            feas = [b for b in pool if b["cal"] <= B]
            res["feasible_rejected_by_est"] = float(np.mean([b["est"] > B for b in feas])) if feas else float("nan")
            return res
        point = stats_for(base)
        bs = collections.defaultdict(list)
        for _ in range(boot):
            idx = rng.integers(0, len(base), len(base))
            r = stats_for([base[i] for i in idx])
            for m in ("near_in_est_budget", "random_in_est_budget", "near_no_budget"):
                if m in r:
                    bs[m + "/violation"].append(r[m]["violation"]); bs[m + "/regret"].append(r[m]["best_feasible_regret"])
            bs["est_admits_but_exceeds"].append(r["est_admits_but_exceeds"]); bs["feasible_rejected_by_est"].append(r["feasible_rejected_by_est"])
        ci = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in bs.items()}
        out[f"B<={B}"] = {"point": point, "ci95": ci}
    return out


# ---------------- E: cost-model fit ----------------
def feature_vector(f):
    g = lambda op, k: f.get(op, {}).get(k, 0)
    macs_std = g("Conv", "macs") - g("Conv", "depthwise_macs")
    return {
        "macs_conv_standard": macs_std,
        "macs_conv_depthwise": g("Conv", "depthwise_macs"),
        "macs_gemm": g("Gemm", "macs") + g("MatMul", "macs"),
        "n_matmul_conv_nodes": g("Conv", "count") + g("Gemm", "count") + g("MatMul", "count"),
        "elem_relu": g("Relu", "elements"),
        "elem_pow": g("Pow", "elements"),
        "elem_mul": g("Mul", "elements"),
        "elem_add_sub": g("Add", "elements") + g("Sub", "elements"),
        "elem_norm": sum(g(o, "elements") for o in ("LayerNormalization", "GroupNormalization", "InstanceNormalization", "BatchNormalization")),
        "elem_pool_reduce": sum(g(o, "elements") for o in ("AveragePool", "GlobalAveragePool", "ReduceMean", "ReduceSum", "MaxPool")),
    }


def cv_metrics(X, y, lr, hand, types, repeats=20, folds=5, seed=0):
    """Repeated k-fold cross-validation of the NNLS cost model over all calibrated networks."""
    rng = np.random.default_rng(seed)
    n = len(y)
    agg = {"fitted": collections.defaultdict(list), "hand": collections.defaultdict(list)}
    by_type = collections.defaultdict(lambda: collections.defaultdict(list))
    for _ in range(repeats):
        fold = rng.permutation(n) % folds
        pred = np.zeros(n)
        for k in range(folds):
            tr, te = fold != k, fold == k
            w = 1.0 / y[tr]
            coef, _ = nnls(X[tr] * w[:, None], np.ones(tr.sum()))
            pred[te] = X[te] @ coef
        for name, pr in (("fitted", pred), ("hand", hand)):
            pl = np.ceil(np.log2(np.maximum(pr, 2))); d = pl - lr
            agg[name]["exact"].append(float((d == 0).mean())); agg[name]["within1"].append(float((abs(d) <= 1).mean()))
            agg[name]["rho"].append(float(stats.spearmanr(pl, lr)[0]))
            for t in set(types):
                m = np.array([x == t for x in types])
                by_type[name + "/" + t]["exact"].append(float((d[m] == 0).mean())); by_type[name + "/" + t]["n"] = int(m.sum())
    out = {"repeats": repeats, "folds": folds, "n": int(n)}
    for name in agg:
        out[name] = {k: float(np.mean(v)) for k, v in agg[name].items()}
    out["by_type"] = {k: {"exact": float(np.mean(v["exact"])), "n": v["n"]} for k, v in by_type.items()}
    return out


def section_E(calib, val_names, arch=None):
    ok = [c for c in calib if c.get("logrows") is not None and c.get("total_assignments") and c.get("features")]
    names = [c["name"] for c in ok]
    fv = [feature_vector(c["features"]) for c in ok]
    cols = [k for k in fv[0] if any(f[k] for f in fv)]
    X = np.array([[f[k] for k in cols] for f in fv], float); y = np.array([c["total_assignments"] for c in ok], float)
    lr = np.array([c["logrows"] for c in ok]); hand = np.array([c["proxy_constraints"] for c in ok], float)
    is_val = np.array([n in val_names for n in names])
    # x3px networks are scored separately: their logrows are driven by calibration range, not cells
    res = {"n_total": len(ok), "n_fit": int((~is_val).sum()), "n_eval": int(is_val.sum()), "features": cols}

    def metrics(pred, mask, label):
        pl = np.ceil(np.log2(np.maximum(pred[mask], 2)))
        d = pl - lr[mask]
        return {"n": int(mask.sum()), "exact_logrows": int((d == 0).sum()), "within1": int((abs(d) <= 1).sum()),
                "spearman_logrows": float(stats.spearmanr(pl, lr[mask])[0]),
                "log2_rmse_assignments": float(np.sqrt(np.mean((np.log2(np.maximum(pred[mask], 2)) - np.log2(y[mask])) ** 2)))}

    # fit on fit split, relative-error weighting (divide rows by target)
    w = 1.0 / y[~is_val]
    coef, _ = nnls(X[~is_val] * w[:, None], np.ones((~is_val).sum()))
    res["coef"] = {c: float(v) for c, v in zip(cols, coef)}
    pred = X @ coef
    res["fitted_eval"] = metrics(pred, is_val, "fitted")
    res["fitted_fit_split"] = metrics(pred, ~is_val, "fitted")
    res["hand_set_eval"] = metrics(hand, is_val, "hand")
    res["hand_set_all"] = metrics(hand, np.ones(len(ok), bool), "hand")
    # per-activation breakdown on the evaluation split (hand-set proxy)
    acts = []
    for n in names:
        acts.append(next((a for a in ("x3px", "x2px", "x2", "relu") if f"_{a}_" in n), "?"))
    acts = np.array(acts)
    res["hand_set_eval_by_act"] = {a: metrics(hand, is_val & (acts == a), a) for a in ("relu", "x2", "x2px", "x3px") if (is_val & (acts == a)).sum() > 3}
    res["fitted_eval_by_act"] = {a: metrics(pred, is_val & (acts == a), a) for a in ("relu", "x2", "x2px", "x3px") if (is_val & (acts == a)).sum() > 3}
    if arch is not None:
        types = [arch[n]["arch_type"] if n in arch else "?" for n in names]
        res["cross_validation"] = cv_metrics(X, y, lr, hand, types)
        res["eval_split_arch_types"] = dict(collections.Counter(t for t, v in zip(types, is_val) if v))
        res["fit_split_arch_types"] = dict(collections.Counter(t for t, v in zip(types, is_val) if not v))
    # does ceil(log2(assignments)) equal logrows? (accounting check)
    d = np.ceil(np.log2(y)) - lr
    res["ceil_log2_vs_logrows_direction"] = {"equal": int((d == 0).sum()), "logrows_below": int((d > 0).sum()), "logrows_above": int((d < 0).sum()), "n": int(len(d))}
    res["assignments_ceil_log2_equals_logrows"] = {a: {"n": int((acts == a).sum()), "equal": int(((d == 0) & (acts == a)).sum())} for a in ("relu", "x2", "x2px", "x3px")}
    return res


def section_G(dump_dirs, arch):
    """Test the four-resource rule  logrows = max(ceil log2 rows, ceil log2 const, ceil log2 lookup span, ceil log2 range-check span)
    on dumped ezkl settings.json files."""
    L = lambda x: math.ceil(math.log2(max(x, 1))) if x > 1 else 0
    recs = {}
    for d in dump_dirs:
        for f in glob.glob(f"{d}/*.settings.json"):
            name = f.split("/")[-1].replace(".settings.json", "")
            st = json.load(open(f)); ra = st["run_args"]
            lo, hi = ra["lookup_range"]
            rcs = st.get("required_range_checks") or []
            need = {"rows": L(st["num_rows"]), "const": L(st["total_const_size"]),
                    "lookup": L(hi - lo + 1) if hi > lo else 0,
                    "range_check": max([L(b - a + 1) for a, b in rcs], default=0)}
            pred = max(need.values())
            act = next((a for a in ("x3px", "x2px", "x2", "relu") if f"_{a}_" in name), "?")
            recs[name] = {"logrows": ra["logrows"], "pred": pred, "need": need, "act": act, "big_rc": need["range_check"] >= 22, "scale": ra.get("input_scale"),
                          "binding": [k for k, v in need.items() if v == pred], "rows_x2_eq_assign": st["total_assignments"] == 2 * st["num_rows"] or abs(st["total_assignments"] - 2 * st["num_rows"]) <= 2,
                          "const": st["total_const_size"], "cells": st["total_assignments"], "params": (arch.get(name) or {}).get("params")}
    n = len(recs)
    out = {"n": n, "exact": sum(r["pred"] == r["logrows"] for r in recs.values()),
           "by_act": {a: {"n": sum(r["act"] == a for r in recs.values()), "exact": sum(r["act"] == a and r["pred"] == r["logrows"] for r in recs.values())} for a in ("relu", "x2", "x2px", "x3px")},
           "big_range_check_by_act": {a: sum(r["act"] == a and r["big_rc"] for r in recs.values()) for a in ("relu", "x2", "x2px", "x3px")},
           "binding_counts": dict(collections.Counter(b for r in recs.values() for b in r["binding"])),
           "pred_minus_logrows": dict(collections.Counter(r["pred"] - r["logrows"] for r in recs.values())),
           "logrows_range": [min(r["logrows"] for r in recs.values()), max(r["logrows"] for r in recs.values())],
           "const_requirement_equals_ceil_log2_cells": sum(r["need"]["const"] == L(r["cells"]) for r in recs.values()),
           "rows_requirement_is_ceil_log2_cells_minus_1": sum(r["need"]["rows"] == L(r["cells"]) - 1 for r in recs.values()),
           "rows_equal_half_assignments": sum(r["rows_x2_eq_assign"] for r in recs.values()),
           "mismatches": [{"name": k, "logrows": r["logrows"], "pred": r["pred"], "need": r["need"]} for k, r in recs.items() if r["pred"] != r["logrows"]][:20]}
    out["const_over_params"] = {}
    for label, pred in (("mlp", lambda k: k.startswith("mlp")), ("conv", lambda k: not k.startswith("mlp"))):
        ratios = [r["const"] / r["params"] for k, r in recs.items() if r["params"] and pred(k)]
        if ratios:
            out["const_over_params"][label] = {"median": float(np.median(ratios)), "min": float(np.min(ratios)), "max": float(np.max(ratios)), "n": len(ratios)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--calib", default=None)
    ap.add_argument("--val73", default="results/proxy_vs_ezkl.json")
    ap.add_argument("--settings", nargs="*", default=[], help="dirs of dumped ezkl settings.json files")
    ap.add_argument("--out", default="results/sweep_summary.json")
    a = ap.parse_args()
    arch = load_sweep(a.sweep)
    out = {"n_archs": len(arch), "n_by_act": dict(collections.Counter(r["act"] for r in arch.values())),
           "n_by_type": dict(collections.Counter(r["arch_type"] for r in arch.values())),
           "A_proxy_vs_accuracy": section_A(arch), "A2_excluding_failed": section_A(arch, True),
           "A3_normalized_only_same_recipe": section_A(arch, False, True),
           "n_failed_by_act": {a: sum(is_failed(r) for r in arch.values() if r["act"] == a) for a in ACTS},
           "n_failed_by_norm": {n: sum(is_failed(r) for r in arch.values() if r["norm"] == n) for n in sorted({r["norm"] for r in arch.values()})},
           "n_failed_by_type": {t: sum(is_failed(r) for r in arch.values() if r["arch_type"] == t) for t in sorted({r["arch_type"] for r in arch.values()})},
           "B_naswot": section_B(arch), "C_seed_noise": section_C(arch),
           "F_zen": section_F(arch)}
    if a.calib:
        calib = [json.loads(l) for l in open(a.calib) if l.strip()]
        out["n_calibration_records"] = len(calib)
        out["n_calibration_failed"] = sum(1 for c in calib if c.get("logrows") is None)
        calib_all = calib
        calib, dropped = apply_skip_rule(arch, calib_all)
        out["calibration_rule"] = {"records": len(calib_all), "dropped_by_uniform_rule": len(dropped), "used": len(calib),
                                   "failed_inside_ezkl": sum(1 for c in calib if c.get("logrows") is None),
                                   "pool_members_failed_or_est_gt_25": 0,
                                   "max_logrows_used": max((c["logrows"] for c in calib if c.get("logrows")), default=None)}
        json.dump(sorted(c["name"] for c in dropped), open("results/calibration_dropped_by_rule.json", "w"), indent=1)
        out["D_search"] = section_D(arch, calib)
        out["D_search_K5"] = section_D(arch, calib, K=5)
        out["D2_estimated_budget"] = section_D2(arch, calib)
        val = {r["name"] for r in json.load(open(a.val73))}
        out["E_cost_model"] = section_E(calib, val, arch)
    if a.settings:
        out["G_logrows_rule"] = section_G(a.settings, arch)
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"{len(arch)} architectures; wrote {a.out}")


if __name__ == "__main__":
    main()
