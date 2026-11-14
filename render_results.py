"""Render paper/results_auto.tex from paper/results_template.tex and the released result files.

Every number in the rendered prose is read from results/sweep_summary.json, results/trained_calibration.jsonl and
(optionally) results/ezkl_timing_trained.jsonl.  The renderer fails if a placeholder is left unresolved.
Usage: python render_results.py   (run make_tables.py first)
"""
import json, re, statistics, sys

S = json.load(open("results/sweep_summary.json"))
A, A2, A3 = S["A_proxy_vs_accuracy"], S["A2_excluding_failed"], S["A3_normalized_only_same_recipe"]
C, F, D, D2, E = S["C_seed_noise"], S["F_zen"], S["D_search"], S["D2_estimated_budget"], S["E_cost_model"]
nf, nfn, na, nt = S["n_failed_by_act"], S["n_failed_by_norm"], S["n_by_act"], S["n_by_type"]
V = {}


def put(k, v):
    V[k] = str(v)


def s2(x):
    return f"{x:+.2f}"


# ----- sweep composition and failures -----
put("n_archs", S["n_archs"])
for k in ("mlp", "cnn", "dws_cnn", "resnet", "dws_resnet"):
    put(f"nt_{k}", nt[k])
for k in ("relu", "x2", "x3px", "x2px"):
    put(f"na_{k}", na[k]); put(f"nf_{k}", nf[k])
for k in ("none", "layernorm", "groupnorm"):
    put(f"nfn_{k}", nfn[k])
put("n_calib", S["n_calibration_records"]); put("n_calib_failed", S["n_calibration_failed"])
put("n_pool", D["n_calibrated_with_near"])
put("sd_median", f"{C['median_sd']:.2f}"); put("sd_p90", f"{C['p90_sd']:.2f}"); put("sd_max", f"{C['max_sd']:.1f}")
put("seed_rho", f"{C['seed_pair_spearman']['0v1']:.2f}")
put("n_chance", C["chance_level_lt15"])

# ----- correlations -----
for tag, d in (("A", A), ("B", A2), ("C", A3)):
    for p in ("near", "synflow", "zico", "meco", "zenas", "naswot", "naswot_sign"):
        for f in ("relu", "x2", "x3px", "x2px"):
            e = d.get(f"{f}/{p}")
            if e and "rho" in e:
                put(f"{tag}_{p}_{f}", s2(e["rho"]))
                put(f"{tag}_{p}_{f}_lo", s2(e["ci95"][0])); put(f"{tag}_{p}_{f}_hi", s2(e["ci95"][1]))
                put(f"{tag}_{p}_{f}_n", e["n"])
                if "partial_rho" in e:
                    put(f"{tag}_{p}_{f}_partial", s2(e["partial_rho"]))
put("n_tests", len([k for k in A if "rho" in A[k]]) )
B = S["B_naswot"]
put("naswot_inf_poly", sum(B[f]["naswot_neg_inf"] for f in ("x2", "x3px", "x2px")))
put("n_poly", sum(B[f]["n"] for f in ("x2", "x3px", "x2px")))

# ----- Zen -----
fz = F["poly_norm_free"]; fr = F["relu_norm_free"]
put("zen_n", fz["n"]); put("zen_acc", s2(fz["rho_zen_acc"])); put("zen_depth", s2(fz["rho_zen_depth"])); put("zen_params", s2(fz["rho_zen_logparams"]))
put("zen_fail_top", f"{100*fz['failure_rate_top_zen_quartile']:.0f}"); put("zen_fail_bot", f"{100*fz['failure_rate_bottom_zen_quartile']:.0f}")
put("zen_fail_all", f"{100*fz['failure_rate_overall']:.0f}"); put("zen_acc_ok", s2(fz["rho_zen_acc_excluding_failed"])); put("zen_n_ok", fz["n_excluding_failed"])
put("zenr_n", fr["n"]); put("zenr_acc", s2(fr["rho_zen_acc"])); put("zenr_params", s2(fr["rho_zen_logparams"]))

# ----- search -----
put("K", D["K"]); put("splits", D["splits"])
for B_ in (20, 21, 22, 23):
    e = D[f"B<={B_}"]
    put(f"pool{B_}", e["pool"]); put(f"oracle{B_}", f"{e['oracle_best_acc']:.1f}")
    for m in ("joint", "near_only", "zk_cost_only", "random"):
        put(f"reg{B_}_{m}", f"{e[m]['mean_regret']:.2f}")
    put(f"lam0_{B_}", e["chosen_lambda_counts"].get("0.0", 0))
    jn = e["joint_vs_near"]; put(f"jvn{B_}", f"{jn['win']}/{jn['tie']}/{jn['loss']}")
    jr = e["joint_vs_random"]; put(f"jvr{B_}", f"{jr['win']}/{jr['tie']}/{jr['loss']}")
D5 = S["D_search_K5"]; put("K5", D5["K"])
for B_ in (20, 21, 22, 23):
    e5 = D5[f"B<={B_}"]
    for m in ("joint", "near_only", "random"):
        put(f"k5reg{B_}_{m}", f"{e5[m]['mean_regret']:.2f}")
    j5 = e5["joint_vs_near"]; put(f"k5jvn{B_}", f"{j5['win']}/{j5['tie']}/{j5['loss']}")
sh = [D[f"B<={b}"]["pool_by_act"].get("relu", 0) / D[f"B<={b}"]["pool"] for b in (20, 21, 22, 23)]
put("relu_share_min", f"{100*min(sh):.0f}"); put("relu_share_max", f"{100*max(sh):.0f}")
for B_ in (20, 21, 22, 23):
    nb = D[f"B<={B_}"]["near_top_K_by_act"]
    put(f"neartop{B_}_poly", sum(v for k, v in nb.items() if k != "relu"))
tot_lam0 = sum(D[f"B<={b}"]["chosen_lambda_counts"].get("0.0", 0) for b in (20, 21, 22, 23)); tot = 4 * D["splits"]
put("lam0_pct", f"{100*tot_lam0/tot:.0f}")
for B_ in (20, 21, 22, 23):
    e = D2[f"B<={B_}"]; p = e["point"]
    put(f"exceeds{B_}", f"{p['est_admits_but_exceeds']:.2f}"); put(f"rejected{B_}", f"{p['feasible_rejected_by_est']:.2f}")
    put(f"viol{B_}", f"{p['near_in_est_budget']['violation']:.1f}")
ex = [D2[f"B<={b}"]["point"]["est_admits_but_exceeds"] for b in (20, 21, 22, 23)]
rj = [D2[f"B<={b}"]["point"]["feasible_rejected_by_est"] for b in (20, 21, 22, 23)]
put("exceeds_min", f"{100*min(ex):.0f}"); put("exceeds_max", f"{100*max(ex):.0f}")
put("rejected_min", f"{100*min(rj):.0f}"); put("rejected_max", f"{100*max(rj):.0f}")

# ----- cost model -----
put("n_fit", E["n_fit"]); put("n_eval", E["n_eval"])
fe, he = E["fitted_eval"], E["hand_set_eval"]
put("fit_exact", fe["exact_logrows"]); put("fit_w1", fe["within1"]); put("fit_rho", f"{fe['spearman_logrows']:.2f}"); put("fit_rmse", f"{fe['log2_rmse_assignments']:.2f}")
put("hand_exact", he["exact_logrows"]); put("hand_w1", he["within1"]); put("hand_rho", f"{he['spearman_logrows']:.2f}"); put("hand_rmse", f"{he['log2_rmse_assignments']:.2f}")
for k, v in E["coef"].items():
    put(f"coef_{k}", f"{v:.2f}")
ce = E["assignments_ceil_log2_equals_logrows"]
for a in ("relu", "x2", "x2px", "x3px"):
    put(f"ceil_{a}", f"{ce[a]['equal']}/{ce[a]['n']}")
nonx3 = sum(ce[a]["equal"] for a in ("relu", "x2", "x2px")); nonx3n = sum(ce[a]["n"] for a in ("relu", "x2", "x2px"))
put("ceil_non_x3", f"{nonx3}/{nonx3n}")
fa = E["fitted_eval_by_act"]
for a in ("relu", "x2", "x2px", "x3px"):
    if a in fa:
        put(f"fit_{a}_exact", f"{fa[a]['exact_logrows']}/{fa[a]['n']}")

# ----- matched ReLU/x^2 pairs on trained networks -----
cal = {}
for l in open("results/trained_calibration.jsonl"):
    q = json.loads(l); cal[q["name"]] = q
pairs = []
for n, q in cal.items():
    if "_relu_" in n and q.get("logrows"):
        m = n.replace("_relu_", "_x2_")
        if m in cal and cal[m].get("logrows"):
            pairs.append((q["logrows"], cal[m]["logrows"], q["total_assignments"], cal[m]["total_assignments"]))
put("n_pairs", len(pairs)); put("pairs_same", sum(a == b for a, b, _, _ in pairs))
put("pairs_x2_larger", sum(b > a for a, b, _, _ in pairs)); put("pairs_x2_smaller", sum(b < a for a, b, _, _ in pairs))
put("pairs_ratio_med", f"{statistics.median(d / c for *_, c, d in pairs):.2f}")
x3 = [(cal[n]["logrows"]) for n in cal if "_x3px_none_" in n and cal[n].get("logrows") and (n.startswith("mlp_x3px") or n.startswith("cnn_x3px"))]
# trained ezkl timing (optional)
try:
    recs = [json.loads(l) for l in open("results/ezkl_timing_trained.jsonl")]
    put("timing_cpu", recs[0]["host"]["cpu"]); put("timing_ezkl", recs[0]["host"]["ezkl"])
    rr = [r for r in recs[1:] if r.get("name") and not r.get("error")]
    pr = [min(r["prove_s"]) for r in rr if r.get("prove_s")]
    put("t_n", len(rr)); put("t_n_prove", len(pr)); put("t_prove_min", f"{min(pr):.0f}"); put("t_prove_max", f"{max(r_ for r in rr if r.get('prove_s') for r_ in r['prove_s']):.0f}")
    mk = [x for r in rr for x in r["mock_s"]]; put("t_mock_min", f"{min(mk):.0f}"); put("t_mock_max", f"{max(mk):.0f}")
    put("t_logrows_min", min(r["logrows"] for r in rr)); put("t_logrows_max", max(r["logrows"] for r in rr))
    spread = max((max(r["mock_s"]) - min(r["mock_s"])) / statistics.median(r["mock_s"]) for r in rr); put("t_mock_spread", f"{100*spread:.0f}")
    qa = [r["quantized_acc"] for r in rr if r.get("quantized_acc")]
    put("t_q_total", sum(q["n"] for q in qa)); put("t_q_agree", sum(q["agree_with_float"] for q in qa))
    put("t_prove_logrows_max", max(r["logrows"] for r in rr if r.get("prove_s")))
except Exception:
    pass

tpl = open("paper/results_template.tex").read()
out = re.sub(r"@@([A-Za-z0-9_]+)@@", lambda m: V[m.group(1)] if m.group(1) in V else sys.exit(f"UNRESOLVED placeholder: {m.group(1)}"), tpl)
open("paper/results_auto.tex", "w").write(out)
json.dump(V, open("results/render_values.json", "w"), indent=0)
print("rendered paper/results_auto.tex;", len(V), "values")
