"""Generate paper/tables_auto.tex from the result files so no table number is typed by hand.

Reads: results/sweep_summary.json, results/trained_calibration.jsonl,
       results/ezkl_timing_trained.jsonl (optional), results/sweep_gpu/*.jsonl (for matched-width trained table)
Writes: paper/tables_auto.tex  (\\input from main.tex)
"""
from __future__ import annotations

import glob, json, math, statistics, sys

S = json.load(open("results/sweep_summary.json"))
OUT = []


def fmt(x, d=2, sign=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def tex(s):
    return s.replace("_", r"\_")


# ---------------- proxy correlation table ----------------
A = S["A_proxy_vs_accuracy"]
names = {"near": "NEAR", "synflow": "Synflow", "zico": "ZiCo", "meco": "MeCo", "zenas": "Zen-style (variant)",
         "naswot": "NASWOT (ReLU patterns)", "naswot_sign": "NASWOT (pre-activation signs)"}
fams = [("relu", "ReLU"), ("x2", r"$x^2$"), ("x3px", r"$x^3{+}x$"), ("x2px", r"$x^2{+}x$")]
n_by = S["n_by_act"]
rows = []
for p, label in names.items():
    cells = []
    for f, _ in fams:
        e = A.get(f"{f}/{p}")
        if not e or "rho" not in e:
            cells.append("n/a")
            continue
        star = r"^{*}" if e.get("p_holm", 1) < 0.05 else ""
        ci = e["ci95"]
        fam_n = A.get(f"{f}/near", {}).get("n", e["n"])
        nn = rf" {{\scriptsize($n{{=}}{e['n']}$)}}" if e["n"] != fam_n else ""
        cells.append(rf"${e['rho']:+.2f}{star}$ {{\scriptsize[{ci[0]:+.2f},{ci[1]:+.2f}]}}{nn}")
    rows.append(f"{label} & " + " & ".join(cells) + r" \\")
hdr = " & ".join([r"\textbf{Proxy}"] + [rf"\textbf{{{l}}} ($n{{=}}{n_by.get(f, 0)}$)" for f, l in fams])
OUT.append(r"""\begin{table}[h]
\centering
\caption{Spearman $\rho$ between proxy score at initialization and mean top-1 accuracy over two training seeds
  (CIFAR-10, regenerated sweep), per activation family, with $95\%$ bootstrap CI over architectures.
  ${}^{*}$ marks Holm-adjusted $p<0.05$ over the """ + str(len([k for k in A if "p_holm" in A[k]])) + r""" activation-family tests in this table.
  A dash or ``n/a'' means fewer than $8$ finite scores.}
\label{tab:corr}
\small
\begin{tabular}{lrrrr}
\toprule
""" + hdr + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# partial correlation table
rows = []
for p, label in names.items():
    cells = []
    for f, _ in fams:
        e = A.get(f"{f}/{p}")
        if not e or "partial_rho" not in e:
            cells.append("--")
        else:
            star = r"^{*}" if e.get("partial_p", 1) < 0.05 else ""
            cells.append(rf"${e['partial_rho']:+.2f}{star}$")
    rows.append(f"{label} & " + " & ".join(cells) + r" \\")
OUT.append(r"""\begin{table}[h]
\centering
\caption{Partial Spearman correlation of proxy and accuracy after removing, by rank regression, the effects of
  architecture type (MLP, CNN, DWS-CNN, ResNet-style), normalization and $\log$ parameter count.  ${}^{*}$: unadjusted
  $p<0.05$ for the partial correlation (descriptive only; the Holm-adjusted tests are in Table~\ref{tab:corr}).}
\label{tab:partial}
\small
\begin{tabular}{lrrrr}
\toprule
\textbf{Proxy} & \textbf{ReLU} & $\bm{x^2}$ & $\bm{x^3{+}x}$ & $\bm{x^2{+}x}$ \\
\midrule
""" + "\n".join(rows).replace(r"\bm", r"\mathbf") + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- correlations excluding failed training runs ----------------
A2 = S.get("A2_excluding_failed", {})
if A2:
    rows = []
    for p_, label in names.items():
        cells = []
        for f, _ in fams:
            e = A2.get(f"{f}/{p_}")
            if not e or "rho" not in e:
                cells.append("n/a")
                continue
            star = r"^{*}" if e.get("p_holm", 1) < 0.05 else ""
            ci = e["ci95"]
            fam_n = A2.get(f"{f}/near", {}).get("n", e["n"]); nn = rf" {{\scriptsize($n{{=}}{e['n']}$)}}" if e["n"] != fam_n else ""
            cells.append(rf"${e['rho']:+.2f}{star}$ {{\scriptsize[{ci[0]:+.2f},{ci[1]:+.2f}]}}{nn}")
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    nf = S["n_failed_by_act"]
    hdr2 = " & ".join([r"\textbf{Proxy}"] + [rf"\textbf{{{l}}} ($n{{=}}{A2.get(f+'/near', {}).get('n', '?')}$)" for f, l in fams])
    OUT.append(r"""\begin{table}[h]
\centering
\caption{As Table~\ref{tab:corr} after excluding networks whose training diverged or ended at chance level
  (mean top-1 $<15\%$; """ + ", ".join(f"{nf[f]} of {n_by[f]} {l}" for f, l in fams) + r""" excluded).  Holm adjustment is applied within this table.}
\label{tab:corrnofail}
\small
\begin{tabular}{lrrrr}
\toprule
""" + hdr2 + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- normalized networks only (identical training recipe for all activations) ----------------
A3 = S.get("A3_normalized_only_same_recipe", {})
if A3:
    rows = []
    for p_, label in names.items():
        cells = []
        for f, _ in fams:
            e = A3.get(f"{f}/{p_}")
            if not e or "rho" not in e:
                cells.append("n/a")
                continue
            star = r"^{*}" if e.get("p_holm", 1) < 0.05 else ""
            ci = e["ci95"]
            fam_n = A3.get(f"{f}/near", {}).get("n", e["n"]); nn = rf" {{\scriptsize($n{{=}}{e['n']}$)}}" if e["n"] != fam_n else ""
            cells.append(rf"${e['rho']:+.2f}{star}$ {{\scriptsize[{ci[0]:+.2f},{ci[1]:+.2f}]}}{nn}")
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    hdr3 = " & ".join([r"\textbf{Proxy}"] + [rf"\textbf{{{l}}} ($n{{=}}{A3.get(f+'/near', {}).get('n', '?')}$)" for f, l in fams])
    OUT.append(r"""\begin{table}[h]
\centering
\caption{As Table~\ref{tab:corr}, restricted to networks with LayerNorm or GroupNorm.  All activations in this subset were trained with the same
  recipe (lr $0.1$), removing the learning-rate difference that applies to norm-free polynomial networks (lr $0.01$ with clipping).
  Holm adjustment is applied within this table.}
\label{tab:corrsame}
\small
\begin{tabular}{lrrrr}
\toprule
""" + hdr3 + r""" \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- NASWOT table ----------------
B = S["B_naswot"]
rows = [f"{lab} & {B[f]['n']} & {B[f]['naswot_neg_inf']} & {B[f]['naswot_sign_neg_inf']} \\\\" for f, lab in fams if f in B]
OUT.append(r"""\begin{table}[h]
\centering
\caption{Networks with a $-\infty$ NASWOT score: standard implementation (scores \texttt{nn.ReLU} outputs) versus the
  pre-activation-sign variant.}
\label{tab:naswot}
\small
\begin{tabular}{lrrr}
\toprule
\textbf{Family} & $n$ & \textbf{standard $-\infty$} & \textbf{sign variant $-\infty$} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- search table ----------------
D = S.get("D_search")
if D:
    rows = []
    for B_ in (20, 21, 22, 23):
        e = D.get(f"B<={B_}")
        if not e:
            continue
        if "note" in e:
            rows.append(rf"$\le{B_}$ & {e['pool']} & \multicolumn{{5}}{{l}}{{{e['note']}}} \\")
            continue
        def c(m):
            x = e[m]
            return rf"{x['mean_regret']:.2f} {{\scriptsize[{x['ci95'][0]:.2f},{x['ci95'][1]:.2f}]}}"
        jn, jr = e["joint_vs_near"], e["joint_vs_random"]
        rows.append(rf"$\le{B_}$ & {e['pool']} & {e['oracle_best_acc']:.1f} & {c('joint')} & {c('near_only')} & {c('zk_cost_only')} & {c('random')} & {jn['win']}/{jn['tie']}/{jn['loss']} \\")
    OUT.append(r"""\begin{table}[h]
\centering
\caption{Budget-constrained search under \emph{ezkl-calibrated} budgets ($\logrows\le B$ from calibration of trained
  networks).  Each of """ + str(D["splits"]) + r""" random $50/50$ splits of the pool tunes $\lambda\in\{0,0.25,0.5,1,2,4\}$ for the joint
  proxy on one half and evaluates all methods on the other, choosing $K{=}""" + str(D["K"]) + r"""$ architectures to train; the outcome is the
  best accuracy among them.  Entries are mean regret in accuracy points against the best architecture in the
  evaluation half, with the $2.5$--$97.5$ percentile range over splits; Random averages $50$ draws per split.  Last
  column: splits where Joint beats / ties / loses to NEAR-only (regret).}
\label{tab:search}
\small
\begin{tabular}{rrrrrrrr}
\toprule
\textbf{$\logrows$} & \textbf{pool} & \textbf{oracle} & \textbf{Joint} & \textbf{NEAR-only} & \textbf{ZK-cost-only} & \textbf{Random} & \textbf{J v N} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- untuned regret per lambda (paired differences) ----------------
if D:
    lams = D["lambda_grid"]
    D5_ = S.get("D_search_K5")
    def block(Dx):
        rows = []
        for B_ in (20, 21, 22, 23):
            e = Dx.get(f"B<={B_}", {})
            if "paired_vs_lambda0" not in e:
                continue
            pv = e["paired_vs_lambda0"]
            cells = []
            for l in lams[1:]:
                x = pv[str(l)]
                cells.append(rf"{x['mean']:+.2f} {{\scriptsize[{x['ci95'][0]:+.2f},{x['ci95'][1]:+.2f}]}} {{\scriptsize({100*x['better_frac']:.0f}/{100*x['worse_frac']:.0f})}}")
            rows.append(rf"$\le{B_}$ & {e['pool']} & {e['regret_by_lambda_untuned']['0.0']:.2f} & " + " & ".join(cells) + r" \\")
        return rows
    OUT.append(r"""\begin{table}[h]
\centering
\caption{Untuned test of the joint proxy.  For each \emph{fixed} $\lambda>0$, the per-split paired difference in regret (accuracy points) relative to $\lambda=0$ (NEAR-only) on the evaluation halves: mean, $2.5$--$97.5$ percentile range over """ + str(D["splits"]) + r""" splits, and in parentheses the percentage of splits in which
  it is better / worse (negative difference = better than NEAR-only).  The third column is NEAR-only's mean regret.  Upper block $K=""" + str(D["K"]) + r"""$, lower block $K=""" + str(D5_["K"]) + r"""$.}
\label{tab:lambda}
\scriptsize
\begin{tabular}{rrr""" + "r" * (len(lams) - 1) + r"""}
\toprule
$B$ & pool & $\lambda{=}0$ & """ + " & ".join(rf"$\lambda{{=}}{l:g}$" for l in lams[1:]) + r""" \\
\midrule
""" + "\n".join(block(D)) + r"""
\midrule
""" + "\n".join(block(D5_)) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- logrows accounting rule ----------------
G = S.get("G_logrows_rule")
if G:
    rows = []
    for a, lab in (("relu", "ReLU"), ("x2", r"$x^2$"), ("x2px", r"$x^2{+}x$"), ("x3px", r"$x^3{+}x$")):
        e = G["by_act"][a]
        if e["n"]:
            rows.append(rf"{lab} & {e['n']} & {e['exact']} \\")
    rows.append(rf"\textbf{{all}} & {G['n']} & {G['exact']} \\")
    bc = G["binding_counts"]
    OUT.append(r"""\begin{table}[h]
\centering
\caption{Test of the four-resource rule $\logrows=\max(\lceil\log_2\text{rows}\rceil,\lceil\log_2\text{const}\rceil,\lceil\log_2\text{lookup span}\rceil,\lceil\log_2\text{range-check span}\rceil)$
  on ezkl's dumped \texttt{settings.json} for """ + str(G["n"]) + r""" trained networks (a random sample of up to $30$ per activation among networks under $2.2$M assigned cells, plus $11$ chosen earlier; $\logrows$ range """ + f"{G['logrows_range'][0]}--{G['logrows_range'][1]}" + r""" ).  The binding resource (ties counted for each) was
  rows in """ + str(bc.get("rows", 0)) + r""", constants in """ + str(bc.get("const", 0)) + r""", lookup table in """ + str(bc.get("lookup", 0)) + r""" and range-check table in """ + str(bc.get("range_check", 0)) + r""" networks.}
\label{tab:accounting}
\small
\begin{tabular}{lrr}
\toprule
\textbf{Family} & $n$ & \textbf{rule exact} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- estimated-budget filtering ----------------
D2 = S.get("D2_estimated_budget")
if D2:
    rows = []
    for B_ in (20, 21, 22, 23):
        e = D2[f"B<={B_}"]; pt = e["point"]; ci = e["ci95"]
        def cc(k, d=2):
            return rf"{{\scriptsize[{ci[k][0]:.{d}f},{ci[k][1]:.{d}f}]}}"
        rows.append(rf"$\le{B_}$ & {pt['n_cal_feasible']} & {pt['n_est_pool']} & {pt['est_admits_but_exceeds']:.2f} {cc('est_admits_but_exceeds')} & {pt['feasible_rejected_by_est']:.2f} {cc('feasible_rejected_by_est')} & {pt['near_in_est_budget']['violation']:.1f} {cc('near_in_est_budget/violation',1)} & {pt['near_no_budget']['violation']:.1f} \\")
    OUT.append(r"""\begin{table}[h]
\centering
\caption{Filtering by the cost model's \emph{estimated} $\logrows$ (no ezkl run) against the \emph{calibrated} budget, on the """ + str(D2["n_pool"]) + r""" calibrated trained networks.
  ``feasible'': calibrated $\logrows\le B$.  ``admitted'': estimated $\logrows\le B$.  ``Exceeds'': share of admitted networks whose calibrated $\logrows>B$.
  ``Rejected'': share of feasible networks the estimate rejects.  ``Violation'': share of the $K{=}10$ highest-NEAR networks, chosen inside the estimated budget
  (or with no budget), whose calibrated $\logrows>B$.  $95\%$ bootstrap intervals over networks.}
\label{tab:estbudget}
\small
\begin{tabular}{rrrrrrr}
\toprule
$B$ & \textbf{feasible} & \textbf{admitted} & \textbf{exceeds} & \textbf{rejected} & \textbf{violation (est.\ budget)} & \textbf{violation (no budget)} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# ---------------- cost-model fit ----------------
E = S.get("E_cost_model")
if E:
    def mrow(label, m):
        return rf"{label} & {m['n']} & {m['exact_logrows']} & {m['within1']} & {m['spearman_logrows']:.2f} & {m['log2_rmse_assignments']:.2f} \\"
    rows = [mrow("Hand-set proxy, evaluation split", E["hand_set_eval"]), mrow("Fitted (NNLS), evaluation split", E["fitted_eval"]),
            mrow("Fitted (NNLS), fit split", E["fitted_fit_split"])]
    for a, lab in (("relu", "ReLU"), ("x2", r"$x^2$"), ("x2px", r"$x^2{+}x$"), ("x3px", r"$x^3{+}x$")):
        if a in E.get("fitted_eval_by_act", {}):
            rows.append(mrow(rf"\quad fitted, eval, {lab}", E["fitted_eval_by_act"][a]))
    cv = E.get("cross_validation")
    if cv:
        rows.append(rf"5-fold CV, all {cv['n']} networks: hand-set & {cv['n']} & {100*cv['hand']['exact']:.0f}\% & {100*cv['hand']['within1']:.0f}\% & {cv['hand']['rho']:.2f} & -- \\")
        rows.append(rf"5-fold CV, all {cv['n']} networks: fitted & {cv['n']} & {100*cv['fitted']['exact']:.0f}\% & {100*cv['fitted']['within1']:.0f}\% & {cv['fitted']['rho']:.2f} & -- \\")
        for t in ("resnet", "dws_resnet", "mlp", "cnn", "dws_cnn"):
            k = "fitted/" + t
            if k in cv["by_type"]:
                rows.append(rf"\quad CV fitted, {t.replace('_', '-')} & {cv['by_type'][k]['n']} & {100*cv['by_type'][k]['exact']:.0f}\% & -- & -- & -- \\")
    coefs = ", ".join(rf"{tex(k)}={v:.3g}" for k, v in E["coef"].items())
    OUT.append(r"""\begin{table}[h]
\centering
\caption{Cost model of Eq.~\eqref{eq:cost} against ezkl-calibrated $\logrows$ on \emph{trained} networks with real inputs.
  Coefficients are fitted by non-negative least squares (relative-error weighting) on """ + str(E["n_fit"]) + r""" networks and evaluated on the
  """ + str(E["n_eval"]) + r""" networks of the earlier validation set, which are excluded from the fit.  ``exact'' and ``within 1'' count
  networks whose predicted $\lceil\log_2 C\rceil$ equals or is within one of the calibrated $\logrows$; $\rho$ is Spearman between predicted and
  calibrated $\logrows$; RMSE is in $\log_2$ of assigned cells.}
\label{tab:costfit}
\small
\begin{tabular}{lrrrrr}
\toprule
 & $n$ & \textbf{exact} & \textbf{within 1} & $\rho$ & \textbf{RMSE} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}

\smallskip
{\scriptsize Fitted coefficients (cells per MAC / per node / per element): """ + coefs + r""".}
\end{table}
""")

# ---------------- ezkl timings (trained) ----------------
try:
    recs = [json.loads(l) for l in open("results/ezkl_timing_trained.jsonl")]
    host = recs[0]["host"]; recs = [r for r in recs[1:] if r.get("name")]
    sw = {}
    for f in glob.glob("results/sweep/cifar10_seed*_shard*.jsonl"):
        for l in open(f):
            r = json.loads(l)
            if "top1" in r:
                sw.setdefault(r["name"], []).append(r["top1"])
    rows = []
    def ms(v, d=1):
        if not v: return "--"
        return f"{statistics.median(v):.{d}f}" + (rf" {{\scriptsize[{min(v):.{d}f},{max(v):.{d}f}]}}" if len(v) > 1 else "")
    for r in recs:
        acc = f"{statistics.mean(sw[r['name']]):.1f}" if r["name"] in sw else "--"
        qa = r.get("quantized_acc")
        q = f"{qa['agree_with_float']}/{qa['n']}" if qa else "--"
        rows.append(rf"{tex(r['name'])} & {acc} & {r.get('logrows','--')} & {ms(r.get('mock_s'))} & {ms(r.get('prove_s'))} & {ms(r.get('verify_s'),2)} & {(f"{r['proof_bytes']/1024:.1f}" if r.get('proof_bytes') else "--")} & {q} \\" if not r.get("error") else rf"{tex(r['name'])} & {acc} & -- & \multicolumn{{5}}{{l}}{{{tex(r['error'][:60])}}} \\")
    OUT.append(r"""\begin{table}[h]
\centering
\caption{ezkl timings on trained seed-0 networks with real CIFAR-10 inputs under one recorded configuration
  (ezkl """ + host["ezkl"] + r""", """ + host["cpu"].replace("&", r"\&") + r""", """ + str(host["logical_cores"]) + r""" logical cores, shared host).  Median [min,max] over
  repeats; seconds.  Mock proofs: $\times3$.  Real proofs: $\times3$ for $\logrows\le21$, $\times1$ for $\logrows=22$.  Last column: quantized-circuit predictions that agree with the float model on real test images.}
\label{tab:ezkl}
\scriptsize
\begin{tabular}{lrrrrrrr}
\toprule
\textbf{Architecture} & \textbf{top-1 (2 seeds)} & $\logrows$ & \textbf{mock} & \textbf{prove} & \textbf{verify} & \textbf{proof KB} & \textbf{q-agree} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
except FileNotFoundError:
    pass

# ---------------- multi-seed validation (released rerun) ----------------
try:
    M = json.load(open("results/multiseed_summary.json"))
    rows = []
    for n, e in M["per_arch"].items():
        def f_(x):
            return "--" if x["mean"] is None else (rf"${x['mean']:.2f}\mathbin{{\pm}}{x['sd']:.2f}$" if x["sd"] is not None else f"${x['mean']:.2f}$")
        rows.append(rf"{tex(n)} & {f_(e['30'] if '30' in e else e[30])} & {f_(e['100'] if '100' in e else e[100])} \\")
    OUT.append(r"""\begin{table}[h]
\centering
\caption{Three-seed (42, 43, 44) test top-1 (mean $\pm$ sample SD) after 30 and after 100 epochs (a fresh cosine schedule for each budget) for the """ + str(M["n_archs"]) + r""" architectures of the earlier validation, rerun with
  \texttt{sweep\_gpu.py}; per-run outputs are released.  Runs that diverged are included at their (chance-level) accuracy.}
\label{tab:robustness}
\scriptsize
\begin{tabular}{lrr}
\toprule
\textbf{Architecture} & \textbf{30 epochs} & \textbf{100 epochs} \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
except FileNotFoundError:
    pass

# ---------------- width-matched, trained networks ----------------
try:
    cal = {}
    for l in open("results/trained_calibration.jsonl"):
        r = json.loads(l); cal[r["name"]] = r
    skel = [("mlp_{a}_none_128x64x32", "MLP 128-64-32"), ("mlp_{a}_none_256x128", "MLP 256-128"), ("mlp_{a}_none_512x256", "MLP 512-256"),
            ("cnn_{a}_none_16_32", "CNN 16-32"), ("cnn_{a}_none_16_32_64", "CNN 16-32-64")]
    rows, ok = [], 0
    for pat, lab in skel:
        cells = []
        for a in ("relu", "x2", "x2px", "x3px"):
            r = cal.get(pat.format(a=a))
            if r and r.get("logrows") is not None:
                cells.append(f"{r['total_assignments']/1e6:.3f}M ({r['logrows']})"); ok += 1
            else:
                cells.append("fail" if r else "--")
        rows.append(f"{lab} & " + " & ".join(cells) + r" \\")
    OUT.append(r"""\begin{table}[h]
\centering
\caption{Width-matched ezkl circuit size on \emph{trained} seed-0 networks calibrated with real CIFAR-10 inputs: assigned cells (calibrated $\logrows$).
  ``fail'': calibration did not complete or errored (recorded in the released file).}
\label{tab:matchedtrained}
\small
\begin{tabular}{lrrrr}
\toprule
\textbf{Skeleton} & \textbf{ReLU} & $\mathbf{x^2}$ & $\mathbf{x^2{+}x}$ & $\mathbf{x^3{+}x}$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
except FileNotFoundError:
    pass

import os, re
os.makedirs("paper/tables", exist_ok=True)
for old in glob.glob("paper/tables/*.tex"):
    os.remove(old)
for t in OUT:
    lab = re.search(r"\\label\{(tab:[^}]+)\}", t).group(1).replace("tab:", "")
    open(f"paper/tables/{lab}.tex", "w").write(t)
print("wrote", len(OUT), "tables to paper/tables/:", [re.search(r"\\label\{tab:([^}]+)\}", t).group(1) for t in OUT])
