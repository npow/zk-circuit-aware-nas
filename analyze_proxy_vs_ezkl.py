"""Summarize results/proxy_vs_ezkl.json and results/matched_width_activation*.json.
Writes results/proxy_vs_ezkl_summary.json.  Pure Python (bootstrap CI, seed 0)."""
import collections, json, random

def rank(x):
    s = sorted(range(len(x)), key=lambda i: x[i]); r = [0.0] * len(x); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and x[s[j + 1]] == x[s[i]]: j += 1
        for k in range(i, j + 1): r[s[k]] = (i + j) / 2 + 1
        i = j + 1
    return r

def spearman(a, b):
    ra, rb = rank(a), rank(b); n = len(a); ma = sum(ra) / n; mb = sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra); vb = sum((y - mb) ** 2 for y in rb)
    return cov / (va * vb) ** 0.5 if va and vb else float("nan")

def boot_ci(a, b, n=2000, seed=0):
    rng = random.Random(seed); idx = range(len(a)); v = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        r = spearman([a[i] for i in s], [b[i] for i in s])
        if r == r: v.append(r)
    v.sort(); return [v[int(0.025 * len(v))], v[int(0.975 * len(v))]]

d = json.load(open("results/proxy_vs_ezkl.json"))
ok = [r for r in d if r.get("ezkl_logrows") is not None and r.get("proxy_logrows") is not None]
out = {"n_total": len(d), "n_calibrated": len(ok), "n_failed": len(d) - len(ok)}
p = [r["proxy_logrows"] for r in ok]; e = [r["ezkl_logrows"] for r in ok]
diff = [a - b for a, b in zip(p, e)]
out["exact"] = sum(x == 0 for x in diff); out["within1"] = sum(abs(x) <= 1 for x in diff)
out["diff_histogram"] = dict(sorted(collections.Counter(diff).items()))
out["spearman_all"] = spearman(p, e); out["spearman_all_ci95"] = boot_ci(p, e)
out["by_activation"] = {}
for act in sorted({r["act"] for r in ok}):
    s = [r for r in ok if r["act"] == act]
    sp_, se_ = [r["proxy_logrows"] for r in s], [r["ezkl_logrows"] for r in s]
    dd = [a - b for a, b in zip(sp_, se_)]
    out["by_activation"][act] = {"n": len(s), "exact": sum(x == 0 for x in dd),
                                 "within1": sum(abs(x) <= 1 for x in dd),
                                 "spearman": spearman(sp_, se_), "ci95": boot_ci(sp_, se_)}
# x^3+x: calibrated logrows by normalization (token after the activation token in the name)
x3 = collections.defaultdict(list)
for r in ok:
    if r["act"] == "x3px":
        toks = r["name"].split("_"); i = toks.index("x3px")
        x3[toks[i + 1]].append(r["ezkl_logrows"])
out["x3px_logrows_by_norm"] = {k: sorted(v) for k, v in x3.items()}
mw = {}
for tag, f in (("default", "results/matched_width_activation.json"), ("noclamp", "results/matched_width_activation_noclamp.json")):
    try: mw[tag] = json.load(open(f))
    except FileNotFoundError: pass
rows = {}
for r in mw.get("default", []): rows[(r["skeleton"], r["act"])] = r
base = collections.defaultdict(dict)
for (sk, act), r in rows.items(): base[sk][act] = r
rel = []
for sk, acts in base.items():
    if "relu" in acts:
        for act in ("x2", "x2px", "x3px"):
            if act in acts:
                rel.append({"skeleton": sk, "act": act,
                            "rows_vs_relu": acts[act]["num_rows"] / acts["relu"]["num_rows"],
                            "logrows_relu": acts["relu"]["logrows"], "logrows_act": acts[act]["logrows"]})
out["matched_width_vs_relu"] = rel
json.dump(out, open("results/proxy_vs_ezkl_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "matched_width_vs_relu"}, indent=1))
for r in rel: print(r["skeleton"].ljust(14), r["act"].ljust(5), "rows/relu = %.3f" % r["rows_vs_relu"], "logrows", r["logrows_relu"], "->", r["logrows_act"])
