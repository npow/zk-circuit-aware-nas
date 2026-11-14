# zkml-bench: ZK-Circuit-Aware NAS

**ZK-Circuit-Aware NAS: A Cost Model, Benchmark, and Proxy Study for Provable Inference**
(paper source: `paper/main.tex`)

A structural cost model for zero-knowledge circuit size, a NAS benchmark spanning ReLU and
polynomial activations, and a study of zero-cost proxies on it. All numbers below are
measured with ezkl 9.5.0 unless stated.

## Results (what the repository can reproduce)

| Result | Value | Script / file |
|---|---|---|
| Proxy-estimated vs ezkl-calibrated logrows, 73 architectures (untrained weights) | exact 49/73, within one 66/73, Spearman 0.80 (95% CI 0.65-0.91); x^3+x family only 0.28 | `validate_proxy_vs_ezkl.py`, `analyze_proxy_vs_ezkl.py`, `results/proxy_vs_ezkl*.json` |
| Width-matched ReLU vs x^2 circuit size (5 skeletons x 3 seeds) | same calibrated logrows in all 15; x^2 has 0.2-6% more assigned cells; seeds identical | `matched_width_activation_cost.py`, `results/matched_width_activation*.json` |
| x^3+x networks | calibrate to logrows 23 in all 5 unnormalized skeletons x 3 seeds (assignments unchanged); with LayerNorm/GroupNorm they range 21-26 in the 73-architecture set | same |
| Real proving, 3 architectures | 47-58s (small polynomial MLPs), 1340s (DWS ReLU, logrows 21); single runs | `measure_verify.py`, `results/prove_verify_times.json` |
| 30 vs 100 epoch ranking, 26 architectures x 3 seeds | Spearman 0.932; selected 75.0% pick reproduces at 74.73+-1.61% | `gpu_validation_experiment.py`, `results/gpu_validation_summary.json` |

A claim in an earlier version of this work, that ReLU is much more expensive than x^2 in
ezkl, is **not supported** by the width-matched measurement above and has been withdrawn.

## Not included

The per-architecture sweep files behind the proxy-correlation tables (n=335 CIFAR-10,
n=84 CIFAR-100, ImageNet-16-120) and the budget-search table, and the raw output of the
multi-seed GPU job, are not in this repository. `generate_sweep.py` generates the sweep
configurations; `results/proxy_sweep_correlations.json` holds the CIFAR-10 correlations.

## Quick start

```bash
pip install -e ".[dev,ezkl]"
python validate_proxy_vs_ezkl.py --n 64 --workers 4 --out results/proxy_vs_ezkl.json
python matched_width_activation_cost.py --out results/matched_width_activation.json
python matched_width_activation_cost.py --clamp none --out results/matched_width_activation_noclamp.json
python matched_width_activation_cost.py --seed 1 --out results/matched_width_activation_seed1.json   # and --seed 2
python analyze_proxy_vs_ezkl.py
```

## Structure

```
src/zkml_bench/metrics/zk_proxy.py        structural ZK cost proxy
src/zkml_bench/nas/zero_cost_proxies.py   NEAR, Synflow, ZiCo, MeCo, ZenNAS, NASWOT
generate_sweep.py                         architecture sweep generator
validate_proxy_vs_ezkl.py                 proxy vs ezkl-calibrated logrows
matched_width_activation_cost.py          width-matched activation comparison
docs/                                     working notes; the paper is authoritative where they differ
```

## Notes on the matched-width experiment

The `--clamp` option mirrors `_wrap_poly_model_for_export`, which only wraps classes whose
name contains "activation"; the sweep's `Square`, `CubicPlusLinear` and `SquarePlusLinear`
classes do not match, so both settings produce identical, unclamped exports.
