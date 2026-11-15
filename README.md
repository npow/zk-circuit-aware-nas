# zkml-bench: ZK-Circuit-Aware NAS

**ZK-Circuit-Aware NAS: A Cost Model, Benchmark, and Proxy Study for Provable Inference**
(paper source: `paper/main.tex`)

A cost model for ezkl circuit size evaluated against ezkl calibrations of trained networks, a CIFAR-10 NAS benchmark of
474 networks (ReLU, x^2, x^3+x, x^2+x activations; no/Layer/Group normalization; MLP, CNN, DWS-CNN, ResNet-style) with two
training seeds, and a study of zero-cost proxies on it. Everything is measured with ezkl 9.5.0 unless stated.

## What the repository reproduces

| Paper element | Script | Data |
|---|---|---|
| Sweep (accuracy, 2 seeds; proxies and cost estimate at seed-0 init) | `sweep_gpu.py` | `results/sweep/*.jsonl` |
| ezkl calibration of trained seed-0 networks with real CIFAR-10 inputs, with ONNX-op features | `calibrate_trained.py` | `results/trained_calibration.jsonl`, `results/trained_calibration_skipped.json` |
| ezkl timings (repeated) and quantized accuracy on trained networks | `time_ezkl.py` | `results/ezkl_timing_trained.jsonl` (used in the paper: 3 repeats up to logrows 21, 1 real proof at logrows 22); `results/ezkl_timing_trained_run1.jsonl` is an earlier run with fewer repeats and is not used |
| Full ezkl `settings.json` for 72 stratified trained networks (logrows accounting rule) | `calibrate_trained.py` with `DUMP_SETTINGS_DIR` | `results/settings_dump/` |
| 3-seed, 30- vs 100-epoch validation of 26 architectures | `sweep_gpu.py --seed {42,43,44} --epochs {30,100} --only ... --no-proxies` | `results/multiseed/`, `results/multiseed_summary.json` (`analyze_multiseed.py`) |
| Width-matched activation comparison, untrained (3 seeds) | `matched_width_activation_cost.py` | `results/matched_width_activation*.json` |
| All sweep statistics (correlations, partial correlations, search, cost-model fit and cross-validation, logrows rule) | `analyze_sweep.py` | `results/sweep_summary.json` |
| LaTeX tables and numbers in the prose | `make_tables.py`, `render_results.py` | `paper/tables/`, `paper/results_auto.tex` |

Regenerate the statistics and the tables from the released data (no GPU or ezkl needed):

```bash
python analyze_sweep.py --sweep results/sweep --calib results/trained_calibration.jsonl --settings results/settings_dump --out results/sweep_summary.json
python analyze_multiseed.py
python make_tables.py && python render_results.py
```

## Main findings

- **Cost model** (Eq. 1 in the paper): fitted by non-negative least squares on trained, calibrated networks and evaluated on a disjoint split and by cross-validation; about one assigned cell per multiply-accumulate and no per-node overhead. ezkl's `logrows` equals the maximum of four requirements (rows, constants, lookup table, range-check table) on all 72 dumped networks; the x^3+x anomaly is a range-check table spanning 2^22+1 values (needing logrows 23).
- **Negative result**: width-matched, ReLU and x^2 networks have the same calibrated logrows (untrained and trained). A claim in an earlier version that ReLU is much more expensive than x^2 is withdrawn.
- **Proxies**: NEAR predicts accuracy only on ReLU networks; a Zen-style score (our variant, not the published Zen-Score protocol) is anti-correlated; a pre-activation-sign NASWOT variant is negatively correlated on polynomial networks; ZiCo's success on x^3+x disappears when failed runs are removed. 39-60% of polynomial networks fail to train.
- **Search**: under ezkl-calibrated budgets a tuned joint proxy shows no consistent gain over NEAR-only (an earlier claim of a gain is withdrawn).

## Not included

Trained weights (about 0.7 GB) and CIFAR-100 / ImageNet-16-120 sweeps (withdrawn: the files could not be recovered). `results/gpu_validation_summary.json` summarizes an earlier three-seed job whose raw output was not retained; the paper no longer uses it (the validation was rerun, `results/multiseed/`). The `results/legacy/` directory holds files from an earlier version of this work; they are superseded.
earlier version of this work; they are superseded and are not used by the current paper.

## Notes

- `results/proxy_vs_ezkl*.json`: an earlier comparison of the proxy with ezkl on untrained networks (73 architectures); superseded by the trained-network calibration above.
- The `--clamp` option of `matched_width_activation_cost.py` mirrors `_wrap_poly_model_for_export`, which only wraps classes whose name contains "activation"; the sweep's `Square`, `CubicPlusLinear` and `SquarePlusLinear` do not match, so both settings produce identical, unclamped exports.
- `sweep_gpu.py` keeps the original training recipe but holds the dataset and augmentation on the GPU; the polynomial MLP x^2+x scores about 3 points higher than in the earlier CPU pipeline.

## Install

```bash
pip install -e ".[dev,ezkl]"
```

`docs/` contains notes from earlier versions of this work (including `docs/theory.md` with proofs that are no longer in the paper); the paper is authoritative.
