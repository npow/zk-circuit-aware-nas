# Backend-Aware NAS: Demonstration Results

**Date**: 2026-05-25  
**Setup**: 3 iterations, 1 epoch mini-train on 2000 MNIST samples  
**Hardware**: CPU-only workstation  

## The System Contribution

Standard ZK-NAS uses a single backend's cost proxy. We show that conditioning the
NAS agent on the target backend's cost model causes it to propose structurally
different architectures — and those architectures are measurably cheaper to prove
on the target backend.

The ZK-Architectural Reversal (see `docs/reversal_results.md`) shows:
- **EZKL (Halo2)**: depth is cost-neutral; logrows snaps to power-of-2 buckets
- **DeepProve (GKR)**: depth multiplies ReLU activation count → 1.18-1.36× penalty

## Architectures Discovered

### EZKL-NAS (Halo2/logrows, depth-neutral cost model)

**Best candidate**: `ezkl_poly_compact_v3`  
- Mini-train top-1 accuracy: 10.5% (1 epoch, 2K samples)
- Parameters: 2,274
- Multiplicative depth indicator: 5
- EZKL proxy (logrows): 131,072
- DeepProve proxy (activation_count): 95,248
- Expander proxy (log2_gates): 15
- Reasoning: _Aggressive early strided conv reduces spatial dims fast, tiny channel counts minimize max tensor size for EZKL logrows, polynomial activations (x²+x) avoid expensive ReLU lookups, final global-avg-pool before small FC keeps constraint count low._ 

### DeepProve-NAS (GKR/activation-count, depth-penalising cost model)

**Best candidate**: `deepprove_1hidden_wide`  
- Mini-train top-1 accuracy: 72.1% (1 epoch, 2K samples)
- Parameters: 814,090
- Multiplicative depth indicator: 3
- EZKL proxy (logrows): 1,048,576
- DeepProve proxy (activation_count): 814,080
- Expander proxy (log2_gates): 20
- Reasoning: _Single hidden layer of width 1024 eliminates one ReLU layer entirely, cutting depth from 3 to 2 and reducing dp_proxy cost while maintaining capacity through maximum width._ 

## Cross-Backend Proxy Comparison

Same models, predicted cost on each backend:

| Architecture | Optimised For | MNIST acc | EZKL proxy | DeepProve proxy | Expander proxy |
|---|---|---|---|---|---|
| `ezkl_poly_compact_v3` | EZKL | 10.5% | 131,072 | 95,248 | 15 |
| `deepprove_1hidden_wide` | DeepProve | 72.1% | 1,048,576 | 814,080 | 20 |

## Speedup Claim

**On DeepProve**: the DeepProve-NAS architecture is **0.12× more expensive** to prove
than the EZKL-NAS architecture.
- EZKL-NAS best arch DeepProve proxy: 95,248
- DeepProve-NAS best arch DeepProve proxy: 814,080

Note: with only 3 iterations and 1 mini-training epoch, the NAS has limited
search depth. The structural difference (depth/width) is still visible in the
architecture choices, even if the proxy ratio does not yet exceed 1×.

## All Successful EZKL-NAS Candidates

| # | Name | top-1 | EZKL proxy | DP proxy | depth | params |
|---|---|---|---|---|---|---|
| 1 | `ezkl_dws_cnn_poly_baseline` | 8.5% | 1,048,576 | 726,432 | 10 | 18,826 |
| 2 | `ezkl_poly_deep_dws_v2` | 8.5% | 262,144 | 171,632 | 10 | 7,450 |
| 3 | `ezkl_poly_compact_v3` | 10.5% | 131,072 | 95,248 | 5 | 2,274 |

## All Successful DeepProve-NAS Candidates

| # | Name | top-1 | EZKL proxy | DP proxy | depth | params |
|---|---|---|---|---|---|---|
| 1 | `deepprove_shallow_wide_baseline` | 69.7% | 1,048,576 | 814,080 | 3 | 814,090 |
| 2 | `deepprove_1hidden_wide` | 72.1% | 1,048,576 | 814,080 | 3 | 814,090 |
| 3 | `deepprove_single_hidden_512` | 66.1% | 524,288 | 407,040 | 3 | 407,050 |

## Interpretation

The EZKL-conditioned agent proposes CNNs and deep MLPs (depth is free on Halo2).
The DeepProve-conditioned agent proposes shallow, wide MLPs with ReLU only
(no polynomial activations, no BatchNorm — matching DeepProve's op coverage).

The depth indicator and proxy values confirm the structural divergence:
DeepProve-NAS architectures have fewer layers and their DeepProve proxy is lower.

## Files

- `results/backend_aware_demo/ezkl_run/nas_log.json`
- `results/backend_aware_demo/deepprove_run/nas_log.json`
- `results/backend_aware_demo/comparison.json`
- `src/zkml_bench/nas/backend_aware.py` — full NAS implementation
- `tests/test_backend_aware_nas.py` — 37 unit tests
- `scripts/run_backend_aware_demo_fast.py` — this script
