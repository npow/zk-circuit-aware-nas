# Proxy Calibration: Reviewer-Defensible Analysis

## Headline result

The ZK constraint proxy, calibrated against actual EZKL constraint counts on 8 baseline models, achieves:

- **In-sample R² = 0.9791** (all 8 models)
- **Held-out R² mean = 0.9727** (50 random 50/50 splits; min 0.907, max 1.0)
- **7/8 exact matches** (proxy snaps to the same logrows EZKL allocates)

Held-out R² being almost identical to in-sample R² means the calibration **generalizes** — it isn't curve-fit to the specific 8 models.

## Ablation: what actually contributes to the R²?

After codex pre-submission review flagged the calibration as potentially curve-fit, we ran a component ablation (`scripts/proxy_calibration_holdout.py`):

| Variant | R² (all 8 models) |
|---------|------------------:|
| MAC count only (no snap, no node overhead) | **0.9581** |
| MAC count + `snap_to_power_of_2()` | **0.9791** |
| MAC count + `NODE_OVERHEAD=4096` (no snap) | **0.9587** |
| MAC + overhead + snap (current proxy) | **0.9791** |

**Interpretation**: `snap_to_power_of_2()` provides essentially all of the +0.021 R² lift. `NODE_OVERHEAD=4096` contributes ≈ 0.0006 in isolation.

The held-out fitting analysis confirms this: the optimizer's fitted overhead value saturates the 65536 search-space ceiling (median over 50 splits = 65535, IQR = [65535, 65535]). The optimizer is not converging to a meaningful value because the snap dominates the error landscape and the function is nearly flat in `overhead` for any value below `O(2^logrows)`.

## What the proxy actually is

**Not** "MAC count × per-op weights gives EZKL constraint count."

**Actually**: "EZKL allocates `2^logrows` constraint rows, with `logrows = ceil(log2(MAC_count × per-op weights × per-node overhead))`."

The dominant source of EZKL cost is the **logrows ceiling**, not the MAC count itself. Two models with MAC counts of 1.6M and 2.0M can land in the same `2^21 = 2,097,152` cell budget. The proxy snaps the continuous estimate to the next power of 2, mirroring EZKL's behavior.

This is a **compiler-aware cost model**, not a multiplier-on-FLOPs model.

## Implications for ZK-friendly architecture design

Conventional ML cost is MAC-dominated. ZK cost (EZKL/Halo2-style) is **logrows-dominated**. Architectures whose total cost lands just above a power of 2 pay for the next snap up; architectures whose total cost lands just below pay only for the current snap. **Designing around the snap boundaries can be more impactful than reducing MAC count.**

Example: `lenet5` (666K constraints raw) and `mlp_small` (1.7M constraints raw) both snap to `2^21 = 2.1M`. Reducing `lenet5` by 50% saves nothing; reducing `mlp_small` by 50% drops it into `2^20 = 1.05M`, a 2× savings.

This generalizes the NAS task: **architecture search under ZK cost should optimize for the snap boundary**, not the raw constraint count.

## Constants used

- `NODE_OVERHEAD = 4096` (per-Conv/Gemm/MatMul row overhead). This was empirically discovered as ~4K rows of per-op overhead, but is a small correction relative to the snap. Held-out fitting suggests any value in `[0, 64K]` produces equivalent R² because the snap absorbs the difference.

## Reviewer-anticipated questions

**"Aren't you reverse-engineering one backend's rounding behavior?"**
Yes, and that's the contribution. EZKL is the most-deployed Halo2-based ZK prover; modeling its logrows ceiling explicitly is a more useful proxy than ignoring it. The same approach generalizes to any lookup-based prover that allocates rows in powers of 2.

**"Why is the calibration set only 8 models?"**
The 8 ReLU baseline models cover the range of architectural patterns we want to validate (MLP, LeNet, ResNet, VGG, BN-free ResNet) at 5 distinct scales (1M, 2M, 4M, 33M, 67M cells). Adding more models (e.g., DWS-CNN, polynomial activations) is the goal of US3 — see CIFAR-10/100 baseline runs.

**"Why does VGG-11 have ratio 0.5?"**
`vgg11_small` proxy = 39.3M, snapped to `2^26 = 67M`; EZKL allocates `2^25 = 33.5M`. The proxy *over*-estimates by 1 logrow. This is an honest miss; the proxy is slightly conservative on this architecture. We hypothesize EZKL is doing constant-folding or peephole optimization that drops the model below the 2^26 boundary; the proxy doesn't model that.

**"How does this generalize beyond EZKL?"**
The `NODE_OVERHEAD + snap_to_power_of_2()` template applies to any lookup-based ZK prover (Halo2 family) since they all allocate constraint rows in powers of 2. For GKR/sumcheck systems (e.g., DeepProve, Expander), the proxy would need a different formula since they don't have the logrows ceiling — they pay roughly linearly in the layer-by-layer cost.

## File pointers

- `src/zkml_bench/metrics/zk_proxy.py` — implementation
- `scripts/proxy_calibration_holdout.py` — held-out validation + ablation
- `results/proxy_holdout_validation.json` — full per-split data
- `proxy_validation_results.json` — the 8-model calibration set
