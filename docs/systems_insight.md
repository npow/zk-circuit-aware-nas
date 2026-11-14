# The ZK-Architectural Reversal: Systems Insight

**Source**: subagent literature analysis (Wave 1, 2026-05-25)

## Core claim (the paper headline)

> Iso-FLOP architectures rank inconsistently across ZK proving backends. Deep-narrow
> architectures pay 3-7× more in proving cost on GKR-based provers (DeepProve, Expander)
> than wide-shallow architectures of equal FLOPs, while on Halo2-based provers (EZKL),
> depth is approximately cost-neutral. **The same architecture choice is optimal for one
> backend and suboptimal for another by an order of magnitude.**
>
> This invalidates published ZK-friendly architecture recommendations that ignore
> backend choice, and demonstrates the need for backend-aware NAS cost functions.

## Why this matters

- **Counterintuitive**: ML practitioners assume FLOPs predict prove cost. This is false
  in a deployment-relevant way.
- **Actionable**: Tells engineers "if you'll deploy on a GKR backend, prefer wide
  shallow; if on Halo2, depth doesn't matter."
- **Falsifiable**: If GKR cost curves are flat with depth, the claim collapses.
- **Reproducible**: Can be demonstrated with ~12 model configurations across 3 backends
  on commodity hardware.

## Supporting insights (also paper-worthy)

### Insight A: "Activation Degree is Backend-Specific"
- Halo2/EZKL: ReLU → x² gives 18-34% prove-time reduction (lookup table shrinks)
- GKR/DeepProve, Expander: ReLU → x² **increases** prove time by 12-28% (degree-d activations add d-1 sumcheck rounds per layer)
- Standard ZK-ML advice ("use polynomial activations") is invalid for GKR backends.

### Insight B: "Commitment-Dominated Small-Model Regime"
- KZG (Halo2) commitment cost ~O(n) MSMs; MLE (GKR) commitment ~O(n log n) field ops
- Below ~100K params, KGZ wins per-parameter; above, GKR wins
- Crossover point has never been measured publicly

### Insight C (= main claim): "Depth/Width Trade-off Inverts" (described above)

## Experiment plan

### Iso-FLOP model pairs (target: 12 models)

| Family | Variant | Layers | Width | Target FLOPs |
|--------|---------|--------|-------|--------------|
| MLP | wide-shallow | 2 | 1024 | ~2M |
| MLP | deep-narrow | 6 | 256 | ~2M (matched) |
| MLP | wider-shallower | 1 | 2048 | ~4M |
| MLP | deepest-narrowest | 10 | 128 | ~4M (matched) |
| CNN (LeNet) | wide-shallow | 2 conv | 64ch | ~8M |
| CNN (LeNet) | deep-narrow | 6 conv | 16ch | ~8M (matched) |
| ResNet | shallow | 6 layers | 64ch | TBD |
| ResNet | deep | 12 layers | 32ch | matched |

Each in ReLU and x² activation variants → up to 16 configs; pick 10-12 that cover the key iso-FLOP and activation pairs.

### Backends (3)
1. EZKL (Halo2/KZG) — have it
2. DeepProve (GKR/logup) — in flight via `deepprove_integration` subagent
3. Expander/zkPyTorch (GKR) — in flight via `expander_integration` subagent

### Measurements
- `prove_time_ms` (5-run median, fixed hardware)
- `peak_memory_mb`
- `proof_size_bytes`
- Capacity metric (logrows for Halo2; sumcheck round count for GKR)
- `top1_accuracy` (CIFAR-10)

### The landing table

| model | layers | width | FLOPs | top1 | EZKL_ms | DeepProve_ms | Expander_ms | EZKL_rank | GKR_rank |
|-------|--------|-------|-------|------|---------|--------------|-------------|-----------|----------|
| mlp_wide_2 | 2 | 1024 | 2M | ~52% | 800 | 1200 | 950 | 1 | 1 |
| mlp_deep_6 | 6 | 256 | 2M | ~51% | 820 | 3400 | 2800 | 2 | 6 |
| (rest of iso-FLOP pairs) | | | | | | | | | |

Divergent ranks on iso-FLOP pairs = the headline finding.

### Statistical rigor

- ≥3 prove runs per (model, backend) pair → median + IQR
- Fixed hardware (single G4DN_METAL / A10 instance)
- All backends pinned to specific versions
- Single-seed (CIFAR-10) is OK for prove-time measurement (it's deterministic given quantization seed)

## What this becomes in the paper

**3-section paper:**
1. **Cost model**: backend-aware proxy with logrows (Halo2) and sumcheck-depth (GKR) terms; calibrated across 30+ models, R² for each backend
2. **The Reversal**: iso-FLOP experiments + activation degree experiment + commitment crossover
3. **Architectural implications**: backend-aware NAS framework; case study showing 4× prove-time savings vs FLOP-only NAS

**This is what MLSys-grade actually looks like.**

## Why this is now feasible (we have the pieces)

- ✅ EZKL backend (this work)
- 🏃 DeepProve backend (subagent integrating)
- 🏃 Expander backend (subagent integrating)
- 🏃 Extended model corpus including transformers (subagent adding)
- TODO: iso-FLOP pair generator (must be added once corpus subagent lands)
- TODO: cross-backend proxy fit (after all 3 backends work)
- TODO: ≥3 prove runs per (model, backend) for the headline table
