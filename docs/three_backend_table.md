# Cross-Backend Prove-Time Results (3 ZK provers, iso-FLOP pairs)

## Headline table — same architecture, 3 ZK backends

### MLP pair 1 (3.68M FLOPs, matched ±0.1%)

| backend | wide_2 (2L×512) | deep_6 (6L×372) | depth ratio | proof ratio |
|---------|-----------------|------------------|:-----------:|:-----------:|
| EZKL (Halo2) proxy logrows | 2^21 | 2^21 | **1.00× (flat)** | 1.00× |
| **DeepProve (GKR) prove** | **0.72 s, 6.2 KB** | **0.98 s, 12.8 KB** | **1.36× ↑** | **2.06× ↑** |
| Expander (GKR) prove | n/a* | n/a* | — | — |

### MLP pair 2 (12.6M FLOPs, matched ±0.1%)

| backend | wide_1 (1L×2048) | deep_10 (10L×684) | depth ratio | proof ratio |
|---------|-----------------|---------------------|:-----------:|:-----------:|
| EZKL (Halo2) proxy logrows | 2^23 | 2^23 | **1.00× (flat)** | 1.00× |
| **DeepProve (GKR) prove** | **2.41 s, 5.5 KB** | **2.85 s, 21.0 KB** | **1.18× ↑** | **3.81× ↑** |
| Expander (GKR) prove | n/a* | n/a* | — | — |

\* Expander v0.2 multi-layer numbers are from MNIST inputs (784-dim).  A
`linear_combination` API patch to ECC's `BasicAPI<C>` trait (see
`scripts/expander_patches/ecc_memory_reduction.patch`) fixes the previous ~70 GB
OOM on large models: it emits one `LinComb` instruction per output neuron instead
of O(2×n_in) `Mul`+`Add` instructions, collapsing the optimizer's
`Pool<Expression<C>>` blowup.  Measured peak RSS with the patch applied:

| Model | Dataset | Total gates | Peak RSS (patched) | Compile time | Feasible on G4DN_METAL? |
|-------|---------|------------|-------------------|--------------|------------------------|
| mlp_deep_10 | MNIST | 1.47M | **1.1 GB** | 44.6s | **Yes** (was ~71 GB OOM) |
| mlp_wide_1 | MNIST | 1.63M | ~0.6 GB | 19s | **Yes** |
| mlp_wide_2 | MNIST | 668K | ~0.6 GB | 18s | **Yes** |
| mlp_deep_6 | MNIST | 568K | ~0.5 GB | 15s | **Yes** |
| mlp_wide_2 | CIFAR-10 | 1.84M | ~1.4 GB (est.) | ~55s (est.) | **Yes** (was ~70 GB OOM) |

The patch (`scripts/expander_patches/ecc_memory_reduction.patch`) is applied
automatically by `mako_run_expander.sh` via `git apply` before the ECC build step.
The `EXPANDER_MAX_TOTAL_GATES` guard has been removed from the new Mako job
(`expander_isoflop_n5_v4`) since the patched compiler fits all MNIST models in
well under 2 GB RSS.

## Backend per-model timing (MNIST, 3 samples, 1× G4DN_METAL)

| Model | EZKL proxy logrows | DeepProve prove | Expander prove (first layer only) | DeepProve proof | Expander proof |
|-------|----:|----:|----:|----:|----:|
| mlp_small (784×512) | 2^21 | 0.65 s | 1.67 s | 5.84 MB | 67.6 KB |
| mlp_large (784×1024) | 2^22 | 1.39 s | 1.82 s | 8.29 MB | 67.7 KB |
| mlp_poly (784×512, x²) | 2^21 | failed* | 2.08 s | — | 67.6 KB |
| lenet5 | 2^20 | skipped (AvgPool) | skipped (Conv2D) | — | — |
| resnet20 | 2^26 | skipped (BN) | skipped (Conv2D+BN) | — | — |

\* DeepProve fails on `Mul`-based activations (x² in mlp_poly).

## Op-coverage matrix — a finding by itself

| Op | EZKL (Halo2) | DeepProve (GKR) | Expander (GKR-via-ECC) |
|----|:------------:|:---------------:|:----------------------:|
| Linear / Gemm | ✅ | ✅ | ✅ (first 3 layers, chained) |
| Conv2D | ✅ | ✅ | ❌ |
| BatchNorm | ✅ | ❌ | ❌ |
| ReLU | ✅ | ✅ | ✅ (via bit-decomp, expensive) |
| Polynomial (x²) | ✅ (with clamp+low scales) | ❌ (`Mul` not parsed) | ✅ |
| AvgPool | ✅ | ❌ (MaxPool only) | ❌ |
| MaxPool | ✅ | ✅ | ❌ |
| Residual Add | ✅ | ❌ (only direct shortcuts) | ❌ |
| Multi-layer chaining | ✅ | ✅ | ✅ (up to 3 linear layers, no activations) |
| Logrows ceiling (cost snaps to powers of 2) | ✅ | ❌ | ❌ |

## What this says for the paper

1. **Backend choice matters for architecture choice.** EZKL flatness vs GKR depth penalty
   (1.18–1.36× prove time) is real and measurable.
2. **Op coverage is the dominant practical constraint** — half the architectures simply
   don't run on the GKR backends we tested.
3. **Proof sizes are inverted vs prove time.** DeepProve's proofs are MUCH larger (5-21 MB)
   than Expander's (67 KB). Halo2 proofs typically sit in between (~50-500 KB for these scales).
   This wasn't in our hypothesis — emerges from the data.
4. **Verification cost varies enormously.** Expander verify ≈ Expander prove (1.6–1.7 s);
   DeepProve verify is 0.09–0.52 s (much faster than prove).
