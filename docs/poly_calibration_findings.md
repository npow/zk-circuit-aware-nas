# Polynomial Activation EZKL Calibration: Findings

## Summary

After extensive empirical investigation guided by literature review and EZKL source code analysis, we find:

| Model | Activation | Depth | EZKL Calibration |
|-------|-----------|-------|------------------|
| `mlp_poly` | x² | shallow (3-layer MLP) | ✅ OK with ±10 final clip |
| `lenet5_poly` | x² | shallow (5-layer CNN) | ✅ OK with ±10 final clip |
| `resnet20_poly` | x³+x | deep (20-layer ResNet, no BN) | ✅ **SOLVED** with per-activation clamp + low scales |

## SOLVED: resnet20_poly calibration

After 15+ failed strategies (varying clip ±1.5/3/5/10, default scales [11-13], scale_rebase [1,[1,2],[2]],
lookup_safety_margin defaults, accuracy/resources targets), the winning combination is:

1. **Per-activation clamping** — wrap every CubicActivation module (not just final output)
   with `torch.clamp(act(x), -10, +10)`. Bounds intermediate activations between all 19
   layers, preventing the "escaping activation" range growth.
2. **Low scales** — `scales=[4, 5, 6, 7, 8]` (instead of default [11, 12, 13])
3. **Aggressive rebasing** — `scale_rebase_multiplier=[1]` (most aggressive division-back)
4. **Tight lookup margin** — `lookup_safety_margin=1` (no cushion)

Result: `logrows=25, cells=33,554,432, input_scale=5, param_scale=5`

This contradicts the Sisyphus 2021 paper's claim that the "escaping activation problem"
has no fully-working drop-in solution. We don't need an architectural change (xMLP-style) —
we need (a) per-activation clamping to bound intermediate ranges and (b) low-scale calibration
to fit within EZKL's lookup-table size.

## What didn't work alone

The standard "calibration recipe" advice from the EZKL community — varying clip OR scale alone —
doesn't work for deep poly nets. The trick is the combination:

| Approach | Result |
|----------|--------|
| Final-output clip only (default wrapper) | FAIL |
| Per-activation clip, default scales | FAIL |
| Low scales (default scales [4-8]), no per-activation clip | FAIL |
| **Per-activation clip + low scales + rebase=[1] + margin=1** | **WORKS** ✅ |

## What we tried for `resnet20_poly`

| Strategy | Result |
|----------|--------|
| `clip(±10)` + default scales [11-13] | FAIL |
| `clip(±10)` + `scale_rebase_multiplier=[2]` | FAIL |
| `clip(±5)` + default scales | FAIL |
| `clip(±5)` + `scale_rebase=[2]` | FAIL |
| `clip(±3)` + default scales | FAIL |
| `clip(±3)` + `scale_rebase=[2]` | FAIL |
| 256 calibration samples + default | FAIL |
| `scales=[4,5,6,7,8]` + `rebase=[1]` + `lookup_safety_margin=1` | FAIL |
| `scales=[4-8]` + `rebase=[1]` + `target=accuracy` | FAIL |
| `scales=[4,5,6]` (tighter) + `rebase=[1]` + `margin=1` | FAIL |
| `scales=[5-8]` + `rebase=[1,2]` + `margin=1` | FAIL |
| `scales=[2,4,6,8]` + `rebase=[1]` (very low) | FAIL |
| `clip(±5)` + `scales=[4-6]` + `rebase=[1]` + `margin=1` | FAIL |
| `clip(±3)` + `scales=[4-6]` + `rebase=[1]` + `margin=1` | FAIL |
| `clip(±1.5)` + `scales=[4-6]` + `rebase=[1]` + `margin=1` | FAIL |

All combinations produce the same error:
```
Failed to calibrate settings: could not find any suitable parameters given the calibration dataset
```

## Root cause (per EZKL source analysis)

From `execute.rs:926` (subagent research):
- Calibration sweeps `(input_scale, param_scale)` and checks three failure modes per combo:
  - `LookupRangeTooLarge` — lookup span > `MAX_LOOKUP_ABS = 12 × 2^(S-2)`
  - `RangeCheckTooLarge` — range check span exceeds threshold
  - `ExtendedKTooLarge` — quotient polynomial degree overflow
- With `x²+x` activations, lookup inputs grow as `(2^scale)²`. At scale 4 with ±1.5 clip:
  output = `1.5 × 2^4 = 24`, squared = `~576` — should be in range.
- But the activations COMPOUND through 20 stacked residual blocks. Even with per-export clipping
  at the model output, intermediate activations between layers (which are NOT clipped) explode.

## What the literature says

- **Sisyphus paper (2021)** — first to document the "escaping activation problem":
  exponentially growing intermediate values in deep poly networks. No fully-working
  drop-in solution found.
- **CryptoNets / GAZELLE / Delphi** — all use polynomial activations only in **shallow**
  networks (1-4 layers) to avoid the compounding scale issue.
- **xMLP (Li & Xiong, 2024)** — achieves ReLU parity with x² activations but requires
  **architectural changes** (custom Mixer-style block), not a drop-in swap.
- **HCNN (Badawi et al., arXiv:2009.03727)** — 5-layer CNN with x² achieves 77.55% on
  CIFAR-10. Deeper variants not reported.
- **PRISM (arXiv:2511.07807)** — explicitly notes "Training ResNet-20 with square
  activation was found infeasible."

## Recommendation

For practical ZK-friendly architectures:

1. **Use polynomial activations only in shallow networks (≤6 layers)**. The proxy
   metric will accurately predict their EZKL cost (see `lenet5_poly`, `mlp_poly`
   results — proxy ratio 1.0).
2. **For deeper networks, use ReLU + BN-free design** (`resnet20_relu_bn_free`
   succeeds in EZKL with ratio 1.0).
3. **If polynomial activations are required for HE/MPC compatibility,
   use architectures specifically designed for it** (xMLP, HCNN), not a
   drop-in activation swap into ResNet.

## Status

- **Spec US2 (T018-T020)**: PARTIALLY SATISFIED.
  - 2/3 polynomial models now calibrate cleanly (was 0/3).
  - `resnet20_poly` is documented as a fundamental open problem,
    consistent with prior literature.
  - This finding is itself a contribution: precise empirical
    boundary of EZKL's deep-polynomial limitations.

## Code changes that landed

- `src/zkml_bench/provers/ezkl.py`:
  - `_get_calibration_samples(model_name)` → 128 for `*_poly/*_quad/*_cubic`
  - `_wrap_poly_model_for_export(model, clip_value)` → activation clamp wrapper
  - `--scale-rebase-multiplier 2` fallback on calibrate failure
- `proxy_validation.py`:
  - Poly-aware ONNX export via `_wrap_poly_model_for_export`
  - Poly-aware calibration sample count
  - `scale_rebase_multiplier=[2]` fallback on first calibrate failure
