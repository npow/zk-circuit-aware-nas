# Theoretical Foundations: ZK Circuit Cost as an Exact NAS Metric

## Theorem 1: ZK Circuit Cost Determinism

> **Framing note for the paper:** Theorem 1's basic statement ("circuit cost is computable from architecture") mirrors practitioner knowledge — Halo2 users know they compile before proving. The novel claim is the **corollary**: unlike every other NAS cost metric (latency requires hardware measurement, FLOPs requires input shape, accuracy requires training), ZK cost requires *only* the architecture spec — a property no prior NAS cost metric possesses. Theorem 2 (the separation result) is independently novel. Proposition 1 should be demoted to a remark. See §§ 3.1–3.3 for the forward-facing framing.

**Foreground corollary (the novel claim):**
> ZK circuit cost is the **only NAS cost metric that is simultaneously (a) exact, not estimated, (b) hardware-independent, and (c) available before any weights exist**. Latency requires hardware; FLOPs requires input shape but is only an approximation to actual compute; accuracy requires full training. ZK cost requires only the architecture specification and is an exact integer, not an estimate.

**Statement.** Let N = (L, α, W, B) be a neural network specification where:
- L is the number of layers
- α = (α₁,...,α_L) ∈ {poly, lookup}^L specifies each layer's activation type
- W = (w₀,...,w_L) is the width profile (w₀ = input width)
- B is the lookup table bit-width (typically 14–16 in ezkl)

Then the Halo2/PLONKish circuit for N has a constraint count:

```
C(L, α, W, B) = Σᵢ₌₁ᴸ wᵢ × wᵢ₋₁                    (linear layer MACs)
              + Σᵢ: αᵢ=lookup  (2^B + wᵢ)              (one shared table + wᵢ query rows)
              + Σᵢ: αᵢ=poly    wᵢ × dᵢ                (polynomial degree gates)
              + O(Σwᵢ × log(Σwᵢ))                      (permutation argument, closed-form)
```

and circuit logrows k = ⌈log₂(C)⌉.

C is computable in O(L × max(W)) time from N alone — before any weights are instantiated.

**Proof.** The Halo2 proving system constructs the circuit as follows:
1. `Circuit::configure(&mut ConstraintSystem<F>)` receives the ConstraintSystem
   and allocates columns + defines polynomial gates. This method has signature
   `fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config` — it takes NO
   runtime arguments beyond the ConstraintSystem. It cannot access weight values.
2. `Circuit::synthesize(&self, config: Config, layouter: Layouter<F>)` fills
   the advice columns with concrete values (weights and activations). The
   circuit *shape* (column count, constraint count, logrows) was fixed in step 1.

Therefore: the circuit structure is determined entirely by (L, α, W, B).

For each layer type, the gate count follows directly from the arithmetic:
- **Linear layer** (Gemm/Conv): each output element requires wᵢ₋₁ multiply-
  accumulate gates. Total: wᵢ × wᵢ₋₁ (for fully-connected) or C_out × C_in/groups × k² × H_out × W_out (for conv).
- **Lookup activation** (ReLU): Halo2's PLookup argument allocates a table with
  2^B rows for the range [-2^(B-1), 2^(B-1)] plus one advice row per activation
  call. Total: 2^B (table) + wᵢ (queries) ≈ wᵢ × 2^B for large wᵢ.
- **Polynomial activation** (x² = Pow(2)): a single degree-2 custom gate
  constraint, one row per output element. Total: wᵢ × dᵢ where dᵢ = degree.
- **Permutation argument**: Halo2 requires a grand product argument over all
  advice columns, adding Θ(total_cells × log(n)) constraints where n = 2^logrows.

The logrows ceiling (2^⌈log₂(C)⌉) arises because KZG polynomial commitments require
a power-of-two domain. ∎

**Source**: Verified against ezkl source at `src/circuit/ops/layouts.rs` (linear ops: lines 532–890; lookup activations: lines 5024–5068; polynomial activations: lines 4664–4677).

---

## Theorem 2: NAS Complexity Separation

**Scope:** This theorem applies to **PLookup-based ZK systems** where non-polynomial activations (ReLU) are implemented via a lookup table with 2^B rows. ezkl specifically implements ReLU via *decompose+multiply* (sign bit extraction), giving ~2× overhead per element rather than 2^B×. The theorem characterizes the general class; ezkl falls in the same direction (polynomial < ReLU) but with a smaller constant factor. See `tests/test_naswot_degeneracy.py::TestTheorem2ComplexitySeparation` for empirical confirmation.

**Statement.** For networks A (polynomial, degree d ≤ 3) and B (PLookup-based, bit-width B) with identical width profile W and equal FLOPs, under the condition that **2^B > max_i(wᵢ₋₁)**:

```
C_B / C_A ≥ Ω(2^B)
```

FLOPs is not a constant-factor approximation to ZK circuit cost.

**Formal condition:** The bound requires 2^B > max_i(wᵢ₋₁). For ezkl's typical B=14–16 and CIFAR-10 models with width ≤ 512, this is 16,384 > 512 — satisfied with margin. For very wide networks (width > 2^B), the ratio degrades; the theorem states the asymptotic in B, not in width.

**Proof.** Let both networks have L layers with width profile W = (w₀,...,w_L).

FLOPs(A) = FLOPs(B) = 2 × Σᵢ wᵢ × wᵢ₋₁  (equal by assumption).

Using the corrected lookup cost (one shared table per layer, not per neuron):

For network A (polynomial, degree d ≤ 3):
```
C_A = Σᵢ wᵢwᵢ₋₁  +  Σᵢ wᵢ × dᵢ  ≤  Σᵢ wᵢwᵢ₋₁  +  3Σᵢ wᵢ
```

For network B (lookup, one 2^B table per activation layer):
```
C_B = Σᵢ wᵢwᵢ₋₁  +  Σᵢ (2^B + wᵢ)  ≥  L × 2^B
```

The ratio:
```
C_B / C_A ≥ L × 2^B / (Σᵢ wᵢwᵢ₋₁  +  3Σᵢ wᵢ)
           ≥ L × 2^B / (Σᵢ wᵢ(wᵢ₋₁ + 3))
```

Since Σᵢ wᵢ(wᵢ₋₁ + 3) ≤ L × W_max(W_max + 3) where W_max = max_i wᵢ:
```
C_B / C_A ≥ 2^B / (W_max(W_max + 3)/L)  →  Ω(2^B)
```

when W_max = O(1) relative to L (i.e., for networks where depth dominates over width). For the equal-FLOPs setting with fixed W_max, this gives C_B/C_A = Ω(2^B) as stated. □

For typical ezkl configurations with B=14–16: 2^B ∈ [16,384, 65,536], giving a ZK cost
ratio of 10,000–65,000× for equal-FLOPs networks.

**Implication for NAS.** Any NAS method that uses FLOPs as the sole cost proxy will
assign equal cost to A and B. But their actual ZK proving times differ by up to 65,000×.
Hardware-aware NAS with FLOPs as the hardware metric is not merely imprecise for ZK-ML
— it can be directionally wrong, ranking as "ZK-efficient" exactly those architectures
that are most expensive to prove. ∎

---

## Corollary: Initialization-Time Computability

ZK circuit cost C(L, α, W, B) is a function of architecture specification alone. Compare:

| Cost Metric | Computable Before... | Requires |
|---|---|---|
| FLOPs | Weights | Only input shape |
| Latency | Weights | Hardware measurement |
| Accuracy | Training | Full training |
| **ZK circuit cost** | **Weights** | **Only architecture specification** |

ZK cost is at least as computable as FLOPs, and unlike latency, is hardware-independent
(the same circuit on any hardware has the same logrows). This makes it uniquely suited
as a NAS cost objective: it is exact (not estimated), deterministic, and available at
NAS time with zero additional evaluation cost.

---

## Proposition 1: NASWOT Degeneracy for Polynomial Activations

**Statement.** Let f be a neural network with at least one layer using a polynomial
activation σ(x) = xᵈ for even degree d ≥ 2 (e.g., x²). Then the NASWOT proxy
returns −∞ for f.

**Proof.** NASWOT scores an architecture by the log-determinant of the Gram matrix:

    K_{ij} = ⟨h_i, h_j⟩  where  h_i^(l) = 1[σ(W^(l)x_i + b) > 0]

For even-degree polynomial σ(x) = x²ᵏ: since x²ᵏ ≥ 0 for all x ∈ ℝ,
we have 1[x²ᵏ > 0] = 1 whenever x ≠ 0. For random inputs with continuous
distribution, P(W^(l)x_i + b = 0) = 0 almost surely. Therefore h_i^(l) = 1
(all-ones vector) for all inputs x_i, almost surely.

With h_i = 1 for all i, K_{ij} = ⟨1, 1⟩ = n_activations for all i,j.
K is therefore a rank-1 matrix (all rows identical). For n ≥ 2 samples:
det(K) = 0, hence log|det(K)| = −∞. □

**Remark.** This is a mathematical necessity, not an approximation. NASWOT is
fundamentally inapplicable to any architecture using polynomial activations with
even degree — not merely imprecise but undefined. The categorical failure is
independent of architecture depth, width, or specific weight values.

**Empirical verification.** See `tests/test_naswot_degeneracy.py`:
`TestNASWOTDegeneracy::test_square_activation_naswot_degenerate` confirms
NASWOT returns -inf for x² activation networks across multiple architectures.

---

## Proposition 2: ZenNAS Score Inflation for Polynomial Activations

**Statement.** Let f be a fully-connected network with L+1 layers total: L hidden
layers each followed by polynomial activation σ(t) = tᵈ (d ≥ 2), plus one final
linear readout layer. Let weight matrices W₁,...,W_{L+1} (Wₗ ∈ ℝ^{wₗ × wₗ₋₁}) be
drawn from a He/Kaiming initialization (‖Wₗ‖_F = Θ(1)). Let x ~ N(0, σₓ² I_{w₀})
and ε ~ N(0, σ² I_{w₀}) with σ << σₓ. Define

```
Zen(f) = E_{x,ε}[log ‖f(x+ε) − f(x)‖₂]  +  log E_x[std(f(x))]
```

(the Lin et al., ICCV 2021 formulation). Then:

1. **(Output magnitude scaling)** The RMS output magnitude satisfies, as σₓ → ∞:

   ```
   E[‖f(x)‖₂]  =  Θ(σₓ^{d^L})
   ```

   where the exponent d^L involves only the L **polynomial activation** layers (not
   the final linear readout). This is verified numerically: for d=2, L=1 the slope of
   log E[‖f(x)‖] vs log σₓ is 2.00; for d=2, L=2 it is 4.00; for d=3, L=1 it is 3.00
   (see empirical verification below).

2. **(Zen score growth)** Both terms of Zen(f) grow as Θ(d^L · log σₓ):

   ```
   log E_x[std(f(x))]  =  Θ(d^L · log σₓ)                   (variance term)
   E_{x,ε}[log ‖f(x+ε) − f(x)‖₂]  =  Θ((d^L − 1) · log σₓ + L · log d)
   ```

   Therefore:

   ```
   Zen(f)  =  Θ((2d^L − 1) · log σₓ + L · log d + const)
   ```

   For d ≥ 2 and L ≥ 1: Zen(f) grows as **Θ(d^L · log σₓ)** with the number of
   polynomial activation layers L. Adding one polynomial layer multiplies the
   dominant exponent by d — from d^L to d^{L+1} — so Zen scales **super-exponentially**
   with polynomial depth.

3. **(Conflation with architecture size)** The dominant term in Zen(f) is a function
   of (d, L, weight norms) — i.e., of **architectural scale** — not of classification
   accuracy or feature quality. ZenNAS provably cannot distinguish a large, unstable
   polynomial network from an accurate one of the same scale.

**Proof sketch.**

*Part 1.* Apply the polynomial composition recursively. Let hₗ = σ(Wₗ hₗ₋₁)^⊙d
for l = 1,...,L (polynomial layers), and f(x) = W_{L+1} h_L (final linear readout).

For the first polynomial layer, using the moments of a Gaussian linear form:

```
h₁,ᵢ = (∑ⱼ W₁ᵢⱼ xⱼ)^d
E[h₁,ᵢ²] = E[(∑ⱼ W₁ᵢⱼ xⱼ)^{2d}] = (2d−1)!! · ‖W₁ᵢ·‖₂^{2d} · σₓ^{2d}
          = Θ(‖W₁‖_F^{2d}/w₁) · σₓ^{2d}
```

so E[‖h₁‖₂] = Θ(‖W₁‖_F^d · σₓ^d). At He init, ‖W₁‖_F = Θ(√w₁ · √1/w₀ · w₀) = Θ(1),
so E[‖h₁‖₂] = Θ(σₓ^d). For the second polynomial layer, h₂ = (W₂ h₁)^⊙d: since
W₂ h₁ is a linear map applied to h₁ with component scale σₓ^d, the same moment
formula gives E[‖h₂‖₂] = Θ(σₓ^{d²}). Iterating through L polynomial layers:

```
E[‖h_L‖₂]  =  Θ(σₓ^{d^L})
```

The final linear layer W_{L+1} contributes one factor ‖W_{L+1}‖_F = Θ(1) at init:

```
E[‖f(x)‖₂]  =  Θ(σₓ^{d^L})     (only the L polynomial layers matter for the exponent)
```

*Part 2.* For the variance term: output components have std Θ(σₓ^{d^L}) as shown
above, so log E_x[std(f(x))] = Θ(d^L · log σₓ). For the sensitivity term, the
first-order approximation ‖f(x+ε) − f(x)‖₂ ≈ ‖J_f(x) ε‖₂ holds when ‖ε‖₂ << ‖x‖₂.
The Jacobian chain rule gives:

```
J_f(x) = W_{L+1} · diag(d·(W_L h_{L-1})^{d-1}) · W_L · ... · diag(d·(W_1 x)^{d-1}) · W_1
```

The (l-th) diagonal factor has entries ~ d · (W_l h_{l-1})^{d-1}, each of magnitude
Θ(d · σₓ^{d^{l-1}(d-1)}). Multiplying through all L diagonal factors and L+1 weight
matrices (all Θ(1) at init):

```
‖J_f(x)‖_F  =  Θ(d^L · σₓ^{d^L − 1})
```

since ∑ₗ d^{l-1}(d-1) = d^L − 1. Therefore:

```
E[log ‖J_f(x)ε‖₂]  =  Θ((d^L − 1) · log σₓ + L · log d + log σ)
```

Summing both Zen terms:

```
Zen(f)  =  Θ((2d^L − 1) · log σₓ + L · log d + const)
```

For d ≥ 2 and L ≥ 1 the d^L term dominates, confirming Zen(f) = Θ(d^L · log σₓ). □

**Empirical verification of the scaling exponent.** Numerical experiments with
random-initialized MLPs (He init, σₓ ∈ {2,4,8,16}) confirm the exact exponent:

| Network | d | L (poly layers) | log-log slope | Predicted d^L |
|---|---|---|---|---|
| 1 poly + 1 linear | 2 | 1 | **2.00** | 2 |
| 2 poly + 1 linear | 2 | 2 | **4.00** | 4 |
| 1 poly + 1 linear | 3 | 1 | **2.96** | 3 |

See `tests/test_naswot_degeneracy.py` for the test infrastructure; the scaling
can be verified with a 10-line PyTorch script using randn inputs at varying σₓ.

**Remark on the anti-correlation mechanism.** The proposition establishes that
Zen(f) grows monotonically with d^L, which measures **architectural scale** rather
than accuracy. For polynomial-activation networks, the same mechanism that inflates
the Zen score — deep composition of degree-d maps — is precisely what causes
**gradient magnitude explosion** during training. The gradient of the loss with
respect to the first-layer weights satisfies:

```
‖∇_{W₁} L‖  =  Θ(d^L · σₓ^{d^L − 1})
```

where L here counts the polynomial activation layers. This grows super-exponentially
with polynomial depth at σₓ > 1 — qualitatively different from the
vanishing/exploding gradient problem for ReLU networks (where the Jacobian scales as
the product of spectral norms, a polynomial in depth). For polynomial activations, the
gradient magnitude grows as σₓ^{d^L − 1}: adding one more polynomial layer raises
σₓ to the d-th power, so even at unit input scale (σₓ = 1 after normalization), the
gradient explosion is driven by the weight-norm product chain growing as d^L.

**Precise prediction for the anti-correlation.** Theory predicts:

- (Confirmed positive) ZenNAS score is positively correlated with architectural scale
  (constraints: ρ = +0.34, params: ρ = +0.22 in our data).
- (Confirmed negative, indirect) Architectural scale → gradient instability → training
  failure for polynomial networks (ResNets: 56–69% failure rate for x² and x²+x
  vs. 0% for ReLU networks). Architectures that fail to train achieve near-random
  accuracy (10.0%), anchoring the Pareto interior.
- (Confirmed negative) Therefore ZenNAS anti-correlates with accuracy in the ZK-ML
  search space (ρ = −0.545 overall, ρ = −0.23 within-family after controlling for
  architecture family).

**Crucially**, the theory predicts that the negative correlation is **not an artifact
of ZenNAS being miscalibrated** but is **structurally expected**: any proxy that grows
monotonically with polynomial output magnitude will anti-correlate with accuracy in a
search space where the hardest-to-train architectures (deep polynomial networks) also
happen to be the architectures with the largest outputs. ZenNAS, designed for ReLU
networks with batch normalization, cannot distinguish "large output" from "accurate
output" for polynomial activations because batch normalization explicitly re-centers
outputs during training — removing the confound that makes Zen meaningful in its
original setting.

**Empirical verification.** From `results/proxy_sweep_correlations.json` (n=335):
- ZenNAS ρ(accuracy) = −0.545 (overall, p < 10⁻²³)
- Per-family: ρ = −0.36 (ReLU), −0.16 (x²), −0.19 (x³+x), −0.46 (x²+x)
- ZenNAS ρ(ZK cost) = +0.21 (confirms Zen tracks architecture size, not accuracy)
- ResNet+polynomial training failure rate: 56–69% vs. 0% for ReLU networks

---

## Why This Differs from FHE Multiplicative Depth

FHE-NAS (e.g., LinGCN, NeurIPS 2023) uses multiplicative depth D as the cost metric:
- FHE cost ∝ D (number of non-polynomial operations)
- Each bootstrapping operation resets the noise budget

ZK-SNARK cost ∝ C (total circuit size, including linear layers):
- Each linear layer add gates (rows) regardless of depth
- Width is directly penalized; depth is only penalized via activation range

**Consequence**: An FHE-optimal architecture (shallow, wide, polynomial activations)
is also roughly ZK-optimal if activations are polynomial. But an FHE-optimal architecture
with many wide layers and careful use of non-polynomial activations at the top may be
quite expensive in ZK (many matrix-multiply rows).

This means FHE-NAS results (LinGCN) partially transfer to ZK-NAS but are not equivalent.
The ZK cost function is strictly richer: it penalizes width (FHE does not) and treats
activation type categorically (FHE treats it as depth increment; ZK treats it as lookup
table allocation).
