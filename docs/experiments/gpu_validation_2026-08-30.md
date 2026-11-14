# GPU publication validation — 2026-08-30

## Completed run

- Hugging Face job: `[redacted job id]`
- Job name: `zknas-cifar10-publication-validation-a10g-large`
- Hardware: 1× NVIDIA A10G (24 GB), 12 vCPU, 46 GB RAM
- Runtime: 2026-08-30 05:48–11:10 UTC (about 5 h 22 min)
- Result repository: `[redacted private dataset]`
  (private personal dataset)
- Result revision: `bbe914d1930bb4e169e2c0223b5270f48be1b982`
- Raw result path: `cifar10-publication-core-20260830/results.json`
- Raw result SHA-256:
  `46ca35d3b8ece06c0e9ed5e20c86adbc844f91cc06d239668b9f4e66d5b58515`
- Executed runner SHA-256:
  `40be564f754dfe61b1be6d97081c7de801300f90c55f2ab7d6c07a62331ecabe`

The job produced all 156 planned records. Of these, 154 completed and two
diverged. Both divergences were
`mlp_x3px_layernorm_128x64x32`, seed 44, during epoch 1: once under the
30-epoch schedule and once under the independently initialized 100-epoch
schedule. No run ended with an infrastructure error.

## Main results

- Spearman ρ between 30- and 100-epoch architecture means: **0.932**
  (`n=26`).
- Spearman ρ across successful paired architecture-seed runs: **0.941**
  (`n=77`).
- Seed-wise Spearman ρ: **0.934**, **0.964**, and **0.929** for seeds 42,
  43, and 44, respectively.
- Mean paired accuracy gain from 30 to 100 epochs: **+5.88 percentage
  points**.
- Original 75.0% joint-proxy pick: **74.73±1.61%** at 30 epochs and
  **79.96±0.40%** at 100 epochs over three seeds.

The 30-epoch ranking is therefore a strong, but not exact, predictor of the
100-epoch ranking. Some deep polynomial CNNs learn late; the largest example
in this suite gains about 18.2 percentage points.

## Central paper architectures

Values are test top-1 accuracy, reported as mean ± sample standard deviation.

| Architecture | 30 epochs | 100 epochs |
|---|---:|---:|
| `mlp_x2px_none_128x64x32` | 50.67±0.52 | 55.67±0.32 |
| `mlp_x2px_layernorm_128x64x32` | 50.87±1.16 | 54.10±1.77 |
| `dws_relu_none_16_32_64` | 64.10±0.52 | 67.52±1.12 |
| `dws_relu_none_32_64_128` | 70.48±1.30 | 75.70±0.71 |
| `dws_relu_none_32_64_128_256` | 74.73±1.61 | 79.96±0.40 |
| `cnn_relu_none_16_32_64` | 76.98±0.72 | 79.80±0.46 |
| `cnn_relu_none_16_32_64_128` | 81.31±0.53 | 84.20±0.32 |
| `cnn_relu_none_64_128_256` | 83.96±0.16 | 86.25±0.21 |

## Protocol

- Dataset: CIFAR-10 with the augmentation and normalization used by
  `generate_sweep.py`.
- Architectures: 26 total. The eight architectures in the paper's central
  accuracy/proving table plus activation-matched controls over ReLU, x²,
  x³+x, and x²+x.
- Seeds: 42, 43, and 44.
- Epoch budgets: fresh 30-epoch and fresh 100-epoch runs. Each budget gets its
  own cosine schedule; epoch 30 of a 100-epoch schedule is not substituted for
  a true 30-epoch run.
- Optimizer: SGD, momentum 0.9, weight decay 5e-4, batch size 128.
- Learning rate: 0.1 normally; 0.01 plus gradient clipping at 1.0 for
  normalization-free polynomial models, matching the paper.
- Checkpoints: epochs 1, 5, 10, 15, 20, 30, 50, 75, and 100 when applicable.
- Additional measurements: NEAR at initialization, parameter count, complete
  learning curves, divergence/error status, runtime, and peak CUDA memory.
- Reproducibility: deterministic algorithms, fixed model/data-loader seeds,
  environment metadata, source Git SHA, and SHA-256 of the executed runner.

## Artifact access

```bash
HF_TOKEN="$HF_TOKEN_1" hf download \
  [redacted private dataset] \
  cifar10-publication-core-20260830/results.json \
  --repo-type dataset \
  --revision bbe914d1930bb4e169e2c0223b5270f48be1b982
```

The temporary organization dataset, CIFAR cache, and shared source snapshot
used to launch the job were removed after the completed artifact was copied to
the personal private repository above.
