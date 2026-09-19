# AdamW vs Muon

Completed 3 epochs with seed=42; the table compares epoch 3 for both runs.

| Optimizer | Train loss | Validation loss | Validation accuracy | Best-accuracy epoch |
|---|---:|---:|---:|---:|
| adamw | 0.076963 | 0.262449 | 90.8257% | 3 |
| muon | 0.136087 | 0.393930 | 84.4037% | 1 |

Muon final accuracy relative to AdamW: -6.4220 percentage points.

Configuration consistency and original AdamW output SHA-256 checks passed.
This is a preliminary single-seed comparison with different optimizer learning rates; no statistical significance is claimed.
See comparison.csv for all epoch results; the Muon run includes AdamW fallback.
