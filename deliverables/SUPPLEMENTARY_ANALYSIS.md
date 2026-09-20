# Supplementary analysis: AdamW vs Muon on DistilBERT/SST-2

This document expands on `REPORT_EN.md` (the one-page submission). It only interprets numbers already present in `outputs/` and `deliverables/`; it adds no new experiments, retraining, or data.

## Why these metrics

- **Gradient norm** (`||grad||`, per parameter group): the raw size of the loss gradient before any optimizer transformation. It is the common input both optimizers act on, so comparing it isolates whether a difference in behavior comes from the gradients themselves or from what each optimizer does with them.
- **Update norm** (`||Δparams||`): the size of the actual step applied to the parameters after the optimizer's transformation (momentum, adaptive scaling for AdamW, orthogonalization for Muon). Comparing update norm to gradient norm shows how much each optimizer reshapes the raw gradient.
- **Relative update norm** (`||Δparams|| / ||params||`): update size normalized by the current parameter scale, so the two optimizers' step sizes are comparable across parameter groups of very different scale (e.g. large hidden matrices vs. small bias vectors) and are not confounded by weight magnitude.
- **Sharpness** (mean loss increase under bounded random perturbation of a saved checkpoint): a proxy for how sensitive the found solution is to small parameter changes, used here as a rough stand-in for local curvature ("flatness") of the loss around each optimizer's final solution.

Together, gradient norm and update norm show *how* each optimizer converts gradients into steps; relative update norm makes that comparable across parameter groups; sharpness asks whether the two optimizers' final solutions differ in a way that might relate to generalization, independent of the accuracy numbers themselves.

## Muon learning-rate selection

`muon_sanity.py` ran a predeclared, training-only stability check before the full experiment: 512 training examples plus a disjoint 128-example holdout (seed 42, batch size 32, 1 epoch per candidate, 16 steps), for three candidate matrix learning rates. The rule (fixed in advance, in `outputs/muon_sanity/protocol.json`) was: *prefer 0.003 if stable; otherwise take the first stable candidate in ascending order*, where "stable" means a finite run, training loss < 0.9, and holdout loss not worse than the initial holdout loss + 0.05. Ranking by accuracy or lowest loss was explicitly excluded.

Recorded results (`outputs/muon_sanity/results.json`):

| lr | train_loss | holdout_loss | holdout_accuracy | stable |
|---|---:|---:|---:|---|
| 0.001 | 0.6905 | 0.6664 | 0.6172 | yes |
| 0.003 | 0.6814 | 0.6280 | 0.6172 | yes |
| 0.01 | 0.6153 | 0.3560 | 0.8750 | yes |

All three candidates were stable, so the rule selected 0.003 as declared — not 0.01, even though 0.01 reached a lower holdout loss and much higher holdout accuracy in this short check. This is a deliberate anti-overfitting design (the rate is fixed before seeing full-run results), but it also means Muon's matrix learning rate was chosen by a stability criterion, while AdamW's 2e-5 reflects standard practice for fine-tuning this model rather than an equivalent stability sweep. The two optimizers therefore do not enter the full comparison with equivalently-tuned learning rates.

## Effective weight decay

Both optimizer configurations use the same nominal `weight_decay=0.01` on 2D (matrix) parameters, applied identically as decoupled multiplicative decay each step: `p *= (1 - lr * weight_decay)` (see `vendor/muon.py` lines 92/124/206/222/268/283, and PyTorch's AdamW for the fallback groups; wiring in `optimizers.py`, group setup in `train.py`).

Because this decay scales with `lr`, the same coefficient produces very different realized decay:

- AdamW matrix groups: `lr * weight_decay = 2e-5 * 0.01 = 2e-7` per step.
- Muon matrix group: `lr * weight_decay = 0.003 * 0.01 = 3e-5` per step — about **150x** larger per step than AdamW's.

Over a full 3-epoch run (6,315 optimizer steps for AdamW; a comparable step count for Muon), this compounds into a materially different amount of shrinkage applied to the hidden weight matrices, on top of the optimizers' different update rules. Any difference observed between AdamW and Muon in this experiment reflects this combined (update rule + effective decay + learning rate) system, not the update rule in isolation.

## 32-step diagnostic: update-norm trend

The matched 32-step run (`outputs/optimizer_diagnostics_32steps/all_steps.csv`) records gradient norm, update norm, and relative update norm at every step for both optimizers, for the "matrix" parameter group (the 37 hidden linear weights Muon optimizes directly) and the "fallback" group (handled by AdamW in both configurations).

Matrix-group **update norm** over the 32 steps:
- AdamW: starts at 0.131 (step 1), decreases through the run, ending near 0.050 (last 5 steps: 0.055, 0.055, 0.053, 0.050, 0.050). Range: 0.044–0.131.
- Muon: starts at 0.343 (step 1), rises over the first ~5 steps to about 0.49, then stays approximately flat for the remainder (last 5 steps: 0.494, 0.497, 0.497, 0.494, 0.497). Range: 0.343–0.518.

Matrix-group **gradient norm** over the same steps fluctuates for both optimizers in an overlapping range (0.5862–1.9162 for AdamW, 0.5494–2.4155 for Muon), with no sustained upward or downward trend and no consistent separation between optimizers. Since the update norms diverge sharply while the gradient norms do not, the difference in update size is attributable to how each optimizer transforms a given gradient — AdamW's per-parameter adaptive scaling versus Muon's orthogonalized (Newton–Schulz) step, which normalizes the update's spectral structure largely independent of the gradient's raw magnitude — rather than to the gradients being systematically larger under one optimizer. This is a description of the observed 32-step pattern, consistent with each optimizer's documented update rule; it is not evidence about *why* the two optimizers reached different final accuracy after 3 full epochs, which this short, separate run cannot establish.

Fallback-group (AdamW-optimized in both configurations) update norms are close between the two configurations (mean 0.00786 for the AdamW run vs 0.00791 for the Muon run), as expected since the same optimizer and rate govern this group in both cases.

## Training time

Wall-clock time for the full 3-epoch runs, from each run's `metrics.json` (`elapsed_seconds`, covering training, per-epoch validation, and checkpoint saving):

| optimizer | elapsed | approx. |
|---|---:|---|
| AdamW | 1603.19 s | ~26 min 43 s |
| Muon + AdamW fallback | 3621.39 s | ~60 min 21 s |

Muon took about **2.26x** as long as AdamW on this machine. This was measured on a single Apple M4 (MPS backend, no CUDA available), using the vendored reference `SingleDeviceMuon` implementation, which applies the Newton–Schulz orthogonalization iteration per parameter tensor in an unfused Python loop. That implementation detail, the MPS backend (which is generally less optimized than CUDA for custom kernels), or both, plausibly explain most of the gap. This is a cost measurement for this specific hardware/implementation pairing, not a general claim that Muon is inherently ~2x more expensive than AdamW; a fused or CUDA implementation could narrow or remove this gap.

## Fairness and reliability, summarized

- **Learning rates** were not chosen by an equivalent procedure for both optimizers (see above) — Muon's rate is stability-selected from a small sweep, AdamW's is a standard default.
- **Effective weight decay** differs by ~150x on matrix parameters despite an identical nominal coefficient, because decay is coupled to `lr` in both implementations.
- **Best checkpoints are at different epochs** (AdamW: epoch 3; Muon: epoch 1) with different baseline losses (0.2253 vs 0.3199), so the sharpness comparison is across different points in each model's training trajectory, not a fixed-stage comparison.
- **Single seed, single run** per optimizer for the full 3-epoch experiments; no variance estimate is available. The 32-step diagnostic uses matched initialization/batches/RNG between optimizers but is itself only one run per optimizer, over a short horizon, and must be read as characterizing early-step *dynamics*, not as reconstructing or explaining the full 3-epoch outcome.
