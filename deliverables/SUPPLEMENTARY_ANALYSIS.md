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

## Hessian top-eigenvalue cross-check

`hessian_sharpness.py` adds a second, independent sharpness estimate for the same two best checkpoints: the top eigenvalue of the loss Hessian at each checkpoint, via power iteration on Hessian-vector products (HVPs) computed by double backward through `torch.autograd.grad` (no explicit Hessian is ever formed). It uses the same 256-example validation subset, seed (42), and mean-cross-entropy loss as `analyze_checkpoints.py`'s random-direction sharpness, over the same parameter scope (all named parameters). No checkpoint, optimizer state, or existing output file is modified; only new files are written under `outputs/hessian_sharpness/`.

**Smoke test and device selection.** A device smoke test (2 iterations on 8 examples) confirmed double backward works on this machine's MPS backend but fails on CPU: DistilBERT's default CPU attention kernel (`_scaled_dot_product_flash_attention_for_cpu`) has no implemented double-backward, so a CPU fallback would additionally require `attn_implementation="eager"`. A batch-size sweep on MPS then found that a single double-backward pass over all 256 examples at once is prohibitively expensive: 1.1s at batch 32, 1.9s at batch 64, 16.9s at batch 128, and an out-of-memory error at 256 (MPS reported needing over 15GB against an ~18GB ceiling once other allocations were counted). Rather than reduce the validation subset, the HVP for the full 256-example subset is computed as a size-weighted sum of 8 mini-batch (32-example) HVPs — exact, not approximate, because the Hessian of a mean loss is linear in the per-example loss (`H = (1/N) sum_i H_i`, so `Hv = sum_c (n_c/N)(H_c v)`). This keeps each mini-batch double-backward within the fast, stable regime the sweep identified and avoids the MPS blowup entirely.

**Avoiding the "largest-magnitude vs. largest-positive" trap.** Plain power iteration converges to whichever eigenvalue has the largest *magnitude*, which can be negative — exactly what happened here for Muon (see below). An initial implementation tried to isolate the top positive eigenvalue by shifting the whole operator (`H + c*I` with `c` large enough to make all eigenvalues non-negative) and re-running power iteration; this technically works but backfired in practice: because the found shift (~1.5x the raw dominant eigenvalue's magnitude) was far larger than the true top eigenvalue, it compressed all relative eigenvalue gaps in the shifted operator toward 1, and a naive relative-change convergence check then falsely reported convergence after 2 iterations before the iterate vector had rotated away from its (random) start — silently returning approximately 0 for both checkpoints. The script now instead uses Hotelling deflation: after the raw pass converges, a second power iteration runs (from a fresh random vector, not the raw eigenvector) on `H - raw_eig*(v v^T)`, which exactly removes the found eigenpair's contribution and searches for the next dominant-by-magnitude eigenvalue. Its sign is checked, and its eigenvector's overlap with the raw eigenvector is checked to confirm it did not collapse back to the same solution.

**Results** (`outputs/hessian_sharpness/hessian_sharpness.json`, full per-iteration Rayleigh quotients in `outputs/hessian_sharpness/hessian_convergence.csv`):

| optimizer | baseline loss | raw dominant eigenvalue | deflated-pass eigenvalue | top eigenvalue estimate | converged | reliable |
|---|---:|---:|---:|---:|---|---|
| AdamW | 0.2253 | 126.19 (10 iters) | 74.36 (12 iters) | **126.19** | yes | yes |
| Muon | 0.3199 | -1180.66 (8 iters) | 492.99 (9 iters) | **492.99** | yes | yes |

For AdamW, the raw pass already converged to a positive value, so it is directly the top algebraic eigenvalue (nothing else can exceed it in magnitude, so nothing else can exceed it in value). For Muon, the raw pass's dominant-magnitude eigenvalue was strongly negative (a large negative-curvature direction dominates its spectrum); the deflated pass found a positive eigenvalue (492.99) with near-zero eigenvector overlap with the raw pass (0.004), confirming it is a genuinely different direction rather than a repeat. All four power-iteration runs (raw and deflated, both checkpoints) show smooth, monotonic Rayleigh-quotient convergence with no oscillation.

**Comparison with the random-direction sharpness, and why they disagree.** The two methods point in different directions here. Random-direction perturbation (`analyze_checkpoints.py`) suggested Muon's checkpoint might be *flatter* (smaller mean loss increase at epsilon=0.1: 0.00435 vs AdamW's 0.00741, though the paired CI was inconclusive at that magnitude). The Hessian top eigenvalue instead shows Muon's checkpoint has *much higher* maximum curvature (492.99 vs AdamW's 126.19) — by this measure, Muon's checkpoint is sharper, not flatter. This is not a contradiction so much as two different notions of "sharpness": 8 random Gaussian directions in a 67M-dimensional space have (with overwhelming probability) negligible overlap with the single top eigenvector, so random-direction perturbation mostly measures an *average-case* curvature over generic directions, while the top Hessian eigenvalue is specifically the *worst-case, single-direction* curvature. A checkpoint can be flat along almost every direction while still having one sharp direction that a random probe would almost never find. Combined with the epoch mismatch already noted (AdamW's checkpoint is from epoch 3, Muon's from epoch 1, with a much higher baseline loss) it is also plausible that Muon's less-trained checkpoint simply sits in a less-settled region of the loss landscape with more extreme curvature in at least one direction. Neither measurement is more "correct" than the other; they characterize different aspects of the local loss surface, and both are subject to the single-seed, single-checkpoint-pair limitations already discussed above.

**Limitations specific to this estimate.** Only the single top eigenvalue is estimated, not the full spectrum or trace; deflation removes only one component, so if a second very-large-magnitude negative eigenvalue existed it would have been found instead of the true positive maximum (not the case here, given the clean sign flip and low eigenvector overlap, but not something this method rules out in general); the shift/deflation choice and Hotelling deflation both assume a symmetric operator, which the Hessian is, but floating-point HVPs make it only approximately so; and — as with the random-direction sharpness — this uses one 256-example subset and one seed per checkpoint, not a distribution over subsets.

### Numerical verification of the Hessian estimate

A follow-up, read-only verification pass (`outputs/hessian_sharpness/verification.json`, log in `outputs/hessian_sharpness/verification_log.txt`) checked the above numbers directly rather than trusting `reliable: true` at face value. It re-derives the eigenvectors (not saved by the original run), which the original script never persisted, and reruns nothing else.

- **HVP correctness.** A central finite-difference check of the gradient along a *generic random direction* was noise-dominated (true curvature there is tiny in a 67M-dim space, so fp32 rounding noise swamps the difference) and is not informative on its own. Repeating the check along the actual high-curvature eigenvector directions (signal ~|eigenvalue|, i.e. ~126 for AdamW and ~493 for Muon) gave a much cleaner signal: cosine similarity between the finite-difference and autograd HVP of 0.86-0.97 (AdamW) and 0.99 (Muon) across three step sizes, with the finite-difference norm within 3-30% of the true eigenvalue magnitude every time — far above what an actually-wrong HVP would produce (near-zero cosine), and consistent with expected finite-difference truncation/rounding error at this scale rather than an implementation bug.
- **Power iteration and deflation.** Rerunning the raw and deflated passes with the original seeds and protocol reproduced both reported eigenvalues to <1e-3 relative difference. The deflation formula was checked directly: the true (undeflated) Hessian's Rayleigh quotient at the deflated pass's final eigenvector matches the reported deflated-operator eigenvalue almost exactly (Muon: 492.974 vs. 492.99, AdamW: 74.362 vs. 74.357), which is expected precisely because the converged eigenvector is nearly orthogonal to the removed one — confirming the Hotelling deflation is implemented correctly.
- **Is Muon's 492.99 a reasonable top-positive-eigenvalue candidate?** Three pieces of evidence say yes, with one caveat. (1) The true-Hessian Rayleigh quotient at the found eigenvector is 492.974 with a small residual (4.996, ~1% of the eigenvalue), i.e. a genuine near-eigenpair of the real Hessian, not a deflation artifact. (2) Rerunning the deflated power iteration from **3 fresh random initializations** converges to the same eigenvalue (492.97-492.97, std 0.001) and the same eigenvector (overlap 0.99997-1.00000 with the original) every time — the mode is isolated and stable, not iteration noise. (3) A **second deflation pass**, removing both the -1180.66 and +492.99 modes, finds a next mode at +271.36 (true-Hessian Rayleigh, residual 1.31) — smaller than 492.99, so no larger positive mode was hiding immediately behind these two. **Caveat:** deflation is a greedy search that can only remove components it has explicitly found; it cannot formally rule out an even-larger-magnitude negative eigenvalue beyond -1180.66 whose removal might reveal a positive eigenvalue larger than 492.99. The evidence above makes that scenario look unlikely for this checkpoint/subset but does not exclude it. 492.99 (and 126.19 for AdamW) should therefore be read as *the largest positive eigenvalue found by this search*, not as a certified global maximum of the Hessian.

Convergence and the final estimates are plotted in [`hessian_convergence.png`](hessian_convergence.png).

## Fairness and reliability, summarized

- **Learning rates** were not chosen by an equivalent procedure for both optimizers (see above) — Muon's rate is stability-selected from a small sweep, AdamW's is a standard default.
- **Effective weight decay** differs by ~150x on matrix parameters despite an identical nominal coefficient, because decay is coupled to `lr` in both implementations.
- **Best checkpoints are at different epochs** (AdamW: epoch 3; Muon: epoch 1) with different baseline losses (0.2253 vs 0.3199), so the sharpness comparison is across different points in each model's training trajectory, not a fixed-stage comparison.
- **Single seed, single run** per optimizer for the full 3-epoch experiments; no variance estimate is available. The 32-step diagnostic uses matched initialization/batches/RNG between optimizers but is itself only one run per optimizer, over a short horizon, and must be read as characterizing early-step *dynamics*, not as reconstructing or explaining the full 3-epoch outcome.
