# AdamW vs Muon — DistilBERT / SST-2

Comparison of AdamW and Muon (with an AdamW fallback for non-matrix parameters) for fine-tuning DistilBERT on SST-2. The repository covers three separate experiments: full 3-epoch fine-tuning, a matched 32-step optimizer diagnostic run, and a checkpoint sharpness (perturbation) analysis.

## Project overview

- **Model / task**: DistilBERT fine-tuned for binary sentiment classification on GLUE SST-2 (67,349 train / 872 validation examples).
- **Comparison**: AdamW (lr 2e-5) vs Muon for the 37 hidden linear weight matrices with AdamW fallback (lr 2e-5) for embeddings, biases, LayerNorm and the classifier head. Muon's fallback learning rate (0.003) was chosen by a predeclared stability check over three candidates, not by picking the highest score (`muon_sanity.py`, `outputs/muon_sanity/`).
- **Diagnostics**: a separate, matched 32-step run records real per-step gradient and update norms for both optimizers under identical initialization, batches, and RNG state (`run_optimizer_diagnostics.py`, `outputs/optimizer_diagnostics_32steps/`).
- **Sharpness**: the best checkpoint from each full run is perturbed along 8 paired random directions at 5 magnitudes to compare local loss curvature (`analyze_checkpoints.py`, `outputs/diagnostics/`).
- Muon is vendored unmodified as `SingleDeviceMuon` (`vendor/muon.py`); its license and source commit/checksum are in `vendor/LICENSE` and `vendor/SOURCE.json`.

## Setup & reproduction

Tested on CPython 3.11.16, macOS arm64 (Apple M4, MPS); CUDA unavailable on this machine. Device selection prefers CUDA, then MPS, then CPU.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

`requirements.txt` pins direct dependencies. `installed-versions.txt` is the full historical environment snapshot (including transitive dependencies); install with `-r installed-versions.txt` instead to reproduce that exact snapshot on a compatible Python/platform. Neither approach guarantees bitwise-identical results across different hardware.

Model/dataset revisions are pinned in `train.py`:

- DistilBERT: `12040accade4e8a0f71eabdb258fecc2e7e948be`
- `nyu-mll/glue`, SST-2: `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`

### Regenerate report tables and figures only (no training)

```bash
.venv/bin/python summarize_experiments.py --output-dir deliverables_repeat
.venv/bin/python render_report.py
```

`summarize_experiments.py` reads the existing files under `outputs/`, hashes them before and after, and writes fresh CSV/Markdown/PNG/PDF outputs — it does not load any model or run training. `render_report.py` renders `deliverables/REPORT_EN.md` into a one-page PDF.

### Full experiment commands (reference only — not rerun for this submission)

These write to **new** output directories so the original results under `outputs/` are never overwritten. Run in a clean copy of this project for a complete end-to-end reproduction (weights are not included in this repository — see [Limitations](#limitations)).

```bash
# Full 3-epoch training
PYTHONHASHSEED=42 .venv/bin/python train.py --optimizer adamw \
  --epochs 3 --batch-size 32 --lr 2e-5 --seed 42 --output-dir outputs/repro_adamw
PYTHONHASHSEED=42 .venv/bin/python train.py --optimizer muon \
  --epochs 3 --batch-size 32 --lr 0.003 --fallback-lr 2e-5 --seed 42 --output-dir outputs/repro_muon

# Matched 32-step optimizer diagnostics (reads original run configs from outputs/adamw, outputs/muon)
PYTHONHASHSEED=42 .venv/bin/python run_optimizer_diagnostics.py \
  --output-dir outputs/repro_diagnostics_32steps

# Checkpoint sharpness (reads original best checkpoints from outputs/{adamw,muon}/best_model)
PYTHONHASHSEED=42 .venv/bin/python analyze_checkpoints.py \
  --output-dir outputs/repro_sharpness
```

Fixed settings: constant learning rate, max length 128, dynamic padding, FP32 model, no warmup/scheduler/clipping. Weight decay 0.01 on matrix parameters, 0 on 1D parameters/biases. Muon's internal orthogonalization runs in BF16.

`muon_sanity.py` reruns the predeclared learning-rate check (0.001 / 0.003 / 0.01) on 512 training + 128 disjoint holdout examples, 16 steps per candidate; it refuses to overwrite an existing `outputs/muon_sanity/`.

## Main results

| Measurement | AdamW | Muon + AdamW fallback |
|---|---:|---:|
| Full-run final validation accuracy | 90.8257% | 84.4037% |
| Full-run final validation loss | 0.262449 | 0.393930 |
| Best validation accuracy (epoch) | 90.8257% (epoch 3) | 87.0413% (epoch 1) |
| Separate 32-step mean training loss | 0.671843 | 0.651614 |
| Separate 32-step mean global update norm | 0.055392 | 0.493242 |
| Best-checkpoint symmetric mean Δloss, ε = 0.1 | 0.007410 | 0.004354 |

Muon showed lower loss and larger parameter updates during the short 32-step run, but ended full training with lower validation accuracy than AdamW under this fixed configuration. Sharpness is inconclusive: the paired directional difference in loss increase between optimizers includes zero across all tested magnitudes. Full tables are in [`deliverables/RESULTS.md`](deliverables/RESULTS.md).

## Report and figures

- **Report**: [`deliverables/REPORT_EN.md`](deliverables/REPORT_EN.md) / [`deliverables/REPORT_EN.pdf`](deliverables/REPORT_EN.pdf)
- **Result tables**: [`deliverables/RESULTS.md`](deliverables/RESULTS.md), with source CSVs [`performance.csv`](deliverables/performance.csv), [`epochs.csv`](deliverables/epochs.csv), [`diagnostics_means.csv`](deliverables/diagnostics_means.csv), [`sharpness.csv`](deliverables/sharpness.csv), [`sharpness_paired.csv`](deliverables/sharpness_paired.csv)
- **Figures**: [`training_loss.png`](deliverables/training_loss.png), [`gradient_update_norms.png`](deliverables/gradient_update_norms.png), [`optimizer_diagnostics.png`](deliverables/optimizer_diagnostics.png), [`sharpness.png`](deliverables/sharpness.png), [`performance.png`](deliverables/performance.png) (vector PDFs of each are also provided alongside the PNGs)
- **Provenance**: [`deliverables/provenance.json`](deliverables/provenance.json) records SHA-256 hashes of the original `outputs/` files used to build the report; [`deliverables/cleanup.json`](deliverables/cleanup.json) and [`deliverables/report_translation_audit.json`](deliverables/report_translation_audit.json) log the non-experimental cleanup and report-text edits applied after the runs completed.

## Repository layout

```text
train.py, optimizers.py, vendor/     training loop and vendored Muon optimizer
muon_sanity.py                       learning-rate stability check
run_optimizer_diagnostics.py         matched 32-step optimizer diagnostic run
diagnostics.py                       gradient/update norm recording helpers
analyze_checkpoints.py               checkpoint perturbation / sharpness analysis
summarize_experiments.py             builds deliverables/ tables and figures from outputs/
render_report.py                     renders the one-page English PDF report
requirements.txt, installed-versions.txt
outputs/                             raw metrics, protocols, and CSVs from each run
  adamw/, muon/                        full 3-epoch runs (config, metrics, checkpoint verification)
  optimizer_diagnostics_32steps/       32-step matched diagnostic data
  diagnostics/                         checkpoint gradients and sharpness perturbation data
  muon_sanity/                         learning-rate stability check results
  comparison.csv, comparison.md        run-to-run comparison summary
deliverables/                        final report, figures, and result tables
```

Saved model checkpoint weights are not included in this repository (see Limitations); everything needed to inspect metrics, regenerate the report, or rerun the experiments end-to-end from a fresh checkpoint is included.

## Limitations

- Single training seed (42) per optimizer; no repeated runs to estimate run-to-run variance.
- Checkpoint weight files (`model.safetensors`) are intentionally excluded from this repository to keep it small; `analyze_checkpoints.py` and the sharpness figures were produced from local checkpoints that are not included here, so that specific analysis step cannot be rerun from a fresh clone without first completing full training.
- Historical gradient/update norms for the original 3-epoch runs were never recorded and cannot be reconstructed; the 32-step diagnostic run is a separate, shorter experiment under matched conditions, not a continuation of the original training trajectory.
- Sharpness is estimated from 8 random paired directions per checkpoint, which do not identify worst-case curvature, and the perturbation normalization is not fully reparameterization-invariant.
- The best AdamW and Muon checkpoints come from different epochs (3 vs 1), which limits direct comparison of sharpness at a fixed training stage.
- The assignment's original four questions were not provided to us; the four themes in the report (design, performance, dynamics, sharpness) are a provisional mapping.
