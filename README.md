# AdamW vs Muon — DistilBERT / SST-2

This project contains three **separate completed experiments**: full fine-tuning, a matched 32-step optimizer diagnostic run, and checkpoint perturbation analysis. Consolidation reads saved results only; it does not train or modify checkpoints.

## Read the results

- [One-page English report (PDF)](deliverables/REPORT_EN.pdf) / [Markdown](deliverables/REPORT_EN.md). The assignment's four questions were not provided; the current four-question mapping is explicitly provisional.
- [Results tables](deliverables/RESULTS.md): [performance](deliverables/performance.csv), [32-step diagnostics](deliverables/diagnostics_means.csv), [sharpness](deliverables/sharpness.csv).
- [Performance figure](deliverables/performance.png), [optimizer diagnostics figure](deliverables/optimizer_diagnostics.png), [sharpness figure](deliverables/sharpness.png). Vector PDFs are also available.

| Measurement | AdamW | Muon + AdamW fallback |
|---|---:|---:|
| Full-run final validation accuracy | 90.8257% | 84.4037% |
| Full-run final validation loss | 0.262449 | 0.393930 |
| Best validation accuracy (epoch) | 90.8257% (3) | 87.0413% (1) |
| Separate 32-step mean training loss | 0.671843 | 0.651614 |
| Separate 32-step mean update norm | 0.055392 | 0.493242 |
| Best-checkpoint symmetric mean Δloss, ε=0.1 | 0.007410 | 0.004354 |

Muon had lower early loss and larger updates in the short run, but worse final accuracy in full training. Sharpness remains inconclusive: the paired directional difference interval includes zero. Historical three-epoch update/gradient norms were never recorded; the short run does not reconstruct them. Best checkpoints represent different epochs.

## Environment

Tested: CPython **3.11.16**, macOS arm64, Apple M4, MPS; CUDA unavailable on this machine. Device selection prefers CUDA, then MPS, then CPU. Use Python 3.11 for this exact dependency set.

On a fresh checkout/environment:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

`requirements.txt` pins direct dependencies, including SciPy used for the directional t interval. `installed-versions.txt` is the complete historical environment snapshot (including transitive dependencies). To reproduce that exact snapshot on compatible Python/platforms, install with `-r installed-versions.txt` instead. Neither approach guarantees bitwise equality across hardware. No environment changes or package installation were performed during consolidation.

The model and dataset revisions used by the experiments are now pinned in `train.py` and reused by the experimental entrypoints:

- DistilBERT: `12040accade4e8a0f71eabdb258fecc2e7e948be`.
- `nyu-mll/glue`, SST-2: `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`.
- Muon: vendored, unmodified `SingleDeviceMuon`; commit/license/checksum in `vendor/SOURCE.json` and `vendor/LICENSE`.

`.cache/huggingface/` holds model/dataset downloads. Use `HF_HUB_OFFLINE=1` when the cache is present; omit it on first use on another machine. `.python/` is the local interpreter backing the existing `.venv`; do not delete it while using this environment.

## Regenerate only figures and tables — no training

```bash
.venv/bin/python summarize_experiments.py --output-dir deliverables_repeat
```

This verifies completed records, writes new CSV/Markdown/PNG/PDF outputs, and hashes all original `outputs/` files before/after. It does not import training code or load models. The English report is editorial text grounded in these tables; `render_report.py` renders `deliverables/REPORT_EN.md` into one A4 PDF page. The renderer does not run experiments.

## Experimental commands — reference only

**These commands run experiments; they were not rerun during consolidation.** Fresh output directories are mandatory. For a complete end-to-end reproduction with default paths, use a separate clean project copy containing the code/dependencies but no `outputs/` results. On this existing project, use the new directories shown below.

### Full runs and smoke tests

```bash
PYTHONHASHSEED=42 .venv/bin/python train.py --smoke-test --output-dir outputs/repro_adamw_smoke
PYTHONHASHSEED=42 .venv/bin/python train.py --optimizer muon --lr 0.003 --smoke-test --output-dir outputs/repro_muon_smoke

PYTHONHASHSEED=42 .venv/bin/python train.py --optimizer adamw \
  --epochs 3 --batch-size 32 --lr 2e-5 --seed 42 --output-dir outputs/repro_adamw
PYTHONHASHSEED=42 .venv/bin/python train.py --optimizer muon \
  --epochs 3 --batch-size 32 --lr 0.003 --fallback-lr 2e-5 --seed 42 --output-dir outputs/repro_muon
```

Constant LR; maximum length128; dynamic padding; FP32 model; no warmup, scheduler or clipping. Matrix weight decay0.01, 1D parameters/biases0. Muon handles 37 hidden linear matrices; embeddings, LayerNorm, biases and final classifier use AdamW. Muon internally orthogonalizes in BF16. Same decay coefficient with different LR means different effective decay.

`muon_sanity.py` reproduces the predeclared LR check (0.001/0.003/0.01), using 512 training examples plus 128 disjoint training-only holdout examples, 16 steps per candidate. It writes `outputs/muon_sanity/` and refuses existing results; run it in a clean reproduction copy. It selects 0.003 if stable, not the highest accuracy. The saved original protocol is in `outputs/muon_sanity/protocol.json`.

### Exactly 32 steps per optimizer

```bash
PYTHONHASHSEED=42 .venv/bin/python run_optimizer_diagnostics.py \
  --output-dir outputs/repro_diagnostics_32steps
```

Reads original optimizer configurations from `outputs/adamw/` and `outputs/muon/`. Verifies identical pretrained initialization, 1024 examples, batch order and per-step RNG states. Records real pre-step gradients and post-step updates, globally and for matching matrix/fallback groups. It does not save a model. `train.py --diagnostics` additionally supports optional global per-step recording for future authorized runs.

### Checkpoint sharpness (no optimizer steps)

```bash
PYTHONHASHSEED=42 .venv/bin/python analyze_checkpoints.py \
  --output-dir outputs/repro_sharpness
```

Reads the original best checkpoints at `outputs/{adamw,muon}/best_model/`. Uses 256 shared validation examples, eight paired directions and ε∈{0,0.001,0.003,0.01,0.03,0.1}; perturbations are normalized per parameter tensor. Restores weights and verifies file hashes. Reported uncertainty is over directions only, not training seeds or datasets.

Analysis/comparison scripts intentionally reference the original run locations. To analyze a new complete reproduction without changing originals, use default run locations in a separate clean project copy. `compare.py` is a legacy original-run verification tool tied to `adamw_preservation.json`; it is not required for new reproductions.

## Files and provenance

```text
train.py, optimizers.py, vendor/       training loop and pinned Muon source
muon_sanity.py                         tiny LR stability check
run_optimizer_diagnostics.py          fixed 32-step matched experiment
diagnostics.py                        global/grouped norm recording
analyze_checkpoints.py                 checkpoint gradients and perturbations
summarize_experiments.py               results-only tables and figures
render_report.py                       one-page English PDF
requirements.txt, installed-versions.txt
outputs/                              original metrics, protocols, raw CSVs, checkpoints
  adamw/, muon/                       full runs (unchanged)
  optimizer_diagnostics_32steps/       independent short-run data
  diagnostics/                        perturbation data and endpoint gradients
  *_smoke/, smoke/, muon_sanity/       earlier verification/stability records
deliverables/                         consolidated report, figures, tables, provenance
*.log, *REPORT*.md, *.zh-CN.md          original logs and detailed stage reports
```

All raw results, checkpoints, logs and earlier reports are retained in place. `outputs/diagnostics_setup_attempt/` contains only an earlier failed setup's audit manifest; it is retained because later provenance manifests reference it. Cleanup removes regenerable project Python bytecode only, not experiment data, model caches or the environment. See `deliverables/provenance.json` and `deliverables/cleanup.json`.


## Detailed stage reports (English)

All previously Chinese stage reports have been translated into English. Their historical `.zh-CN.md` filenames are retained to preserve existing paths and links. Statements about work not yet performed describe the stage at which each report was written.

- [Environment setup](REPORT.md)
- [AdamW baseline](BASELINE_REPORT.md)
- [Muon comparison](MUON_REPORT.md)
- [Checkpoint diagnostics and sharpness](DIAGNOSTICS_REPORT.md)
- [32-step optimizer diagnostics](OPTIMIZER_DIAGNOSTICS_32STEPS.md)
- [Original comparison summary](outputs/comparison.md)

Translation changes documentation only. Numerical experimental files and checkpoints remain unchanged. Earlier provenance manifests retain their historical hashes; `deliverables/report_translation_audit.json` records the authorized report-text changes, including the derived `outputs/comparison.md`.
