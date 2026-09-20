# Archive

Historical materials kept for provenance. Nothing here is required to reproduce the results in `deliverables/`; see the top-level `README.md` for the current reproduction pipeline.

- `reports/` — stage-by-stage process reports written during the project (environment setup, AdamW baseline, Muon comparison, checkpoint diagnostics, 32-step diagnostics). Superseded by `deliverables/REPORT_EN.md`.
- `smoke_test/` — the original smoke-test script and its recorded output, used to verify the training pipeline before running full experiments. Paths inside `smoke_test.py` are relative to its original location at the project root and are not updated for its new location here. It also imports `evaluate`, `scikit-learn`, and `tqdm`, which are not required by the current pipeline and were removed from `requirements.txt`; install them separately to run this script.
- `legacy/` — `compare.py`, an early original-run verification tool tied to `adamw_preservation.json`; superseded by `analyze_checkpoints.py` and `outputs/comparison.*`.
