# AdamW vs Muon experiment report

The full experiment completed, with the original AdamW results preserved. Under this fixed configuration, Muon + AdamW fallback achieved lower final validation accuracy than AdamW. No additional tuning or reruns were performed in response to the full validation results.

## Results

| Optimizer | Epoch | Train loss | Validation loss | Validation accuracy |
|---|---:|---:|---:|---:|
| AdamW | 1 | 0.220440 | 0.257660 | 88.8761% |
| AdamW | 2 | 0.114025 | 0.287294 | 89.5642% |
| AdamW | 3 | 0.076963 | 0.262449 | **90.8257%** |
| Muon + AdamW fallback | 1 | 0.254595 | 0.369564 | **87.0413%** |
| Muon + AdamW fallback | 2 | 0.169230 | 0.570577 | 83.8303% |
| Muon + AdamW fallback | 3 | 0.136087 | 0.393930 | **84.4037%** |

Comparing epoch 3 for both runs, Muon accuracy was **6.4220 percentage points lower**, and validation loss was **0.131481 higher**. The best checkpoints were AdamW epoch 3 and Muon epoch 1; the best-to-best accuracy difference was **−3.7844 percentage points**.

Muon training, per-epoch validation and saving took 3,621.39 seconds (approximately 60 minutes 21 seconds), compared with 1,603.19 seconds for AdamW. On this machine with this implementation, Muon took approximately 2.26 times as long; this is not a general hardware-performance conclusion.

## Settings held constant

- The same `distilbert-base-uncased` model revision, `12040accade4e8a0f71eabdb258fecc2e7e948be`, starting afresh from base pretrained weights.
- GLUE/SST-2: 67,349 training examples and 872 validation examples; batch size=32, epochs=3, seed=42.
- Maximum length 128, dynamic padding, FP32 model, MPS GPU and constant learning rate, without a scheduler.
- The original `load_data()`, `create_model()`, `seed_everything()` and `evaluate()` functions were unchanged at this stage. The training loop only gained stronger smoke-test assertions for parameter updates.
- Python and package versions were not upgraded, and no large dependencies were added. Full training did not inherit model or optimizer states from smoke or sanity checks.
- Training and validation losses remained sample-weighted; accuracy used the same evaluation function.
- Parameter grouping and optimizer construction did not consume random numbers; the training dataset and shuffle seed were identical.

## Muon implementation and parameter groups

The implementation uses `SingleDeviceMuon` from the [original KellerJordan/Muon repository](https://github.com/KellerJordan/Muon/tree/f98f1cacc0263b04290753e32be8d498c1efc806), pinned to commit `f98f1cacc0263b04290753e32be8d498c1efc806`. The source file was unchanged, and its MIT license was retained. Since torch 2.8 did not include a built-in Muon optimizer, the reference source was placed in `vendor/` rather than upgrading PyTorch.

| Group | Tensors | Parameters | LR | Weight decay |
|---|---:|---:|---:|---:|
| Muon | 37 | 43,057,152 | 0.003 | 0.01 |
| AdamW fallback, 2D parameters | 3 | 23,835,648 | 0.00002 | 0.01 |
| AdamW fallback, 1D parameters/biases | 64 | 62,210 | 0.00002 | 0 |

Muon handles only hidden `Linear.weight` matrices: the attention/FFN matrices in 6 Transformer layers, plus `pre_classifier.weight`. Word/position embeddings, all biases, LayerNorm and the final `classifier.weight` use AdamW. Every trainable parameter belongs to exactly one group, with no omissions or overlaps.

Muon uses momentum=0.95, Nesterov updates, 5 Newton–Schulz steps and the author's `sqrt(max(1, rows/cols))` update scaling. Model parameters and momentum are FP32; temporary orthogonalization matrices use BF16. The fallback is native `torch.optim.AdamW`, with betas=(0.9,0.999) and eps=1e-8.

Parameter names, shapes, counts and hyperparameters were printed to `training_muon.log` and saved in `outputs/muon/parameter_groups.json`.

## Smoke test and LR sanity check

First, the same model and batch size were run for 1 epoch on 64 training and 64 validation examples. Forward passes, backward passes, optimizer steps and validation all passed. Assertions confirmed updates to representative attention matrices, embeddings, LayerNorm, classification-head weights and biases. Smoke validation loss was 0.684763, with accuracy 64.0625%.

Each of three learning rates then ran for 16 steps on 512 training examples, with a disjoint set of 128 training examples as a holdout. The official validation set was not used to select LR. Each candidate restarted with the same model initialization, fresh optimizer, random seed and batch order.

| Muon LR | Train loss | Holdout loss | Holdout accuracy | Stability check |
|---|---:|---:|---:|---|
| 0.001 | 0.690455 | 0.666385 | 61.7188% | Passed |
| **0.003** | 0.681401 | 0.628045 | 61.7188% | Passed; selected |
| 0.01 | 0.615311 | 0.355985 | 87.5000% | Passed |

Initial holdout loss was 0.703467 for all candidates. The selection rule was saved before execution: require finite values, mean training loss <0.9, and holdout loss no greater than the initial value+0.05; prefer 0.003, falling back to another stable candidate only if needed. Although 0.01 achieved higher small-sample accuracy, the predeclared rule selected 0.003 rather than tuning for the highest score.

The rule, sample IDs and candidate results are in `outputs/muon_sanity/`. A small-sample check rules out obvious numerical problems, but does not guarantee stable generalization during full training.

## Commands actually executed

From the project root:

```bash
# Muon smoke test
PYTHONHASHSEED=42 HF_HUB_OFFLINE=1 .venv/bin/python -u train.py --optimizer muon --lr 0.003 --smoke-test > training_muon_smoke.log 2>&1

# Small-sample check of three candidate learning rates
PYTHONHASHSEED=42 HF_HUB_OFFLINE=1 .venv/bin/python -u muon_sanity.py > training_muon_sanity.log 2>&1

# Full 3-epoch experiment
PYTHONHASHSEED=42 HF_HUB_OFFLINE=1 .venv/bin/python -u train.py --optimizer muon --lr 0.003 --fallback-lr 2e-5 --epochs 3 --batch-size 32 --seed 42 --output-dir outputs/muon > training_muon.log 2>&1

# Generate comparison tables and verify that original AdamW files are unchanged
.venv/bin/python compare.py
```

Repeating a full experiment requires a new `--output-dir`; existing metrics are protected from overwriting. Offline mode depends on the project's existing cache. Omit `HF_HUB_OFFLINE=1` for initial downloads on a new machine.

## Files and verification

```text
train.py                          # Preserved AdamW branch; added Muon entrypoint
optimizers.py                     # Parameter routing; Muon + native AdamW
muon_sanity.py                    # Three-LR check on small training subsets
compare.py                        # CSV/Markdown comparison and preservation checks
vendor/{muon.py,LICENSE,SOURCE.json}
adamw_preservation.json           # Original AdamW output SHA-256 hashes
training_muon{,_smoke,_sanity}.log
outputs/
  adamw/                          # Original contents unchanged
  muon_smoke/
  muon_sanity/{protocol.json,sample_ids.json,results.json,...}
  muon/
    config.json
    parameter_groups.json
    metrics.json
    checkpoint_verification.json
    best_model/                   # Epoch 1, not epoch 3
  data_equivalence.json
  comparison.csv
  comparison.md
  comparison_checks.json
```

Model and data configuration checks passed. Validation cache fingerprints differed, but row-by-row verification of input_ids, attention_mask, labels and order produced identical content SHA-256 hashes. The training data passed the same verification; see `data_equivalence.json`. Original AdamW output files, including its checkpoint, passed SHA-256 verification and were not overwritten.

## Interpretation

Under this configuration, Muon's training loss decreased, but validation accuracy fell from 87.04% at epoch 1 to 83.83% at epoch 2, then recovered to 84.40% at epoch 3. These results were retained without changing LR or adding experiments to maximize accuracy. No OOM errors, NaNs or training failures occurred.

This is one result for **Muon + AdamW fallback, LR=0.003, seed=42**, not evidence that Muon is universally better or worse than AdamW. Both used the same weight-decay coefficient, but their different learning rates produced different effective per-step decay: `lr*weight_decay=3e-5` for Muon matrices versus `2e-7` for AdamW. Consequently, the result difference cannot be attributed entirely to orthogonalization. Update magnitudes and cumulative decay were not matched, and no extensive hyperparameter search was performed.

As requested at this stage, no sharpness, gradient-norm or multiple-seed experiments were added.

The final best checkpoint was reloaded offline and evaluated on all 872 validation examples using MPS: loss=0.3695635207749288 and accuracy=0.8704128440366973, exactly matching epoch 1. Verification evidence is saved in `outputs/muon/checkpoint_verification.json`.
