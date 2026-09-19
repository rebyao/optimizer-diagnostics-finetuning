# Stage-one baseline: AdamW / DistilBERT / SST-2

All 3 epochs completed. Epoch 3 was both the final and best epoch, with validation accuracy **90.825688%** (792/872) and validation loss **0.262449**. Reloading the saved checkpoint reproduced the full validation loss and accuracy exactly.

## Configuration and implementation

- Model: `distilbert-base-uncased`; revision `12040accade4e8a0f71eabdb258fecc2e7e948be`.
- Dataset: `nyu-mll/glue`, configuration `sst2`; 67,349 training examples and 872 validation examples.
- Device: Apple M4 GPU through PyTorch MPS; no CUDA.
- AdamW, 3 epochs, batch size 32, lr=2e-5, seed=42.
- Maximum sequence length 128, dynamic padding, FP32; constant learning rate, without a scheduler, warmup, gradient accumulation or clipping.
- AdamW betas=(0.9, 0.999), eps=1e-8; weight decay=0.01 for matrices, with no decay on one-dimensional parameters or biases.
- Fixed Python, NumPy, PyTorch/MPS and DataLoader shuffle seeds; deterministic algorithms enabled in warn-only mode. Bitwise equality across devices or dependency versions is not guaranteed.
- Explicit `zero_grad()` → forward → `backward()` → `step()`; no Trainer.
- Separate functions for the five main stages: `load_data()`, `create_model()`, `build_optimizer()`, `train_epoch()` and `evaluate()`.
- Muon, sharpness, gradient norms and additional optimizer diagnostics were not implemented at this stage.

## Verification and results

First, the same DistilBERT and batch size 32 were tested for 1 epoch on 64 randomly selected training examples and 64 validation examples:

- Forward pass, backward pass and optimizer step succeeded; an assertion confirmed an actual classifier-weight update.
- Smoke train loss=0.699324, validation loss=0.676999, accuracy=62.5%.
- The saved smoke checkpoint and tokenizer reloaded successfully for inference.
- The full experiment started independently from the base pretrained weights, not the smoke-test fine-tuned weights.

| Epoch | Mean training loss | Validation loss | Validation accuracy |
|---|---:|---:|---:|
| 1 | 0.220440 | 0.257660 | 88.8761% |
| 2 | 0.114025 | 0.287294 | 89.5642% |
| 3 | 0.076963 | 0.262449 | 90.8257% |

Batch size remained 32 throughout, with no out-of-memory errors or numerical failures. The run completed 6,315 optimizer steps. Training, per-epoch validation and saving took 1,603.19 seconds (approximately 26 minutes 43 seconds), excluding initial downloads, tokenization and the final reload verification. Checkpoint selection used validation accuracy rather than validation loss: epoch 3 had the highest accuracy, while epoch 1 had the lowest validation loss.

## Commands actually executed

Working directory: `/Users/rebeccayao/Desktop/Optimization_FT`.

```bash
.venv/bin/python -m py_compile train.py

PYTHONHASHSEED=42 .venv/bin/python -u train.py --smoke-test > training_smoke.log 2>&1

PYTHONHASHSEED=42 .venv/bin/python -u train.py --optimizer adamw --epochs 3 --batch-size 32 --lr 2e-5 --seed 42 > training_adamw.log 2>&1
```

Existing results are protected from overwriting. To repeat the full experiment, specify a new directory:

```bash
PYTHONHASHSEED=42 .venv/bin/python -u train.py \
  --optimizer adamw --epochs 3 --batch-size 32 --lr 2e-5 --seed 42 \
  --output-dir outputs/adamw_repeat
```

Command used for the final checkpoint verification:

```bash
PYTHONHASHSEED=42 HF_HUB_OFFLINE=1 .venv/bin/python - <<'PY'
import json
from pathlib import Path
import torch
from train import load_data, create_model, evaluate, seed_everything
seed_everything(42)
path = Path('outputs/adamw')
metrics = json.loads((path / 'metrics.json').read_text())
_, loaders, _ = load_data(str(path / 'best_model'), 32, 42, 128)
model = create_model(str(path / 'best_model'), torch.device('mps'))
loss, accuracy = evaluate(model, loaders['validation'], torch.device('mps'))
best = metrics['epochs'][metrics['best_epoch'] - 1]
assert accuracy == best['validation_accuracy']
assert abs(loss - best['validation_loss']) < 1e-6
result = {'checkpoint_reload': 'passed', 'validation_loss': loss,
          'validation_accuracy': accuracy, 'device': 'mps', 'validation_rows': 872}
(path / 'checkpoint_verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
PY
```

## File structure and artifacts

```text
Optimization_FT/
├── train.py
├── README.md
├── requirements.txt
├── installed-versions.txt
├── BASELINE_REPORT.zh-CN.md
├── REPORT.zh-CN.md                 # Original environment report
├── smoke_test.py                   # Original environment loading test
├── smoke_test.log
├── smoke_test_results.json
├── training_smoke.log
├── training_adamw.log
├── outputs/
│   ├── smoke/
│   │   ├── config.json
│   │   ├── metrics.json
│   │   └── best_model/
│   └── adamw/
│       ├── config.json
│       ├── metrics.json
│       ├── checkpoint_verification.json
│       └── best_model/
│           ├── model.safetensors
│           ├── config.json
│           ├── tokenizer.json
│           ├── tokenizer_config.json
│           ├── special_tokens_map.json
│           └── vocab.txt
├── .venv/
├── .python/
└── .cache/
```

Checkpoint weights occupy approximately 255 MiB. The tokenizer is included, and the model can be loaded directly with Hugging Face `from_pretrained()`. Only the model and tokenizer were saved; optimizer state for interrupted-training resumption was not saved.

## Issues and limitations

- No training failures, OOM errors or dependency conflicts; no new dependencies or smaller batch size were needed.
- The warning about randomly initialized DistilBERT classification-head weights was expected; that head was trained in this experiment. The base model supports downstream fine-tuning; see the [official model card](https://huggingface.co/distilbert/distilbert-base-uncased).
- Final verification ran offline, so datasets reported using its local cache. This was expected.
- Higher accuracy does not necessarily mean lower validation loss. Both metrics were retained without adjusting hyperparameters to improve the result.
- This was a single-seed baseline; multiple-seed statistics and comparisons with other optimizers had not yet been performed at this stage.
