"""Load a tiny model and SST-2, then run inference only (no training)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import platform
from importlib.metadata import version
import json
import torch
import transformers
import datasets
import evaluate
import sklearn
import numpy
import pandas
import matplotlib
import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset


def main():
    cuda = torch.cuda.is_available()
    mps = torch.backends.mps.is_available()
    device = "cuda" if cuda else "mps" if mps else "cpu"
    report = {
        "python": platform.python_version(),
        "packages": {name: version(name) for name in (
            "torch", "transformers", "datasets", "evaluate", "scikit-learn",
            "numpy", "pandas", "matplotlib", "tqdm")},
        "cuda_available": cuda,
        "cuda_build": torch.version.cuda,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": mps,
        "gpu_name": torch.cuda.get_device_name(0) if cuda else
            "Apple M4 (Metal/MPS)" if mps else None,
        "device": device,
    }
    print(json.dumps(report, indent=2), flush=True)
    a = torch.ones((32, 32), device=device)
    assert torch.allclose((a @ a).cpu(), torch.full((32, 32), 32.0))
    model_id = "hf-internal-testing/tiny-random-bert"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, use_safetensors=True
    ).to(device).eval()
    dataset = load_dataset("nyu-mll/glue", "sst2")
    assert set(dataset) == {"train", "validation", "test"}
    batch = tokenizer(dataset["validation"][:2]["sentence"],
                      return_tensors="pt", padding=True, truncation=True)
    with torch.inference_mode():
        logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
    assert logits.shape == (2, 2)
    assert torch.isfinite(logits).all().item()
    report.update({
        "model": model_id,
        "model_revision": model.config._commit_hash,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "dataset": "nyu-mll/glue/sst2",
        "dataset_rows": {split: len(data) for split, data in dataset.items()},
        "logits_shape": list(logits.shape),
        "inference_device": str(logits.device),
        "smoke_test": "passed",
        "training_started": False,
    })
    (ROOT / "smoke_test_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
