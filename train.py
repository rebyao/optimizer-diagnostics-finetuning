"""Explicit AdamW/Muon SST-2 comparison; use --smoke-test before a full run."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
DATASET_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import random
import time
from importlib.metadata import version

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_data(model_name, batch_size, seed, max_length, smoke_test=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=MODEL_REVISION if model_name == "distilbert-base-uncased" else None)
    raw = load_dataset("nyu-mll/glue", "sst2", revision=DATASET_REVISION)
    splits = {}
    for name in ("train", "validation"):
        data = raw[name]
        if smoke_test:
            data = data.shuffle(seed=seed).select(range(64))
        data = data.map(lambda batch: tokenizer(batch["sentence"], truncation=True,
                                               max_length=max_length), batched=True,
                        remove_columns=["sentence", "idx"])
        splits[name] = data.rename_column("label", "labels")
    collate = DataCollatorWithPadding(tokenizer)
    generator = torch.Generator().manual_seed(seed)
    loaders = {name: DataLoader(data, batch_size=batch_size, shuffle=name == "train",
                               collate_fn=collate, num_workers=0,
                               generator=generator if name == "train" else None)
               for name, data in splits.items()}
    return tokenizer, loaders, {name: data._fingerprint for name, data in splits.items()}


def create_model(model_name, device):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, use_safetensors=True,
        revision=MODEL_REVISION if model_name == "distilbert-base-uncased" else None,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1}).to(device)


def build_optimizer(model, optimizer_name, lr, fallback_lr=2e-5):
    if optimizer_name.lower() == "muon":
        from optimizers import build_muon
        return build_muon(model, lr, fallback_lr)
    if optimizer_name.lower() != "adamw":
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    # Keep the baseline explicit: constant LR; no decay on biases/LayerNorm.
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (no_decay if parameter.ndim < 2 else decay).append(parameter)
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": 0.01},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr, betas=(0.9, 0.999), eps=1e-8)


def train_epoch(model, loader, optimizer, device, epoch, verify_update=False, diagnostics=None):
    model.train()
    total_loss, count = 0.0, 0
    start = time.monotonic()
    for step, batch in enumerate(loader, 1):
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        before = None
        if verify_update and step == 1:
            names = ["classifier.weight"]
            if hasattr(optimizer, "muon"):
                names += ["distilbert.transformer.layer.0.attention.q_lin.weight",
                          "distilbert.embeddings.word_embeddings.weight",
                          "distilbert.embeddings.LayerNorm.weight", "classifier.bias"]
            parameters = dict(model.named_parameters())
            before = {name: parameters[name].detach().clone() for name in names}
        loss = model(**batch).loss
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        if diagnostics is not None:
            diagnostics.before_step(model)
        optimizer.step()
        if diagnostics is not None:
            diagnostics.after_step(model, epoch, loss.item())
        if before is not None:
            for name, old in before.items():
                current = parameters[name].detach()
                assert parameters[name].grad is not None, f"No gradient: {name}"
                assert torch.isfinite(current).all().item(), f"Non-finite parameter: {name}"
                assert not torch.equal(old, current), f"No parameter update: {name}"
                print(f"Smoke: forward/backward/step, parameter updated: {name}", flush=True)
        size = batch["labels"].size(0)
        total_loss += loss.item() * size
        count += size
        if step == 1 or step % 100 == 0 or step == len(loader):
            print(f"epoch={epoch} step={step}/{len(loader)} train_loss={total_loss/count:.4f} "
                  f"elapsed={time.monotonic()-start:.0f}s", flush=True)
    return total_loss / count


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        output = model(**batch)
        if not torch.isfinite(output.loss).item():
            raise RuntimeError("Non-finite validation loss")
        size = batch["labels"].size(0)
        loss_sum += output.loss.item() * size
        correct += (output.logits.argmax(-1) == batch["labels"]).sum().item()
        count += size
    return loss_sum / count, correct / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--fallback-lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--diagnostics", action="store_true", help="Record actual per-step gradient/update norms")
    args = parser.parse_args()
    if args.lr is None:
        if args.optimizer == "muon":
            parser.error("Muon requires an explicit --lr chosen by a small sanity check")
        args.lr = 2e-5
    if min(args.epochs, args.batch_size, args.max_length) < 1 or args.lr <= 0 or args.fallback_lr <= 0:
        parser.error("epochs, batch size, max length and lr must be positive")
    output_dir = args.output_dir or ROOT / "outputs" / (("smoke" if args.optimizer == "adamw" else "muon_smoke")
                                                       if args.smoke_test else args.optimizer)
    if (output_dir / "metrics.json").exists():
        raise FileExistsError(f"Existing run at {output_dir}; choose a fresh --output-dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer, loaders, fingerprints = load_data(args.model, args.batch_size, args.seed,
                                                args.max_length, args.smoke_test)
    model = create_model(args.model, device)
    optimizer = build_optimizer(model, args.optimizer, args.lr, args.fallback_lr)
    from optimizers import save_groups
    save_groups(optimizer, output_dir)
    epochs = 1 if args.smoke_test else args.epochs
    config = {**vars(args), "output_dir": str(output_dir), "epochs": epochs,
              "device": str(device), "weight_decay": 0.01, "lr_schedule": "constant",
              "precision": "float32", "deterministic_warn_only": True,
              "model_revision": model.config._commit_hash,
              "dataset_fingerprints": fingerprints,
              "train_rows": len(loaders["train"].dataset),
              "validation_rows": len(loaders["validation"].dataset),
              "versions": {name: version(name) for name in ("torch", "transformers", "datasets", "numpy")}}
    if args.optimizer == "muon":
        config["muon_source"] = json.loads((ROOT / "vendor/SOURCE.json").read_text())
        config["muon_groups_file"] = "parameter_groups.json"
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"device={device} train={config['train_rows']} validation={config['validation_rows']} "
          f"batch_size={args.batch_size} epochs={epochs}", flush=True)
    diagnostics = None
    if args.diagnostics:
        from diagnostics import StepDiagnostics
        diagnostics = StepDiagnostics(output_dir / "step_diagnostics.csv")
    history, best_accuracy, best_epoch = [], -1.0, None
    start = time.monotonic()
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, loaders["train"], optimizer, device, epoch, args.smoke_test, diagnostics)
        val_loss, val_accuracy = evaluate(model, loaders["validation"], device)
        metrics = {"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                   "validation_accuracy": val_accuracy}
        history.append(metrics)
        if val_accuracy > best_accuracy:
            best_accuracy, best_epoch = val_accuracy, epoch
            model.save_pretrained(output_dir / "best_model", safe_serialization=True)
            tokenizer.save_pretrained(output_dir / "best_model")
        result = {"epochs": history, "best_epoch": best_epoch,
                  "best_validation_accuracy": best_accuracy,
                  "checkpoint": str(output_dir / "best_model"),
                  "completed": epoch == epochs, "elapsed_seconds": time.monotonic() - start}
        (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(metrics), flush=True)
    print(f"Done. best_epoch={best_epoch} best_accuracy={best_accuracy:.4f} "
          f"metrics={output_dir / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
