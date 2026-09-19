"""Small, predeclared Muon LR stability check using only SST-2 training rows."""
import gc
import json
from pathlib import Path

from train import MODEL_REVISION, DATASET_REVISION
from train import (ROOT, seed_everything, create_model, build_optimizer,
                   train_epoch, evaluate)
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from optimizers import save_groups


def main():
    output = ROOT / 'outputs/muon_sanity'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'results.json').exists():
        raise FileExistsError('Preserving existing sanity results')
    candidates = [0.001, 0.003, 0.01]
    protocol = {
        'candidates': candidates, 'preferred_lr': 0.003, 'fallback_lr': 2e-5,
        'train_rows': 512, 'holdout_rows': 128, 'seed': 42, 'batch_size': 32,
        'max_length': 128, 'epochs_per_candidate': 1,
        'data_source': 'nyu-mll/glue sst2 train only; disjoint seeded subsets',
        'selection_rule': 'Prefer 0.003 if stable; otherwise first stable candidate in ascending order. '
                          'Stable = finite run, train loss < 0.9, and holdout loss <= initial loss + 0.05. '
                          'Do not rank by accuracy or lowest validation loss.',
    }
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', revision=MODEL_REVISION)
    raw = load_dataset('nyu-mll/glue', 'sst2', split='train',revision=DATASET_REVISION).shuffle(seed=42).select(range(640))
    (output / 'sample_ids.json').write_text(json.dumps({
        'train_idx': raw[:512]['idx'], 'holdout_idx': raw[512:]['idx']}, indent=2) + '\n')
    data = raw.map(lambda batch: tokenizer(batch['sentence'], truncation=True, max_length=128),
                   batched=True, remove_columns=['sentence', 'idx']).rename_column('label', 'labels')
    collate = DataCollatorWithPadding(tokenizer)
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    results = []
    for lr in candidates:
        seed_everything(42)
        train_loader = DataLoader(data.select(range(512)), batch_size=32, shuffle=True,
                                  generator=torch.Generator().manual_seed(42), collate_fn=collate)
        holdout = DataLoader(data.select(range(512, 640)), batch_size=32, collate_fn=collate)
        model = create_model('distilbert-base-uncased', device)
        optimizer = build_optimizer(model, 'muon', lr)
        candidate_dir = output / f'lr_{lr:g}'
        candidate_dir.mkdir(exist_ok=True)
        save_groups(optimizer, candidate_dir)
        initial_loss, _ = evaluate(model, holdout, device)
        # Reset RNG consumed by the initial evaluation's DataLoader iterator.
        seed_everything(42)
        loss = train_epoch(model, train_loader, optimizer, device, 1, verify_update=True)
        holdout_loss, accuracy = evaluate(model, holdout, device)
        stable = loss < 0.9 and holdout_loss <= initial_loss + 0.05
        row = {'lr': lr, 'initial_holdout_loss': initial_loss, 'train_loss': loss,
               'holdout_loss': holdout_loss, 'holdout_accuracy': accuracy, 'stable': stable}
        print(json.dumps(row), flush=True)
        results.append(row)
        (output / 'results.json').write_text(json.dumps({'protocol': protocol, 'candidates': results,
                                                        'completed': False}, indent=2) + '\n')
        del model, optimizer, train_loader, holdout
        gc.collect()
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif device.type == 'cuda':
            torch.cuda.empty_cache()
    stable_lrs = [row['lr'] for row in results if row['stable']]
    if not stable_lrs:
        raise RuntimeError('No stable LR; do not launch full training')
    selected = 0.003 if 0.003 in stable_lrs else stable_lrs[0]
    result = {'protocol': protocol, 'candidates': results, 'selected_lr': selected, 'completed': True}
    (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'Selected stable Muon LR: {selected}; full validation not used', flush=True)


if __name__ == '__main__':
    main()
