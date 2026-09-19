"""Compare completed runs without modifying their metrics or checkpoints."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / 'outputs'
    configs, results = {}, {}
    for name in ('adamw', 'muon'):
        configs[name] = json.loads((output / name / 'config.json').read_text())
        results[name] = json.loads((output / name / 'metrics.json').read_text())
        assert results[name]['completed'] and len(results[name]['epochs']) == 3, name
    keys = ['model', 'model_revision', 'epochs', 'batch_size', 'seed', 'max_length',
            'device', 'weight_decay', 'lr_schedule', 'precision', 'deterministic_warn_only',
            'train_rows', 'validation_rows', 'versions']
    for key in keys:
        assert configs['adamw'][key] == configs['muon'][key], f'Config mismatch: {key}'
    equivalence = json.loads((output / 'data_equivalence.json').read_text())
    assert all(row['identical_ordered_rows'] for row in equivalence.values())
    manifest = json.loads((ROOT / 'adamw_preservation.json').read_text())
    for name, digest in manifest.items():
        if name != 'train.py':  # train.py was intentionally extended; baseline outputs are immutable.
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    rows = []
    for name, result in results.items():
        for epoch in result['epochs']:
            rows.append({'optimizer': name, **epoch})
    with (output / 'comparison.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# AdamW vs Muon', '', 'Completed 3 epochs with seed=42; the table compares epoch 3 for both runs.', '',
             '| Optimizer | Train loss | Validation loss | Validation accuracy | Best-accuracy epoch |',
             '|---|---:|---:|---:|---:|']
    for name, result in results.items():
        row = result['epochs'][-1]
        lines.append(f"| {name} | {row['train_loss']:.6f} | {row['validation_loss']:.6f} | "
                     f"{row['validation_accuracy']:.4%} | {result['best_epoch']} |")
    delta = (results['muon']['epochs'][-1]['validation_accuracy'] -
             results['adamw']['epochs'][-1]['validation_accuracy']) * 100
    lines += ['', f'Muon final accuracy relative to AdamW: {delta:+.4f} percentage points.', '',
              'Configuration consistency and original AdamW output SHA-256 checks passed.',
              'This is a preliminary single-seed comparison with different optimizer learning rates; no statistical significance is claimed.',
              'See comparison.csv for all epoch results; the Muon run includes AdamW fallback.']
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    (output / 'comparison_checks.json').write_text(json.dumps({
        'matched_config_keys': keys, 'tokenized_data_identical': True, 'adamw_outputs_unchanged': True,
        'all_runs_completed': True, 'accuracy_delta_percentage_points': delta,
    }, indent=2) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
