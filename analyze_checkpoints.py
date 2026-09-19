"""Read-only checkpoint gradient and random-direction sharpness analysis. No optimizer steps."""
import os
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.path.dirname(__file__), '.cache/matplotlib'))
from train import MODEL_REVISION, DATASET_REVISION
from train import ROOT, seed_everything, evaluate
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from diagnostics import l2_norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalized_direction(base, seed):
    generator = torch.Generator().manual_seed(seed)
    result = {}
    for name, p in base.items():
        noise = torch.randn(p.shape, generator=generator, dtype=torch.float32)
        norm = p.norm().item()
        result[name] = noise * (norm / noise.norm().item()) if norm else torch.zeros_like(noise)
    return result


@torch.no_grad()
def perturb(model, base, direction=None, amount=0):
    for name, p in model.named_parameters():
        value = base[name] if direction is None else base[name] + amount * direction[name]
        p.copy_(value)


def make_plots(out, gradients, aggregate):
    colors = {'adamw': '#2378b5', 'muon': '#df7924'}
    labels = {'adamw': 'AdamW (best epoch 3)', 'muon': 'Muon (best epoch 1)'}
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in colors:
        rows = json.loads((ROOT / f'outputs/{name}/metrics.json').read_text())['epochs']
        ax.plot([r['epoch'] for r in rows], [r['train_loss'] for r in rows],
                'o-', label=name, color=colors[name])
    ax.set(xlabel='Epoch', ylabel='Mean training loss', title='Recorded epoch means (historical runs)', xticks=[1,2,3])
    ax.legend(); fig.tight_layout(); fig.savefig(out/'training_loss.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    names = list(colors)
    axes[0].bar([labels[n] for n in names], [gradients[n]['gradient_norm'] for n in names], color=list(colors.values()))
    axes[0].set(title='Checkpoint validation gradient', ylabel='L2 norm of mean-loss gradient')
    for ax, title in zip(axes[1:], ['Historical update norm', 'Historical relative update norm']):
        ax.set_title(title)
        ax.text(.5,.5,'Not recorded\nCannot recover from checkpoints\nNo retraining performed',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Endpoint gradient is NOT a historical training-gradient trace', fontsize=11)
    fig.tight_layout(); fig.savefig(out/'gradient_update_norms.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in names:
        rows = [r for r in aggregate if r['optimizer'] == name]
        x = np.array([r['epsilon'] for r in rows])
        y = np.array([r['mean_symmetric_delta'] for r in rows])
        sd = np.array([r['std_symmetric_delta'] for r in rows])
        axes[0].plot(x,y,'o-',label=labels[name],color=colors[name])
        axes[0].fill_between(x,y-sd,y+sd,alpha=.15,color=colors[name])
        axes[1].plot(x,[r['max_sampled_delta'] for r in rows],'o-',label=labels[name],color=colors[name])
    for ax in axes:
        ax.set_xscale('symlog', linthresh=.001)
        ax.set_xlabel('Relative perturbation epsilon (per tensor)')
        ax.set_ylabel('Validation loss increase'); ax.axhline(0,color='gray',lw=.5); ax.legend()
    axes[0].set_title('Symmetric mean over 8 directions ± 1 SD')
    axes[1].set_title('Maximum observed over sampled ± directions')
    fig.tight_layout(); fig.savefig(out/'sharpness.png',dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/diagnostics')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f'Use an empty output directory: {out}')
    seed_everything(42)
    protected = {str(p.relative_to(ROOT)): sha(p) for name in ('adamw','muon')
                 for p in (ROOT/f'outputs/{name}').rglob('*') if p.is_file()}
    (out/'preservation_before.json').write_text(json.dumps(protected,indent=2)+'\n')
    radii = [0.001,0.003,0.01,0.03,0.1]
    seeds = list(range(2026,2034))
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT/'outputs/adamw/best_model'),local_files_only=True)
    other = AutoTokenizer.from_pretrained(str(ROOT/'outputs/muon/best_model'),local_files_only=True)
    assert tokenizer.get_vocab() == other.get_vocab()
    raw = load_dataset('nyu-mll/glue','sst2',split='validation',revision=DATASET_REVISION).shuffle(seed=42).select(range(256))
    encoded = tokenizer(list(raw['sentence']),truncation=True,max_length=128)
    assert encoded == other(list(raw['sentence']),truncation=True,max_length=128)
    subset = raw.map(lambda batch: tokenizer(batch['sentence'],truncation=True,max_length=128),
                     batched=True,remove_columns=['sentence','idx']).rename_column('label','labels')
    loader = DataLoader(subset,batch_size=32,shuffle=False,collate_fn=DataCollatorWithPadding(tokenizer))
    protocol = {'subset_size':256,'subset_seed':42,'sample_ids':list(raw['idx']), 'labels':list(raw['label']),
                'direction_seeds':seeds,'radii':[0]+radii,'batch_size':32,'max_length':128,
                'normalization':'delta_tensor = epsilon * ||theta_tensor||_2 * z_tensor / ||z_tensor||_2',
                'zero_norm_policy':'Zero-norm tensors stay zero', 'scope':'All named parameters, including embeddings, norms and biases',
                'symmetric_delta':'(L(theta+delta)+L(theta-delta))/2 - L(theta)',
                'uncertainty':'Sample standard deviation across 8 independent paired directions; not across datasets',
                'gradient':'Gradient of average validation-subset loss in eval mode; not historical training gradients',
                'historical_gradient_norm':None,'historical_update_norm':None,'historical_relative_update_norm':None,
                'optimizer_steps':0,'checkpoints':{'adamw':'best, epoch 3','muon':'best, epoch 1'}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    gradients, raw_rows, aggregate = {}, [], []
    for name in ('adamw','muon'):
        model = AutoModelForSequenceClassification.from_pretrained(str(ROOT/f'outputs/{name}/best_model'),
                    local_files_only=True).to(device).eval()
        baseline, accuracy = evaluate(model,loader,device)
        model.zero_grad(set_to_none=True)
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()}
            (model(**batch).loss * (len(batch['labels'])/len(subset))).backward()
        gradient = l2_norm(p.grad for p in model.parameters() if p.grad is not None)
        assert np.isfinite(gradient)
        gradients[name]={'gradient_norm':gradient,'parameter_norm':l2_norm(model.parameters()),
                         'validation_loss':baseline,'validation_accuracy':accuracy,
                         'historical_update_norm':None,'historical_relative_update_norm':None}
        model.zero_grad(set_to_none=True)
        print(name, 'baseline', gradients[name], flush=True)
        base={n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        aggregate.append(dict(optimizer=name,epsilon=0,baseline_loss=baseline,mean_symmetric_delta=0,
                              std_symmetric_delta=0,max_sampled_delta=0))
        try:
            for seed in seeds:
                direction=normalized_direction(base,seed)
                for epsilon in radii:
                    losses=[]
                    for sign in (1,-1):
                        perturb(model,base,direction,sign*epsilon)
                        loss,_=evaluate(model,loader,device)
                        raw_rows.append(dict(optimizer=name,direction_seed=seed,epsilon=epsilon,sign=sign,
                                             loss=loss,baseline_loss=baseline,delta_loss=loss-baseline))
                        losses.append(loss)
                    print(f'{name} seed={seed} epsilon={epsilon:g} symmetric_delta={np.mean(losses)-baseline:.6g}',flush=True)
                del direction
                save_csv(out/'sharpness_raw.csv',raw_rows)
        finally:
            perturb(model,base)
        assert all(torch.equal(p.detach().cpu(),base[n]) for n,p in model.named_parameters())
        restored_loss,_=evaluate(model,loader,device)
        assert abs(restored_loss-baseline)<1e-6
        gradients[name]['restoration_verified']=True
        for epsilon in radii:
            rows=[r for r in raw_rows if r['optimizer']==name and r['epsilon']==epsilon]
            paired=[np.mean([r['delta_loss'] for r in rows if r['direction_seed']==seed]) for seed in seeds]
            aggregate.append(dict(optimizer=name,epsilon=epsilon,baseline_loss=baseline,
                                  mean_symmetric_delta=float(np.mean(paired)),std_symmetric_delta=float(np.std(paired,ddof=1)),
                                  max_sampled_delta=max(r['delta_loss'] for r in rows)))
        del model,base
        gc.collect()
        if device.type=='mps': torch.mps.empty_cache()
        if device.type=='cuda': torch.cuda.empty_cache()
    save_csv(out/'sharpness_summary.csv',aggregate)
    from scipy.stats import t
    paired_rows = []
    for radius in radii:
        differences = []
        for seed in seeds:
            values = {name: np.mean([r['delta_loss'] for r in raw_rows
                       if r['optimizer']==name and r['epsilon']==radius and r['direction_seed']==seed])
                      for name in ('adamw','muon')}
            differences.append(values['muon']-values['adamw'])
        mean = float(np.mean(differences))
        margin = float(t.ppf(.975,len(seeds)-1)*np.std(differences,ddof=1)/np.sqrt(len(seeds)))
        paired_rows.append(dict(epsilon=radius,muon_minus_adamw_mean=mean,
                                ci95_low=mean-margin,ci95_high=mean+margin))
    save_csv(out/'paired_direction_difference.csv',paired_rows)
    (out/'checkpoint_gradients.json').write_text(json.dumps(gradients,indent=2)+'\n')
    make_plots(out,gradients,aggregate)
    assert all(sha(ROOT/p)==digest for p,digest in protected.items())
    (out/'verification.json').write_text(json.dumps({'original_files_unchanged':True,'optimizer_steps':0,
                 'parameter_restoration_verified':True,'device':str(device),'completed':True},indent=2)+'\n')
    print('Completed. Existing checkpoints/metrics unchanged; no optimizer steps.',flush=True)


if __name__=='__main__':
    main()
