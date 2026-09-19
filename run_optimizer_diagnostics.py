"""Exactly 32 steps per optimizer, identical pretrained initialization and batches."""
import os
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.path.dirname(__file__), '.cache/matplotlib'))
from train import MODEL_REVISION, DATASET_REVISION
from train import ROOT, seed_everything, create_model, build_optimizer
from diagnostics import GroupedStepDiagnostics
import argparse
import csv
import gc
import hashlib
import json
import math
import time
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def rng_hash(device):
    value = torch.mps.get_rng_state() if device.type == 'mps' else (
        torch.cuda.get_rng_state(device) if device.type == 'cuda' else torch.get_rng_state())
    return hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()


def plots(frame, out):
    colors = {'adamw':'#2378b5', 'muon':'#df7924'}
    metrics = ['training_loss','gradient_norm','update_norm','relative_update_norm']
    fig, axes = plt.subplots(2, 2, figsize=(11,7))
    for ax, metric in zip(axes.flat, metrics):
        for name, color in colors.items():
            data = frame[(frame.optimizer==name)&(frame.group=='all')]
            ax.plot(data.step,data[metric],label=name,color=color)
        ax.set(xlabel='Optimizer step',ylabel=metric.replace('_',' ')); ax.legend()
        if metric.endswith('update_norm'): ax.set_yscale('log')
    fig.suptitle('32-step matched experiment | actual pre-step gradients and post-step updates')
    fig.tight_layout(); fig.savefig(out/'global_diagnostics.png',dpi=180); plt.close(fig)
    fig, axes = plt.subplots(3,2,figsize=(12,10))
    for column, group in enumerate(['matrix','fallback']):
        for row, metric in enumerate(metrics[1:]):
            ax=axes[row,column]
            for name,color in colors.items():
                data=frame[(frame.optimizer==name)&(frame.group==group)]
                ax.plot(data.step,data[metric],label=name,color=color)
            ax.set(xlabel='Optimizer step',ylabel=metric.replace('_',' ')); ax.legend()
            if metric.endswith('update_norm'): ax.set_yscale('log')
        axes[0,column].set_title('Muon-compatible matrices (both runs)' if group=='matrix' else 'Fallback parameters (both runs)')
    fig.tight_layout(); fig.savefig(out/'group_diagnostics.png',dpi=180); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/optimizer_diagnostics_32steps')
    args=parser.parse_args(); out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise FileExistsError(f'Choose an empty directory: {out}')
    protected={str(p.relative_to(ROOT)):file_hash(p) for p in (ROOT/'outputs').rglob('*')
               if p.is_file() and not p.is_relative_to(out.resolve())}
    (out/'preservation_before.json').write_text(json.dumps(protected,indent=2)+'\n')
    configs={n:json.loads((ROOT/f'outputs/{n}/config.json').read_text()) for n in ('adamw','muon')}
    groups_source=json.loads((ROOT/'outputs/muon/parameter_groups.json').read_text())
    groups={'matrix':[p['name'] for p in groups_source['muon']['parameters']],
            'fallback_decay':[p['name'] for p in groups_source['adamw_decay']['parameters']],
            'fallback_no_decay':[p['name'] for p in groups_source['adamw_no_decay']['parameters']]}
    groups['fallback']=groups['fallback_decay']+groups['fallback_no_decay']
    assert set(groups['matrix']).isdisjoint(groups['fallback'])
    seed_everything(42)
    tokenizer=AutoTokenizer.from_pretrained(configs['adamw']['model'],revision=configs['adamw']['model_revision'])
    raw=load_dataset('nyu-mll/glue','sst2',split='train',revision=DATASET_REVISION).shuffle(seed=42).select(range(1024))
    sample_ids=list(raw['idx'])
    encoded=raw.map(lambda b:tokenizer(b['sentence'],max_length=128,truncation=True),
                    batched=True,remove_columns=['idx','sentence']).rename_column('label','labels')
    collator=DataCollatorWithPadding(tokenizer)
    # Materialize identical CPU batches once. No DataLoader RNG or reshuffling.
    batches=[collator([encoded[i] for i in range(start,start+32)]) for start in range(0,1024,32)]
    assert len(batches)==32 and len(set(sample_ids))==1024
    batch_hashes=[]
    for batch in batches:
        h=hashlib.sha256()
        for key in sorted(batch): h.update(key.encode());h.update(batch[key].numpy().tobytes())
        batch_hashes.append(h.hexdigest())
    protocol={'steps_per_optimizer':32,'batch_size':32,'seed':42,'max_length':128,
              'sample_ids_in_batch_order':sample_ids,'batch_sha256':batch_hashes,
              'initialization':'fresh pretrained model + identical seeded classification head',
              'source_configs':configs,'group_names':groups,
              'update_definition':'||theta_after - theta_before||_2 including weight decay',
              'relative_update_definition':'update_norm / pre-step parameter_norm within each group',
              'gradient_definition':'pre-optimizer-step gradient L2; batch mean loss, train mode',
              'group_note':'matrix + fallback partitions all; fallback_decay/no_decay subdivide fallback',
              'checkpoint_saved':False,'diagnostics_overhead':'parameter snapshots and device synchronization'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    device=torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    initial_hashes={}; rng_traces={}; all_rows=[]; durations={}
    for name in ('adamw','muon'):
        seed_everything(42)
        model=create_model(configs[name]['model'],device)
        assert model.config._commit_hash==configs[name]['model_revision']
        initial_hashes[name]=state_hash(model)
        if name=='muon': assert initial_hashes[name]==initial_hashes['adamw']
        optimizer=build_optimizer(model,name,configs[name]['lr'],configs[name].get('fallback_lr',2e-5))
        params={n for n,p in model.named_parameters() if p.requires_grad}
        assert params==set(groups['matrix']+groups['fallback'])
        if name=='muon':
            assert optimizer.grouping==groups_source
        recorder=GroupedStepDiagnostics(model,groups)
        model.train(); seed_everything(42); rng_traces[name]=[]
        start=time.monotonic(); run_rows=[]
        for step,cpu_batch in enumerate(batches,1):
            rng_traces[name].append(rng_hash(device))
            if name=='muon': assert rng_traces[name][-1]==rng_traces['adamw'][step-1]
            batch={k:v.to(device) for k,v in cpu_batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss=model(**batch).loss
            assert torch.isfinite(loss).item()
            loss.backward()
            recorder.before_step()
            optimizer.step()
            rows=recorder.after_step(step,loss.item())
            for row in rows:
                assert all(math.isfinite(row[k]) for k in ['training_loss','gradient_norm','update_norm','relative_update_norm'])
                assert row['update_norm']>0
                row['optimizer']=name
            global_row=next(row for row in rows if row['group']=='all')
            for metric in ('parameter_norm','gradient_norm','update_norm'):
                partition=sum(row[metric]**2 for row in rows if row['group'] in ('matrix','fallback'))
                assert math.isclose(global_row[metric]**2,partition,rel_tol=1e-6)
            run_rows.extend(rows)
            with (out/f'{name}_steps.csv').open('a',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]))
                if step==1: writer.writeheader()
                writer.writerows(rows)
            if step==1 or step%8==0:
                print(f'{name} step={step}/32 loss={loss.item():.5f} gradient={global_row["gradient_norm"]:.5f} update={global_row["update_norm"]:.5f} relative={global_row["relative_update_norm"]:.6g}',flush=True)
        durations[name]=time.monotonic()-start
        all_rows.extend(run_rows)
        del model,optimizer,recorder,batch,loss
        gc.collect()
        if device.type=='mps': torch.mps.empty_cache()
        if device.type=='cuda': torch.cuda.empty_cache()
    frame=pd.DataFrame(all_rows)
    assert len(frame)==320
    frame.to_csv(out/'all_steps.csv',index=False)
    summary=frame.groupby(['optimizer','group'])[['training_loss','gradient_norm','update_norm','relative_update_norm']].mean()
    summary.to_csv(out/'mean_summary.csv')
    plots(frame,out)
    assert all(file_hash(ROOT/path)==digest for path,digest in protected.items())
    verification={'completed':True,'steps':{'adamw':32,'muon':32},'device':str(device),
                  'initial_state_sha256':initial_hashes,'identical_initialization':True,
                  'identical_batches':True,'identical_per_step_rng_states':True,'rng_state_hashes':rng_traces,
                  'all_norms_finite':True,'group_partition_norm_checks':True,
                  'existing_outputs_unchanged':True,'elapsed_seconds_with_diagnostics':durations}
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(summary.to_string(),flush=True)
    print('Completed: 32 steps each; existing outputs unchanged.',flush=True)


if __name__=='__main__': main()
