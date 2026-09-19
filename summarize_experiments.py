"""Regenerate publication artifacts from saved results only; never load/train a model."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import argparse
import hashlib
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'deliverables')
    args=parser.parse_args();out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()): raise FileExistsError('Choose an empty output directory')
    original={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'outputs').rglob('*') if p.is_file()}
    names=['adamw','muon'];colors={'adamw':'#2378b5','muon':'#df7924'}
    runs={n:json.loads((ROOT/f'outputs/{n}/metrics.json').read_text()) for n in names}
    assert all(r['completed'] for r in runs.values())
    performance=[];epochs=[]
    for name,r in runs.items():
        final=r['epochs'][-1];best=r['epochs'][r['best_epoch']-1]
        performance.append(dict(optimizer=name,final_epoch=final['epoch'],final_train_loss=final['train_loss'],
            final_validation_loss=final['validation_loss'],final_validation_accuracy=final['validation_accuracy'],
            best_epoch=r['best_epoch'],best_validation_loss=best['validation_loss'],
            best_validation_accuracy=best['validation_accuracy']))
        epochs.extend(dict(optimizer=name,**e) for e in r['epochs'])
    performance=pd.DataFrame(performance);epoch_frame=pd.DataFrame(epochs)
    steps=pd.read_csv(ROOT/'outputs/optimizer_diagnostics_32steps/all_steps.csv')
    assert all(steps[(steps.optimizer==n)&(steps.group=='all')].step.tolist()==list(range(1,33)) for n in names)
    diagnostics=steps.groupby(['optimizer','group'])[['training_loss','gradient_norm','update_norm','relative_update_norm']].mean().reset_index()
    sharp=pd.read_csv(ROOT/'outputs/diagnostics/sharpness_summary.csv')
    paired=pd.read_csv(ROOT/'outputs/diagnostics/paired_direction_difference.csv')
    for filename,frame in [('performance.csv',performance),('epochs.csv',epoch_frame),
                            ('diagnostics_means.csv',diagnostics),('sharpness.csv',sharp),('sharpness_paired.csv',paired)]:
        frame.to_csv(out/filename,index=False)
    tables=['# Results tables','', 'Accuracy is a fraction (0–1). Diagnostics are from a separate 32-step run.', '']
    for title,frame in [('Full-run performance',performance),('Recorded epochs',epoch_frame),
                         ('32-step arithmetic means by group',diagnostics),('Checkpoint sharpness',sharp),('Paired directional uncertainty',paired)]:
        tables+=['## '+title,'','| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
        for row in frame.itertuples(index=False,name=None):
            tables.append('| '+' | '.join(f'{x:.6g}' if isinstance(x,float) else str(x) for x in row)+' |')
        tables.append('')
    (out/'RESULTS.md').write_text('\n'.join(tables)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,metric,label in zip(axes,['train_loss','validation_loss','validation_accuracy'],['Training loss (epoch mean)','Validation loss','Validation accuracy (%)']):
        for n in names:
            d=epoch_frame[epoch_frame.optimizer==n]
            ax.plot(d.epoch,d[metric]*(100 if metric.endswith('accuracy') else 1),'o-',label=n,color=colors[n])
        ax.set(xlabel='Epoch',ylabel=label,xticks=[1,2,3]);ax.legend()
    fig.suptitle('Full training: 3 epochs, SST-2, seed 42')
    fig.tight_layout();fig.savefig(out/'performance.png',dpi=200);fig.savefig(out/'performance.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    panels=[('all','training_loss','Batch training loss'),('all','gradient_norm','Global gradient norm'),
            ('all','relative_update_norm','Global relative update norm'),('all','update_norm','Global update norm'),
            ('matrix','update_norm','Matching matrix-group update norm'),('fallback','update_norm','Fallback-group update norm')]
    for ax,(group,metric,title) in zip(axes.flat,panels):
        for n in names:
            d=steps[(steps.optimizer==n)&(steps.group==group)]
            ax.plot(d.step,d[metric],label=n,color=colors[n])
        ax.set(xlabel='Step',title=title);ax.legend()
        if 'update' in metric: ax.set_yscale('log')
    fig.suptitle('Separate matched 32-step experiment (not the historical 3-epoch trajectory)')
    fig.tight_layout();fig.savefig(out/'optimizer_diagnostics.png',dpi=200);fig.savefig(out/'optimizer_diagnostics.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for n in names:
        d=sharp[sharp.optimizer==n];x=d.epsilon.to_numpy();y=d.mean_symmetric_delta.to_numpy();sd=d.std_symmetric_delta.to_numpy()
        label=f'{n} (best epoch {runs[n]["best_epoch"]})'
        axes[0].plot(x,y,'o-',label=label,color=colors[n]);axes[0].fill_between(x,y-sd,y+sd,color=colors[n],alpha=.15)
        axes[1].plot(x,d.max_sampled_delta,'o-',label=label,color=colors[n])
    for ax in axes:
        ax.set_xscale('symlog',linthresh=.001);ax.axhline(0,lw=.6,color='gray')
        ax.set(xlabel='Relative perturbation epsilon',ylabel='Validation loss change');ax.legend()
    axes[0].set_title('Symmetric mean ± direction SD (n=8)');axes[1].set_title('Maximum sampled one-sided increase')
    fig.suptitle('Same 256 validation samples; tensor-relative perturbations')
    fig.tight_layout();fig.savefig(out/'sharpness.png',dpi=200);fig.savefig(out/'sharpness.pdf');plt.close(fig)
    assert all(digest(ROOT/p)==h for p,h in original.items())
    (out/'provenance.json').write_text(json.dumps({'source_files_sha256':original,'source_files_unchanged':True,
         'training_executed':False,'checkpoint_analysis_executed':False,
         'generator':'summarize_experiments.py'},indent=2)+'\n')
    print('Read-only consolidation complete; all original output files unchanged.')


if __name__=='__main__': main()
