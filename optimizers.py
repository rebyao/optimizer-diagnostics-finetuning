"""Muon for hidden Linear weights; native AdamW for all other parameters."""
import json
from pathlib import Path

import torch
from vendor.muon import SingleDeviceMuon


class MuonWithAdamW:
    """Expose the two operations used by our explicit training loop."""
    def __init__(self, muon, adamw, grouping):
        self.muon, self.adamw, self.grouping = muon, adamw, grouping

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adamw.step()


def build_muon(model, lr, fallback_lr=2e-5):
    # Explicit module allowlist: exclude embeddings, norms and final classifier.
    eligible = {id(module.weight) for name, module in model.named_modules()
                if isinstance(module, torch.nn.Linear) and name != 'classifier'
                and module.weight.ndim == 2}
    groups = {'muon': [], 'adamw_decay': [], 'adamw_no_decay': []}
    records = {name: [] for name in groups}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = ('muon' if id(parameter) in eligible else
                 'adamw_decay' if parameter.ndim >= 2 else 'adamw_no_decay')
        groups[group].append(parameter)
        records[group].append({'name': name, 'shape': list(parameter.shape),
                               'numel': parameter.numel()})
    ids = [id(p) for params in groups.values() for p in params]
    assert len(ids) == len(set(ids)), 'Duplicate optimizer ownership'
    assert set(ids) == {id(p) for p in model.parameters() if p.requires_grad}
    assert all(groups.values()), 'Expected nonempty Muon and AdamW groups'
    grouping = {
        'muon': {'lr': lr, 'weight_decay': 0.01, 'momentum': 0.95,
                 'nesterov': True, 'ns_steps': 5, 'ns_dtype': 'bfloat16',
                 'lr_scaling': 'sqrt(max(1, rows/cols))', 'parameters': records['muon']},
        'adamw_decay': {'lr': fallback_lr, 'weight_decay': 0.01,
                        'betas': [0.9, 0.999], 'eps': 1e-8,
                        'parameters': records['adamw_decay']},
        'adamw_no_decay': {'lr': fallback_lr, 'weight_decay': 0.0,
                           'betas': [0.9, 0.999], 'eps': 1e-8,
                           'parameters': records['adamw_no_decay']},
    }
    return MuonWithAdamW(
        SingleDeviceMuon(groups['muon'], lr=lr, weight_decay=0.01, momentum=0.95),
        torch.optim.AdamW([
            {'params': groups['adamw_decay'], 'weight_decay': 0.01},
            {'params': groups['adamw_no_decay'], 'weight_decay': 0.0},
        ], lr=fallback_lr, betas=(0.9, 0.999), eps=1e-8), grouping)


def save_groups(optimizer, output_dir):
    if not isinstance(optimizer, MuonWithAdamW):
        return
    for name, group in optimizer.grouping.items():
        params = group['parameters']
        print(f"{name}: {len(params)} tensors, {sum(p['numel'] for p in params):,} parameters; "
              f"lr={group['lr']}, weight_decay={group['weight_decay']}", flush=True)
        for parameter in params:
            print(f"  {parameter['name']} {parameter['shape']}", flush=True)
    Path(output_dir, 'parameter_groups.json').write_text(
        json.dumps(optimizer.grouping, indent=2) + '\n')
