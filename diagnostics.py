"""Opt-in measurements around a real optimizer step; no gradient mutation."""
import csv
import math
import torch


def l2_norm(tensors):
    # Reduce in FP32 on device; accumulate tensor contributions in Python float64.
    return math.sqrt(sum(t.detach().float().square().sum().item() for t in tensors))


class StepDiagnostics:
    def __init__(self, path):
        self.path = path
        if path.exists():
            raise FileExistsError(path)
        self.step = 0

    def before_step(self, model):
        params = [p for p in model.parameters() if p.requires_grad]
        self.gradient_norm = l2_norm(p.grad for p in params if p.grad is not None)
        self.parameter_norm = l2_norm(params)
        self.before = [p.detach().clone() for p in params]

    def after_step(self, model, epoch, loss):
        params = [p for p in model.parameters() if p.requires_grad]
        update = l2_norm(p.detach() - old for p, old in zip(params, self.before))
        self.step += 1
        row = dict(step=self.step, epoch=epoch, loss=loss,
                   gradient_norm=self.gradient_norm, parameter_norm=self.parameter_norm,
                   update_norm=update,
                   relative_update_norm=update/self.parameter_norm if self.parameter_norm else None)
        with self.path.open('a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            if self.step == 1:
                writer.writeheader()
            writer.writerow(row)
        del self.before


class GroupedStepDiagnostics:
    """Record global and named-group norms around the same actual optimizer step."""
    def __init__(self, model, groups):
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.groups = {'all': list(self.params), **groups}
        for names in self.groups.values():
            if len(names) != len(set(names)) or not set(names) <= self.params.keys():
                raise ValueError('Invalid parameter group')

    def before_step(self):
        self.before = {n: p.detach().clone() for n, p in self.params.items()}
        self.param_sq = {n: p.detach().float().square().sum().item() for n, p in self.params.items()}
        # Must precede Muon.step(), whose reference implementation mutates gradients.
        self.grad_sq = {n: p.grad.detach().float().square().sum().item() if p.grad is not None else 0.
                        for n, p in self.params.items()}

    def after_step(self, step, loss):
        update_sq = {n: (p.detach() - self.before[n]).float().square().sum().item()
                     for n, p in self.params.items()}
        rows = []
        for group, names in self.groups.items():
            parameter = math.sqrt(sum(self.param_sq[n] for n in names))
            update = math.sqrt(sum(update_sq[n] for n in names))
            row = dict(step=step, group=group, training_loss=loss,
                       gradient_norm=math.sqrt(sum(self.grad_sq[n] for n in names)),
                       parameter_norm=parameter, update_norm=update,
                       relative_update_norm=update / parameter if parameter else None)
            rows.append(row)
        del self.before
        return rows
