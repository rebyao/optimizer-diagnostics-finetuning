"""Top-Hessian-eigenvalue sharpness estimate via power iteration + Hessian-vector products.

Read-only cross-check against the random-direction perturbation sharpness in
analyze_checkpoints.py. No optimizer steps, no retraining, no changes to any
existing checkpoint or output file. Uses the same 256-example validation
subset, tokenizer, and loss (mean cross-entropy over the subset) as
analyze_checkpoints.py's protocol.

The Hessian-vector product (HVP) for the subset's mean loss is computed as a
size-weighted sum of per-mini-batch HVPs (H = (1/N) sum_i H_i, so
H v = sum_c (n_c/N) (H_c v)), rather than one HVP over all 256 examples at
once. A device smoke test found that a single-batch double backward over the
full subset OOMs on this machine's MPS backend (needs >15GB, over its ~18GB
ceiling once other allocations are counted) and is drastically slower even
below that (16.9s at batch 128 vs 1.1s at batch 32); chunking keeps the exact
same 256-example subset and loss while avoiding that blowup.
"""
import os
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.path.dirname(__file__), '.cache/matplotlib'))
from train import DATASET_REVISION, ROOT, seed_everything
import argparse
import csv
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_subset_chunks(chunk_size=32, size=256, seed=42, max_length=128):
    """Same 256-example validation subset, seed, and tokenization as analyze_checkpoints.py,
    split into fixed mini-batches for chunked HVP computation."""
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / 'outputs/adamw/best_model'), local_files_only=True)
    other = AutoTokenizer.from_pretrained(str(ROOT / 'outputs/muon/best_model'), local_files_only=True)
    assert tokenizer.get_vocab() == other.get_vocab()
    raw = load_dataset('nyu-mll/glue', 'sst2', split='validation',
                        revision=DATASET_REVISION).shuffle(seed=seed).select(range(size))
    subset = raw.map(lambda batch: tokenizer(batch['sentence'], truncation=True, max_length=max_length),
                      batched=True, remove_columns=['sentence', 'idx']).rename_column('label', 'labels')
    collate = DataCollatorWithPadding(tokenizer)
    chunks = []
    for i in range(0, len(subset), chunk_size):
        idxs = range(i, min(i + chunk_size, len(subset)))
        batch = collate([subset[j] for j in idxs])
        chunks.append((batch, len(idxs)))
    return tokenizer, chunks, list(raw['idx'])


def to_device(chunks, device):
    return [({k: v.to(device) for k, v in batch.items()}, n) for batch, n in chunks]


@torch.no_grad()
def subset_loss(model, chunks):
    total, count = 0.0, 0
    for batch, n in chunks:
        loss = model(**batch).loss.item()
        total += loss * n
        count += n
    return total / count


def hvp_chunked(model, params, chunks, vector_flat):
    """Weighted sum of per-mini-batch Hessian-vector products; never materializes the full Hessian
    and never needs a single graph over all 256 examples at once."""
    total_n = sum(n for _, n in chunks)
    acc = torch.zeros_like(vector_flat)
    for batch, n in chunks:
        loss = model(**batch).loss
        grads = torch.autograd.grad(loss, params, create_graph=True)
        flat_grad = torch.cat([g.reshape(-1) for g in grads])
        grad_dot_v = torch.dot(flat_grad, vector_flat)
        hv = torch.autograd.grad(grad_dot_v, params, retain_graph=False)
        hv_flat = torch.cat([h.reshape(-1) for h in hv]).detach()
        acc += hv_flat * (n / total_n)
        del loss, grads, flat_grad, grad_dot_v, hv, hv_flat
    return acc


def power_iteration(hvp_fn, dim, device, seed, max_iters, tol, deflate=None, init_vector=None):
    """Plain power iteration. If `deflate=(vector, eigenvalue)` is given, applies Hotelling
    deflation (Hv -> Hv - eigenvalue*(vector.v)*vector) so a dominant eigenpair already found can
    be removed before searching for the next one, instead of shifting the whole spectrum (which,
    when the dominant eigenvalue is far larger in magnitude than the one being searched for,
    compresses relative eigenvalue gaps toward 1 and can silently stall convergence)."""
    generator = torch.Generator().manual_seed(seed)
    v = init_vector if init_vector is not None else torch.randn(dim, generator=generator)
    v = v.to(device)
    v = v / v.norm()
    history = []
    prev_eig = None
    converged = False
    for i in range(max_iters):
        Hv = hvp_fn(v)
        if deflate is not None:
            deflate_vec, deflate_eig = deflate
            Hv = Hv - deflate_eig * torch.dot(deflate_vec, v) * deflate_vec
        if not torch.isfinite(Hv).all():
            history.append({'iteration': i, 'rayleigh_quotient': float('nan'), 'residual_norm': float('nan')})
            break
        eig = torch.dot(v, Hv).item()
        v_new = Hv / Hv.norm()
        residual = (Hv - eig * v).norm().item()
        history.append({'iteration': i, 'rayleigh_quotient': eig, 'residual_norm': residual})
        if prev_eig is not None and abs(eig - prev_eig) < tol * (abs(prev_eig) + 1e-12):
            v, converged = v_new, True
            break
        v, prev_eig = v_new, eig
    final_eig = history[-1]['rayleigh_quotient'] if history else float('nan')
    return final_eig, v.detach(), history, converged


def run_for_checkpoint(name, device, chunk_size, max_iters, tol, seed, log):
    tokenizer, chunks_cpu, sample_ids = load_subset_chunks(chunk_size=chunk_size)
    chunks = to_device(chunks_cpu, device)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ROOT / f'outputs/{name}/best_model'), local_files_only=True).to(device)
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    dim = sum(p.numel() for p in params)
    baseline_loss = subset_loss(model, chunks)

    def hvp_fn(v):
        return hvp_chunked(model, params, chunks, v)

    t0 = time.monotonic()
    raw_eig, raw_vec, raw_history, raw_converged = power_iteration(
        hvp_fn, dim, device, seed, max_iters, tol)
    t_raw = time.monotonic() - t0
    log(f'{name}: raw power iteration done in {t_raw:.1f}s, '
        f'{len(raw_history)} iters, converged={raw_converged}, eig~{raw_eig:.6g}')

    # If raw's dominant-by-magnitude eigenvalue is already positive, it IS the top algebraic
    # eigenvalue (nothing else can out-rank it in magnitude, and anything negative is smaller in
    # value). If it is negative, run a second, DEFLATED power iteration on H - raw_eig*vv^T (a
    # fresh random start, not warm-started from raw_vec) to remove that component and find the
    # next dominant-by-magnitude eigenpair. An earlier version of this script instead shifted the
    # whole operator by a large constant (H + c*I); with c forced far larger than the sought
    # eigenvalue to guarantee non-negativity, that compresses relative eigenvalue gaps toward 1
    # and can silently stall power iteration after only 1-2 "converged" iterations without ever
    # rotating away from the initial vector. Deflation avoids that failure mode.
    t0 = time.monotonic()
    second_eig, second_vec, second_history, second_converged = power_iteration(
        hvp_fn, dim, device, seed + 5000, max_iters, tol, deflate=(raw_vec, raw_eig))
    t_second = time.monotonic() - t0
    same_eigenvector = abs(torch.dot(raw_vec, second_vec).item())
    log(f'{name}: deflated power iteration done in {t_second:.1f}s, '
        f'{len(second_history)} iters, converged={second_converged}, '
        f'eig~{second_eig:.6g}, |raw.deflated eigenvector overlap|={same_eigenvector:.4f}')

    collapsed_to_raw_eigenvector = same_eigenvector > 0.99
    if raw_eig > 0:
        top_algebraic_eig = raw_eig
        reliable = bool(raw_converged)
        note = 'raw dominant eigenvalue was already positive; it is the top algebraic eigenvalue.'
    elif second_eig > 0 and not collapsed_to_raw_eigenvector:
        top_algebraic_eig = second_eig
        reliable = bool(raw_converged and second_converged)
        note = 'raw dominant eigenvalue was negative; using the deflated pass, which found a positive eigenvalue.'
    else:
        top_algebraic_eig = second_eig
        reliable = False
        note = ('raw dominant eigenvalue was negative and the deflated pass did not find a clear '
                'positive eigenvalue within budget; top-eigenvalue estimate is UNRELIABLE.')
    log(f'{name}: {note} top_algebraic_eigenvalue_estimate={top_algebraic_eig:.6g}, reliable={reliable}')

    result = {
        'optimizer': name,
        'dim': dim,
        'baseline_loss': baseline_loss,
        'raw_dominant_eigenvalue': raw_eig,
        'raw_converged': raw_converged,
        'raw_iterations': len(raw_history),
        'raw_elapsed_seconds': t_raw,
        'deflated_dominant_eigenvalue': second_eig,
        'deflated_converged': second_converged,
        'deflated_iterations': len(second_history),
        'deflated_elapsed_seconds': t_second,
        'raw_deflated_eigenvector_overlap': same_eigenvector,
        'collapsed_to_raw_eigenvector': collapsed_to_raw_eigenvector,
        'top_algebraic_eigenvalue_estimate': top_algebraic_eig,
        'note': note,
        'reliable': reliable,
    }
    del model, params, chunks
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result, raw_history, second_history


def select_device(chunk_size, log):
    """Smoke-test double-backward support/stability on candidate devices; fall back on failure.
    CUDA/MPS are tried before CPU; CPU is checked with eager attention since its flash-attention
    kernel has no double-backward implementation in this torch version."""
    tokenizer, chunks_cpu, _ = load_subset_chunks(chunk_size=chunk_size, size=chunk_size)
    candidates = []
    if torch.cuda.is_available():
        candidates.append(('cuda', {}))
    if torch.backends.mps.is_available():
        candidates.append(('mps', {}))
    candidates.append(('cpu', {'attn_implementation': 'eager'}))
    for device_type, extra_kwargs in candidates:
        device = torch.device(device_type)
        try:
            chunks = to_device(chunks_cpu, device)
            model = AutoModelForSequenceClassification.from_pretrained(
                str(ROOT / 'outputs/adamw/best_model'), local_files_only=True, **extra_kwargs).to(device)
            model.eval()
            params = [p for p in model.parameters() if p.requires_grad]
            dim = sum(p.numel() for p in params)
            t0 = time.monotonic()
            eig, _, history, _ = power_iteration(
                lambda v: hvp_chunked(model, params, chunks, v), dim, device, 2026, 2, 1e-6)
            elapsed = time.monotonic() - t0
            ok = np.isfinite(eig) and all(np.isfinite(h['rayleigh_quotient']) for h in history)
            log(f'device smoke test: {device} (kwargs={extra_kwargs}) -> eig={eig}, finite={ok}, '
                f'{elapsed:.2f}s for 2 iterations on {chunk_size} examples')
            del model, params, chunks
            gc.collect()
            if ok:
                return device, extra_kwargs
        except Exception as exc:
            log(f'device smoke test: {device} (kwargs={extra_kwargs}) failed with '
                f'{type(exc).__name__}: {exc}')
    raise RuntimeError('No device passed the Hessian-vector-product smoke test.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/hessian_sharpness')
    parser.add_argument('--chunk-size', type=int, default=32)
    parser.add_argument('--max-iters', type=int, default=25)
    parser.add_argument('--tol', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--time-budget-seconds', type=float, default=1800,
                         help='Abort before starting the full run if the timed smoke test projects '
                              'a total cost above this budget.')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f'Use an empty output directory: {out}')

    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    seed_everything(args.seed)
    protected = {str(p.relative_to(ROOT)): sha(p) for name in ('adamw', 'muon')
                 for p in (ROOT / f'outputs/{name}').rglob('*') if p.is_file()}

    device, model_kwargs = select_device(args.chunk_size, log)

    # Time one full-subset chunked HVP on the chosen device to project total cost before committing.
    tokenizer, chunks_cpu, _ = load_subset_chunks(chunk_size=args.chunk_size)
    chunks = to_device(chunks_cpu, device)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ROOT / 'outputs/adamw/best_model'), local_files_only=True, **model_kwargs).to(device)
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    dim = sum(p.numel() for p in params)
    v0 = torch.randn(dim, generator=torch.Generator().manual_seed(args.seed)).to(device)
    v0 = v0 / v0.norm()
    t0 = time.monotonic()
    _ = hvp_chunked(model, params, chunks, v0)
    single_hvp_seconds = time.monotonic() - t0
    projected_total = single_hvp_seconds * args.max_iters * 2 * 2  # raw+shifted, 2 checkpoints
    log(f'single full-subset chunked HVP on {device}: {single_hvp_seconds:.2f}s; '
        f'projected total for {args.max_iters} iters x 2 passes x 2 checkpoints: {projected_total:.0f}s')
    del model, params, chunks, v0
    gc.collect()

    if projected_total > args.time_budget_seconds:
        report = {
            'aborted': True,
            'reason': 'projected_cost_exceeds_time_budget',
            'device': str(device),
            'single_hvp_seconds': single_hvp_seconds,
            'projected_total_seconds': projected_total,
            'time_budget_seconds': args.time_budget_seconds,
        }
        (out / 'hessian_sharpness.json').write_text(json.dumps(report, indent=2) + '\n')
        log(f'ABORTED: projected cost {projected_total:.0f}s exceeds budget '
            f'{args.time_budget_seconds:.0f}s. Reduce --max-iters or raise --time-budget-seconds.')
        (out / 'run_log.txt').write_text('\n'.join(log_lines) + '\n')
        return

    results = {}
    histories = {}
    for name in ('adamw', 'muon'):
        result, raw_history, deflated_history = run_for_checkpoint(
            name, device, args.chunk_size, args.max_iters, args.tol, args.seed, log)
        results[name] = result
        histories[name] = {'raw': raw_history, 'deflated': deflated_history}

    assert all(sha(ROOT / p) == digest for p, digest in protected.items())

    convergence_rows = []
    for name in ('adamw', 'muon'):
        for phase in ('raw', 'deflated'):
            for row in histories[name][phase]:
                convergence_rows.append(dict(optimizer=name, phase=phase, **row))
    save_csv(out / 'hessian_convergence.csv', convergence_rows)

    protocol = {
        'method': 'Power iteration on Hessian-vector products (double backward), no explicit Hessian',
        'subset_size': 256, 'subset_seed': 42, 'max_length': 128, 'chunk_size': args.chunk_size,
        'hvp_construction': 'H v computed as a size-weighted sum of 8 per-chunk (batch=32) HVPs '
                             '(H = (1/N) sum_i H_i is linear in the per-example loss, so this is '
                             'exact, not an approximation); avoids an MPS OOM/blowup found when '
                             'differentiating all 256 examples in one graph.',
        'loss': 'mean cross-entropy over the 256-example validation subset, eval mode (dropout off)',
        'parameter_scope': 'all named parameters (embeddings, norms, biases, classifier, hidden matrices)',
        'device': str(device), 'model_kwargs': model_kwargs,
        'random_seed': args.seed,
        'max_iters': args.max_iters,
        'tolerance': args.tol,
        'sign_check': 'Plain power iteration converges to the eigenvalue of largest MAGNITUDE, which '
                      'can be negative. If the raw pass finds a positive value it is already the top '
                      'algebraic eigenvalue; otherwise a second, freshly-initialized power iteration is '
                      'run on the Hotelling-deflated operator H - raw_eig*(v v^T) to remove that '
                      'component and search for the next dominant eigenpair, checked for a positive '
                      'sign and for not re-converging to the same eigenvector as the raw pass.',
        'caveat': 'Deflation only removes one eigencomponent; if a second large-magnitude negative '
                  'eigenvalue exists it would be found next instead of the true positive maximum. '
                  'Non-convergent, still-negative, or eigenvector-collapsed results are marked '
                  'unreliable rather than reported as a confirmed curvature estimate.',
        'checkpoints': {'adamw': 'best, epoch 3', 'muon': 'best, epoch 1'},
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    (out / 'hessian_sharpness.json').write_text(json.dumps(results, indent=2) + '\n')
    (out / 'run_log.txt').write_text('\n'.join(log_lines) + '\n')
    (out / 'verification.json').write_text(json.dumps({
        'original_files_unchanged': True, 'optimizer_steps': 0,
        'device': str(device), 'completed': True}, indent=2) + '\n')
    log('Completed. Existing checkpoints/metrics unchanged; no optimizer steps.')


if __name__ == '__main__':
    main()
