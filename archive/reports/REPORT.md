# Environment report: NLP optimizer comparison

The environment was set up and the smoke test passed. No training had started at this stage.

## Machine and GPU

- macOS Darwin 24.3.0, ARM64, Apple M4, 16 GB RAM, 10-core GPU.
- Python 3.11.16; the project virtual environment is `.venv`, with its standalone interpreter in `.python`.
- CUDA is unavailable and `torch.version.cuda` is `None`: this machine has an Apple GPU rather than NVIDIA CUDA hardware.
- Both PyTorch MPS build support and runtime availability are True. GPU matrix multiplication passed, and BERT forward inference produced output on `mps:0`.
- This verifies GPU computation, not training compatibility or speed. MPS is PyTorch's official Apple GPU backend: https://docs.pytorch.org/docs/main/notes/mps.html

## Installed versions of the requested dependencies

| Package | Version |
|---|---|
| torch | 2.8.0 |
| transformers | 4.57.1 |
| datasets | 4.3.0 |
| evaluate | 0.4.6 |
| scikit-learn | 1.7.2 |
| numpy | 2.2.6 |
| pandas | 2.3.3 |
| matplotlib | 3.10.7 |
| tqdm | 4.67.1 |

At this setup stage, `requirements.txt` contained only these nine pinned direct dependencies. The complete set of 59 packages, including necessary transitive dependencies, is recorded in `installed-versions.txt`. torchvision, torchaudio, TensorFlow, JAX and NVIDIA CUDA components were not installed. uv 0.12.17 was a bootstrap tool in a temporary directory, outside the experiment's virtual environment.

## Commands used for setup and verification

Working directory: `/Users/rebeccayao/Desktop/Optimization_FT`. These are the main commands actually executed; requirements.txt and smoke_test.py were written before running them.

```sh
# Inspect the machine and existing Python installations
pwd
ls -la
uname -a
command -v python3 python3.11 python3.10 uv brew nvidia-smi
python3 --version
system_profiler SPHardwareDataType SPDisplaysDataType

# Check connectivity (failed in the sandbox; succeeded with approved network access)
curl -I --max-time 15 https://pypi.org/simple/torch/
curl -I --max-time 20 https://pypi.org/simple/torch/

# Install the lightweight bootstrap tool and standalone Python 3.11; create .venv
/opt/homebrew/bin/python3.12 -m pip install --target /private/tmp/optimization-ft-bootstrap uv
UV_PYTHON_INSTALL_DIR="$PWD/.python" UV_CACHE_DIR="$PWD/.cache/uv" /private/tmp/optimization-ft-bootstrap/bin/uv venv --python 3.11 .venv

# Install pinned dependencies, check compatibility and export installed versions
UV_CACHE_DIR="$PWD/.cache/uv" /private/tmp/optimization-ft-bootstrap/bin/uv pip install --python .venv/bin/python -r requirements.txt
UV_CACHE_DIR="$PWD/.cache/uv" /private/tmp/optimization-ft-bootstrap/bin/uv pip check --python .venv/bin/python
UV_CACHE_DIR="$PWD/.cache/uv" /private/tmp/optimization-ft-bootstrap/bin/uv pip freeze --python .venv/bin/python > installed-versions.txt

# Verify (initially failed on the dataset repository name; succeeded after correction)
.venv/bin/python smoke_test.py > smoke_test.log 2>&1
```

Commands used to investigate the dataset access issue:

```sh
.venv/bin/python -c 'from huggingface_hub import HfApi; api=HfApi(); print(api.dataset_info("stanfordnlp/glue"))'
.venv/bin/python - <<'PY'
import requests
for repo in ['stanfordnlp/glue', 'glue', 'nyu-mll/glue']:
    r = requests.get('https://huggingface.co/api/datasets/' + repo, timeout=30)
    print(repo, r.status_code, r.json().get('id'))
PY
```

## Loading and inference results

- Model: `hf-internal-testing/tiny-random-bert`, with 87,995 parameters and safetensors weights. This randomly initialized test model verifies loading and execution, not sentiment-classification quality.
- Model revision: `f171d7baecaf37b5da5a3616d8833b9969753535`.
- Dataset: `load_dataset("nyu-mll/glue", "sst2")`.
- Rows: train 67,349; validation 872; test 1,821.
- Forward inference on two validation examples: output shape `[2, 2]`, finite values, device `mps:0`.
- Used `eval()` and `torch.inference_mode()`, without backward passes, optimizer updates or training.
- Full output: `smoke_test.log`; structured results: `smoke_test_results.json`.

## Compatibility issues and resolutions

- System Python 3.9.6 and the existing Homebrew Python 3.12 did not meet the requested versions. A project-local Python 3.11.16 was downloaded without modifying system Python.
- `uv pip check` checked 59 packages and reported `All installed packages are compatible`. All nine direct dependencies imported successfully.
- The initial `stanfordnlp/glue` request returned HTTP 401. The public `glue` alias resolved to `nyu-mll/glue`; using that repository succeeded without authentication.
- The initial sandbox network request failed during DNS resolution. Downloads succeeded with approved network access.
- CUDA does not apply to this machine; subsequent experiments should use `mps`. GPU inference was verified, but training compatibility had not yet been tested because training was explicitly excluded at this stage.

Activate the environment with `source .venv/bin/activate`, or invoke `.venv/bin/python` directly. Rerun the smoke test with `.venv/bin/python smoke_test.py`. Model and dataset caches are stored in `.cache/huggingface`.
