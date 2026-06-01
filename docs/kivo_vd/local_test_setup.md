# Kivo-VD Local Test Setup (Mac/CPU)

This document provides a minimal local path to run only Kivo-VD CPU-only unit
tests in an isolated suite that does not load `tests/conftest.py`:

- `tests_kivo/test_kivo_vd_observer.py`
- `tests_kivo/test_kivo_vd_sketch.py`
- `tests_kivo/test_kivo_vd_sketch_math.py`

## Why this is needed

Importing `vllm.v1.core.*` goes through `import vllm`, and `vllm/__init__.py`
imports `vllm.env_override`, which imports `torch`.  
So even these pure-Python tests need `torch` importability.

## Option A (minimal, fastest for Kivo-VD tests)

This installs only what these tests need (`torch` + `pytest`) and avoids full
test/CUDA dependency sets.

```bash
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate

# macOS CPU torch comes from standard torch package version used by repo.
uv pip install "torch==2.11.0" "pytest"

python -m pytest tests_kivo -q
```

## Option B (repo-supported CPU dependency path)

If you want closer alignment with repository docs for CPU installs:

```bash
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate

uv pip install -r requirements/cpu.txt --index-strategy unsafe-best-match
uv pip install pytest

python -m pytest tests_kivo -q
```

Notes:
- `requirements/cpu.txt` is the repo-defined CPU dependency set.
- This step does **not** install CUDA/Triton GPU stacks.
- No scheduler/runtime behavior changes are required for these tests.
- `tests_kivo/` is outside `tests/`, so pytest does not pick up
  `tests/conftest.py` and its heavyweight dependencies (for example `tblib`).
- Running tests under the main `tests/` tree may require full vLLM dev/test
  dependencies. Failures from `tests/conftest.py` (for example missing
  `tblib`) are environment/dependency issues, not Kivo-only code failures.

## Optional: HuggingFace Q/K Offline Eval (Phase 1.5)

The HF script is optional and not required for `tests_kivo`.

Install optional dependencies:

```bash
uv pip install torch transformers
```

Run the optional eval:

```bash
.venv/bin/python scripts/kivo_vd/run_hf_qk_sketch_eval.py \
  --model-name sshleifer/tiny-gpt2 \
  --sketch-type random_projection \
  --sketch-dim 64 \
  --topk-blocks 4 \
  --device cpu
```

Optional layer/head sweep:

```bash
.venv/bin/python scripts/kivo_vd/run_hf_qk_head_sweep.py \
  --model-name distilgpt2 \
  --sketch-type count_sketch \
  --sketch-dim 64 \
  --layers 0,1 \
  --heads 0,1 \
  --max-tokens 512
```

## Troubleshooting

- If `pytest` is missing: `uv pip install pytest`
- If `torch` import fails: verify `torch==2.11.0` is installed in `.venv`.
- Run tests from repo root so `vllm` source package is importable.
