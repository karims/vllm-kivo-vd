# Kivo-VD Phase 3.1: Linux/NVIDIA Runtime Validation Plan

Phase 3.1 defines the reproducible path for validating Kivo-VD runtime dry-run
on a proper vLLM runtime environment.

For the first successful RunPod result and source-overlay details, see
[Phase 5.1: Linux Runtime Validation Result](phase5_1_linux_runtime_validation_result.md).

This phase does not change scheduler behavior, GPUModelRunner, attention
metadata, block tables, kernels, model architecture, training, or model output
logic.

## Why The Mac Runtime Attempt Failed

The Phase 3.0 dry-run script reached real vLLM model/config initialization on
Mac, but failed during CPU worker startup:

```text
'_OpNamespace' '_C' object has no attribute 'init_cpu_memory_env'
```

This indicates the local source tree does not have the compiled vLLM CPU
extension needed by that runtime path. It is a local build/runtime limitation,
not evidence that Kivo-VD changes attention behavior.

## Why Linux/NVIDIA Is The Target

vLLM is primarily optimized and validated on Linux with NVIDIA GPUs. Kivo-VD
runtime dry-run validation should happen there because:

- the standard CUDA vLLM runtime path is available;
- compiled vLLM extensions are expected to be present;
- future real memory and latency measurements require GPU execution;
- candidate-block attention work will eventually require CUDA/Triton/backend
  changes, even though Phase 3.1 does not implement them.

## Success Criteria

Runtime dry-run succeeds when:

- baseline inference completes with Kivo-VD disabled;
- Kivo-enabled inference completes;
- greedy generated outputs match exactly;
- dry-run routing events export successfully;
- no attention behavior, block table, slot mapping, attention metadata, or
  kernel behavior is changed.

Expected event types include:

- `after_allocate_slots`
- `dry_run_routing_decision`
- `free_request`, depending on request completion/export timing

## Setup Option A: Native Linux/NVIDIA

```bash
git clone https://github.com/<your-org-or-user>/vllm-kivo-vd.git
cd vllm-kivo-vd
git checkout chore/sync-upstream-main

uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install pytest

.venv/bin/python -m pytest tests_kivo -q
.venv/bin/python scripts/kivo_vd/check_vllm_runtime_env.py
```

If using a prebuilt vLLM wheel plus repo source overlay, run:

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"
.venv/bin/python scripts/kivo_vd/setup_runtime_source_overlay.py \
  --repo-root "$PWD"
```

Tiny-model dry-run:

```bash
.venv/bin/python scripts/kivo_vd/run_vllm_kivo_dry_run.py \
  --model sshleifer/tiny-gpt2 \
  --max-tokens 8 \
  --enable-kivo-vd \
  --gpu-memory-utilization 0.05 \
  --max-model-len 128 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 1

.venv/bin/python scripts/kivo_vd/analyze_dry_run_events.py \
  --input outputs/kivo_vd/vllm_kivo_dry_run_events.jsonl
```

Small-model follow-up:

```bash
.venv/bin/python scripts/kivo_vd/run_vllm_kivo_dry_run.py \
  --model gpt2 \
  --max-tokens 16 \
  --enable-kivo-vd \
  --gpu-memory-utilization 0.05 \
  --max-model-len 128 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 1
```

## Setup Option B: Docker-Based Linux/NVIDIA

Use a CUDA-capable container with NVIDIA Container Toolkit enabled. From inside
the container:

```bash
git clone https://github.com/<your-org-or-user>/vllm-kivo-vd.git
cd vllm-kivo-vd
git checkout chore/sync-upstream-main

uv venv --python 3.12
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
uv pip install pytest

.venv/bin/python -m pytest tests_kivo -q
.venv/bin/python scripts/kivo_vd/check_vllm_runtime_env.py
.venv/bin/python scripts/kivo_vd/setup_runtime_source_overlay.py \
  --repo-root "$PWD"
.venv/bin/python scripts/kivo_vd/run_vllm_kivo_dry_run.py \
  --model sshleifer/tiny-gpt2 \
  --max-tokens 8 \
  --enable-kivo-vd \
  --gpu-memory-utilization 0.05 \
  --max-model-len 128 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 1

.venv/bin/python scripts/kivo_vd/analyze_dry_run_events.py \
  --input outputs/kivo_vd/vllm_kivo_dry_run_events.jsonl
```

## Setup Option C: Cloud GPU Instance

Use a Linux GPU instance with an NVIDIA driver and CUDA-compatible PyTorch.
Then follow the native Linux/NVIDIA commands above.

Recommended first pass:

```bash
nvidia-smi
.venv/bin/python scripts/kivo_vd/check_vllm_runtime_env.py
.venv/bin/python scripts/kivo_vd/setup_runtime_source_overlay.py \
  --repo-root "$PWD"
.venv/bin/python scripts/kivo_vd/run_vllm_kivo_dry_run.py \
  --model sshleifer/tiny-gpt2 \
  --max-tokens 8 \
  --enable-kivo-vd \
  --gpu-memory-utilization 0.05 \
  --max-model-len 128 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 1

.venv/bin/python scripts/kivo_vd/analyze_dry_run_events.py \
  --input outputs/kivo_vd/vllm_kivo_dry_run_events.jsonl
```

These runtime limits are intentionally low for validation. On large GPUs, a
tiny model can otherwise produce an unexpectedly large KV-cache allocation plan.
Use the conservative command first, then increase limits only after dry-run
event export and output equality are confirmed.

Note: the first successful RunPod validation used `gpt2`, not
`sshleifer/tiny-gpt2`, because the tiny model can hit an attention embedding
dimension limitation in the selected vLLM backend. If `tiny-gpt2` fails with a
small `FLEX_ATTENTION` embedding-dimension error, retry with `gpt2` and the same
conservative runtime limits.

## Expected Dry-Run JSON

The dry-run script prints compact JSON. Important fields:

- `model`
- `prompt_token_length`
- `kivo_enabled`
- `gpu_memory_utilization`
- `max_model_len`
- `max_num_batched_tokens`
- `max_num_seqs`
- `baseline_text`
- `kivo_text`
- `outputs_match`
- `event_output`
- `num_events_exported`
- `observer_counters`
- `observer_note`
- `dry_run_only`

The default event export path is:

```text
outputs/kivo_vd/vllm_kivo_dry_run_events.jsonl
```

Inspect event types with:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/kivo_vd/vllm_kivo_dry_run_events.jsonl")
print([json.loads(line)["event_type"] for line in path.read_text().splitlines()])
PY
```

## Failure Checklist

If validation fails, check:

- vLLM build issue: missing compiled extension or custom op;
- CUDA unavailable: `torch.cuda.is_available()` is false;
- model download issue: HuggingFace network, token, cache, or rate limit;
- observer not reachable: V1 multiprocessing may hide scheduler internals;
- no events exported: confirm `--enable-kivo-vd` and in-process engine core;
- outputs differ unexpectedly: confirm greedy sampling, same seed, same prompt,
  and no non-Kivo config differences.

## Future Work

- Add a public Kivo runtime config path.
- Add an engine-core utility RPC for event export in multiprocessing mode.
- Validate runtime dry-run on modern RoPE/GQA models.
- Add quality benchmarks after dry-run stability is established.
- Measure real GPU memory and latency only after candidate-block attention or
  active residency mechanisms exist.
