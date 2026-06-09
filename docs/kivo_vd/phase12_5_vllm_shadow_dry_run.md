# Phase 12.5: vLLM Shadow Dry-Run

## Purpose

Phase 12.5 combines three checks in one report:

1. vLLM, Torch, CUDA, and compiled-extension import status;
2. an optional conservative baseline vLLM generation;
3. validator-compatible Phase 12 events emitted after generation through the
   standalone runtime touchpoint.

This is not an active runtime hook. Generation uses normal vLLM behavior, and
the runtime-adjacent shadow events are produced separately afterward.

## Safety Boundary

The workflow does not modify:

- scheduler decisions;
- GPUModelRunner;
- attention kernels or metadata;
- KV cache allocation or contents;
- block tables or slot mappings;
- generated output.

Every report states:

- `dry_run_only=true`;
- `shadow_only=true`;
- `active_routing=false`;
- `measured_runtime_reduction=false`;
- `no_attention_kernel_change=true`;
- `no_kv_cache_mutation=true`;
- `no_scheduler_change=true`.

## Environment-Only Check

```bash
.venv/bin/python scripts/kivo_vd/run_phase12_vllm_shadow_dry_run.py \
  --model gpt2 \
  --skip-vllm-generation \
  --output-json outputs/kivo_vd/runs/phase12_5_env_check.json \
  --output-md outputs/kivo_vd/runs/phase12_5_env_check.md \
  --continue-on-error
```

The report records Python, Torch, CUDA, GPU, vLLM, `vllm._C`,
`vllm._C_stable_libtorch`, and `vllm.vllm_flash_attn` import status. Optional
extension failures are reported rather than hidden.

## Baseline Generation

```bash
.venv/bin/python scripts/kivo_vd/run_phase12_vllm_shadow_dry_run.py \
  --model gpt2 \
  --prompt "Kivo Phase 12 vLLM baseline smoke." \
  --max-tokens 8 \
  --gpu-memory-utilization 0.05 \
  --max-model-len 128 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 1 \
  --output-json \
    outputs/kivo_vd/runs/phase12_5_vllm_generation_no_shadow.json \
  --output-md \
    outputs/kivo_vd/runs/phase12_5_vllm_generation_no_shadow.md
```

No Phase 12 event file is created when shadow mode is disabled.

## Generation With Shadow Events

```bash
.venv/bin/python scripts/kivo_vd/run_phase12_vllm_shadow_dry_run.py \
  --model gpt2 \
  --prompt "Kivo Phase 12 vLLM shadow smoke." \
  --max-tokens 8 \
  --gpu-memory-utilization 0.05 \
  --max-model-len 128 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 1 \
  --enable-shadow \
  --shadow-output-jsonl \
    outputs/kivo_vd/runs/phase12_5_shadow_events.jsonl \
  --output-json \
    outputs/kivo_vd/runs/phase12_5_vllm_generation_with_shadow.json \
  --output-md \
    outputs/kivo_vd/runs/phase12_5_vllm_generation_with_shadow.md
```

The script uses actual prompt token count when vLLM exposes it. It then emits
four runtime-adjacent events for research layers `0`, `5`, `8`, and `11`.
Block scores remain synthetic; no query, key, value, or KV-cache tensor is
read.

Validate independently:

```bash
.venv/bin/python scripts/kivo_vd/validate_phase12_shadow_event.py \
  --input outputs/kivo_vd/runs/phase12_5_shadow_events.jsonl \
  --output-json \
    outputs/kivo_vd/runs/phase12_5_shadow_event_validation.json \
  --output-md \
    outputs/kivo_vd/runs/phase12_5_shadow_event_validation.md
```

## Readiness Interpretation

`phase12_6_runtime_hook_ready=true` requires all of:

- vLLM import succeeds;
- baseline generation succeeds;
- shadow mode is explicitly enabled;
- all emitted events pass validation.

Environment-only or synthetic-event-only success is insufficient. A passing
report permits consideration of one separately reviewed opt-in runtime hook;
it does not authorize active routing.

## Caveats

- Shadow events are emitted after generation, not from a real runtime hook.
- Scores are synthetic.
- No runtime memory reduction or latency improvement is measured.
- Generation success is not a quality-preservation claim.
- Active selected attention remains outside Phase 12.
