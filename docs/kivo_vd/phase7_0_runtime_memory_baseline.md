# Kivo-VD Phase 7.0: Runtime Memory Baseline

Phase 7.0 adds a real vLLM GPU memory measurement script for normal generation
and optional Kivo-VD dry-run generation.

This phase measures memory checkpoints only. It does not reduce KV memory,
change attention, route selected blocks, alter block tables or slot mapping, or
claim latency or quality improvements.

## Why Memory Should Not Decrease Yet

Kivo-VD runtime integration remains dry-run only. vLLM still allocates its
normal KV cache and attention still consumes the normal KV representation.
Kivo candidate decisions are observed and ignored.

Therefore, a Kivo-enabled Phase 7.0 run should not be expected to use less GPU
memory than its baseline. Small differences can arise from allocator state,
engine initialization, event metadata, and normal run-to-run variation.

## Script

The script is:

```text
scripts/kivo_vd/run_vllm_memory_baseline.py
```

It records these CUDA allocator checkpoints:

- `process_start`
- `before_llm_init`
- `after_llm_init`
- `before_generate`
- `after_generate`
- `after_request_or_cleanup`

Each checkpoint includes current and peak allocated/reserved bytes, plus
free/total device memory from `torch.cuda.mem_get_info()` when available. The
output also records vLLM, torch, CUDA, GPU, model, prompt, and conservative
engine configuration metadata.

CUDA allocator counters describe the current process. The script defaults to
an in-process V1 engine core both for cleaner measurement and for optional Kivo
observer export.

## Baseline Generation

```bash
.venv/bin/python scripts/kivo_vd/run_vllm_memory_baseline.py \
  --model gpt2 \
  --max-tokens 32 \
  --gpu-memory-utilization 0.05 \
  --max-model-len 256 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 1 \
  --output outputs/kivo_vd/phase7_0_gpt2_baseline_memory.json
```

## Kivo Dry-Run Generation

```bash
.venv/bin/python scripts/kivo_vd/run_vllm_memory_baseline.py \
  --model gpt2 \
  --max-tokens 32 \
  --enable-kivo-vd \
  --gpu-memory-utilization 0.05 \
  --max-model-len 256 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 1 \
  --output outputs/kivo_vd/phase7_0_gpt2_kivo_dry_run_memory.json
```

When Kivo is enabled, the script attempts to export observer events beside the
main JSON artifact. It records observer counters and the event path when the
in-process observer is accessible.

## Interpreting Checkpoints

- Initialization deltas show allocator growth while constructing the LLM and
  its configured KV cache.
- Generation deltas show current allocator changes across one request.
- Peak fields are cumulative high-water marks after the script resets torch's
  peak statistics.
- Cleanup deltas show what the Python/torch allocator reports after releasing
  request and engine references. Reserved memory may remain cached.
- `gpu_memory_utilization` constrains vLLM planning; it is not a direct
  measurement of actual KV bytes used by the request.

For useful comparisons, run baseline and Kivo dry-run with the same model,
prompt, token count, engine limits, software versions, and otherwise idle GPU.
Multiple repetitions are preferable to interpreting one small delta.

## Caveat

These artifacts are runtime memory baselines, not memory-reduction results.
Kivo-VD has no active KV residency or candidate-attention mechanism in Phase
7.0. Any future memory-reduction claim requires a behavior-changing mechanism,
quality validation, repeated measurements, and direct GPU evidence.
