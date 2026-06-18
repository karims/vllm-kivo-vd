# Sk-1.3c GPU Pod Smoke

This runbook is for the first real decode validation of the live
`random_projection` sketch path.

Local macOS runs are not the validation target. The local runner is only for
unit tests, argument checks, and JSON output structure. Real decode, sketch
build, demotion, ownership removal, and free-to-pool validation should happen
on a GPU pod with a working source-built vLLM runtime.

## Setup assumptions

- GPU is available and usable by vLLM.
- This repo is installed from source in the pod environment.
- The vLLM source build is working in that environment.
- A model is either already cached on the pod or can be downloaded there.
- Recommended tiny smoke model:
  - `Qwen/Qwen2.5-0.5B-Instruct`
- Fallback model:
  - `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- A local model path is also acceptable via `--model /path/to/model`.

## Required env path behavior

The smoke runner sets these internally before constructing `LLM(...)`:

- `KIVO_KV_SKETCH_ENABLE=1`
- `KIVO_KV_SKETCH_BACKEND=random_projection`
- `KIVO_KV_SKETCH_DIM=<--sketch-dim>`
- `KIVO_KV_SKETCH_SEED=<--sketch-seed>`
- `KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ENABLE=1`
- `KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ACTION=apply_block_table_only`
- `KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_POLICY=<--runtime-policy>`
- `KIVO_KV_RUNTIME_BLOCK_TABLE_KEEP_RECENT_BLOCKS=<--keep-recent-blocks>`
- `KIVO_KV_RUNTIME_BLOCK_TABLE_MAX_FULL_BLOCKS=<--max-full-blocks>`
- `KIVO_KV_DEMOTION_TRANSPORT_ENABLE=1`
- `KIVO_KV_DEMOTION_TRANSPORT_ACTION=apply_core_mark_demoted`
- `KIVO_KV_CORE_DEMOTION_ENABLE=1`
- `KIVO_KV_CORE_DEMOTION_ACTION=mark_demoted_only`
- `KIVO_KV_OWNERSHIP_REMOVE_ENABLE=1`
- `KIVO_KV_OWNERSHIP_REMOVE_ACTION=remove_marked_demoted_only`
- `KIVO_KV_FREE_TO_POOL_ENABLE=1`
- `KIVO_KV_FREE_TO_POOL_ACTION=free_removed_demoted_only`
- `KIVO_KV_DEMOTION_COUNTERS_ENABLE=1`
- `KIVO_KV_DEMOTION_COUNTERS_EXPORT_FILE=<--counter-export-file or <output>.counters.json>`

## Primary smoke command

Start with one request and a small decode budget:

```bash
.venv/bin/python scripts/kivo_vd/run_sk1_3c_live_sketch_smoke.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --max-tokens 16 \
  --prompt-repeats 24 \
  --num-prompts 1 \
  --max-model-len 512 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.05 \
  --sketch-dim 16 \
  --keep-recent-blocks 2 \
  --max-full-blocks 2 \
  --counter-export-file /tmp/sk1_3c_live_sketch_smoke.counters.json \
  --output /tmp/sk1_3c_live_sketch_smoke.json
```

If the recommended model is not available, try:

```bash
.venv/bin/python scripts/kivo_vd/run_sk1_3c_live_sketch_smoke.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --max-tokens 16 \
  --prompt-repeats 24 \
  --num-prompts 1 \
  --max-model-len 512 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.05 \
  --sketch-dim 16 \
  --keep-recent-blocks 2 \
  --max-full-blocks 2 \
  --counter-export-file /tmp/sk1_3c_live_sketch_smoke.counters.json \
  --output /tmp/sk1_3c_live_sketch_smoke.json
```

For a pre-downloaded local model path:

```bash
.venv/bin/python scripts/kivo_vd/run_sk1_3c_live_sketch_smoke.py \
  --model /models/Qwen2.5-0.5B-Instruct \
  --max-tokens 16 \
  --prompt-repeats 24 \
  --num-prompts 1 \
  --max-model-len 512 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.05 \
  --sketch-dim 16 \
  --keep-recent-blocks 2 \
  --max-full-blocks 2 \
  --counter-export-file /tmp/sk1_3c_live_sketch_smoke.counters.json \
  --output /tmp/sk1_3c_live_sketch_smoke.json
```

## PASS criteria

The smoke passes when the output JSON shows:

- `generation_success == true`
- `counter_export_file_found == true`
- `counter_summary.sketch_backend == "random_projection"`
- `counter_summary.sketch_build_attempted > 0`
- `counter_summary.sketch_build_succeeded > 0`
- `counter_summary.sketched_blocks_total > 0`
- `counter_summary.freed_after_sketch_blocks_total > 0`
- `counter_summary.sketch_missing_prevented_free == 0`
- `counter_summary.invariants_clean == true`

## Acceptable debug states

### 1. Decode succeeds but no sketch attempts

If:

- `generation_success == true`
- `counter_summary.sketch_build_attempted == 0`

Interpretation:

- demotion candidates were not triggered in this tiny run.

Smallest next adjustment:

- increase `--prompt-repeats`
- or reduce `--keep-recent-blocks`
- or reduce `--max-full-blocks`

### 2. Sketch attempts happen but all sketch builds fail

If:

- `counter_summary.sketch_build_attempted > 0`
- `counter_summary.sketch_build_succeeded == 0`

Interpretation:

- likely real KV block extraction/shape mismatch.

Next step:

- inspect runner JSON and counter export for shape/debug signals
- do not relax sketch gating

### 3. Sketch builds succeed but no free-to-pool after sketch

If:

- `counter_summary.sketch_build_succeeded > 0`
- `counter_summary.freed_after_sketch_blocks_total == 0`

Interpretation:

- sketching worked, but transport/core/remove/free did not fully progress.

Next step:

- inspect transport/core counters
- inspect ownership removal counters
- inspect free-to-pool counters

## Not in scope

- broad serving benchmark matrix
- CountSketch or structured sketch comparison
- performance tuning
- memory reduction claims
- quality claims

If this smoke passes, the next step can be a tiny baseline-vs-sketch quality
comparison on a handful of prompts.
