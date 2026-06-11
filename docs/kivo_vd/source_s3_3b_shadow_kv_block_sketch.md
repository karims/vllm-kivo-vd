# Phase S3.3B Shadow KV Block Sketch

## Purpose

Phase S3.3B is the first real sketch-construction phase. It uses the actual
runtime `kv_cache` tensor observed at `unified_attention_with_output(...)` and
computes tiny bounded K/V block sketches in shadow mode.

This phase does not mutate runtime behavior. It does not edit attention
metadata, slot mappings, KV cache contents, scheduler state, or model outputs.

## Policy

Enable the shadow sketch observer with:

```bash
KIVO_SOURCE_ENABLE=1
KIVO_SOURCE_POLICY=shadow_kv_block_sketch
KIVO_SOURCE_OBSERVE_PATH=/path/to/events.jsonl
KIVO_SOURCE_SKETCH_DIM=8
KIVO_SOURCE_MAX_SKETCH_BLOCKS=4
KIVO_SOURCE_BUDGET_RATIO=0.5
```

Events use:

```text
schema_version = kivo_source_s3_3b_shadow_kv_block_sketch_v1
policy_name = shadow_kv_block_sketch
hook_point = unified_attention_with_output
sketch_source = kv_cache
sketch_method = random_projection_l2
```

## What It Computes

The observer derives a current physical block id from the current valid
`slot_mapping` entry:

- `current_slot_id = last valid slot_mapping value`
- `current_physical_block_id = current_slot_id // block_size`

It then sketches a bounded recent window of candidate physical blocks:

- current physical block
- previous physical blocks up to `max_sketch_blocks`

For each candidate block, it assumes the observed runtime layout:

```text
[num_blocks, 2, block_size, num_heads, head_dim]
```

and treats:

- `kv_cache[block_id, 0, ...]` as the K block
- `kv_cache[block_id, 1, ...]` as the V block

This assumption is guarded. If the layout does not match `ndim == 5` and
`shape[1] == 2`, the observer fails closed and records a blocker reason.

The actual sketch is a deterministic, tiny sign-based random projection over
the flattened K and V block tensors, with bounded summaries only.

## Selection

S3.3B remains shadow-only. It computes a simple score:

```text
score = k_l2_norm + v_l2_norm
```

and records selected versus excluded candidate blocks under the configured
budget ratio. The newest block is always kept when candidates exist.

This is a real sketch path, but it is not yet a final retrieval policy.

## RunPod Probe

```bash
cd /workspace/vllm-kivo-vd

PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.run_source_s3_3b_shadow_kv_block_sketch \
  --model gpt2 \
  --max-tokens 8 \
  --sketch-dim 8 \
  --max-sketch-blocks 4 \
  --budget-ratio 0.5 \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_3b_shadow_kv_block_sketch.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_3b_shadow_kv_block_sketch.md \
  --events-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_3b_shadow_kv_block_sketch_events.jsonl \
  --continue-on-error
```

Validate:

```bash
PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.validate_source_s3_3b_shadow_kv_block_sketch \
  --input-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_3b_shadow_kv_block_sketch.json \
  --events-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_3b_shadow_kv_block_sketch_events.jsonl \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_3b_shadow_kv_block_sketch_validation.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_3b_shadow_kv_block_sketch_validation.md
```

## Success Criteria

S3.3B passes when:

- all baseline and shadow generations succeed;
- shadow outputs match baseline outputs;
- at least one S3.3B event is written;
- at least one event computes a real block sketch;
- at least one event has a non-empty `block_sketch_sample`;
- all mutation and runtime-change flags remain false.

## Boundary

Passing S3.3B means that tiny real KV-cache block sketches can be built at
runtime from the Python-level hook. It does not prove selected attention,
memory reduction, latency reduction, or quality preservation.
