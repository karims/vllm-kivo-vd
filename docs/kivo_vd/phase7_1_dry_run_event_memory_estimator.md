# Kivo-VD Phase 7.1: Dry-Run Event Memory Estimator

Phase 7.1 converts exported Kivo dry-run routing counts into theoretical active
KV byte estimates. It complements the measured CUDA checkpoints from Phase 7.0
but does not claim that vLLM actually released or avoided those bytes.

## Purpose

Current vLLM execution still allocates and attends over the normal/full KV
cache. Kivo routing decisions are computed and ignored. The estimator asks a
narrow counterfactual question:

> If only the blocks selected by a dry-run decision were active, how many
> model KV bytes would those blocks represent?

This is useful for memory-policy planning before active routing exists.

## Bytes Per KV Block

The estimator uses:

```text
bytes_per_block =
    2 * num_layers * num_kv_heads * head_dim * block_size * dtype_bytes
```

The factor `2` represents K and V. `num_kv_heads` is the number of KV heads,
which can differ from query heads for GQA/MQA models. The formula estimates
model tensor payload only; it excludes allocator overhead, fragmentation,
metadata, sketch storage, and backend-specific padding.

## Relationship To Phase 7.0

Phase 7.0 records real CUDA allocator checkpoints during unchanged vLLM
generation. Phase 7.1 reads observer events and performs theoretical active-KV
accounting. A Phase 7.0 JSON can be supplied for model metadata when it contains
the required dimensions, but its measured CUDA deltas are not reinterpreted as
Kivo savings.

## GPT-2 Example

```bash
.venv/bin/python scripts/kivo_vd/estimate_kivo_memory_from_events.py \
  --events outputs/kivo_vd/vllm_kivo_dry_run_events.jsonl \
  --memory-baseline outputs/kivo_vd/phase7_0_gpt2_kivo_dry_run_memory.json \
  --model gpt2 \
  --num-layers 12 \
  --num-kv-heads 12 \
  --head-dim 64 \
  --block-size 16 \
  --dtype-bytes 2 \
  --output-json outputs/kivo_vd/phase7_1_gpt2_event_memory_estimate.json \
  --output-md outputs/kivo_vd/phase7_1_gpt2_event_memory_estimate.md
```

The script focuses on `dry_run_routing_decision` events. It treats selected
blocks as the active union. Recent blocks are already represented in that union
and are not added a second time.

## Output

The JSON and Markdown outputs include:

- resolved model/KV metadata;
- estimated bytes per block;
- average selected, recent, and skipped blocks;
- average active and skipped KV bytes;
- average and percentile theoretical reduction ratios;
- request IDs, sources, warnings, and a capped per-event sample;
- `estimated_only: true`;
- `measured_runtime_reduction: false`.

## Interpretation

A high estimated reduction ratio means the dry-run selector omitted many of the
blocks it considered. It does not mean those bytes were absent from GPU memory:
the current runtime still owns the full KV cache and executes normal attention.

Compare policies using identical model dimensions, block size, dtype, prompts,
and selector settings. Treat events with saturated or tiny block counts
carefully, and retain quality and latency validation as separate future work.

## Proven Vs Not Proven

This phase proves that exported routing counts can drive reproducible,
model-aware theoretical KV accounting.

It does not prove:

- measured vLLM runtime memory reduction;
- active KV residency or candidate-block attention;
- latency improvement;
- quality preservation.
