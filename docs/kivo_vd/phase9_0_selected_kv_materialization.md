# Kivo-VD Phase 9.0: Selected-KV Materialization

Phase 8 established that compact sketch-buffer payload is small relative to
the cumulative theoretical skipped-KV opportunity in the validated GPT-2
dry-run. Its readiness gate authorized a narrow next experiment.

Phase 9.0 materializes selected synthetic KV blocks into temporary tensors
outside attention. It measures temporary payload, copy time, and CUDA allocator
deltas without reading or modifying the real vLLM KV cache.

## What Materialization Means

The script creates synthetic block-major tensors:

```text
K: [layers, KV heads, pool blocks, block size, head dimension]
V: [layers, KV heads, pool blocks, block size, head dimension]
```

For each exported routing decision, selected block IDs index the pool dimension
and produce temporary selected K and V tensors.

This is a gather/copy experiment only. The selected tensors are not passed to
attention, and they do not replace the full synthetic or runtime KV pool.

## Event Export Limitation

Current runtime observer events store only an eight-ID
`selected_block_preview` by default, even when `selected_block_count` is
larger. The script prefers complete `selected_block_ids` when present.

When only a preview exists, it materializes exactly those exported IDs and
reports:

- the requested selected-block count;
- the smaller materialized count;
- `selected_ids_preview_only: true`;
- an explicit warning.

It does not invent missing physical block IDs. Preview-only results therefore
underestimate the temporary payload for that routing decision.

Complete selected block IDs are required before using Phase 9 materialization
ratios for Phase 10 planning. Preview-only exports remain useful for debugging
but undercount payload.

Opt in during the Kivo runtime event run with:

```bash
--export-full-block-ids
```

The flag sets `KIVO_EXPORT_FULL_BLOCK_IDS=1` before engine construction. The
environment variable can also be set directly. Enabled events retain previews
and additionally contain:

- `selected_block_ids_full`
- `recent_block_ids_full`
- `skipped_block_ids_full`
- `full_block_ids_exported: true`

Default events contain `full_block_ids_exported: false` and do not include the
full arrays.

## Memory Formula

```text
bytes_per_block =
    2 * layers * KV_heads * block_size * head_dim * dtype_bytes
```

The factor of two accounts for K and V.

When skipped-block counts are available:

```text
full_considered_blocks = requested_selected_blocks + skipped_blocks
materialization_ratio =
    materialized_selected_bytes / full_considered_KV_bytes
```

For preview-only rows, this ratio describes the exported preview payload, not
the complete candidate set.

## Run On CPU

```bash
.venv/bin/python scripts/kivo_vd/materialize_selected_kv.py \
  --events \
    outputs/kivo_vd/runs/phase7_gpt2_medium_memory_accounting/\
kivo_dry_run_events.jsonl \
  --model gpt2 \
  --num-layers 12 \
  --num-kv-heads 12 \
  --head-dim 64 \
  --block-size 16 \
  --dtype-bytes 2 \
  --device cpu \
  --max-events 32 \
  --output-json \
    outputs/kivo_vd/phase9_0_gpt2_selected_kv_materialization_cpu.json \
  --output-md \
    outputs/kivo_vd/phase9_0_gpt2_selected_kv_materialization_cpu.md
```

CPU mode validates parsing, tensor shapes, payload accounting, and gather
timing. CPU timings are not GPU runtime predictions.

## Run On RunPod CUDA

First regenerate the Phase 7 events with complete IDs:

```bash
.venv/bin/python scripts/kivo_vd/run_memory_accounting_pipeline.py \
  --model gpt2 \
  --prompt "$LONG_PROMPT" \
  --max-tokens 32 \
  --gpu-memory-utilization 0.05 \
  --max-model-len 768 \
  --max-num-batched-tokens 768 \
  --max-num-seqs 1 \
  --num-layers 12 \
  --num-kv-heads 12 \
  --head-dim 64 \
  --block-size 16 \
  --dtype-bytes 2 \
  --export-full-block-ids \
  --run-name phase7_gpt2_medium_memory_accounting_full_ids
```

```bash
.venv/bin/python scripts/kivo_vd/materialize_selected_kv.py \
  --events \
    outputs/kivo_vd/runs/phase7_gpt2_medium_memory_accounting_full_ids/\
kivo_dry_run_events.jsonl \
  --model gpt2 \
  --num-layers 12 \
  --num-kv-heads 12 \
  --head-dim 64 \
  --block-size 16 \
  --dtype-bytes 2 \
  --device cuda \
  --max-events 32 \
  --output-json \
    outputs/kivo_vd/phase9_0_gpt2_selected_kv_materialization.json \
  --output-md \
    outputs/kivo_vd/phase9_0_gpt2_selected_kv_materialization.md
```

Use `--num-pool-blocks` to force a specific synthetic pool size. Otherwise,
the script infers the minimum from exported IDs and falls back to `256` when
there are no usable IDs.

## Interpreting Results

- Selected KV bytes describe temporary materialized payload.
- Copy time measures one synchronized gather per event.
- CUDA allocated/reserved deltas describe allocator behavior around the
  temporary tensors.
- Materialization ratio compares the copied payload with selected-plus-skipped
  KV bytes when skipped counts are available.
- `full_block_ids_exported_count` and `preview_only_event_count` identify
  whether aggregate ratios use complete candidate sets.

These are microbenchmark measurements, not end-to-end latency or memory-saving
results.

## Caveats

- KV tensors are synthetic.
- Materialization is outside the attention path.
- Full KV remains allocated.
- No active routing is implemented.
- No real vLLM KV cache is accessed.
- No measured runtime KV memory reduction is claimed.
- No latency or quality claim follows from this experiment.

## Next Steps

First validate complete block-ID export or another safe source of full selected
IDs. Then compare temporary payload and copy cost on CUDA. Any later real-KV
capture must remain observational and outside attention before considering
behavior-changing routing.
