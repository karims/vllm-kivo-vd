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

## RunPod Full-ID Validation

The complete-ID path was validated on an NVIDIA L40S using driver
`580.126.09`, CUDA `13.0`, torch `2.11.0+cu130`, and vLLM `0.22.1` with the
repository source overlay. The GPT-2 run used a 632-token prompt, generated 32
tokens, and used:

- maximum model length: `768`;
- maximum batched tokens: `768`;
- maximum sequences: `1`;
- layers: `12`;
- KV heads: `12`;
- head dimension: `64`;
- block size: `16`;
- dtype bytes: `2`.

The Phase 7 analyzer found 97 events, including 32 routing decisions. All 32
routing events exported complete selected, recent, and skipped block-ID
arrays:

- average selected blocks: `16.0`;
- average recent blocks: `8.0`;
- average skipped blocks: `24.9375`;
- candidate budget: `16`;
- recent window: `8`;
- `full_block_ids_exported_count`: `32`;
- `preview_only_routing_event_count`: `0`;
- `all_routing_events_have_full_block_ids`: `true`;
- warnings: none.

Phase 9.0 processed all 32 events and materialized all 16 selected blocks per
event on average:

| metric | result |
| --- | ---: |
| Average requested selected blocks | `16.0` |
| Average materialized selected blocks | `16.0` |
| Average selected KV bytes | `9,437,184` |
| Total selected KV bytes materialized | `301,989,888` |
| Average full considered KV bytes | `24,145,920` |
| Average materialization ratio | `0.390955` |
| Average copy time | `0.047969 ms` |
| P50 copy time | `0.041623 ms` |
| P90 copy time | `0.043966 ms` |
| Maximum copy time | `0.233020 ms` |
| Average CUDA allocated delta | `9,437,184 bytes` |
| Maximum CUDA allocated delta | `9,437,184 bytes` |
| Average CUDA reserved delta | `655,360 bytes` |
| Maximum CUDA reserved delta | `20,971,520 bytes` |

This replaces the earlier preview-limited result, which materialized only 8
of 16 selected blocks and reported an artificially low ratio of about
`0.195`. The corrected full-ID ratio of about `0.391` is less aggressive but
remains a promising synthetic materialization signal.

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

Complete block-ID export and synthetic CUDA materialization are validated.
The next authorized experiment is standalone selected-KV torch reference
attention on synthetic tensors outside vLLM. Real-KV capture, active routing,
and behavior-changing attention remain unauthorized.
