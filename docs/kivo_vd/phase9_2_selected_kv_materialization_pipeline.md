# Kivo-VD Phase 9.2: Selected-KV Materialization Pipeline

Phase 9.2 provides one command for the Phase 9.0 synthetic selected-KV
materialization and Phase 9.1 comparison workflow.

This is orchestration and reporting only. It does not access real vLLM KV,
change attention, or enable active routing.

## Relationship To Prior Artifacts

The pipeline consumes:

- Phase 7 dry-run routing events;
- the Phase 7 event memory estimate;
- optionally, Phase 8 event-aware sketch-buffer accounting.

Phase 9.0 creates and gathers synthetic temporary K/V tensors. Phase 9.1
compares their payload and copy timing with theoretical skipped-KV opportunity
and optional sketch-buffer overhead.

## Pipeline Stages

1. Synthetic selected-KV materialization outside attention.
2. Materialization comparison and report generation.

Each stage records its command, timestamps, status, return code, stdout/stderr
previews, and output paths in `pipeline_summary.json`.

## Run On CPU

```bash
.venv/bin/python \
  scripts/kivo_vd/run_selected_kv_materialization_pipeline.py \
  --events \
    outputs/kivo_vd/runs/phase7_gpt2_medium_memory_accounting/\
kivo_dry_run_events.jsonl \
  --event-estimate \
    outputs/kivo_vd/runs/phase7_gpt2_medium_memory_accounting/\
kivo_event_memory_estimate.json \
  --sketch-accounting \
    outputs/kivo_vd/runs/phase8_gpt2_sketch_buffer_accounting/\
event_aware_sketch_buffer_accounting.json \
  --device cpu \
  --run-name phase9_gpt2_selected_kv_materialization_cpu
```

CPU mode validates parsing, tensor shapes, accounting, report generation, and
pipeline behavior. CPU copy timing is not a GPU runtime prediction.

## Run On RunPod CUDA

```bash
.venv/bin/python \
  scripts/kivo_vd/run_selected_kv_materialization_pipeline.py \
  --events \
    outputs/kivo_vd/runs/phase7_gpt2_medium_memory_accounting_full_ids/\
kivo_dry_run_events.jsonl \
  --event-estimate \
    outputs/kivo_vd/runs/phase7_gpt2_medium_memory_accounting_full_ids/\
kivo_event_memory_estimate.json \
  --sketch-accounting \
    outputs/kivo_vd/runs/phase8_gpt2_sketch_buffer_accounting_full_ids/\
event_aware_sketch_buffer_accounting.json \
  --model gpt2 \
  --num-layers 12 \
  --num-kv-heads 12 \
  --head-dim 64 \
  --block-size 16 \
  --dtype-bytes 2 \
  --device cuda \
  --max-events 32 \
  --run-name phase9_gpt2_selected_kv_materialization_full_ids
```

## RunPod Full-ID Pipeline Result

The L40S run completed successfully:

- pipeline success: `true`;
- selected-KV materialization: succeeded;
- selected-KV materialization comparison: succeeded;
- events processed: `32`;
- full-ID events: `32`;
- preview-only events: `0`;
- average requested and materialized blocks: `16.0`;
- average materialization ratio: `0.390955`;
- average copy time: `0.047969 ms`;
- warnings: none.

The preceding Phase 8 full-ID accounting pipeline also succeeded for sketch
measurement, overhead comparison, and event-aware accounting. Its savings
remain theoretical only. The Phase 9 pipeline preserved:

- `synthetic_kv: true`;
- `outside_attention_path: true`;
- `full_kv_still_allocated: true`;
- `active_routing: false`;
- `measured_runtime_reduction: false`;
- `quality_not_measured: true`.

## Dry-Run Planning

Dry-run creates only the run directory and `pipeline_summary.json`. It does not
read the artifacts, import torch, or execute either stage.

```bash
.venv/bin/python \
  scripts/kivo_vd/run_selected_kv_materialization_pipeline.py \
  --events \
    outputs/kivo_vd/runs/phase7_gpt2_medium_memory_accounting/\
kivo_dry_run_events.jsonl \
  --event-estimate \
    outputs/kivo_vd/runs/phase7_gpt2_medium_memory_accounting/\
kivo_event_memory_estimate.json \
  --run-name phase9_gpt2_selected_kv_materialization_dry_run \
  --dry-run
```

Use `--continue-on-error` to run later stages for diagnostics after an earlier
failure. Normally, dependent stages are marked skipped.

## Expected Outputs

- `selected_kv_materialization.json`
- `selected_kv_materialization.md`
- `selected_kv_materialization_comparison.json`
- `selected_kv_materialization_comparison.md`
- `pipeline_summary.json`

Unless `--output-dir` supplies an exact directory, outputs are written under:

```text
outputs/kivo_vd/runs/<run-name>/
```

## Interpretation

Selected payload and copy timing describe synthetic temporary gathers.
Comparison ratios place that payload beside Phase 7 theoretical skipped-KV
bytes. Optional Phase 8 rows add named sketch-pool overhead.

Preview-only event exports undercount copied payload. Phase 9.1 carries this
warning forward and does not recommend a strong repeated-run conclusion until
complete selected block IDs are available.

Complete selected block IDs are required before using Phase 9 ratios for Phase
10 planning. Regenerate the Phase 7 event artifact with:

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

Then point Phase 9.2 at the resulting `kivo_dry_run_events.jsonl`. Full-ID
events preserve the normal previews and add complete selected, recent, and
skipped arrays. This workflow was validated successfully on RunPod.

## Caveats

- KV tensors are synthetic.
- Materialization occurs outside the attention path.
- Full KV remains allocated.
- No real vLLM KV cache is accessed.
- No active routing is implemented.
- No measured runtime memory reduction is claimed.
- Quality is not measured.
- No latency improvement is claimed.
