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
    outputs/kivo_vd/runs/phase7_gpt2_medium_memory_accounting/\
kivo_dry_run_events.jsonl \
  --event-estimate \
    outputs/kivo_vd/runs/phase7_gpt2_medium_memory_accounting/\
kivo_event_memory_estimate.json \
  --sketch-accounting \
    outputs/kivo_vd/runs/phase8_gpt2_sketch_buffer_accounting/\
event_aware_sketch_buffer_accounting.json \
  --model gpt2 \
  --num-layers 12 \
  --num-kv-heads 12 \
  --head-dim 64 \
  --block-size 16 \
  --dtype-bytes 2 \
  --device cuda \
  --max-events 32 \
  --run-name phase9_gpt2_selected_kv_materialization
```

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

## Caveats

- KV tensors are synthetic.
- Materialization occurs outside the attention path.
- Full KV remains allocated.
- No real vLLM KV cache is accessed.
- No active routing is implemented.
- No measured runtime memory reduction is claimed.
- Quality is not measured.
- No latency improvement is claimed.
