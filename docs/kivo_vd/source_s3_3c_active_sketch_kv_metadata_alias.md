# Phase S3.3C Active Sketch KV Metadata Alias

## Purpose

Phase S3.3C is the first active sketch-controlled runtime phase. It combines:

- the real KV-cache block sketch construction from S3.3B; and
- the cloned attention-metadata aliasing mechanism from S3.2A.

The result is an active metadata-control experiment driven by real runtime
KV-cache sketches, while keeping scheduler-owned block tables, slot mappings,
and KV allocation untouched.

## Policy

Enable the phase with:

```bash
KIVO_SOURCE_ENABLE=1
KIVO_SOURCE_POLICY=active_sketch_kv_metadata_alias
KIVO_SOURCE_OBSERVE_PATH=/path/to/events.jsonl
KIVO_SOURCE_SKETCH_DIM=8
KIVO_SOURCE_MAX_SKETCH_BLOCKS=4
KIVO_SOURCE_BUDGET_RATIO=0.5
KIVO_SOURCE_ACTIVE_FILTER_MODE=alias_excluded_blocks_to_sketch_selected
```

This phase emits two event types:

```text
kivo_source_s3_3c_active_sketch_plan_v1
kivo_source_s3_3c_active_sketch_metadata_alias_v1
```

The tensor hook computes the real sketch plan from `kv_cache`. The metadata
hook consumes the latest compatible sketch plan and aliases excluded visible
block IDs to a selected target block in cloned attention metadata.

## Matching Strategy

S3.3C uses a bounded in-memory latest-plan cache. The metadata hook does not
attempt exact request identity reconstruction. Instead it uses a conservative
compatibility rule:

- extract visible physical block IDs from metadata;
- load the latest sketch plan for the current process;
- require overlap between visible metadata blocks and sketch candidate blocks;
- keep selected visible blocks from the sketch plan intersection;
- if selected overlap is empty but candidate overlap exists, keep the most
  recent visible block as a conservative fallback;
- alias excluded visible blocks to the selected target in cloned metadata.

If the overlap is ambiguous or absent, the hook fails closed.

## RunPod Probe

```bash
cd /workspace/vllm-kivo-vd

PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.run_source_s3_3c_active_sketch_kv_metadata_alias \
  --model gpt2 \
  --max-tokens 8 \
  --sketch-dim 8 \
  --max-sketch-blocks 4 \
  --budget-ratio 0.5 \
  --active-filter-mode alias_excluded_blocks_to_sketch_selected \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_3c_active_sketch_kv_metadata_alias.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_3c_active_sketch_kv_metadata_alias.md \
  --events-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_3c_active_sketch_kv_metadata_alias_events.jsonl \
  --continue-on-error
```

Validate:

```bash
PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.validate_source_s3_3c_active_sketch_kv_metadata_alias \
  --input-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_3c_active_sketch_kv_metadata_alias.json \
  --events-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_3c_active_sketch_kv_metadata_alias_events.jsonl \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_3c_active_sketch_kv_metadata_alias_validation.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_3c_active_sketch_kv_metadata_alias_validation.md
```

## Success Criteria

S3.3C passes when:

- all baseline and active generations succeed;
- real sketch-plan events are written;
- metadata alias events are written;
- at least one sketch plan is used by the metadata hook;
- at least one metadata mutation is attempted and applied;
- at least one active-routing event is recorded;
- no event claims measured memory reduction, quality preservation, or
  performance improvement.

Output preservation is not required. Output change is allowed.

## Boundary

Passing S3.3C means real runtime KV sketches can actively affect cloned
attention metadata behavior. It does not reduce KV allocation. It does not
prove latency improvement, quality preservation, or final selected attention.
