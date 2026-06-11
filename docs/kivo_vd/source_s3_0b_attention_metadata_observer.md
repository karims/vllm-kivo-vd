# Kivo-VD Phase S3.0B Attention Metadata Observer

Phase S3.0A traced the path from block tables and slot mappings into the
attention metadata builders. Phase S3.0B adds an observer-only hook at that
backend-agnostic boundary.

The first S3.0B attempt reached healthy runtime execution and preserved model
outputs, but wrote zero observer events. The likely issue was that the initial
observer was placed only on `build_attn_metadata(...)`, while the active GPT-2 /
FlashAttention runtime path was more reliably visible from the higher-level
`_build_attention_metadata(...)` path in `gpu_model_runner.py`.

The fix is conservative: observe both
`gpu_model_runner._build_attention_metadata(...)` and
`gpu/attn_utils.build_attn_metadata(...)`, and record the `hook_point` in each
event so the runtime path can be verified explicitly.

The JSONL event stream may also contain records from other Kivo hook schemas in
the same file. The S3.0B validator filters to
`schema_version == "kivo_source_s3_0b_attention_metadata_observer_v1"` and
reports the ignored non-S3 events separately.

This phase records what metadata is visible at `build_attn_metadata(...)`
without mutating the runtime.

## What This Phase Observes

For each observed KV cache group, the hook records:

- block-table tensor presence, shape, dtype, and device
- slot-mapping presence, shape, dtype, and device
- `query_start_loc` shape and metadata
- `seq_lens` shape and metadata
- `positions` shape if present
- `max_query_len` and `max_seq_len` when available
- a bounded sample of block IDs and slot IDs, when safe
- a conservative visible-block-count estimate only when it can be computed
  safely without large tensor transfers
- env-debug visibility fields:
  - `kivo_source_enable_seen`
  - `kivo_source_policy_seen`
  - `observe_path_present`

The hook is observation-only:

- `mutation_attempted = false`
- `mutation_applied = false`
- `active_routing = false`
- `runtime_behavior_changed = false`
- `measured_runtime_reduction = false`

## What This Phase Does Not Prove

- It does not prove selected attention.
- It does not mutate slot mappings or block tables.
- It does not reduce KV memory.
- It does not reduce latency.
- It does not claim quality preservation.

## How To Run

The intended RunPod command pattern is:

```bash
cd /workspace/vllm-kivo-vd

PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.run_source_s3_0b_attention_metadata_observer \
  --model gpt2 \
  --max-tokens 8 \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_0b_attention_metadata_observer.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_0b_attention_metadata_observer.md \
  --events-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_0b_attention_metadata_observer_events.jsonl \
  --continue-on-error
```

Validation command:

```bash
PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.validate_source_s3_0b_attention_metadata_observer \
  --input-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_0b_attention_metadata_observer.json \
  --events-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_0b_attention_metadata_observer_events.jsonl \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_0b_attention_metadata_observer_validation.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_0b_attention_metadata_observer_validation.md
```

## Interpretation

If the observer pass succeeds, we know:

- the metadata boundary is observable in source form
- the tensors and shapes needed for future filtering are visible
- observation does not perturb model output

That is the correct boundary for the next research step, but it is still not a
selected-attention implementation.
