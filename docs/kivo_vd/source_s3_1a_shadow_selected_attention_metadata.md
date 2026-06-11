# Kivo-VD Phase S3.1A Shadow Selected-Attention Metadata

Phase S3.1A computes a shadow selected-attention plan at the real attention
metadata boundary confirmed in Phase S3.0B.

This phase is still planning only:

- it does not mutate `block_table_tensor`
- it does not mutate `slot_mapping`
- it does not mutate attention metadata
- it does not change model outputs
- it does not claim memory reduction, latency reduction, or selected attention

## What It Computes

At `gpu_model_runner._build_attention_metadata(...)`, the shadow policy records:

- visible block count and a bounded visible block ID sample when safe
- selected block count and selected block ID sample
- excluded block count and excluded block ID sample
- theoretical reduction in attention-visible blocks
- reduction ratio relative to visible blocks

The current selection policy is a deterministic placeholder:

- policy name: `deterministic_placeholder_block_score`
- keep the most recent `keep_recent_blocks`
- use a stable placeholder score for older visible blocks
- apply a budget ratio such as `0.5`

This is not the final sketch selector. It only proves that a future
metadata-filtering decision can be computed at the correct backend-agnostic
hook.

## Runtime Boundary

The primary planning hook runs at:

- `vllm/v1/worker/gpu_model_runner.py::_build_attention_metadata(...)`

The lower-level `build_attn_metadata(...)` observer path may still exist for
Phase S3.0B visibility, but S3.1A planning is intentionally anchored to the
higher-level metadata boundary.

## How To Run

RunPod command pattern:

```bash
cd /workspace/vllm-kivo-vd

PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.run_source_s3_1a_shadow_selected_attention_metadata \
  --model gpt2 \
  --max-tokens 8 \
  --budget-ratio 0.5 \
  --keep-recent-blocks 1 \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_1a_shadow_selected_attention_metadata.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_1a_shadow_selected_attention_metadata.md \
  --events-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_1a_shadow_selected_attention_metadata_events.jsonl \
  --continue-on-error
```

Validation command:

```bash
PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.validate_source_s3_1a_shadow_selected_attention_metadata \
  --input-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_1a_shadow_selected_attention_metadata.json \
  --events-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_1a_shadow_selected_attention_metadata_events.jsonl \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_1a_shadow_selected_attention_metadata_validation.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_1a_shadow_selected_attention_metadata_validation.md
```

## Interpretation

If S3.1A passes, we know:

- the real metadata boundary can support a future selected-attention planning step
- visible blocks can be summarized there without mutating runtime behavior
- a future metadata filter can be reasoned about before any real attention change

This phase does not prove selected attention, memory savings, latency
improvement, or quality preservation.
