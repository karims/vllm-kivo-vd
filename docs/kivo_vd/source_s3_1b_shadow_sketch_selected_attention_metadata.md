# Kivo-VD Phase S3.1B Shadow Sketch Selected-Attention Metadata

Phase S3.1B replaces the S3.1A deterministic placeholder score with a more
meaningful metadata-derived proxy score at the real attention metadata
boundary.

This phase is still shadow-only:

- it does not mutate `block_table_tensor`
- it does not mutate `slot_mapping`
- it does not mutate attention metadata
- it does not change model outputs
- it does not claim memory reduction, latency reduction, quality preservation,
  or selected attention

## What It Computes

At `gpu_model_runner._build_attention_metadata(...)`, the shadow sketch policy
records:

- visible block count and a bounded visible block ID sample when safe
- selected block count and selected block ID sample
- excluded block count and excluded block ID sample
- theoretical reduction in attention-visible blocks
- reduction ratio relative to visible blocks
- a bounded `block_score_sample`
- whether fallback to recency-only scoring was required

The current selection policy is:

- policy name: `slot_coverage_recency_proxy`
- keep the most recent `keep_recent_blocks`
- estimate per-block slot coverage from `slot_mapping` when safe
- combine normalized coverage and recency into a shadow score
- fall back to recency-only scoring if safe coverage cannot be derived

This is still not the final sketch selector. It is a safer proxy score that
uses real runtime metadata rather than the S3.1A placeholder hash score.

## Runtime Boundary

The primary planning hook runs at:

- `vllm/v1/worker/gpu_model_runner.py::_build_attention_metadata(...)`

S3.1B stays at the same backend-agnostic boundary as S3.1A. The only change is
the shadow scoring policy.

## How To Run

RunPod command pattern:

```bash
cd /workspace/vllm-kivo-vd

PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.run_source_s3_1b_shadow_sketch_selected_attention_metadata \
  --model gpt2 \
  --max-tokens 8 \
  --budget-ratio 0.5 \
  --keep-recent-blocks 1 \
  --coverage-weight 0.6 \
  --recency-weight 0.4 \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_1b_shadow_sketch_selected_attention_metadata.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_1b_shadow_sketch_selected_attention_metadata.md \
  --events-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_1b_shadow_sketch_selected_attention_metadata_events.jsonl \
  --continue-on-error
```

Validation command:

```bash
PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.validate_source_s3_1b_shadow_sketch_selected_attention_metadata \
  --input-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_1b_shadow_sketch_selected_attention_metadata.json \
  --events-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_1b_shadow_sketch_selected_attention_metadata_events.jsonl \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_1b_shadow_sketch_selected_attention_metadata_validation.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_1b_shadow_sketch_selected_attention_metadata_validation.md
```

## Interpretation

If S3.1B passes, we know:

- the real metadata boundary can support a more meaningful proxy-scored shadow
  selection plan
- metadata-derived coverage and recency are available often enough to drive a
  useful shadow policy
- fallback behavior can be measured without mutating runtime state

This phase still does not prove selected attention, memory savings, latency
improvement, or quality preservation. It is the final shadow step before any
future active metadata filtering experiment.
