# Kivo-VD Phase 3.6: Modern Model Reporting

Phase 3.6 updates the offline reporting path so modern HuggingFace extraction
metadata is preserved from Q/K sweep rows through active-KV simulation and
Markdown reports.

This is offline scripts/docs only. It does not change vLLM runtime behavior,
scheduler logic, GPUModelRunner, attention metadata, block tables, slot mapping,
kernels, model architecture, or training.

## Why Metadata Matters

Modern models can differ substantially from GPT-2-style models:

- separate `q_proj` / `k_proj` instead of fused `c_attn`;
- RoPE applied after linear projection;
- grouped-query or multi-query attention;
- query heads and KV heads may not match one-to-one.

Benchmark rows need to record this context so future results are comparable and
not overclaimed.

## Preserved Fields

The active-KV simulator now passes through:

- `model_name`
- `extraction_mode`
- `qk_space`
- `num_query_heads`
- `num_key_value_heads`
- `selected_query_head`
- `selected_kv_head`

The report generator includes a `Model and Extraction Metadata` section and
groups retrieval summaries by model/extraction fields when present.

## Pre-RoPE Limitation

For separate Q/K projection models, Phase 3.5 extracts Q/K before RoPE. Reports
now explicitly warn when `qk_space=pre_rope_projection`.

Pre-RoPE results are useful for offline sketch diagnostics, but runtime
post-RoPE behavior may differ. They should not be treated as final vLLM runtime
claims.

## GQA/MQA Interpretation

For GQA/MQA models:

- `selected_query_head` is the query head used for scoring;
- `selected_kv_head` is the mapped KV head;
- multiple query heads may map to the same KV head;
- `num_query_heads` and `num_key_value_heads` make the grouping visible.

## Future Qwen/TinyLlama Pipeline Commands

Qwen dry run:

```bash
.venv/bin/python scripts/kivo_vd/run_offline_benchmark_pipeline.py \
  --dry-run \
  --model-name Qwen/Qwen2.5-0.5B \
  --extraction-mode auto
```

TinyLlama dry run:

```bash
.venv/bin/python scripts/kivo_vd/run_offline_benchmark_pipeline.py \
  --dry-run \
  --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --extraction-mode auto
```

Remove `--dry-run` only in an environment where model downloads and optional HF
dependencies are available.

## Interpretation

Modern-model reports are still offline evidence. They do not prove measured
runtime memory reduction, latency improvement, output quality preservation, or
candidate-block attention behavior.
