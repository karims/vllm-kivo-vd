# Kivo-VD Phase 6.3: Modern Model Structured Check

Phase 6.3 defines a small offline structured-sketch check on a modern
HuggingFace causal LM, with `Qwen/Qwen2.5-0.5B` as the recommended target.

This phase is offline only. It does not run vLLM inference, change scheduler
behavior, modify GPUModelRunner, alter attention kernels, change block tables or
slot mapping, add active routing, or claim measured runtime memory reduction.

## Why GPT-2 Is Not Enough

GPT-2-style models are useful for early validation because their attention
projection path is simple and lightweight. However, GPT-2 is not representative
of many modern serving targets:

- GPT-2 uses fused `c_attn` Q/K/V projection.
- GPT-2 does not exercise common separate `q_proj` / `k_proj` extraction paths.
- GPT-2 does not cover RoPE-positioned Q/K behavior.
- GPT-2 does not cover grouped-query or multi-query attention metadata.
- GPT-2 head dimensions can make some sketch dimensions full-dimensional rather
  than compressed.

Modern-model offline checks are needed before treating any structured sketch as
more than a GPT-2-specific signal.

## Why Qwen/Qwen2.5-0.5B

`Qwen/Qwen2.5-0.5B` is a good next offline target because it is relatively small
while exercising modern architecture features:

- separate Q/K projection modules;
- RoPE-oriented model family;
- query-head and KV-head metadata that can expose GQA/MQA-style mapping;
- larger modern head dimensions than GPT-2 in many configurations.

The goal is not to prove production behavior. The goal is to check whether the
structured sketch variants remain plausible on modern projected Q/K tensors.

## Important Caveats

### Pre-RoPE Projection Caveat

The current HF extraction path for separate `q_proj` / `k_proj` models evaluates
projected Q/K before RoPE is applied. Rows should include:

- `extraction_mode`
- `qk_space`
- `head_dim`
- compression metadata

If `qk_space=pre_rope_projection`, results are useful but limited. Runtime
post-RoPE attention behavior may differ.

### GQA/MQA Metadata Caveat

For models with different query-head and KV-head counts, the extraction helper
maps query heads to KV heads conservatively. Rows should preserve:

- `num_query_heads`
- `num_key_value_heads`
- `selected_query_head`
- `selected_kv_head`

When several query heads map to one KV head, head-level conclusions should be
interpreted with care.

### Retrieval-Only Caveat

This check measures offline retrieval metrics such as block recall and score
correlation. It is not:

- real vLLM runtime memory reduction;
- active KV routing;
- candidate-routed attention;
- a quality benchmark;
- a latency benchmark.

## Small Qwen Smoke Sweep

```bash
python scripts/kivo_vd/run_structured_sketch_param_sweep.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --sketch-types bidiagonal_sign_subsample,bidiagonal_sign,tridiagonal_sign \
  --sketch-dims 16,32 \
  --alphas 0.0,0.25,0.5,0.75,1.0 \
  --coordinate-strategies uniform,stride,low,high,alternating \
  --layers 0,1 \
  --heads 0,1,2,3 \
  --max-tokens 256 \
  --output outputs/kivo_vd/runs/phase6_3_qwen_structured_smoke/structured_param_sweep.jsonl
```

This is intentionally small. It is meant to confirm that modern extraction and
structured sketch parameterization work before running a larger sweep.

## Summarize The Smoke Sweep

```bash
python scripts/kivo_vd/summarize_structured_sketch_param_sweep.py \
  --input outputs/kivo_vd/runs/phase6_3_qwen_structured_smoke/structured_param_sweep.jsonl \
  --output-json outputs/kivo_vd/runs/phase6_3_qwen_structured_smoke/structured_param_summary.json \
  --output-md outputs/kivo_vd/runs/phase6_3_qwen_structured_smoke/structured_param_summary.md
```

The summary groups by:

- `sketch_type`
- `sketch_dim`
- `structured_alpha`
- `structured_coordinate_strategy`

It preserves modern-model metadata when present, including model name,
extraction mode, Q/K space, query/KV head counts, selected heads, head dimension,
and compression metadata.

## How To Interpret Results

Good signs:

- recall@2x remains strong for compressed dimensions;
- recall@4x is high enough to support candidate-retrieval hypotheses;
- block score correlation is not severely degraded;
- a coordinate strategy improves retrieval or timing without being
  full-dimensional;
- metadata confirms the run used the intended modern extraction path.

Bad signs:

- structured variants collapse on Qwen while CountSketch/RP remain strong;
- only full-dimensional rows look good;
- pre-RoPE rows differ sharply from later post-RoPE validation;
- GQA/MQA head mappings show highly head-dependent behavior.

Conservative conclusion language:

```text
This Qwen smoke sweep is offline pre-RoPE Q/K retrieval evidence only. It does
not prove active KV routing, quality preservation, latency improvement, or vLLM
runtime memory reduction.
```
