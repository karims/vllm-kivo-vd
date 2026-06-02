# Kivo-VD Phase 3.5: Modern HuggingFace Q/K Extraction

Phase 3.5 extends the offline HuggingFace Q/K sketch evaluation scripts beyond
GPT-2-style fused attention projections.

This is still offline only. It does not change vLLM runtime behavior,
scheduler logic, GPUModelRunner, attention metadata, block tables, slot
mapping, kernels, model architecture, or training.

## Extraction Modes

The HF scripts now accept:

```bash
--extraction-mode auto
--extraction-mode gpt2_fused_c_attn
--extraction-mode separate_qk_proj
```

Default: `auto`.

### GPT-2 Fused `c_attn`

For GPT-2-style models, attention exposes a fused `c_attn` projection. The
script splits the fused QKV output into Q and K and reports:

```json
"qk_space": "gpt2_projection"
```

This preserves the existing `distilgpt2` / `gpt2` path.

### Separate Q/K Projection

For Llama/Qwen-style models, attention commonly exposes:

- `q_proj`
- `k_proj`
- `num_heads` or `num_attention_heads`
- `num_key_value_heads`, for GQA/MQA models

The script applies `q_proj` and `k_proj` to hidden states entering the selected
layer, reshapes Q and K into heads, and selects one query head plus the mapped
KV head.

It reports:

```json
"qk_space": "pre_rope_projection"
```

## Pre-RoPE Limitation

For the first modern-model implementation, Q/K are extracted before rotary
position embedding is applied. This is useful for offline sketch stress testing,
but it is not identical to post-RoPE attention-space Q/K.

Future work should compare pre-RoPE and post-RoPE extraction where possible,
likely by instrumenting model attention forward paths or using model-specific
hooks. Phase 3.5 intentionally avoids patching model internals.

## GQA/MQA Head Mapping

When query heads and KV heads differ, the script maps query heads to KV heads by
integer grouping:

```text
kv_head = query_head // (num_query_heads // num_key_value_heads)
```

For example, 8 query heads and 2 KV heads map query heads `0..3` to KV head `0`
and query heads `4..7` to KV head `1`.

Head sweep `--heads all` iterates over query heads. Multiple query heads may map
to the same KV head.

## Output Fields

HF eval and head sweep rows now include:

- `extraction_mode`
- `qk_space`
- `num_query_heads`
- `num_key_value_heads`
- `selected_query_head`
- `selected_kv_head`

Head sweep also records:

- `extraction_mode_requested`

## Suggested Future Commands

Qwen:

```bash
.venv/bin/python scripts/kivo_vd/run_hf_qk_sketch_eval.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --extraction-mode auto \
  --sketch-type count_sketch \
  --sketch-dim 64 \
  --topk-blocks 4 \
  --max-tokens 512
```

TinyLlama:

```bash
.venv/bin/python scripts/kivo_vd/run_hf_qk_head_sweep.py \
  --model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --extraction-mode auto \
  --sketch-types count_sketch,random_projection \
  --sketch-dims 64,128 \
  --layers 0,1 \
  --heads 0,1,2,3 \
  --max-tokens 512 \
  --include-ranked-blocks
```

Pipeline dry run with explicit extraction mode:

```bash
.venv/bin/python scripts/kivo_vd/run_offline_benchmark_pipeline.py \
  --dry-run \
  --extraction-mode separate_qk_proj
```

## Interpretation

Modern-model Q/K results should be treated as an offline signal only,
especially while using pre-RoPE projections. They do not prove runtime memory
reduction, latency improvement, quality preservation, or candidate-block
attention behavior.
