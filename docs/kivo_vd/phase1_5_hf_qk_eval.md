# Kivo-VD Phase 1.5: Optional HuggingFace Q/K Sketch Evaluation

Phase 1.5 adds an optional offline script that evaluates sketch quality on real
query/key tensors extracted from a small HuggingFace causal LM.

## What this phase adds

- `scripts/kivo_vd/run_hf_qk_sketch_eval.py`
  - Loads a GPT-2 style model (`sshleifer/tiny-gpt2`, `distilgpt2`, `gpt2`).
  - Extracts Q/K from a selected layer/head using GPT-2 fused `c_attn`.
  - Supports configurable query position (not only final token).
  - Uses only causal keys before the selected query position.
  - Runs exact and sketched score comparisons using existing NumPy sketch math.
  - Prints compact JSON metrics to stdout.

## Important scope limits

- Optional and offline only.
- Requires `torch` + `transformers` only for this script.
- Does not modify vLLM runtime behavior.
- No scheduler/GPUModelRunner/attention-kernel changes.
- No CUDA/Triton requirements for CPU runs.
- Core Kivo modules remain NumPy-only and do not import `torch`.

## Example commands

Start with a tiny model:

```bash
.venv/bin/python scripts/kivo_vd/run_hf_qk_sketch_eval.py \
  --model-name sshleifer/tiny-gpt2 \
  --sketch-type random_projection \
  --sketch-dim 64 \
  --topk-blocks 4 \
  --device cpu
```

For long prompts on GPT-2 style models (context often 1024), cap tokens:

```bash
.venv/bin/python scripts/kivo_vd/run_hf_qk_sketch_eval.py \
  --model-name distilgpt2 \
  --max-tokens 1024 \
  --truncate-side right
```

Try larger GPT-2 variants:

```bash
.venv/bin/python scripts/kivo_vd/run_hf_qk_sketch_eval.py \
  --model-name distilgpt2 \
  --layer 1 \
  --head 0 \
  --sketch-type count_sketch \
  --sketch-dim 128
```

Evaluate an earlier query position (not last token):

```bash
.venv/bin/python scripts/kivo_vd/run_hf_qk_sketch_eval.py \
  --model-name distilgpt2 \
  --query-position 100
```

Sweep positions in one run (25%, 50%, 75%, last):

```bash
.venv/bin/python scripts/kivo_vd/run_hf_qk_sketch_eval.py \
  --model-name sshleifer/tiny-gpt2 \
  --sweep-query-positions
```

## Output metrics

The script prints one JSON object with:

- model/setup: `model_name`, `prompt_num_tokens`, `layer`, `head`,
  `sketch_type`, `sketch_dim`, `block_size`, `topk_blocks`
- prompt/context safety: `original_prompt_num_tokens`, `prompt_num_tokens`,
  `truncated`, `max_context_tokens`
- query selection: `query_position`, `num_keys_used`
- retrieval metrics: `token_topk_recall`, `block_topk_recall`,
  `block_recall_at_2x_budget`, `block_recall_at_4x_budget`, `block_mrr`
- score diagnostics: `token_score_correlation`, `block_score_correlation`
- rankings: `exact_top_block_ids`, `approx_top_block_ids`

## Why this exists

Structured synthetic experiments were useful, but this phase checks whether real
transformer Q/K tensors behave closer to:

- random Gaussian baseline (hard for sketches), or
- structured regimes where candidate-budget retrieval is strong.

This helps validate the Kivo-VD candidate-retrieval + exact-rerank direction
before any runtime integration.

## Query-position behavior

Final-token queries often emphasize recency, especially in causal models. That
can overstate how much the metric reflects long-range retrieval quality.

This script supports:
- `--query-position last` (default, backward compatible)
- `--query-position <int>` (absolute token index)
- `--query-position <negative int>` (Python-style from end, for example `-1`)
- `--sweep-query-positions` (25%, 50%, 75%, and last token; one JSON line each)

`num_keys_used` indicates the causal key prefix size for the selected query
position (future tokens are excluded).

## Context limit behavior

GPT-2 family models can fail on overlong prompts if tokenized length exceeds
the model context window (for example 1024 for many GPT-2 variants).

The script now handles this defensively:
- Determines context limit from `model.config.n_positions` when available.
- Otherwise uses `tokenizer.model_max_length` only when it looks reasonable.
- Otherwise does not auto-truncate unless `--max-tokens` is set.
- If truncation is needed, default is `--truncate-side right` (keep prompt
  prefix). Use `--truncate-side left` to keep prompt suffix instead.
