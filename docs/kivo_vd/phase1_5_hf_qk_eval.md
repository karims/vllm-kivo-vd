# Kivo-VD Phase 1.5: Optional HuggingFace Q/K Sketch Evaluation

Phase 1.5 adds an optional offline script that evaluates sketch quality on real
query/key tensors extracted from a small HuggingFace causal LM.

## What this phase adds

- `scripts/kivo_vd/run_hf_qk_sketch_eval.py`
  - Loads a GPT-2 style model (`sshleifer/tiny-gpt2`, `distilgpt2`, `gpt2`).
  - Extracts Q/K from a selected layer/head using GPT-2 fused `c_attn`.
  - Uses the final-token query against prior-token keys.
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

Try larger GPT-2 variants:

```bash
.venv/bin/python scripts/kivo_vd/run_hf_qk_sketch_eval.py \
  --model-name distilgpt2 \
  --layer 1 \
  --head 0 \
  --sketch-type count_sketch \
  --sketch-dim 128
```

## Output metrics

The script prints one JSON object with:

- model/setup: `model_name`, `prompt_num_tokens`, `layer`, `head`,
  `sketch_type`, `sketch_dim`, `block_size`, `topk_blocks`
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
