# Kivo-VD Phase 10.1: Real-QKV Selected-Attention Evaluation

Phase 10.0 compared full attention with selected-KV attention on synthetic
Q/K/V tensors. Phase 10.1 is the first real-model correctness signal: it
extracts real GPT-2 Q/K/V projections and compares full versus selected
attention output outside vLLM.

This still does not run generation through selected attention. It does not
evaluate logits, does not access the real vLLM KV cache, and does not modify
vLLM runtime behavior.

## What The Script Does

`scripts/kivo_vd/run_real_qkv_selected_attention_eval.py`:

1. Loads a HuggingFace GPT-2-style causal LM and tokenizer.
2. Tokenizes one prompt or one prompt per line from `--prompts-file`.
3. Runs the model with hidden states enabled.
4. Selects the input to a chosen GPT-2 transformer block.
5. Applies the block's `ln_1`, then GPT-2's fused `attn.c_attn` projection.
6. Splits fused Q/K/V and reshapes to:

```text
Q: [batch, heads, tokens, head dim]
K: [batch, heads, tokens, head dim]
V: [batch, heads, tokens, head dim]
```

7. Uses the last query token by default.
8. Computes full scaled dot-product attention over all causal K/V tokens.
9. Selects KV blocks, gathers those blocks, and computes selected attention.
10. Reports output similarity and attention-mass coverage.

The extracted Q/K/V space is `gpt2_projection_after_ln_1`, which matches the
input convention to GPT-2's attention projection. This script currently
targets GPT-2-style fused `c_attn` models. Modern separate q/k/v projection
models should be added later as a separate extension.

## Selection Policies

- `recent`: select the final `candidate_budget_blocks` blocks.
- `first`: select the earliest blocks.
- `random`: select deterministic random blocks from `--seed`.
- `oracle_topk`: select blocks with the largest full-attention mass.

`oracle_topk` is an upper-bound diagnostic and is not deployable because it
uses full attention probabilities to choose blocks. If oracle top-k performs
poorly, selected attention is risky even under best-case block selection. If
oracle top-k performs well but recent/random/first are weak, candidate
selection is the likely bottleneck.

Explicit `--selected-blocks` overrides policy selection and is useful for
debugging known block sets.

## Metrics

Per prompt, the report includes:

- token length;
- layer index;
- block count;
- selected block IDs;
- selected block and token ratios;
- attention mass captured by selected blocks;
- cosine similarity between full and selected attention outputs;
- relative L2 error;
- mean and maximum absolute error;
- full and selected output norms.

Aggregate metrics include the number of prompts, average cosine similarity,
average relative L2 error, average attention mass captured, minimum cosine
similarity, and maximum relative L2 error.

## Recent Policy Example

```bash
.venv/bin/python scripts/kivo_vd/run_real_qkv_selected_attention_eval.py \
  --model gpt2 \
  --selection-policy recent \
  --candidate-budget-blocks 16 \
  --block-size 16 \
  --layer-idx 0 \
  --device cuda \
  --output-json outputs/kivo_vd/phase10_1_gpt2_recent_real_qkv_eval.json \
  --output-md outputs/kivo_vd/phase10_1_gpt2_recent_real_qkv_eval.md
```

## Oracle Policy Example

```bash
.venv/bin/python scripts/kivo_vd/run_real_qkv_selected_attention_eval.py \
  --model gpt2 \
  --selection-policy oracle_topk \
  --candidate-budget-blocks 16 \
  --block-size 16 \
  --layer-idx 0 \
  --device cuda \
  --output-json outputs/kivo_vd/phase10_1_gpt2_oracle_real_qkv_eval.json \
  --output-md outputs/kivo_vd/phase10_1_gpt2_oracle_real_qkv_eval.md
```

## Prompt Files

To evaluate multiple prompts:

```bash
.venv/bin/python scripts/kivo_vd/run_real_qkv_selected_attention_eval.py \
  --model gpt2 \
  --prompts-file prompts.txt \
  --selection-policy oracle_topk \
  --candidate-budget-blocks 16 \
  --block-size 16 \
  --device cuda
```

`prompts.txt` should contain one prompt per non-empty line.

## Interpreting Results

High cosine similarity and low relative L2 error indicate that selected
attention output is close to full attention for the extracted last-token Q/K/V
vectors. High attention mass captured indicates the selected blocks cover much
of the full attention distribution.

Useful comparisons:

- Run `oracle_topk` first to estimate an upper-bound signal.
- Compare `recent`, `first`, and `random` against oracle.
- Increase or decrease `candidate_budget_blocks` to find sensitivity.
- Repeat across layers before drawing model-level conclusions.

This is not generation quality. It does not compare logits, sampled text, or
benchmark answers.

## Caveats

- Q/K/V projections come from a real GPT-2-style model.
- Evaluation runs outside vLLM.
- Evaluation runs outside production attention kernels.
- The real vLLM KV cache is not used.
- No block tables or slot mappings are mutated.
- No active routing is implemented.
- No measured runtime memory reduction is claimed.
- No latency improvement is claimed.
- No generation quality or quality preservation is measured.

## Next Steps

If oracle top-k selected attention is close to full attention across prompts
and layers, the next standalone step is to compare candidate policies and
possibly add modern-model Q/K/V extraction. If oracle top-k is weak, Kivo-VD
should revisit candidate budgets or the selected-attention approximation
before any vLLM-adjacent prototype.
