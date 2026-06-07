# Kivo-VD Phase 10.0: Selected-Attention Equivalence

Phase 9 validated complete selected block-ID export and synthetic selected-KV
materialization outside attention. Phase 10 moves from memory accounting to a
correctness-style question:

```text
Does selected-KV attention approximate full attention output?
```

Phase 10.0 answers this only for synthetic PyTorch tensors outside vLLM. It
does not use the real vLLM KV cache, does not call vLLM attention backends, and
does not change runtime behavior.

## What The Prototype Does

`scripts/kivo_vd/run_selected_attention_equivalence.py` creates synthetic
Q/K/V tensors, partitions K/V tokens into fixed-size blocks, computes full
scaled dot-product attention over all K/V tokens, gathers a selected subset of
K/V blocks, and computes selected attention over only those selected tokens.

It then compares the selected output with the full output using:

- cosine similarity;
- relative L2 error;
- maximum absolute error;
- mean absolute error;
- selected token and block ratios;
- attention mass captured by the selected blocks;
- full and selected output norms.

The tensor layout is:

```text
Q: [batch, query heads, query length, head dim]
K: [batch, KV heads, KV tokens, head dim]
V: [batch, KV heads, KV tokens, head dim]
```

When query heads outnumber KV heads, the script expands KV heads with the same
integer grouping used for conservative GQA/MQA analysis. This remains a
standalone reference experiment, not a model-specific implementation.

## Selection Policies

The script supports:

- `recent`: select the last `candidate_budget_blocks` blocks;
- `first`: select the first `candidate_budget_blocks` blocks;
- `random`: select deterministic random blocks from `--seed`;
- `oracle_topk`: select blocks with the largest full-attention mass.

`oracle_topk` is not deployable because it uses full attention weights to pick
blocks. It is an upper-bound diagnostic. If oracle top-k is weak, selected
attention itself or the candidate budget may be risky. If oracle top-k is
strong but recent/random/first are weak, the selection policy is likely the
problem.

Explicit `--selected-blocks` overrides the policy and is useful for debugging
known block sets.

## CPU Example

```bash
.venv/bin/python scripts/kivo_vd/run_selected_attention_equivalence.py \
  --selection-policy recent \
  --candidate-budget-blocks 16 \
  --num-blocks 64 \
  --block-size 16 \
  --head-dim 64 \
  --device cpu \
  --output-json outputs/kivo_vd/phase10_0_recent_selected_attention.json \
  --output-md outputs/kivo_vd/phase10_0_recent_selected_attention.md
```

## CUDA Example

```bash
.venv/bin/python scripts/kivo_vd/run_selected_attention_equivalence.py \
  --selection-policy oracle_topk \
  --candidate-budget-blocks 16 \
  --num-blocks 64 \
  --block-size 16 \
  --head-dim 64 \
  --device cuda \
  --output-json outputs/kivo_vd/phase10_0_oracle_selected_attention.json \
  --output-md outputs/kivo_vd/phase10_0_oracle_selected_attention.md
```

## Interpreting Metrics

High cosine similarity and low relative L2 error mean the selected attention
output is close to full attention for that synthetic tensor draw. High
attention mass captured means the selected blocks cover much of the full
attention distribution.

Useful comparisons:

- Run `oracle_topk` first to establish an upper-bound sanity signal.
- Compare `recent`, `first`, and `random` against oracle to separate policy
  quality from selected-attention feasibility.
- Vary `candidate_budget_blocks` to see how much budget is needed before
  selected output approaches full output.
- Vary `num_query_heads`, `num_kv_heads`, and `head_dim` to stress GQA/MQA-like
  layouts.

These are synthetic reference signals only. They do not prove real model
quality, runtime latency, or memory reduction.

## Output Artifacts

The JSON report contains:

- `config`;
- `selected_block_ids`;
- selected block and token ratios;
- full and selected output shapes;
- metrics;
- caveats.

The Markdown report contains the same information in tables plus an
interpretation section.

## Caveats

- Q/K/V tensors are synthetic.
- The experiment runs outside vLLM.
- The experiment runs outside production attention kernels.
- The real vLLM KV cache is not accessed.
- No block tables or slot mappings are mutated.
- No active routing is implemented.
- No measured runtime memory reduction is claimed.
- No latency improvement is claimed.
- No real model quality or quality preservation is measured.

## Next Steps

If oracle-selected attention is consistently close to full attention, the next
standalone step is to compare selected and full outputs across more synthetic
regimes and candidate budgets. If oracle is weak, the selected-KV approximation
needs more scrutiny before any vLLM-adjacent prototype.

Only after standalone correctness signals are understood should Kivo-VD
consider isolated, non-production vLLM-adjacent experiments.
