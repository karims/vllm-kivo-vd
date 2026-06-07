# Phase 10.3: Sketch-Based Real-Q/K/V Selectors

## Status

Phase 10.3 adds standalone selector diagnostics for real GPT-2 projected
Q/K/V tensors. It runs with HuggingFace and PyTorch outside vLLM.

This phase does not change vLLM scheduling, KV allocation, block tables, slot
mapping, attention metadata, or attention kernels. It does not implement
active routing.

## Why This Follows Phase 10.2

The Phase 10.2 sweep showed that oracle top-k block selection produced much
stronger selected-attention outputs than recent-only or random selection.
That result identified candidate selection as the current bottleneck.

Oracle top-k is not deployable because it selects blocks using full attention
probabilities. Phase 10.3 therefore compares oracle against selectors that use
projected Q/K tensors without using full attention probabilities.

## Selector Policies

### `query_key_block_score`

For the final query, this selector computes scaled Q/K dot products for each
token in each block. It reduces token/head scores with `max`, `mean`, or
`logsumexp`, then selects the highest-scoring blocks.

This is a useful Q/K baseline, but it is not yet a cheap runtime algorithm.

### `count_sketch`

The selector averages K vectors within each block, applies a deterministic
CountSketch to the block means and final query, then ranks blocks by sketch
dot product.

### `random_projection`

The selector applies a deterministic normalized Gaussian projection to the
block-mean K vectors and final query, then ranks projected dot products.

### `bidiagonal_sign_subsample`

This experimental selector applies deterministic sign flips, adjacent
bidiagonal mixing, and coordinate subsampling to block-mean K vectors and the
final query.

It remains a research variant and is not a default.

## Sketch Dimension

`--sketch-dim` controls the compressed vector width used by sketch policies.
The policy sweep accepts comma-separated dimensions through `--sketch-dims`.
Non-sketch policies run once with `sketch_dim: null`; sketch policies expand
over each requested dimension.

## Comparing With Oracle

The sweep reports:

- selected-attention output cosine similarity;
- relative L2 error;
- captured full-attention mass;
- oracle gaps for cosine, relative L2, and attention mass;
- summaries grouped by policy, sketch dimension, layer, and budget.

The "best deployable selector" excludes `oracle_topk`. It aggregates each
policy/sketch-dimension configuration, then ranks configurations first by
average cosine similarity and then by maximum relative L2 error. This is a
diagnostic ordering, not a production recommendation.

## Commands

Full RunPod sweep:

```bash
python scripts/kivo_vd/run_real_qkv_policy_sweep.py \
  --model gpt2 \
  --layers 0,3,5,8,11 \
  --budgets 4,8,16 \
  --block-sizes 16 \
  --policies recent,random,oracle_topk,query_key_block_score,count_sketch,random_projection,bidiagonal_sign_subsample \
  --sketch-dims 16,32,64 \
  --device cuda \
  --output-dir outputs/kivo_vd/runs/phase10_3_gpt2_sketch_selector_sweep
```

Faster sweep:

```bash
python scripts/kivo_vd/run_real_qkv_policy_sweep.py \
  --model gpt2 \
  --layers 0,5,11 \
  --budgets 4,8 \
  --block-sizes 16 \
  --policies oracle_topk,query_key_block_score,count_sketch,random_projection,bidiagonal_sign_subsample \
  --sketch-dims 16,32 \
  --device cuda \
  --output-dir outputs/kivo_vd/runs/phase10_3_gpt2_sketch_selector_sweep_fast
```

Single selector example:

```bash
python scripts/kivo_vd/run_real_qkv_selected_attention_eval.py \
  --model gpt2 \
  --selection-policy count_sketch \
  --sketch-dim 32 \
  --candidate-budget-blocks 8 \
  --device cuda
```

## Claims Boundary

- Q/K/V projections come from a real GPT-2-style model.
- All selector and selected-attention evaluation runs outside vLLM.
- No real vLLM KV cache is read or modified.
- No active routing is implemented.
- No logits or generation quality is evaluated.
- No measured runtime memory reduction is claimed.
- No latency improvement is claimed.

Strong standalone selector results would justify further quality evaluation.
They would not, by themselves, authorize vLLM attention integration.
