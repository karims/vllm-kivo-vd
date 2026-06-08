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
  --budgets 8,16,32,64 \
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
  --budgets 8,16 \
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

Stress testing at budgets `2` or `4` may still be useful for failure analysis,
but those budgets should not be presented as practical enterprise defaults.

## RunPod Result

The full Phase 10.3 GPT-2 sweep completed `195` standalone
HuggingFace/PyTorch real-Q/K/V runs with no failures. It covered layers
`0,3,5,8,11`, budgets `4,8,16`, block size `16`, seven selector policies, and
sketch dimensions `16,32,64`.

| policy | avg cosine | min cosine | avg relative L2 | max relative L2 | avg attention mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| `oracle_topk` | `0.987684` | `0.950780` | `0.132672` | `0.320883` | `0.914064` |
| `query_key_block_score` | `0.986247` | `0.944010` | `0.145825` | `0.336095` | `0.902847` |
| `count_sketch` | `0.957279` | `0.502508` | `0.318132` | `1.887915` | `0.782310` |
| `bidiagonal_sign_subsample` | `0.944366` | `0.454973` | `0.376663` | `2.094204` | `0.757161` |
| `random_projection` | `0.941393` | `0.454973` | `0.366456` | `2.094204` | `0.767317` |
| `recent` | `0.822501` | `0.454973` | `0.908913` | `2.088797` | `0.493023` |
| `random` | `0.618329` | `0.194487` | `1.196096` | `2.546401` | `0.179596` |

`query_key_block_score` was the best deployable selector and nearly matched
the oracle aggregate. CountSketch, random projection, and the structured
bidiagonal selector were promising on average but retained catastrophic
worst cases, especially at small budgets. They need improvement before any
runtime-use discussion.

The result supports adaptive, safe reduction rather than maximum compression.
Future practical sweeps should focus on budgets `8,16,32,64`; budgets `2` and
`4` remain failure-oriented stress tests.

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
