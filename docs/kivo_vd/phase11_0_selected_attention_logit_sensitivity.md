# Phase 11.0: Selected-Attention Logit Sensitivity

## Status

Phase 11.0 begins logits-level evaluation outside vLLM. It uses a real
GPT-2-style HuggingFace model and patches one transformer layer's last-token
attention contribution before continuing the remaining model computation.

This is not vLLM integration and is not a full generation-quality result.

## Purpose

Phase 10 showed that selected attention can preserve attention outputs when
the correct blocks are chosen. `query_key_block_score` nearly matched the
oracle upper bound in the GPT-2 real-Q/K/V sweep.

Phase 11.0 asks the next question: how much does a selected-attention error at
one layer change the final next-token logits?

## Patch Procedure

For each prompt, the script:

1. runs the unmodified model and saves baseline next-token logits;
2. reaches the selected GPT-2 layer;
3. applies `ln_1` and the fused `c_attn` projection;
4. computes real Q/K/V for the layer;
5. selects KV blocks for the final query;
6. computes selected attention for the final token;
7. applies the normal GPT-2 attention output projection `c_proj`;
8. applies the layer residual and MLP to the patched final token;
9. continues through all remaining transformer layers;
10. compares patched and baseline next-token logits.

All other tokens at the selected layer use the normal block output.

## Supported Selectors

- `query_key_block_score` (default);
- `oracle_topk`;
- `recent`;
- `count_sketch`;
- `random_projection`;
- `bidiagonal_sign_subsample`.

Oracle remains an undeployable upper-bound diagnostic. Practical initial
experiments should focus on `query_key_block_score` with budgets `8` and `16`.

## Metrics

Per prompt:

- selected block IDs and ratio;
- attention-output cosine similarity and relative L2 error;
- final-logit cosine similarity and relative L2 error;
- KL divergence between next-token distributions;
- top-1 token match;
- top-5 and top-10 overlap;
- baseline and patched top tokens and probabilities;
- probability change for the baseline top token.

Aggregate results include average logit and attention metrics, KL divergence,
top-1 match rate, and average top-k overlap.

## Example Commands

Default `query_key_block_score` run:

```bash
.venv/bin/python scripts/kivo_vd/run_selected_attention_logit_sensitivity.py \
  --model gpt2 \
  --layer-idx 0 \
  --candidate-budget-blocks 16 \
  --selection-policy query_key_block_score \
  --device cuda
```

Compare the oracle upper bound:

```bash
.venv/bin/python scripts/kivo_vd/run_selected_attention_logit_sensitivity.py \
  --model gpt2 \
  --layer-idx 0 \
  --candidate-budget-blocks 16 \
  --selection-policy oracle_topk \
  --device cuda \
  --output-json outputs/kivo_vd/phase11_0_oracle_logits.json \
  --output-md outputs/kivo_vd/phase11_0_oracle_logits.md
```

## Interpretation

- High top-1 agreement and low KL for `query_key_block_score` support broader
  standalone quality tests.
- A stable oracle with an unstable deployable selector means selection is
  still the bottleneck.
- An unstable oracle means selected attention may be risky at that layer and
  budget even under strong selection.

No single threshold proves generation quality. Layer, prompt, context-length,
and budget sweeps remain necessary.

## Caveats

- Evaluation is standalone HuggingFace/PyTorch outside vLLM.
- No vLLM integration is implemented or authorized.
- Only one layer and the final token's attention contribution are patched.
- No real vLLM KV cache is used.
- No active routing is implemented.
- No measured runtime memory reduction is claimed.
- No latency improvement is claimed.
- Full autoregressive generation quality is not measured.

## Next Step

If Phase 11.0 remains stable across practical budgets, the next experiment
should sweep layers and budgets before moving to controlled generation-level
comparisons. vLLM attention integration remains out of scope.

## RunPod Result

The initial GPT-2 RunPod check tested layers `0,5,8,11`, budget `16`, block
size `16`, five prompts, and both `query_key_block_score` and `oracle_topk`.

Top-1 next-token match was `1.0` for every layer and policy. Top-5 overlap was
`5.0` throughout, while top-10 overlap was generally `10.0` and was `9.8` for
both policies at layer 11. Logit cosine similarity was effectively `1.0`, and
KL divergence remained very low.

This is a logits-level green signal for broader offline testing. It is not a
generation-quality result and does not authorize vLLM integration.
