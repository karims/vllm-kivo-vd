# Phase 10.4: Practical-Budget Decision Gate

## Purpose

Phase 10.4 converts the Phase 10.3 real-Q/K/V selector sweep into a
conservative gate before Phase 11 quality evaluation.

The gate authorizes only standalone logits and generation-quality experiments
outside vLLM. It does not authorize vLLM attention integration, active KV
routing, block-table mutation, or runtime memory claims.

## Phase 10.3 Decision

The `195`-run GPT-2 sweep completed without failures. Oracle top-k remained
the upper bound, while `query_key_block_score` was the strongest selector
that did not use full attention probabilities.

| selector | avg cosine | min cosine | max relative L2 | avg mass |
| --- | ---: | ---: | ---: | ---: |
| `oracle_topk` | `0.987684` | `0.950780` | `0.320883` | `0.914064` |
| `query_key_block_score` | `0.986247` | `0.944010` | `0.336095` | `0.902847` |

The direct Q/K selector clears the Phase 10.4 thresholds:

- average cosine at least `0.95`;
- minimum cosine at least `0.90`;
- maximum relative L2 no more than `0.50`.

Sketch selectors remain research candidates. Their average metrics were
promising, but their worst cases are not safe enough for runtime use as-is.

## Practical Budget Policy

| candidate budget | interpretation |
| ---: | --- |
| `2` | experimental failure analysis only |
| `4` | aggressive stress testing only |
| `8` | minimum practical candidate |
| `16` | safer baseline |
| `32`, `64` | long-context enterprise-oriented candidates |

The goal is adaptive, reliable active-set reduction. It is not maximum
compression. In long-context serving, a dependable `15-25%` active-KV
reduction could still be useful, but this phase does not measure or claim such
a runtime reduction.

## Readiness Helper

```bash
.venv/bin/python scripts/kivo_vd/check_phase10_readiness.py \
  --summary outputs/kivo_vd/runs/phase10_3_gpt2_sketch_selector_sweep/policy_sweep_summary.json \
  --output-json outputs/kivo_vd/phase10_readiness.json \
  --output-md outputs/kivo_vd/phase10_readiness.md \
  --min-practical-budget 8 \
  --recommended-budgets 8,16,32,64
```

The helper requires:

- a successful sweep;
- an `oracle_topk` summary;
- a non-oracle best deployable selector;
- deployable average cosine at least `0.95`;
- deployable minimum cosine at least `0.90`;
- deployable maximum relative L2 at most `0.50`.

It warns when all observed budgets are below the practical minimum.

## Phase 11 Allowed Scope

If the gate passes, Phase 11 may evaluate:

- logits differences under selected attention;
- greedy or controlled generation comparisons;
- `query_key_block_score` at practical budgets;
- failure behavior across layers, prompts, and context lengths.

All of that remains standalone HuggingFace/PyTorch work outside vLLM.

## Not Authorized

- vLLM attention integration;
- scheduler or GPUModelRunner changes;
- attention-kernel changes;
- block-table or slot-mapping mutation;
- active KV routing;
- measured runtime memory-reduction claims;
- latency claims;
- claims of generation-quality preservation before Phase 11 evidence exists.
