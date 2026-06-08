# Phase 11.4: Adaptive Multi-Layer Generation Sweep

## Status

Phase 11.4 adds a reproducible offline sweep across adaptive layer budgets,
selection policies, generation lengths, and prompt sets. It builds on the
Phase 11.3 multi-layer GPT-2 generation probe.

This experiment runs outside vLLM. It does not implement active routing,
change vLLM attention, or demonstrate runtime memory or latency improvement.

## Purpose

A single clean generation run is useful but narrow. The sweep asks whether an
adaptive map remains stable across:

- `query_key_block_score` and the diagnostic `oracle_topk` upper bound;
- one or more layer-budget maps;
- generation lengths such as 32 and 64 tokens;
- the original five retrieval/system prompts and a larger mixed prompt set.

The default adaptive map is:

```text
0:12,5:8,8:8,11:12
```

Oracle results are not deployable. Their purpose is to separate a selector
failure from a budget that is intrinsically too aggressive.

## Outputs

The output directory contains:

- `adaptive_multilayer_generation_runs.jsonl`
- `adaptive_multilayer_generation_summary.json`
- `adaptive_multilayer_generation_summary.md`

Each run records exact sequence match, token match, prefix length, normalized
edit distance, per-step KL divergence, per-step top-1 match, and the selected
block ratio across patched layers.

The summary includes:

- per-policy, per-map, and per-generation-length tables;
- policy/map/generation-length combinations;
- worst cases under each quality metric;
- non-oracle versus oracle gaps;
- the best non-oracle configuration;
- conservative Phase 11.5 and Phase 12 readiness fields.

`phase12_ready` is always false in this phase. A clean query-key sweep can
suggest Phase 11.5, but it does not authorize vLLM integration.

## Dry-Run Planning

Dry-run mode builds the complete planned matrix and writes all three artifacts
without loading or downloading a model:

```bash
.venv/bin/python \
  scripts/kivo_vd/run_adaptive_multilayer_generation_sweep.py \
  --dry-run
```

## Fast Run

```bash
.venv/bin/python \
  scripts/kivo_vd/run_adaptive_multilayer_generation_sweep.py \
  --model gpt2 \
  --layer-budget-maps "0:12,5:8,8:8,11:12" \
  --policies query_key_block_score,oracle_topk \
  --max-new-tokens-values 32 \
  --prompt-set default \
  --device cuda \
  --output-dir \
  outputs/kivo_vd/runs/phase11_4_gpt2_adaptive_multilayer_sweep_fast
```

## Full Run

```bash
.venv/bin/python \
  scripts/kivo_vd/run_adaptive_multilayer_generation_sweep.py \
  --model gpt2 \
  --layer-budget-maps "0:12,5:8,8:8,11:12" \
  --policies query_key_block_score,oracle_topk \
  --max-new-tokens-values 32,64 \
  --prompt-set extended \
  --device cuda \
  --output-dir \
  outputs/kivo_vd/runs/phase11_4_gpt2_adaptive_multilayer_sweep_full
```

The extended set has 12 prompts. It retains the five Phase 11 prompts and adds
list continuation, reasoning, API design, documentation, mathematical
explanation, systems diagnosis, and code-oriented prompts.

## Safer-Map Comparison

```bash
.venv/bin/python \
  scripts/kivo_vd/run_adaptive_multilayer_generation_sweep.py \
  --model gpt2 \
  --layer-budget-maps \
  "0:12,5:8,8:8,11:12;0:16,5:8,8:8,11:12;0:16,5:8,8:8,11:16" \
  --policies query_key_block_score,oracle_topk \
  --max-new-tokens-values 32 \
  --prompt-set default \
  --device cuda \
  --output-dir \
  outputs/kivo_vd/runs/phase11_4_gpt2_adaptive_map_comparison
```

## Failure Flags

The sweep marks a successful run when any of these research thresholds fail:

- `exact_sequence_match_below_1`
- `token_match_below_0.99`
- `normalized_edit_distance_above_0`
- `average_kl_above_0.01`
- `per_step_top1_below_1`

These intentionally strict thresholds make regressions visible. They are not
a claim that a future production policy must satisfy exact equality.

## Interpretation

A useful result has stable non-oracle rows, small oracle gaps, and no hidden
worst-case prompt. If oracle passes while query-key selection fails, selector
quality remains the likely bottleneck. If both fail, the layer budget is
probably too aggressive for that prompt or generation length.

Even a clean GPT-2 sweep should be expanded to more prompts and a larger model
before any vLLM work is considered.

## Caveats

- The evaluator runs outside vLLM.
- No vLLM integration is implemented.
- Generation uses greedy decoding only.
- No active routing is implemented.
- No measured runtime memory reduction is claimed.
- No latency claim is made.
- This is a generation-quality probe, not a preservation claim.
- Results apply to GPT-2 unless another model is explicitly configured.
