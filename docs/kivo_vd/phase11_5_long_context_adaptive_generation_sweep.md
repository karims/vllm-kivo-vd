# Phase 11.5: Long-Context Adaptive Generation Sweep

## Status

Phase 11.5 adds a standalone HuggingFace/PyTorch sweep for testing adaptive
multi-layer selected attention on longer GPT-2 contexts.

It does not use the vLLM runtime or KV cache. It does not implement active
routing, change attention kernels, or demonstrate measured memory or latency
improvement.

## Motivation

Phase 11.4 preserved greedy GPT-2 generation for the tested adaptive map:

```text
0:12,5:8,8:8,11:12
```

The fast five-prompt run was exact over 32 generated tokens for query-key and
oracle selection. The extended 12-prompt run was exact over both 32 and 64
generated tokens.

The selected-block ratio varied substantially with context:

- the default prompt set averaged about `0.394`, corresponding to roughly
  `0.606` theoretical inactive blocks in the patched layers;
- the extended prompt set averaged about `0.74`, corresponding to roughly
  `0.26` theoretical inactive blocks.

Those complements are estimates only. The standalone experiment does not
change KV allocation, and they are not measured runtime memory reductions.

Phase 11.5 asks whether quality remains clean when prompts contain many more
KV blocks and fixed budgets therefore select a smaller fraction.

## Prompt Sources

The script supports:

- `synthetic`: deterministic long prompts built from five semantic templates;
- `long_builtin`: the same controlled construction with a shifted template
  order for a second built-in sample;
- `file`: line-delimited source prompts expanded to target lengths.

The synthetic templates cover:

1. early-key retrieval;
2. system debugging;
3. code and documentation;
4. long explanatory context;
5. structured facts.

Prompts are assembled in token space. The prefix and final question are kept,
while filler tokens are repeated and trimmed to approach the requested target.
The maximum prompt size is:

```text
max_length - max(max_new_tokens_values)
```

Actual tokenizer lengths are recorded in every run row and in the saved prompt
artifact.

## Outputs

The output directory contains:

- `long_context_generation_runs.jsonl`
- `long_context_generation_summary.json`
- `long_context_generation_summary.md`
- `long_context_prompts.json` for an executed run

Each successful row records quality metrics, target and actual prompt lengths,
selected-block ratio, and:

```text
estimated_active_block_reduction_ratio = 1 - selected_block_ratio
```

The summary includes per-policy, target-length, map, and generation-length
tables; selected-ratio estimates; worst cases; oracle gaps; and the best
non-oracle configuration.

## Dry-Run Planning

Dry-run creates the planned matrix without loading or downloading a model:

```bash
.venv/bin/python \
  scripts/kivo_vd/run_long_context_adaptive_generation_sweep.py \
  --dry-run
```

## Fast RunPod Run

```bash
.venv/bin/python \
  scripts/kivo_vd/run_long_context_adaptive_generation_sweep.py \
  --model gpt2 \
  --target-token-lengths 768 \
  --num-prompts-per-length 2 \
  --layer-budget-maps "0:12,5:8,8:8,11:12" \
  --policies query_key_block_score,oracle_topk \
  --max-new-tokens-values 32 \
  --device cuda \
  --output-dir \
  outputs/kivo_vd/runs/phase11_5_gpt2_long_context_fast
```

## Full RunPod Run

```bash
.venv/bin/python \
  scripts/kivo_vd/run_long_context_adaptive_generation_sweep.py \
  --model gpt2 \
  --target-token-lengths 768,896 \
  --num-prompts-per-length 3 \
  --layer-budget-maps "0:12,5:8,8:8,11:12" \
  --policies query_key_block_score,oracle_topk \
  --max-new-tokens-values 32 \
  --device cuda \
  --output-dir \
  outputs/kivo_vd/runs/phase11_5_gpt2_long_context_full
```

GPT-2 has a 1024-token context limit. Targets 768 and 896 leave room for 32
generated tokens and avoid operating directly at the boundary.

## Optional Safer-Map Comparison

```bash
.venv/bin/python \
  scripts/kivo_vd/run_long_context_adaptive_generation_sweep.py \
  --model gpt2 \
  --target-token-lengths 896 \
  --num-prompts-per-length 2 \
  --layer-budget-maps \
  "0:12,5:8,8:8,11:12;0:16,5:8,8:8,11:12;0:16,5:8,8:8,11:16" \
  --policies query_key_block_score,oracle_topk \
  --max-new-tokens-values 32 \
  --device cuda \
  --output-dir \
  outputs/kivo_vd/runs/phase11_5_gpt2_long_context_map_comparison
```

## Failure Flags

The report emits:

- `exact_sequence_match_below_1`
- `token_match_below_0.99`
- `normalized_edit_distance_above_0`
- `average_kl_above_0.01`
- `per_step_top1_below_1`
- `selected_ratio_above_0.85`
- `actual_prompt_length_too_short`

The prompt-length flag is raised when any actual prompt is below 95 percent of
its requested target.

## Readiness

`phase11_6_ready` is true only when all successful
`query_key_block_score` rows pass the strict thresholds. `phase12_ready`
remains false by design.

A clean result recommends broader prompt coverage or a larger model before any
vLLM work. It does not authorize active routing or runtime integration.

## Interpretation

The most useful comparison is quality against selected ratio as context grows.
If oracle remains clean while query-key selection diverges, selector quality
is the likely bottleneck. If both diverge, the budget or layer map is too
aggressive for the tested context.

## Caveats

- Evaluation runs outside vLLM.
- No vLLM integration is implemented.
- Generation uses greedy decoding only.
- Prompts are synthetic unless a file is provided.
- No active routing is implemented.
- No measured runtime memory reduction is claimed.
- No latency claim is made.
- This is a generation-quality probe, not a preservation claim.
- GPT-2's context limit applies.
