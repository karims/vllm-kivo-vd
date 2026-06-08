# Phase 11.5: Long-Context Adaptive Generation Sweep

## Status

Phase 11.5 adds a standalone HuggingFace/PyTorch sweep for testing adaptive
multi-layer selected attention on longer GPT-2 contexts.

It does not use the vLLM runtime or KV cache. It does not implement active
routing, change attention kernels, or demonstrate measured memory or latency
improvement.

## Motivation

Phase 11.4 produced exact baseline sequence matches for the tested adaptive
map:

```text
0:12,5:8,8:8,11:12
```

The fast five-prompt run matched the baseline over 32 generated tokens for
query-key and oracle selection. The extended 12-prompt run matched over both
32 and 64 generated tokens.

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

## RunPod Results

### Fixed Short-Context Map Fails At Long Context

The first RunPod test used the Phase 11.4 map:

```text
0:12,5:8,8:8,11:12
```

Two target-768 prompts tokenized to about 734 tokens. The selected ratio was
about `0.211399`, corresponding to a theoretical complement of `0.788601`.

| policy | exact | token match | edit distance | average KL | step top-1 |
| --- | --- | --- | --- | --- | --- |
| `query_key_block_score` | `0.0` | `0.125` | `0.796875` | `11.861486` | `0.125` |
| `oracle_topk` | `0.5` | `0.5` | `0.343750` | `8.179362` | `0.5` |

Oracle also failed, so this was primarily a budget/risk failure rather than
only a selector failure. A selected ratio near `0.21` was too aggressive.

The full target-768/896 sweep confirmed this result. Across four successful
experiment executions, actual prompt lengths were about 733 and 855 tokens.
The average selected ratio was about `0.197`, but generation quality failed:

| policy | exact | token match | edit distance | average KL |
| --- | --- | --- | --- | --- |
| `query_key_block_score` | `0.0` | `0.239583` | `0.734375` | `9.747772` |
| `oracle_topk` | `0.333333` | `0.489583` | `0.406250` | `7.579497` |

Fixed budgets of 8 or 12 blocks should therefore be treated as long-context
stress tests, not practical defaults.

### 768-Target Safe-Boundary Comparison

At about 734 actual prompt tokens, four safer maps were tested:

- `0:16,5:16,8:16,11:16`
- `0:24,5:16,8:16,11:24`
- `0:24,5:24,8:24,11:24`
- `0:32,5:24,8:24,11:32`

Oracle matched the baseline exactly for every map. Query-key selection failed
for the first three maps with exact match `0.5`, token match `0.5`, edit
distance `0.343750`, and KL between about `7.25` and `8.48`.

Query-key selection passed for:

```text
0:32,5:24,8:24,11:32
```

Its exact and token match rates were `1.0`, edit distance was `0.0`, average
KL was `0.001076`, and per-step top-1 match was `1.0`. The selected ratio was
`0.591917`, corresponding to a theoretical active-block reduction estimate of
`0.408083`.

The oracle/query-key difference on smaller maps shows that selector margin
matters even after the budget becomes large enough for oracle selection.

### Near GPT-2 Context Limit

A second comparison targeted 960 tokens, producing prompts around 917 tokens,
with 16 generated tokens. All tested maps passed exact greedy generation for
both query-key and oracle selection:

| map | query-key KL | oracle KL | selected ratio | estimated reduction |
| --- | --- | --- | --- | --- |
| `0:32,5:24,8:24,11:32` | `0.002257` | `0.000334` | `0.480713` | `0.519287` |
| `0:40,5:32,8:32,11:40` | `0.000684` | `0.000102` | `0.618060` | `0.381940` |
| `0:48,5:40,8:40,11:48` | `0.000122` | `0.000034` | `0.755406` | `0.244594` |

The largest map is the safest by KL. The first map is the more interesting
quality/savings tradeoff because it matched the baseline in the tested
continuations while retaining a theoretical complement of about `51.9%`.

### Context-Scaled Budget Interpretation

The evidence supports a context-scaled rather than globally fixed policy:

- sensitive layers 0 and 11 require higher budgets;
- middle layers 5 and 8 can use lower budgets;
- budgets should grow with context block count;
- selected ratios around `0.48-0.60` were safe in these GPT-2 tests;
- selected ratios around `0.18-0.21` were unsafe.

These observations are specific to the tested GPT-2 prompts and standalone
patching implementation. They are not production thresholds.

## Long-Context Readiness Helper

The readiness helper accepts one or more Phase 11.5 summary JSON or run JSONL
artifacts:

```bash
.venv/bin/python \
  scripts/kivo_vd/check_phase11_long_context_readiness.py \
  --inputs \
  outputs/kivo_vd/runs/phase11_5_gpt2_long_context_full/long_context_generation_summary.json \
  outputs/kivo_vd/runs/phase11_5_gpt2_long_context_map_comparison/long_context_generation_summary.json
```

It reports the lowest-KL passing configuration separately from the best
quality/savings tradeoff under the selected-ratio ceiling.

## Recommended Next Phase

Phase 11.6 should move from fixed layer-budget maps to ratio/context-scaled
maps, or test a longer-context small model. GPT-2 is near its context limit;
further enterprise-relevant testing requires a model with 2K-8K+ context.

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
