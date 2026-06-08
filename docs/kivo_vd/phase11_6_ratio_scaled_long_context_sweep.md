# Phase 11.6: Ratio-Scaled Long-Context Sweep

## Status

Phase 11.6 adds a standalone HuggingFace/PyTorch sweep that generates
layer-budget maps from context block count and layer-specific ratios.

This remains outside vLLM. It does not use the vLLM KV cache, modify block
tables, change attention kernels, implement active routing, or demonstrate
measured memory or latency improvement.

## Motivation

Phase 11.5 showed that fixed short-context budgets do not transfer to longer
GPT-2 contexts. The map `0:12,5:8,8:8,11:12` matched short-context greedy
generation but failed badly around 734-855 prompt tokens. Oracle also failed
at those aggressive budgets, so the failure was a budget/risk issue, not only
selector quality.

Phase 11.5 also showed that `0:32,5:24,8:24,11:32` matched the tested baseline
continuations around 734 and 917 prompt tokens. The near-limit test had a
selected ratio around `0.4807`, whose complement is a theoretical active-block
reduction estimate around `51.9%`.

The emerging rule is context-scaled:

- sensitive layers 0 and 11 need higher budgets;
- middle layers 5 and 8 can use lower budgets;
- selected ratios around `0.18-0.21` were unsafe;
- selected ratios around `0.48-0.60` were safe in the tested GPT-2 runs.

Phase 11.6 turns that rule into a repeatable ratio sweep.

## Ratio Policies

Ratio policies use:

```text
name=layer:ratio,layer:ratio
```

The default policies are:

```text
balanced=0:0.60,5:0.45,8:0.45,11:0.60
safer=0:0.70,5:0.55,8:0.55,11:0.70
aggressive=0:0.50,5:0.40,8:0.40,11:0.50
```

For each target length, the script builds prompts, measures average actual
token length, estimates context blocks as:

```text
ceil(average_actual_prompt_tokens / block_size)
```

Then each layer budget is:

```text
rounding(num_blocks * ratio)
```

Budgets are clamped by `--min-budget`, optional `--max-budget`, and
`num_blocks`.

## Outputs

The output directory contains:

- `ratio_scaled_generation_runs.jsonl`
- `ratio_scaled_generation_summary.json`
- `ratio_scaled_generation_summary.md`
- `ratio_scaled_prompts.json` for executed runs

Each run records the ratio policy, derived map, target length, average actual
prompt tokens, estimated context blocks, quality metrics, selected-block
ratio, estimated active-block reduction ratio, and failure flags.

The summary includes derived maps, per-ratio-policy tables, per-model-policy
tables, target-length and generation-length tables, oracle gaps, best
deployable tradeoff, and safest passing deployable configuration.

## Dry Run

Dry-run mode derives maps from target lengths and writes planned artifacts
without loading a model:

```bash
.venv/bin/python \
  scripts/kivo_vd/run_ratio_scaled_long_context_sweep.py \
  --dry-run
```

## Fast RunPod Command

```bash
.venv/bin/python \
  scripts/kivo_vd/run_ratio_scaled_long_context_sweep.py \
  --model gpt2 \
  --target-token-lengths 768 \
  --num-prompts-per-length 2 \
  --ratio-policies \
  "balanced=0:0.60,5:0.45,8:0.45,11:0.60;safer=0:0.70,5:0.55,8:0.55,11:0.70" \
  --policies query_key_block_score,oracle_topk \
  --max-new-tokens-values 16 \
  --device cuda \
  --output-dir outputs/kivo_vd/runs/phase11_6_gpt2_ratio_scaled_fast
```

## Full RunPod Command

```bash
.venv/bin/python \
  scripts/kivo_vd/run_ratio_scaled_long_context_sweep.py \
  --model gpt2 \
  --target-token-lengths 768,960 \
  --num-prompts-per-length 2 \
  --ratio-policies \
  "aggressive=0:0.50,5:0.40,8:0.40,11:0.50;balanced=0:0.60,5:0.45,8:0.45,11:0.60;safer=0:0.70,5:0.55,8:0.55,11:0.70" \
  --policies query_key_block_score,oracle_topk \
  --max-new-tokens-values 16,32 \
  --device cuda \
  --output-dir outputs/kivo_vd/runs/phase11_6_gpt2_ratio_scaled_full
```

## Failure Flags

The report emits:

- `exact_sequence_match_below_1`
- `token_match_below_0.99`
- `normalized_edit_distance_above_0`
- `average_kl_above_0.01`
- `per_step_top1_below_1`
- `selected_ratio_above_0.85`
- `estimated_reduction_below_0.20`
- `actual_prompt_length_too_short`

## Readiness

`phase11_7_ready` is true when at least one `query_key_block_score` ratio
policy passes the strict standalone thresholds. `phase12_ready` remains false
by design.

The best deployable tradeoff excludes oracle and ranks passing query-key rows
by estimated active-block reduction descending, then KL ascending. The safest
passing configuration excludes oracle and ranks by KL ascending.

## RunPod Results

### Fast Result

The fast RunPod run used target length `768`, two synthetic prompts, and
`max_new_tokens=16`.

Derived maps:

| ratio policy | actual tokens | blocks | derived map | selected ratio | estimated reduction |
| --- | --- | --- | --- | --- | --- |
| `balanced` | `~734` | `46` | `0:28,5:21,8:21,11:28` | `0.523401` | `0.476599` |
| `safer` | `~734` | `46` | `0:33,5:26,8:26,11:33` | `0.630218` | `0.369782` |

Oracle top-k passed for both ratio policies with exact match `1.0`, token
match `1.0`, and edit distance `0.0`.

Query-key block scoring failed for `balanced`:

| metric | value |
| --- | --- |
| exact sequence match | `0.5` |
| token match | `0.5` |
| normalized edit distance | `0.5` |
| average KL | `7.979269` |

Query-key block scoring passed for `safer`:

| metric | value |
| --- | --- |
| exact sequence match | `1.0` |
| token match | `1.0` |
| prefix match length | `16.0` |
| normalized edit distance | `0.0` |
| average KL | `0.001453` |
| per-step top-1 match | `1.0` |
| selected ratio | `0.630218` |
| estimated reduction | `0.369782` |

Interpretation: ratio-scaled map generation works, but query-key selection
needs more margin than oracle. At about 734 prompt tokens, `safer` passed and
`balanced` was not reliable.

### Full Result

The full RunPod run used target lengths `768,960`, two synthetic prompts per
length, ratio policies `aggressive`, `balanced`, and `safer`, and
`max_new_tokens=16,32`.

Derived maps:

| target | actual tokens | blocks | ratio policy | derived map |
| --- | --- | --- | --- | --- |
| `768` | `~734` | `46` | `aggressive` | `0:23,5:19,8:19,11:23` |
| `768` | `~734` | `46` | `balanced` | `0:28,5:21,8:21,11:28` |
| `768` | `~734` | `46` | `safer` | `0:33,5:26,8:26,11:33` |
| `960` | `~917` | `58` | `aggressive` | `0:29,5:24,8:24,11:29` |
| `960` | `~917` | `58` | `balanced` | `0:35,5:27,8:27,11:35` |
| `960` | `~917` | `58` | `safer` | `0:41,5:32,8:32,11:41` |

The full run completed `24` runs with no script failures. It reported
`phase11_7_ready=true` and `phase12_ready=false`.

Per-ratio-policy summary:

| ratio policy | exact | token match | edit | KL | selected ratio | estimated reduction |
| --- | --- | --- | --- | --- | --- | --- |
| `aggressive` | `0.75` | `0.75` | `0.210938` | `3.953345` | `0.449662` | `0.550338` |
| `balanced` | `0.875` | `0.875` | `0.105469` | `2.049755` | `0.525318` | `0.474682` |
| `safer` | `1.0` | `1.0` | `0.0` | `0.000488` | `0.625461` | `0.374539` |

Oracle top-k passed all ratio policies with exact match `1.0`, token match
`1.0`, edit distance `0.0`, and average KL around `0.000248`.

Query-key block scoring had failures in the aggressive policy and in the
balanced target-768 cases. Balanced passed at target 960 for both 16 and 32
new tokens. For target 960 with `max_new_tokens=32`, balanced produced:

| metric | value |
| --- | --- |
| derived map | `0:35,5:27,8:27,11:35` |
| exact sequence match | `1.0` |
| token match | `1.0` |
| normalized edit distance | `0.0` |
| average KL | `0.001344` |
| selected ratio | `0.527726` |
| estimated reduction | `0.472274` |

### Best Tradeoff And Safest Config

The best deployable tradeoff found in this run was:

| field | value |
| --- | --- |
| ratio policy | `balanced` |
| target | `960` |
| derived map | `0:35,5:27,8:27,11:35` |
| max new tokens | `32` |
| exact/token match | `1.0 / 1.0` |
| average KL | `0.001344` |
| selected ratio | `0.527726` |
| estimated reduction | `0.472274` |

The safest passing deployable config by KL was:

| field | value |
| --- | --- |
| ratio policy | `safer` |
| target | `960` |
| derived map | `0:41,5:32,8:32,11:41` |
| max new tokens | `16` |
| average KL | `0.000408` |
| selected ratio | `0.626644` |
| estimated reduction | `0.373356` |

### Interpretation

Ratio-scaled budgets are better than fixed maps for these long-context GPT-2
probes. The results support budget selection as a function of layer, context
block count, and risk tolerance.

`aggressive` is too risky for query-key selection. `balanced` can be viable,
but it was not universally safe because it failed at target 768. `safer`
currently looks like the reliable GPT-2 default. Oracle passing where
query-key fails indicates selector-margin risk rather than selected-attention
impossibility.

## Readiness Helper

Use the readiness helper to summarize Phase 11.6 artifacts:

```bash
.venv/bin/python \
  scripts/kivo_vd/check_phase11_ratio_scaled_readiness.py \
  --inputs \
  outputs/kivo_vd/runs/phase11_6_gpt2_ratio_scaled_full/ratio_scaled_generation_summary.json
```

The helper reports `phase11_7_ready`, keeps `phase12_ready=false`, separates
the best quality/savings tradeoff from the safest passing config, and emits
warnings for aggressive failures, balanced target-768 failures, and
oracle/query-key margin gaps.

## Recommended Next Phase

Phase 11.7 should either test a longer-context small model or improve selector
margin. GPT-2 is near its context limit; further enterprise-relevant testing
requires 2K-8K+ context. vLLM integration remains out of scope until
longer-context offline evidence exists.

## Caveats

- Evaluation runs outside vLLM.
- No vLLM integration is implemented.
- Generation uses greedy decoding only.
- Prompts are synthetic long prompts.
- No active routing is implemented.
- No measured runtime memory reduction is claimed.
- No latency claim is made.
- This is a generation-quality probe, not a preservation claim.
- GPT-2's context limit applies.
