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
