# Kivo-VD Phase 2.8: Active KV Policy Simulation

Phase 2.8 adds an offline simulator that translates HF Q/K sketch retrieval
rows into estimated active KV block residency.

This is not a vLLM runtime memory measurement. It is a policy sanity check for
whether sketch-ranked candidate blocks plus a recent-window fallback could keep
the exact top-attended blocks active while reducing the number of active KV
blocks.

## Script

```bash
.venv/bin/python scripts/kivo_vd/simulate_active_kv_policy.py
```

Defaults:

- Input: `outputs/kivo_vd/hf_qk_head_sweep.jsonl`
- Output: `outputs/kivo_vd/active_kv_policy_simulation.jsonl`
- Recent windows: `4,8,16`
- Candidate budgets: `8,16,32`

## Required Sweep Input

The simulator needs a full approximate block ranking, not only the compact
top-k block list. Generate suitable HF sweep rows with:

```bash
.venv/bin/python scripts/kivo_vd/run_hf_qk_head_sweep.py \
  --model-name distilgpt2 \
  --sketch-types count_sketch,random_projection \
  --sketch-dims 32,64,128 \
  --layers 0,1 \
  --heads 0,1 \
  --max-tokens 512 \
  --include-ranked-blocks
```

Without `--include-ranked-blocks`, the simulator fails closed with a clear
message because `approx_top_block_ids` is too small for larger candidate-budget
policies.

## Policy Model

For each HF sweep row and each policy pair:

- `num_total_blocks = ceil(num_keys_used / block_size)`
- Recent blocks are the last `recent_window_blocks` blocks available to that
  causal query.
- Candidate blocks are the first `candidate_budget_blocks` entries from
  `approx_ranked_block_ids`.
- Active blocks are the union of recent and candidate blocks.
- `active_block_ratio = active_block_count / num_total_blocks`
- `estimated_kv_reduction = 1 - active_block_ratio`
- `exact_top_recall_in_active` is the fraction of exact top-attended blocks
  included in the active set.

## Output Rows

The simulator writes one JSONL row per input row per policy. Key fields:

- `sketch_type`
- `sketch_dim`
- `layer`
- `head`
- `query_position`
- `num_total_blocks`
- `recent_window_blocks`
- `candidate_budget_blocks`
- `active_block_count`
- `active_block_ratio`
- `estimated_kv_reduction`
- `exact_top_recall_in_active`

It also prints an aggregate summary grouped by sketch type, sketch dimension,
recent window, and candidate budget.

## Interpreting Results

The target signal is high `exact_top_recall_in_active` with low
`active_block_ratio`.

This estimates an active GPU KV residency policy. It does not prove actual vLLM
memory savings yet because real savings require runtime support for candidate
block attention, block residency management, or offload/compression paths.

## Scope Limits

- Offline only.
- No vLLM runtime changes.
- No scheduler, GPUModelRunner, attention, block table, CUDA, Triton, or kernel
  changes.
- No claim of measured runtime KV memory reduction.
