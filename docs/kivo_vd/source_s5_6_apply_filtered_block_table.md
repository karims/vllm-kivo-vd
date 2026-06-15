# Phase S5.6: Apply Filtered Block Table

Phase S5.5 defined the worker-side filtered-view plan. Phase S5.6 adds the
smallest local-safe apply primitive for that plan: replacing a CPU-side
`BlockTable` row with a validated filtered subsequence.

## What S5.6 implements

- Pure filtered-row apply validation:
  - `vllm/v1/worker/kivo_block_table_sync.py::apply_filtered_block_row_if_safe(...)`
- CPU-side row replacement helper:
  - `vllm/v1/worker/block_table.py::BlockTable.replace_row_block_ids_if_safe(...)`

The helper:

- preserves natural order
- rejects unknown ids
- rejects duplicates
- rejects empty filtered rows
- clears trailing row entries back to the table convention (`0`)
- modifies only the targeted row

## What S5.6 does not do

- It does not wire this mutation into the normal runtime hot path.
- It does not rewrite slot mappings automatically during inference.
- It does not free live KV blocks.
- It does not claim memory reduction.

## Why this still is not the full live-demotion step

This phase proves the worker block-table row can be rewritten safely in a local
unit-tested context. The remaining missing piece is pairing that worker row
rewrite with the ownership-side `req_to_blocks` mutation in the same runtime
transition.

## Exact remaining blocker before live KV ownership demotion/free

The remaining blocker is a synchronized request-to-row/runtime update point
that can safely pair:

- ownership-side kept blocks in
  `vllm/v1/core/single_type_kv_cache_manager.py`
- worker-side row rewrite in
  `vllm/v1/worker/block_table.py`
- and refreshed slot mappings through
  `vllm/v1/worker/gpu_model_runner.py::_get_slot_mappings(...)`

Without that synchronized step, filtered worker rows and ownership-side live KV
demotion can still diverge.

## Next step

S5.7 should pair filtered block-table application with live KV ownership
demotion/free in one gated transition, or fail closed if request-to-row mapping
and slot-mapping refresh cannot be proven safe.
