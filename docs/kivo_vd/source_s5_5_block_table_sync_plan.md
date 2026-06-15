# Phase S5.5: Block Table Sync Plan

S5.4 showed why ownership mutation alone is unsafe: the core KV manager owns
`req_to_blocks`, but worker-visible attention state is built from separate
block-table rows and downstream slot mappings.

## What worker block-table sync must guarantee

For any future live-block demotion, the worker-visible block-table row must
match the ownership-side kept block sequence:

- filtered block ids must be a subsequence of the original row
- natural order must be preserved
- protected/current/recent block ids must remain visible
- removed block ids must be exactly the original row minus the filtered row

If these conditions are not satisfied, attention metadata can diverge from the
ownership-side view.

## What S5.5 implements

- A new pure planner:
  - `vllm/v1/worker/kivo_block_table_sync.py`
- A tiny CPU-side accessor:
  - `vllm/v1/worker/block_table.py::BlockTable.get_row_block_ids(...)`
  - `vllm/v1/worker/block_table.py::MultiGroupBlockTable.get_row_block_ids(...)`

The planner computes a filtered worker-row view from:

- `original_block_ids`
- `keep_block_ids`
- optional `protected_block_ids`

and returns a fail-closed `KivoBlockTableSyncPlan`.

## Why S5.5 is still plan-only

`apply_filtered_view_if_safe` is recognized but still fails closed locally.
This phase does not mutate worker block-table buffers, does not rewrite slot
mapping, and does not change runtime behavior by default.

## Exact next mutation point for S5.6

The next mutation step must synchronize the worker-side path around:

- `vllm/v1/worker/gpu_input_batch.py::InputBatch.add_request(...)`
- `vllm/v1/worker/block_table.py::BlockTable.add_row(...)`
- `vllm/v1/worker/block_table.py::BlockTable.commit_block_table(...)`
- `vllm/v1/worker/gpu_model_runner.py::_get_slot_mappings(...)`

S5.6 should only apply filtered block-table rows if ownership-side kept blocks
and worker row updates can be committed together.

## What this phase does not claim

- No live KV free/demotion is applied yet.
- No measured memory reduction.
- No latency improvement.
- No quality preservation claim.
