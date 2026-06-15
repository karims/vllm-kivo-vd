# Phase S5.8: Runtime Block Table Apply

S5.8 adds the first runtime-facing helper for applying a filtered worker
block-table row, but keeps it outside the default hot path.

## Why `_get_slot_mappings(...)` was the candidate

`_get_slot_mappings(...)` is close to where attention-visible state is exposed
to the backend, so it was the natural first place to inspect for a runtime
integration hook.

## What S5.8 found

`_get_slot_mappings(...)` is too late for a correct first integration.

`gpu_model_runner._prepare_inputs(...)` already calls:

- `input_batch.block_table.compute_slot_mapping(...)`

before `_get_slot_mappings(...)` slices and returns the slot-mapping tensors.

That means replacing the block-table row inside `_get_slot_mappings(...)` would
not automatically rebuild slot mappings for the current step.

## What S5.8 implements

- Runtime-facing helper module:
  - `vllm/v1/worker/kivo_runtime_block_table_apply.py`
- Small `InputBatch` accessors:
  - `get_req_index(...)`
  - `get_req_block_row_ids(...)`

The helper can:

- inspect request-to-row mapping from `InputBatch`
- build a retention decision for one worker row
- build a sync-apply decision
- locally apply a filtered worker row through the S5.6 primitive when
  slot-mapping refresh is explicitly marked available

## Whether S5.8 actually wires into `_get_slot_mappings(...)`

No.

The helper is runtime-facing, but not wired into `_get_slot_mappings(...)`
because that point is downstream of slot-mapping computation and therefore too
late for a correct first apply step.

## Why live KV free remains disabled

Even with local block-table-only apply available, S5.8 still does not pair:

- worker filtered row application
- slot-mapping rebuild for the same step
- ownership-side live KV mutation/free

Without that synchronized step, live KV free remains unsafe.

## Exact remaining blocker before S5.9

The remaining blocker is a gated runtime hook *before*
`compute_slot_mapping(...)` runs, where:

- request-to-row identity is known
- filtered worker rows can be applied
- slot mappings can then be rebuilt from the filtered rows

The next concrete runtime integration point should therefore move earlier than
`_get_slot_mappings(...)`, most likely into the `_prepare_inputs(...)` path in
`vllm/v1/worker/gpu_model_runner.py`.
