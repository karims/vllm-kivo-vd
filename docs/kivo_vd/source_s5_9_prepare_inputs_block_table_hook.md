# Phase S5.9: Prepare Inputs Block Table Hook

S5.8 established that `_get_slot_mappings(...)` is too late for the first
runtime block-table apply step, because slot mapping is already computed
earlier in `_prepare_inputs(...)`.

## Why `_get_slot_mappings(...)` was too late

`gpu_model_runner._prepare_inputs(...)` already calls:

- `input_batch.block_table.compute_slot_mapping(...)`

before `_get_slot_mappings(...)` exposes the per-group/per-layer slot-mapping
views. Rewriting block-table rows inside `_get_slot_mappings(...)` would not
rebuild the current step's slot mapping.

## Why `_prepare_inputs(...)` is the right hook

S5.9 places the Kivo hook immediately before:

- `input_batch.block_table.compute_slot_mapping(...)`

inside `vllm/v1/worker/gpu_model_runner.py::_prepare_inputs(...)`.

That is the first correct runtime location where:

- request-to-row mapping is already present in `InputBatch`
- worker block-table rows are available
- a filtered row can be applied
- slot mapping is about to be recomputed from the updated row

## What S5.9 applies

S5.9 applies only:

- block-table-only filtered worker rows

through the existing S5.8 helper and S5.6 row-replacement primitive.

## What remains disabled

- live KV ownership mutation
- `req_to_blocks` mutation
- `BlockPool.free_blocks(...)`
- KV tensor mutation

Live KV free remains disabled in this phase.

## Exact remaining blocker before S5.10

The remaining blocker is pairing this pre-slot-mapping worker row rewrite with
a synchronized ownership-side live KV mutation/free decision in the same step.

That future step must connect:

- core ownership in `SingleTypeKVCacheManager`
- worker row rewrite in `BlockTable`
- slot-mapping rebuild in `_prepare_inputs(...)`

without letting ownership and attention-visible state diverge.
