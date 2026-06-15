# Phase S5.7: KV Sync Apply

S5.7 introduces a conservative coordinator for pairing three pieces of the
future live-demotion path:

1. live-block demotion/retention decision
2. worker filtered block-table row
3. ownership-side demotion/free intent

## What synchronization means here

The synchronized decision answers:

- which blocks are kept
- which blocks are demotion candidates
- whether the worker block-table row can be filtered safely
- whether slot mapping refresh is required and available
- whether ownership-side mutation is safe enough to pair with that worker view

## What S5.7 implements

- New coordinator module:
  - `vllm/v1/worker/kivo_kv_sync_apply.py`
- Pure decision object:
  - `KivoKVSyncApplyDecision`
- Local block-table-only apply helper:
  - `apply_block_table_only_if_safe(...)`

This allows a direct local `BlockTable` row replacement when:

- the keep/demote split is valid
- the filtered row is non-empty and order-preserving
- slot-mapping refresh is not required, or is marked available

## What S5.7 does not do

- No default runtime behavior change.
- No live KV ownership mutation or free.
- No hot-path integration into `InputBatch` or `GPUModelRunner`.
- No memory reduction claim.

## Whether S5.7 can apply block-table-only locally

Yes. In a direct local `BlockTable` context, S5.7 can apply a filtered row
through the existing S5.6 helper when the coordinator marks the decision safe.

This is still a local primitive, not a full runtime integration.

## Why live KV ownership mutation/free remains blocked

Live ownership mutation still needs a synchronized runtime point that pairs:

- ownership-side mutation in
  `vllm/v1/core/single_type_kv_cache_manager.py`
- worker block-table mutation in
  `vllm/v1/worker/block_table.py`
- slot-mapping refresh through
  `vllm/v1/worker/gpu_model_runner.py::_get_slot_mappings(...)`

`apply_block_table_and_mark_ownership` therefore remains fail-closed locally.

## Exact next runtime integration point for S5.8

The next runtime step should be a gated integration point near the worker path
that already owns block-table rows and slot-mapping rebuild:

- `vllm/v1/worker/gpu_input_batch.py`
- `vllm/v1/worker/block_table.py`
- `vllm/v1/worker/gpu_model_runner.py::_get_slot_mappings(...)`

S5.8 should only proceed if that runtime point can pair row replacement with a
provable slot-mapping refresh and request-to-row identity.
