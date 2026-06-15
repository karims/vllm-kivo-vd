# Phase S5.4: Live Block Demotion Plan

Phase S5.3 only filtered blocks already entering vLLM's existing
skipped/removable free path. That can help at the ownership boundary, but it
does not reduce peak KV memory because older live blocks are still present in
the request block list and in the worker-visible block table.

## Why live demotion is harder

Real online KV reduction needs both sides to stay consistent:

1. Core ownership:
   - `vllm/v1/core/single_type_kv_cache_manager.py`
   - `req_to_blocks[request_id]`
2. Worker visibility:
   - `vllm/v1/worker/gpu_input_batch.py::InputBatch.add_request(...)`
   - `vllm/v1/worker/block_table.py::BlockTable.add_row(...)`
   - `vllm/v1/worker/block_table.py::BlockTable.commit_block_table(...)`
   - `vllm/v1/worker/block_table.py::BlockTable.compute_slot_mapping(...)`
   - `vllm/v1/worker/gpu_model_runner.py::_get_slot_mappings(...)`

The worker block-table rows are populated from `request.block_ids` on the
worker path, then copied to GPU, and then used to build slot mappings and
attention metadata. Removing a block from `req_to_blocks` inside the core KV
manager does not automatically rewrite the worker block-table row or slot
mapping for the same request.

## What S5.4 implements

- A plan-only live demotion module:
  - `vllm/v1/core/kivo_kv_live_block_plan.py`
- A manager accessor:
  - `SingleTypeKVCacheManager.build_kivo_live_block_plan(request_id)`
- A stored last-plan accessor:
  - `SingleTypeKVCacheManager.get_last_kivo_live_block_plan()`

The plan reuses the S5.2 retention policy and classifies current live blocks
into:

- keep
- candidate demote
- protected
- blocked due to block-table sync unavailability
- blocked due to shared/prefix-style ownership (`ref_cnt > 1`)

## Current behavior

S5.4 is still plan-only in practice.

Even if `KIVO_KV_LIVE_DEMOTION_ACTION=apply_live_demotion_if_safe`, the local
implementation fail-closes because block-table synchronization is not available
from the core KV manager path. `safe_to_apply` therefore remains `False` under
the default conservative assumption.

## Exact blocker

The exact blocker is that the core manager cannot safely mutate the worker row
that was populated by:

- `vllm/v1/worker/gpu_input_batch.py::InputBatch.add_request(...)`
  via `self.block_table.add_row(request.block_ids, req_index)`

and later consumed by:

- `vllm/v1/worker/gpu_model_runner.py::_get_slot_mappings(...)`
- `vllm/v1/worker/gpu_model_runner.py::_build_attention_metadata(...)`

Without a synchronized update at that worker path, removing a live block from
`req_to_blocks` alone would make ownership and worker-visible attention state
diverge.

## Next exact implementation point

S5.5 should add a synchronized worker-side update path around:

- `vllm/v1/worker/gpu_input_batch.py::InputBatch.block_table`
- `vllm/v1/worker/block_table.py::BlockTable.add_row(...)`
- `vllm/v1/worker/block_table.py::BlockTable.commit_block_table(...)`
- `vllm/v1/worker/gpu_model_runner.py::_get_slot_mappings(...)`

so that a live demotion decision can update request ownership and worker block
table state together.

## What this phase does not claim

- No measured memory reduction.
- No latency claim.
- No quality preservation claim.
- No live mutation is applied by default.
