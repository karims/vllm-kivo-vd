# Phase S5.0: KV Memory Intervention Map

This phase is static reconnaissance only. It identifies the smallest real
vLLM code paths that could move Kivo-VD from metadata aliasing to actual KV
memory reduction.

## Executive Summary

The S3/S4 hooks proved we can observe attention metadata, tensor metadata, and
even drive behavior through `slot_mapping` and metadata aliasing. That is not
enough for real KV memory reduction because:

- the KV cache is still allocated and owned by the normal vLLM cache manager
- attention still consumes the normal `block_table`, `slot_mapping`, and
  `kv_cache`
- metadata aliasing changes what attention *sees*, but it does not reclaim or
  shrink KV storage

The next subsystem that must change for real memory work is the KV cache
allocation/free ownership path, with attention metadata changes following only
if we want selected or compressed blocks to remain semantically valid.

## Files Inspected

### KV cache ownership and allocation

- `vllm/v1/core/kv_cache_manager.py`
  - `KVCacheManager.allocate_slots(...)`
  - `KVCacheManager.free(...)`
  - `KVCacheManager.cache_blocks(...)`
  - `KVCacheManager.remove_skipped_blocks(...)`
  - `KVCacheManager.get_blocks(...)`
  - `KVCacheManager.get_block_ids(...)`
  - `KVCacheManager.take_new_block_ids(...)`
- `vllm/v1/core/kv_cache_coordinator.py`
  - `KVCacheCoordinator.get_num_blocks_to_allocate(...)`
  - `KVCacheCoordinator.allocate_new_blocks(...)`
  - `KVCacheCoordinator.allocate_new_computed_blocks(...)`
  - `KVCacheCoordinator.cache_blocks(...)`
  - `KVCacheCoordinator.free(...)`
  - `KVCacheCoordinator.remove_skipped_blocks(...)`
- `vllm/v1/core/single_type_kv_cache_manager.py`
  - `SingleTypeKVCacheManager.get_num_blocks_to_allocate(...)`
  - `SingleTypeKVCacheManager.allocate_new_computed_blocks(...)`
  - `SingleTypeKVCacheManager.allocate_new_blocks(...)`
  - `SingleTypeKVCacheManager.cache_blocks(...)`
  - `SingleTypeKVCacheManager.remove_skipped_blocks(...)`
  - `SingleTypeKVCacheManager.free(...)`
  - `SingleTypeKVCacheManager.get_num_skipped_tokens(...)`
- `vllm/v1/core/block_pool.py`
  - `BlockPool.get_new_blocks(...)`
  - `BlockPool.free_blocks(...)`
  - `BlockPool.cache_full_blocks(...)`
  - `BlockPool.evict_blocks(...)`
  - `BlockPool.take_events(...)`

### Block table / slot mapping construction

- `vllm/v1/worker/block_table.py`
  - `BlockTable.append_row(...)`
  - `BlockTable.add_row(...)`
  - `BlockTable.clear_row(...)`
  - `BlockTable.compute_slot_mapping(...)`
  - `MultiGroupBlockTable.compute_slot_mapping(...)`
- `vllm/v1/worker/gpu_input_batch.py`
  - `InputBatch.block_table`
  - `InputBatch.add_request(...)` and batch update paths that move rows
- `vllm/v1/worker/gpu_model_runner.py`
  - `_get_slot_mappings(...)`
  - `_build_attention_metadata(...)`
  - `initialize_input_batch(...)`
  - `maybe_reinitialize_input_batch(...)`
  - `initialize_metadata_builders(...)`

### Attention metadata and backend handoff

- `vllm/v1/attention/backend.py`
  - `CommonAttentionMetadata`
  - `AttentionMetadataBuilder`
  - `AttentionMetadataBuilder.build(...)`
  - `AttentionMetadataBuilder.update_block_table(...)`
- `vllm/v1/attention/backends/utils.py`
  - `CommonAttentionMetadata` construction helpers
  - `make_*_common_attn_metadata(...)`
  - `split_decodes_and_prefills(...)`
  - `subclass_attention_metadata(...)`
- `vllm/model_executor/layers/attention/attention.py`
  - `get_attention_context(...)`
  - `unified_kv_cache_update(...)`
  - `unified_attention_with_output(...)`
  - `maybe_transfer_kv_layer(...)`
- `vllm/v1/attention/backends/*`
  - backend-specific `build(...)` and `update_block_table(...)`
  - `FlashInferMetadataBuilder`
  - `FlexAttentionMetadataBuilder`
  - `CPUAttentionMetadataBuilder`
  - `RocmAttentionMetadataBuilder`
  - MLA builders and sparse variants

## Allocation / Free Lifecycle

The relevant lifecycle is:

1. The scheduler selects requests.
2. `KVCacheManager.allocate_slots(...)` asks the coordinator how many blocks
   are needed.
3. `KVCacheCoordinator.allocate_new_computed_blocks(...)` and
   `KVCacheCoordinator.allocate_new_blocks(...)` delegate to the per-type
   managers.
4. `SingleTypeKVCacheManager.allocate_new_blocks(...)` or
   `allocate_new_computed_blocks(...)` obtains physical blocks from
   `BlockPool.get_new_blocks(...)`.
5. `SingleTypeKVCacheManager.req_to_blocks` becomes the durable ownership map
   for the request.
6. `KVCacheManager.cache_blocks(...)` and `BlockPool.cache_full_blocks(...)`
   attach prefix-cache metadata and hash ownership to the allocated blocks.
7. `gpu_model_runner._get_slot_mappings(...)` reads `BlockTable.slot_mapping`
   and `BlockTable.block_table`.
8. `_build_attention_metadata(...)` packages `block_table_tensor`,
   `slot_mapping`, `seq_lens`, and `query_start_loc` into
   `CommonAttentionMetadata`.
9. The attention backend builder turns `CommonAttentionMetadata` into
   backend-specific metadata.
10. `unified_attention_with_output(...)` hands the metadata plus `kv_cache`
    to the backend implementation.
11. `KVCacheManager.free(...)` and `SingleTypeKVCacheManager.free(...)`
    release ownership, and `BlockPool.free_blocks(...)` returns blocks to the
    free queue.

## Existing S3/S4 Hook Placement

Current Kivo hooks sit in two places:

- `vllm/v1/worker/gpu_model_runner.py::_build_attention_metadata(...)`
  - observer-only metadata hook
  - sees `block_table_tensor`, `slot_mapping`, `seq_lens`, `query_start_loc`
  - can affect control decisions if used for mutation, but it does not by
    itself reduce KV memory
- `vllm/model_executor/layers/attention/attention.py::unified_attention_with_output(...)`
  - tensor observer hook
  - sees `query`, `key`, `value`, `kv_cache`, `slot_mapping`, and the layer
    backend
  - useful for observation and dry-run selection, but still not the real
    memory ownership point

These hooks are valuable for visibility and shadow experiments, but they do
not move the block ownership boundary.

## Real Intervention Candidates

### Candidate 1: KV cache ownership / block manager level

- Files:
  - `vllm/v1/core/single_type_kv_cache_manager.py`
  - `vllm/v1/core/kv_cache_manager.py`
  - `vllm/v1/core/block_pool.py`
- Functions/classes:
  - `SingleTypeKVCacheManager.remove_skipped_blocks(...)`
  - `SingleTypeKVCacheManager.free(...)`
  - `SingleTypeKVCacheManager.allocate_new_blocks(...)`
  - `BlockPool.free_blocks(...)`
  - `BlockPool.get_new_blocks(...)`
- What it can do:
  - mark blocks evictable or demoted
  - actually free blocks back to the pool
  - establish a real two-tier ownership policy
- What it breaks if wrong:
  - request state bookkeeping
  - prefix cache correctness
  - reuse of blocks still needed by attention
- Memory impact:
  - yes, this is the first place that can reduce real reserved KV ownership
    if paired with a valid replacement representation

### Candidate 2: Block table / GPU input path

- Files:
  - `vllm/v1/worker/block_table.py`
  - `vllm/v1/worker/gpu_input_batch.py`
  - `vllm/v1/worker/gpu_model_runner.py`
- Functions/classes:
  - `BlockTable.compute_slot_mapping(...)`
  - `MultiGroupBlockTable.append_row(...)`
  - `gpu_model_runner._get_slot_mappings(...)`
  - `gpu_model_runner._build_attention_metadata(...)`
- What it can do:
  - skip or remap old blocks in metadata
  - compact the table fed to attention
  - change which blocks attention sees
- What it affects:
  - mostly compute-path visibility and attention semantics
  - only indirect memory impact unless paired with real block freeing
- Suitability:
  - safer than kernel changes, but still not enough alone for real memory

### Candidate 3: Attention backend / kernel path

- Files:
  - `vllm/v1/attention/backend.py`
  - `vllm/v1/attention/backends/*`
  - `vllm/model_executor/layers/attention/attention.py`
- Functions/classes:
  - backend builders that consume `CommonAttentionMetadata`
  - `AttentionMetadataBuilder.build(...)`
  - `AttentionMetadataBuilder.update_block_table(...)`
  - `unified_attention_with_output(...)`
- What it can do:
  - implement real selected attention or compressed attention
  - define exactly which blocks are attended
- Why it is harder:
  - likely needs backend-specific logic
  - may require CUDA/Triton work for performance
  - safest correctness path, but not the smallest change

### Candidate 4: Two-tier KV cache

- Files likely involved:
  - `vllm/v1/core/single_type_kv_cache_manager.py`
  - `vllm/v1/core/block_pool.py`
  - `vllm/v1/worker/block_table.py`
  - `vllm/v1/attention/backends/*`
- Behavior:
  - recent blocks stay full KV
  - old blocks are demoted to compressed/sketched summaries
  - attention sees either full or compressed representation depending on age
- This is the first path that is plausibly both memory-reducing and
  semantics-preserving, but it requires real ownership plus a real attention
  representation for demoted blocks.

## Recommendation

S5.1 should implement a minimal real KV block intervention at
`vllm/v1/core/single_type_kv_cache_manager.py::remove_skipped_blocks(...)`
with coordinated ownership bookkeeping in `BlockPool.free_blocks(...)`.

Why this is the direct path:

- it is the first place where blocks can be truly released from request
  ownership
- it already represents the boundary between live and no-longer-needed blocks
- it is the smallest place to attach a real demotion/eviction policy before
  changing attention semantics

Expected behavior:

- keep recent blocks as full KV
- identify blocks that can be demoted or freed
- return only truly no-longer-needed blocks to the free queue
- preserve request correctness while preparing for a future compressed
  representation

Primary failure mode to guard against:

- freeing blocks that attention still needs, which would corrupt request state
  or produce wrong outputs

Local test that can run without GPU:

- unit tests for block-ownership bookkeeping and demotion decisions using fake
  block sequences and mocked `KVCacheBlock` ownership state

GPU validation needed later:

- confirm the chosen ownership policy does not change outputs
- confirm the demotion/free path actually reduces reserved GPU KV memory
- confirm the attention backend still receives valid metadata for the live
  portion of the cache

## Do Not Do Next

Stop doing these as the primary path:

- more shadow paths
- more metadata-only aliasing
- more JSONL-heavy instrumentation
- more GPT-2 tiny latency benchmarks until a real memory path exists
- more observation-only probes that do not change KV ownership

The next step should change actual KV ownership or a concrete two-tier cache,
not another visibility layer.
