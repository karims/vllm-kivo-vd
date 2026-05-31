# Kivo-VD Phase 0: vLLM KV Cache Map

This document maps the current vLLM v1 KV-cache flow and identifies safe Phase 0 observer hook points.

## 1) KV cache block allocation

- `vllm/v1/core/sched/scheduler.py`
  - `Scheduler.schedule()`
  - Calls `self.kv_cache_manager.allocate_slots(...)` for running and waiting requests.
- `vllm/v1/core/kv_cache_manager.py`
  - `KVCacheManager.allocate_slots(...)`
  - Public allocation entrypoint used by scheduler.
- `vllm/v1/core/kv_cache_coordinator.py`
  - Coordinator layer used by `KVCacheManager` to delegate per-KV-group logic.
- `vllm/v1/core/single_type_kv_cache_manager.py`
  - Per-group block math and allocation details (including block bookkeeping).
- `vllm/v1/core/block_pool.py`
  - `BlockPool.get_new_blocks(...)`/queue operations own physical free-block pool behavior.

## 2) Block tables / logical-to-physical mapping

- `vllm/v1/worker/block_table.py`
  - `BlockTable.append_row/add_row/clear_row/move_row/swap_row`
  - Stores per-request logical block-id rows.
  - `compute_slot_mapping(...)` transforms positions + block table into slot mappings consumed by attention backends.
  - `MultiGroupBlockTable` manages one table per KV-cache group.
- `vllm/v1/worker/gpu_model_runner.py`
  - `_build_attention_metadata(...)`
  - Reads block tables (`_get_block_table`) and injects `block_table_tensor` + `slot_mapping` into `CommonAttentionMetadata`.

## 3) Freeing/reusing KV blocks

- `vllm/v1/core/sched/scheduler.py`
  - `_preempt_request(...)` and finish paths call `self.kv_cache_manager.free(request)`.
- `vllm/v1/core/kv_cache_manager.py`
  - `free(...)` public free path.
  - `get_computed_blocks(...)` and `cache_blocks(...)` coordinate prefix-cache reuse.
- `vllm/v1/core/block_pool.py`
  - `free_blocks(...)`, `evict_blocks(...)`, hash map (`BlockHashToBlockMap`) for prefix-cache reuse and eviction order.

## 4) Scheduler interaction with KV cache

- Main interaction is in `Scheduler.schedule()`:
  - Prefix hit query: `kv_cache_manager.get_computed_blocks(request)`
  - Allocation: `kv_cache_manager.allocate_slots(...)`
  - Common-prefix metrics: `get_num_common_prefix_blocks(...)`
  - Zeroing list extraction: `take_new_block_ids()`
  - Per-step lifecycle: `kv_cache_manager.new_step_starts()`
- Output handoff:
  - `SchedulerOutput` carries `new_block_ids`/`block_ids` to workers.

## 5) Model runner path where KV metadata enters attention

- `vllm/v1/worker/gpu_model_runner.py`
  - `_prepare_inputs(...)` updates block table rows.
  - `self.input_batch.block_table.commit_block_table(num_reqs)`
  - `self.input_batch.block_table.compute_slot_mapping(...)`
  - `_build_attention_metadata(...)` creates `CommonAttentionMetadata` with:
    - `block_table_tensor`
    - `slot_mapping`
    - `seq_lens` and related lengths
  - Per-layer metadata builders consume `CommonAttentionMetadata` and pass backend-specific metadata into attention kernels.

## Safe Phase 0 observer attachment points

- Scheduler-level (lowest risk, no kernel/math impact):
  - Before/after `kv_cache_manager.allocate_slots(...)` in `Scheduler.schedule()`.
  - Before `kv_cache_manager.free(request)` in `_preempt_request(...)` and finish/free paths.
- Runner metadata-level (still non-invasive):
  - Inside `_build_attention_metadata(...)` after block table tensors are created and before builder invocation.
- Avoid attaching in kernels/backends for Phase 0.

## Phase 0.1 wiring status (implemented)

- Observer creation:
  - `vllm/v1/core/sched/scheduler.py::_create_kivo_vd_observer(...)`
  - `Scheduler.__init__` creates observer only when `vllm_config.enable_kivo_vd` is `True`.
- Allocation hooks wired:
  - `Scheduler.schedule()` running-request allocation loop:
    - `on_before_allocate_slots(...)` immediately before `kv_cache_manager.allocate_slots(...)`
    - `on_after_allocate_slots(...)` immediately after call (including `None` return case)
  - `Scheduler.schedule()` waiting-request allocation path:
    - same before/after hook pattern around `allocate_slots(...)`
- Free hooks wired:
  - `_preempt_request(...)` before `kv_cache_manager.free(request)`
  - `_free_blocks(...)` before `kv_cache_manager.free(request)`
  - `_update_waiting_for_remote_kv(...)` failure path before `kv_cache_manager.free(request)`

## What remains unwired (intentional)

- No hooks in `vllm/v1/worker/gpu_model_runner.py`.
- No hooks in `_build_attention_metadata(...)`.
- No hooks in block table tensor population or slot mapping computation.
- No hooks in kernels/backend custom ops.

## Why GPUModelRunner/attention metadata hooks are deferred

- Phase 0.1 scope is scheduler/block lifecycle only to keep risk low.
- Scheduler hooks provide lifecycle visibility without touching attention execution.
- Deferring runner/attention metadata hooks avoids accidental performance or
  numerical behavior changes in critical forward paths.

## What should NOT be modified in Phase 0

- Attention kernels / CUDA / Triton / ROCm custom ops.
- Attention score/value math or causal/sliding-window behavior.
- Block hashing semantics and prefix-cache eviction policy.
- KV cache layout contracts used by backend metadata builders.
- Model architecture code paths.
- KV compression/offload/sparse-attention implementations.

## Proposed minimal hook design

- New no-op observer class:
  - `vllm/v1/core/kivo_vd_observer.py::KivoVDObserver`
  - Methods:
    - `on_before_allocate_slots(...)`
    - `on_after_allocate_slots(...)`
    - `on_free_request(...)`
    - `on_build_attention_metadata(...)`
- Config gate:
  - `vllm/config/vllm.py::VllmConfig.enable_kivo_vd` (default `False`).
- Phase 0 behavior:
  - Default is disabled and no behavior change.
  - Observer can be instantiated and wired later with minimal scheduler/runner touchpoints.
