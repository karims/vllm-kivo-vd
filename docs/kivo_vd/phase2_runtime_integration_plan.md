# Kivo-VD Phase 2.0 Runtime Integration Plan (Design Only)

## Scope

This document proposes a runtime integration path for Kivo-VD in vLLM v1,
without changing model weights, training, tokenizer behavior, or architecture.
It is a design/plan only; no runtime behavior changes are included in Phase 2.0.

## Files inspected

- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/block_pool.py`
- `vllm/v1/worker/block_table.py`
- `vllm/v1/worker/gpu/block_table.py`
- `vllm/v1/worker/gpu/model_runner.py`
- `vllm/v1/worker/gpu/attn_utils.py`
- `vllm/v1/worker/gpu/model_states/default.py`
- `vllm/v1/worker/gpu/model_states/interface.py`
- `vllm/v1/attention/backend.py`
- `vllm/v1/attention/backends/flash_attn.py`

## Current runtime map (where data flows today)

1. Scheduler decides tokens to run and alloc/free lifecycle:
- `Scheduler` calls `kv_cache_manager.allocate_slots(...)` in running/waiting paths.
- `Scheduler` calls `kv_cache_manager.free(request)` via `_preempt_request`, `_free_blocks`, and remote-KV failure cleanup.
- Phase 0/1 hooks already exist around these points via `KivoVDObserver`.

2. KV block ownership/storage:
- `KVCacheManager` + coordinator + `BlockPool` own physical block lifecycle.
- `KVCacheBlocks.get_block_ids()` gives per-group physical block ids.
- `BlockPool` manages free queue, hash map for prefix cache, and block metadata.

3. Request logical-to-physical mapping:
- GPU path uses `BlockTables` in `vllm/v1/worker/gpu/block_table.py`.
- Scheduler side passes `new_block_ids`; runner updates persistent staged tables.
- Runtime attention uses gathered block tables + slot mappings per batch.

4. Attention metadata build path:
- `GPUModelRunner.prepare_attn()` gathers block tables and computes slot mappings.
- `ModelState.prepare_attn(...)` calls `build_attn_metadata(...)`.
- `build_attn_metadata(...)` creates `CommonAttentionMetadata` and backend-specific metadata per layer.

5. Where Q/K/V exist:
- Attention backend `forward(...)` receives `query`, `key`, `value`, `kv_cache`, and `attn_metadata`.
- Example: `FlashAttentionImpl.forward(...)` uses `query/key/value` tensors and `attn_metadata.block_table`.

## Answers to required design questions

### 1) Where can Kivo-VD store per-block key sketches?

Primary choice (safest initial):
- Store in a Kivo-owned Python-side runtime index keyed by
  `(request_id, kv_cache_group_id, block_id, layer_id/head_id optional)`.
- Host this in a dedicated runtime service object attached to scheduler/worker
  lifecycle, not in core `KVCacheBlock` initially.

Possible later optimization:
- Add lightweight metadata slots near block manager state for faster lookup,
  but defer until behavior is validated.

### 2) How can sketches be updated when new KV blocks are allocated/written?

Allocation-time metadata:
- Scheduler hooks (`on_after_allocate_slots`) already expose request/block ids.
- Use this to create placeholder entries and expected ownership.

Write-time sketch materialization (future):
- Real key sketch updates need actual key tensors after KV write.
- Likely insertion points are in attention backend paths after K/V projections and
  cache update paths (`do_kv_cache_update` / backend forward update stages).
- Phase 2 should only define interface contracts here, not compute sketches yet.

### 3) Where can current query vector be accessed?

Best runtime location:
- Attention backend `forward(...)` where `query` tensor is directly available
  per layer call (`query: [num_tokens, num_heads, head_size]`).

Alternative (harder):
- Earlier model layer hooks before backend call, but this is more model-specific
  and riskier.

### 4) Where can sketch scoring happen?

Two-tier plan:
- Phase 2 dry-run scoring: Python-side in worker before backend invocation,
  using detached lightweight summaries and no behavior change.
- Phase 3 real path: backend-integrated scoring/selection on device side (or
  low-overhead host-device path), because per-token Python scoring is too slow.

### 5) How can candidate block IDs be represented?

Recommended representation:
- Per request, per KV group (and eventually per layer/head if needed):
  `candidate_block_ids: list[int]` plus optional ranking/score array.
- Keep this separate from canonical full block table at first.
- For dry-run tracing, store as side metadata attached to observer traces.

### 6) Can selected-block attention be implemented by modifying:

a. block tables
- Yes, but this is behavior-changing and correctness-sensitive.
- Would require constructing reduced block tables and consistent slot mappings.
- High risk for prefill/decode/chunked prefill/cudagraph invariants.

b. attention metadata
- Yes, likely safest entry for a controlled behavior change.
- Backends already consume metadata objects; adding optional candidate fields is
  cleaner than mutating scheduler state globally.

c. attention backend/kernel
- Yes, likely required for real speedup.
- Kernel/backend must actually skip non-candidate blocks during attention.

d. scheduler/block manager
- Scheduler can produce advisory candidates, but should not own final
  layer/head attention behavior.
- Block manager should remain source of truth for allocation/free.

### 7) Safest path for Phase 2.1 dry-run

Safest: metadata-only advisory path in worker/attention metadata build.
- Keep full attention unchanged.
- Generate candidate sets and compare with actual accessed/full blocks.
- Emit traces/metrics only.

### 8) Likely path for Phase 3 real speedup

Most likely:
- Extend attention metadata + backend/kernel to consume candidate block subsets,
  and run exact attention only over selected blocks (+ required recency window).
- This requires backend/kernel work and careful fallback behavior.

## Concrete phased runtime plan

## Phase 2.1: Sketch backend abstraction

Deliverables:
- Introduce runtime sketch backend interface (Python only):
  - `update_block_metadata(...)`
  - `update_block_sketch(...)` (stub)
  - `score_candidates(query_meta, block_meta)`
  - `select_candidates(...)`
- Add config surface for mode selection (`off`, `dry_run`) and limits.

No behavior change:
- Attention still uses full block tables.

## Phase 2.2: Runtime sketch storage metadata

Deliverables:
- Integrate with existing scheduler lifecycle hooks to maintain request/block
  metadata index in live runtime.
- Add request/block cleanup on preempt/free/finish.
- Add minimal memory accounting and bounded retention.

No behavior change:
- Candidate sets are computed or prepared, but not used by kernels.

## Phase 2.3: Dry-run candidate selection (no behavior change)

Deliverables:
- At worker attention-prep or backend-pre-forward stage, obtain query metadata
  and compute candidate block ids using sketch backend.
- Record per step/layer/head:
  - selected candidates
  - full available blocks
  - recall-at-budget diagnostics (offline-style metrics online)

No behavior change:
- Backend continues full attention.

## Phase 2.4: Trace selected vs full-attention blocks

Deliverables:
- Add structured runtime trace export (bounded, optional).
- Correlate candidate sets with observed sequence lengths, layer/head, request
  status (prefill/decode), and cache mode (prefix/sliding window).

No behavior change:
- Tracing/observability only.

## Phase 3.0: Candidate-block attention path

Deliverables:
- Introduce candidate-aware attention execution path:
  - metadata carries selected block ids
  - backend/kernel executes exact attention over candidates (+ mandatory recent)
- Add robust fallback to full attention for unsupported shapes/modes.
- Validate throughput/latency, quality, and memory overhead.

Behavior change:
- Yes, controlled by config and guarded by fallback.

## Risk analysis

Performance overhead:
- Python-side per-layer/head scoring can erase gains if done naively.
- Mitigation: Phase 2 stays dry-run, bounded sampling, then move compute closer
  to backend/kernel for Phase 3.

FlashAttention/PagedAttention compatibility:
- Current backends assume standard block table/metadata invariants.
- Candidate filtering must preserve expected tensor shapes or provide a separate
  backend path.

Batching complexity:
- Requests in a batch have different seq lens, query lens, and block histories.
- Candidate sets must be per-request and compatible with batched kernels.

Prefix cache interaction:
- Prefix-cached blocks can be shared/hashed and ref-counted independently.
- Candidate logic must not violate cache ownership or eviction assumptions.

Sliding-window attention:
- Mandatory recent window must always be included.
- Candidate pruning cannot remove required local window blocks.

MQA/GQA:
- Fewer KV heads than Q heads changes head-wise sketch policy assumptions.
- Might require per-KV-head or shared-per-group policies rather than per-Q-head.

Layer/head-specific behavior:
- Offline sweeps already show heterogeneity.
- Runtime policy likely needs per-layer/head budgeting and thresholds.

CUDAGraph constraints:
- Dynamic candidate counts can conflict with graph capture/static-shape paths.
- Need fixed-capacity metadata buffers and padding conventions.

## Explicit non-goals (for Phase 2)

- No training or finetuning.
- No model architecture changes.
- No tokenizer changes.
- No semantic memory/retrieval system.
- No KV compression/offload/sketch-value compression changes yet.
- No attention math change in Phase 2 dry-run.

## Recommended implementation direction

- Phase 2.x: keep integration observational and metadata-only.
- First behavior-changing step should be backend/metadata gated prototype with
  strict fallback, not scheduler-level hard pruning.
- Treat scheduler as allocator/orchestrator, not attention policy engine.
