# Kivo-VD Phase 1.0 / 1.1: Sketch Index Interfaces and Metadata Population

Phase 1.0 adds Python-side interfaces and in-memory metadata structures only.
Phase 1.1 populates that metadata index from scheduler lifecycle hooks.

## Scope in this phase

- Added `vllm/v1/core/kivo_vd_sketch.py` with:
  - `KivoVDSketchConfig`
  - `KivoVDSketchType`
  - `KivoVDBlockSketch`
  - `KivoVDBlockScore`
  - `KivoVDRoutingDecision`
  - `KivoVDSketchIndex`
- Added placeholder index methods:
  - `add_or_update_block_sketch(...)`
  - `remove_request(request_id)`
  - `get_request_block_sketches(request_id)`
  - `score_blocks_placeholder(...)`
  - `route_blocks_placeholder(...)`
  - `reset()`
- Added unit tests for index behavior under `tests/v1/core/test_kivo_vd_sketch.py`.
- Added observer-side metadata population from allocation/free hooks:
  - `on_after_allocate_slots(...)` upserts `KivoVDBlockSketch` records
  - `on_free_request(...)` removes request records from the sketch index
  - population is metadata-only (block ids/group ids/logical order/source/tokens)

## Explicit non-goals for Phase 1.0

- No real sketch computation is performed.
- No key/value tensor access is performed.
- No scheduler decision changes are made.
- No CUDA/Triton/kernel/attention/GPUModelRunner changes are made.
- No model architecture changes are made.
- No routing/scheduling behavior changes are introduced in Phase 1.1.

## Placeholder behavior details

- `score_blocks_placeholder(...)` returns deterministic dummy scores derived
  from block metadata (IDs and logical indices only).
- `route_blocks_placeholder(...)` combines:
  - recent block preference (`recent_window_blocks`)
  - top dummy scores
- This is metadata-only plumbing for interface validation.

## Phase 1.1 lifecycle population notes

- Hook source: scheduler allocation/free observer calls (running/waiting/preempt/free).
- Available metadata at hook point:
  - request id
  - per-group block id lists from `new_blocks.get_block_ids(allow_none=True)`
  - source path (`running`, `waiting`, `preempt`, `free_blocks`, ...)
  - `num_new_tokens` where provided by scheduler
- Logical block index:
  - derived from block position within each per-group list at the hook point.
- Missing (not captured in this phase):
  - real sketch vectors
  - KV tensor data
  - layer-level exact mapping
  - explicit “only newly allocated physical ids” marker from allocator internals.

If exact new-vs-existing block identity needs stronger guarantees later, the
safest future hook point is a dedicated allocator return payload carrying
explicitly new block ids per request/group.

## Why this index is separate from the observer

- The observer captures lifecycle events for instrumentation and validation.
- The sketch index models future retrieval/routing state and contracts.
- Keeping them separate allows:
  - independent testing
  - low-risk iteration on data model and API
  - future wiring of real sketches without changing scheduler behavior first.
