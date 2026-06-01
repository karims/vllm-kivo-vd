# Kivo-VD Phase 1.0: Sketch Index Interfaces

Phase 1.0 adds Python-side interfaces and in-memory metadata structures only.

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

## Explicit non-goals for Phase 1.0

- No real sketch computation is performed.
- No key/value tensor access is performed.
- No scheduler decision changes are made.
- No CUDA/Triton/kernel/attention/GPUModelRunner changes are made.
- No model architecture changes are made.

## Placeholder behavior details

- `score_blocks_placeholder(...)` returns deterministic dummy scores derived
  from block metadata (IDs and logical indices only).
- `route_blocks_placeholder(...)` combines:
  - recent block preference (`recent_window_blocks`)
  - top dummy scores
- This is metadata-only plumbing for interface validation.

## Why this index is separate from the observer

- The observer captures lifecycle events for instrumentation and validation.
- The sketch index models future retrieval/routing state and contracts.
- Keeping them separate allows:
  - independent testing
  - low-risk iteration on data model and API
  - future wiring of real sketches without changing scheduler behavior first.
