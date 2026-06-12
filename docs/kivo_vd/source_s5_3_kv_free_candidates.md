# Phase S5.3: KV Free Candidates

Phase S5.3 adds the first gated Kivo-controlled KV ownership mutation path.

## What changed

- The retention policy now accepts:
  - `KIVO_KV_RETENTION_ACTION=plan_only`
  - `KIVO_KV_RETENTION_ACTION=free_candidates`
- `vllm/v1/core/single_type_kv_cache_manager.py::remove_skipped_blocks(...)`
  now applies the retention decision only on the already-safe
  skipped/removable candidate path.
- A compact mutation summary is stored on the manager for local inspection and
  tests.

## Safety boundary

This phase only acts on blocks that vLLM had already identified as skipped and
removable through the existing ownership/free path.

It does not:

- evict arbitrary live blocks
- alter block-table semantics
- change attention-visible metadata
- mutate KV tensors directly
- claim memory reduction or quality preservation

If retention policy or action configuration is invalid, the Kivo-specific
retention layer fails closed and frees no Kivo-filtered candidates.

## Behavior

- Disabled or `plan_only`:
  - current/default behavior is unchanged
- `free_candidates + recent_only`:
  - protects recent request blocks and frees only older skipped candidates
- `free_candidates + countsketch_online`:
  - protects recent blocks
  - protects blocks with missing scores conservatively
  - frees only lower-scored skipped candidates that remain safe to remove

## What this still does not prove

- No measured memory reduction yet.
- No latency improvement claim.
- No quality preservation claim.
- No arbitrary online eviction of still-visible KV blocks.

## Next step

S5.4 should focus on block-table-consistent live-block demotion or equivalent
online retention mechanisms once attention/block-table consistency is handled.
