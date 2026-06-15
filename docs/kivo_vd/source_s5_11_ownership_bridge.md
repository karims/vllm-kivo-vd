# Phase S5.11: Ownership Bridge

S5.10 proved that the worker-side pre-slot-mapping hook can safely plan a
filtered block-table row rewrite and pair it with a live demotion decision.

The remaining gap was that the worker hook still could not safely reach core
ownership truth in `SingleTypeKVCacheManager.req_to_blocks`.

S5.11 adds the narrowest possible bridge API for that gap.

## What S5.11 adds

S5.11 introduces:

- `vllm/v1/core/kivo_ownership_bridge.py`

This module builds a fail-closed core ownership bridge decision using:

- `demote_block_ids`
- `visible_after_block_ids`
- `ownership_before_block_ids`
- `protected_block_ids`
- proof that worker block-table apply happened
- proof that slot mapping will be rebuilt

It also adds narrow manager helpers in:

- `vllm/v1/core/single_type_kv_cache_manager.py`

These helpers expose the current ownership-side block ids for one request and
build a bridge decision in core terms.

## What S5.11 checks

The ownership bridge validates that:

- demote ids are present in ownership-before
- demote ids are absent from worker-visible after-state
- worker-visible after-state is a subset of ownership-before
- protected ids are not demoted
- block-table apply is proven when required
- slot-mapping refresh is guaranteed when required

## Does S5.11 mutate ownership?

No.

S5.11 adds the bridge decision and the manager-side accessors, but it still
stops before mutating `req_to_blocks`.

`mark_demoted_if_safe` remains explicitly blocked with:

- `ownership_mark_demoted_not_implemented`

This is intentional. Removing or rewriting live ownership bookkeeping without a
proven synchronized runtime step would be riskier than useful here.

## Why free-to-pool is still disabled

S5.11 is only an ownership-bridge phase.

It does not:

- free live blocks to the pool
- call a new live `BlockPool.free_blocks(...)` path
- mutate KV tensors
- claim memory savings

`safe_to_free` remains `false` by design in this phase.

## Exact remaining blocker before S5.12

The remaining blocker is not the lack of a decision object anymore. It is the
lack of a locally proven mutation step that can update core ownership state in
sync with:

- the worker filtered block-table row
- the upcoming slot-mapping rebuild
- future pool-free eligibility

S5.12 should focus on the first gated free-to-pool path for already-demoted
ownership-side blocks, only after this synchronized mutation step is proven
safe.
