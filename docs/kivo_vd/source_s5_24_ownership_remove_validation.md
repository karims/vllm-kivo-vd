# Phase S5.24: Ownership Remove Validation

Phase S5.23 proved that gated ownership removal could happen live:

- demoted block ids were marked in core
- those ids were removed from `req_to_blocks`
- at least one owned block remained
- no pool free occurred

Phase S5.24 hardens that path before any future free-to-pool step.

## Goal

Validate the ownership-removal invariants around:

```text
marked demoted blocks
-> removed from req_to_blocks
-> remaining owned blocks still non-empty
-> free_to_pool still disabled
```

## What S5.24 Adds

Manager-side invariant checks now confirm that successful removal keeps these
properties true:

- removed ids are a subset of marked demoted ids
- removed ids are a subset of currently owned ids
- removed ids are absent from the remaining owned ids
- remaining owned ids stay non-empty
- duplicate remaining ids are not introduced
- removed ids are cleared from demoted bookkeeping

Observed counters include:

- `ownership_remove_invariant_checked`
- `ownership_remove_invariant_failed`
- `ownership_removed_subset_of_marked`
- `ownership_removed_subset_of_owned`
- `ownership_removed_absent_after`
- `ownership_remaining_nonempty`
- `ownership_removed_reintroduced`
- `ownership_demoted_bookkeeping_cleared`
- `ownership_removed_blocks_total`
- `last_owned_before_remove_count`
- `last_owned_after_remove_count`
- `last_marked_demoted_before_remove_count`
- `last_removed_after_absent`

## Runner And Validator

S5.24 adds a thin wrapper around the existing transport probe so the output can
be phase-specific without changing the shared runtime path:

- `/Users/ksnaik/StudioProjects/vllm-kivo-vd/scripts/kivo_vd/run_source_s5_24_ownership_remove_validation.py`
- `/Users/ksnaik/StudioProjects/vllm-kivo-vd/scripts/kivo_vd/validate_source_s5_24_ownership_remove_validation.py`

The validator passes strict ownership-removal mode only when:

- generation succeeds
- transport is observed
- ownership removal succeeds at least once
- `req_to_blocks_removed > 0`
- `ownership_remaining_blocks_last > 0`
- `ownership_remove_invariant_failed == 0`
- `free_to_pool_calls == 0`

If ownership removal is not enabled, the validator can still pass in no-op mode
and reports `ownership_remove_observed=false`.

## Safety Boundary

S5.24 still does **not**:

- free blocks to the pool
- mutate KV tensors
- prove runtime memory reduction
- prove latency improvement
- prove quality preservation

## Next Phase

If pod validation keeps:

- `ownership_remove_succeeded > 0`
- `req_to_blocks_removed > 0`
- `ownership_remaining_blocks_last > 0`
- `ownership_remove_invariant_failed == 0`
- `free_to_pool_calls == 0`

then S5.25 can attempt the first gated free-to-pool step for already-removed
demoted blocks.
