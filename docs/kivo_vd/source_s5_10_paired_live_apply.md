# Phase S5.10: Paired Live Apply

S5.10 is the first phase that pairs two pieces of Kivo-VD state in one local
runtime decision:

- worker-side filtered block-table row apply
- ownership-side live-block demotion intent

The goal is not to free live KV yet. The goal is to make the pairing explicit,
fail-closed, and reviewable before any real live-KV ownership mutation is
allowed.

## What S5.10 adds

S5.10 introduces a pure paired decision layer in:

- `vllm/v1/core/kivo_live_ownership_apply.py`

and wires it into the existing S5.9 pre-slot-mapping runtime helper in:

- `vllm/v1/worker/kivo_runtime_block_table_apply.py`

The paired decision checks whether all locally visible invariants line up:

- candidate demote ids were visible before filtering
- candidate demote ids are no longer visible after filtering
- protected/recent ids remain visible after filtering
- the filtered row is non-empty
- slot mapping refresh is guaranteed
- block-table apply happened when required
- a trustworthy ownership-side request/block mapping is available

## What S5.10 actually mutates

S5.10 may still apply the existing gated worker-side block-table-only filtered
row rewrite from S5.9 when explicitly enabled.

S5.10 does **not** mutate live ownership bookkeeping locally.

Specifically, S5.10 does **not**:

- mutate `req_to_blocks`
- mutate live core ownership state
- call a new live `BlockPool.free_blocks(...)` path
- mutate KV tensors

## Why ownership mutation remains disabled

The S5.9 hook point is inside the worker path immediately before
`compute_slot_mapping(...)` rebuilds slot mapping. That makes it the right place
for a worker row rewrite, but not yet a safe place to mutate core ownership.

At that point we do not have a narrow, proven bridge to the core
`SingleTypeKVCacheManager.req_to_blocks` ownership truth. Applying a worker row
rewrite without a synchronized core ownership mutation is acceptable only when
we keep ownership mutation disabled and report the blocker explicitly.

S5.10 therefore remains paired-plan only for the ownership side.

## Main blocker reasons now surfaced

When live paired apply is enabled, the new decision reports blocker reasons such
as:

- `ownership_mapping_unavailable`
- `ownership_mutation_not_enabled_locally`
- `block_table_apply_required`
- `slot_mapping_refresh_not_guaranteed`
- `candidate_demote_still_visible_after`
- `protected_ids_missing_after`
- `empty_visible_after_blocks`

This gives us an honest local readiness signal without pretending live KV free
is safe.

## Default behavior

Default vLLM behavior remains unchanged.

All S5.10 behavior is gated. If the live apply env flags are not enabled, the
new paired decision path stays inactive.

## Current conclusion

S5.10 successfully pairs:

- the original worker-visible row
- the filtered worker-visible row
- the candidate demote ids
- the protected ids
- the slot-mapping refresh guarantee
- the block-table apply result

But S5.10 still stops before live ownership mutation.

That is the correct fail-closed outcome for this phase.

## Remaining blocker before real live KV free

The remaining blocker is a narrow, synchronized bridge from the S5.9 worker
pre-slot-mapping hook to the core ownership truth in
`SingleTypeKVCacheManager.req_to_blocks`.

That future step must prove that:

- worker-visible filtered rows
- slot-mapping rebuild
- core ownership mutation
- any eventual live block free path

all stay synchronized in the same step.

Until that bridge exists, live KV free must remain disabled.
