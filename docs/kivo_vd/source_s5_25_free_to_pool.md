# Phase S5.25: Free To Pool

Phase S5.24 validated the ownership-removal invariants:

- demoted ids were marked first
- those ids were removed from `req_to_blocks`
- removed ids were absent afterward
- remaining ownership stayed non-empty
- free-to-pool was still disabled

Phase S5.25 adds the first strictly gated attempt to return those already
removed blocks back to the block pool.

## Goal

Attempt a real `BlockPool.free_blocks(...)` call only for blocks that were
already:

1. marked demoted
2. removed from request ownership
3. validated absent from remaining request ownership

## Safety Rules

The manager-side free helper fails closed unless all of these hold:

- free-to-pool is explicitly enabled
- the action is `free_removed_demoted_only`
- removed block objects are available, not just ids
- none of those blocks are still owned by the same request
- none of those blocks are owned by another active request
- none of those blocks were already freed earlier

The helper never mutates KV tensors directly.

## Observability

New counters include:

- `free_to_pool_attempted`
- `free_to_pool_succeeded`
- `free_to_pool_rejected`
- `free_to_pool_blocks`
- `free_to_pool_double_free_prevented`
- `free_to_pool_calls`
- `last_freed_block_ids_sample`
- `last_free_rejected_block_ids_sample`

`free_to_pool_calls` increments only when a real `BlockPool.free_blocks(...)`
call happens.

## Interpretation

S5.25 is still not a runtime memory claim.

Even if pool free succeeds, this phase does **not** prove:

- measured GPU memory reduction
- latency improvement
- quality preservation
- production-ready selected attention

It only proves whether a first ownership-safe pool-free call can be made after
S5.24 removal.

## Next Step

- If pool free succeeds safely, the next step is S5.26 block-pool and memory
  counters.
- If pool free fails closed, the next step is S5.26 analysis of the exact
  blocker path.
