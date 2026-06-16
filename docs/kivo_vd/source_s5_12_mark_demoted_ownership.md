# Phase S5.12: Mark Demoted Ownership

S5.11 added the narrow ownership bridge API but still stopped at pure bridge
planning.

S5.12 is the first phase that actually mutates Kivo-side ownership bookkeeping
for live demotion candidates.

## What S5.12 adds

S5.12 adds Kivo demoted-block bookkeeping inside
`SingleTypeKVCacheManager`:

- `kivo_req_to_demoted_block_ids`
- `get_kivo_demoted_block_ids(request_id)`
- `mark_kivo_demoted_blocks_if_safe(...)`
- `clear_kivo_demoted_blocks(request_id)`

This bookkeeping tracks which live-owned blocks have become Kivo-demoted after
worker visibility and slot-mapping preconditions are proven.

## What S5.12 does not do

S5.12 does **not**:

- remove blocks from `req_to_blocks`
- free blocks to the pool
- call a new live `BlockPool.free_blocks(...)` path during demotion marking
- mutate KV tensors
- claim memory or latency reduction

`safe_to_free` remains `false`.

## Why this is safer than immediate free

The worker/runtime path and the core ownership path are still only partially
connected. Marking demoted ids in separate Kivo bookkeeping is much safer than
trying to immediately:

- rewrite core ownership arrays
- free physical blocks
- assume pool eligibility

This gives us an intermediate state that is locally testable and easy to clear
on request free.

## Current runtime boundary

The actual demotion marking helper now exists in core manager code, but the
worker-side S5.9 runtime hook still does not have a direct core KV manager
reference.

That boundary is reported explicitly as:

- `worker_path_lacks_core_kv_manager_reference`

So S5.12 implements manager-side marking, but does not wire runtime hot-path
mutation yet.

## Exact next step for S5.13

S5.13 should either:

- wire the existing runtime worker hook to the manager-side mark-demoted helper
  through a provably safe core manager reference path

or

- prepare the first gated free-to-pool phase only after that synchronized
  runtime-to-core mutation path is proven.
