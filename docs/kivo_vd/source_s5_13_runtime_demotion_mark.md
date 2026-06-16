# Phase S5.13: Runtime Demotion Mark

S5.12 introduced manager-side Kivo demoted-block bookkeeping, but the S5.9
worker hook still had no way to safely call it.

S5.13 adds the first narrow runtime-to-core adapter for that path.

## What S5.13 adds

S5.13 introduces:

- `vllm/v1/worker/kivo_runtime_demotion_mark.py`

This adapter takes:

- request id
- demote block ids
- visible-after block ids
- block-table apply proof
- slot-mapping refresh proof
- an optional core manager reference

and attempts to call the manager-side:

- `mark_kivo_demoted_blocks_if_safe(...)`

only when all local invariants hold.

## Did runtime worker path find safe core manager access?

Not in the current default path.

`GPUModelRunner` and the S5.9 pre-slot-mapping helper still do not expose an
obvious safe `SingleTypeKVCacheManager` reference. So the integrated runtime
summary still fail-closes when no manager is provided.

The explicit blocker is now:

- `kv_cache_manager_unavailable`

This is slightly sharper than the earlier generic blocker because the adapter
itself now exists and can succeed if a safe manager reference is supplied.

## What S5.13 can do now

- Adapter-only runtime bridge: implemented
- Manager-side demotion marking: already implemented from S5.12
- Runtime marking with an injected safe manager object: testable and works
- Default worker hot path marking: still blocked by missing manager reference

## What S5.13 does not do

S5.13 does **not**:

- remove blocks from `req_to_blocks`
- free blocks to pool
- mutate KV tensors
- claim memory reduction or latency improvement

Free-to-pool remains disabled.

## Exact next step

S5.14 should focus on one of two things:

- find and prove a safe runtime path that can supply the active core KV manager
  to the S5.13 adapter

or

- if that reference is not safely reachable from worker runtime code, design
  the first explicit handoff path from scheduler/core into worker-side demotion
  marking before any free-to-pool experiment begins.
