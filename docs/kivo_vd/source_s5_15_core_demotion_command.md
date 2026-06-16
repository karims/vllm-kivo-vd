# Phase S5.15: Core Demotion Command

S5.14 confirmed that the worker/runtime path does not own the active core KV
manager.

That means the worker should not mutate KV ownership directly. Core must own
that mutation.

## What S5.15 adds

S5.15 introduces a core-owned command path in:

- `vllm/v1/core/kivo_demotion_command.py`

and core-side application APIs in:

- `vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/v1/core/kv_cache_coordinator.py`
- `vllm/v1/core/kv_cache_manager.py`

The public gated entry point is:

- `KVCacheManager.apply_kivo_demotion_command(...)`

## Why worker cannot mutate core ownership directly

The worker pre-slot-mapping hook sees:

- worker-visible filtered rows
- request ids
- slot-mapping timing

But it does not hold the scheduler-owned `KVCacheManager` object.

Core ownership lives in:

- `Scheduler`
- `KVCacheManager`
- `KVCacheCoordinator`
- `SingleTypeKVCacheManager.req_to_blocks`

So the mutation path has to be core-owned.

## What the command does

The command validates and applies only:

- mark-demoted bookkeeping through S5.12 manager helpers

It does **not**:

- remove blocks from `req_to_blocks`
- free blocks to pool
- mutate KV tensors

## Where the command API lives

Ownership-layer command application happens in this order:

- `KVCacheManager.apply_kivo_demotion_command(...)`
- `KVCacheCoordinator.apply_kivo_demotion_command(...)`
- `SingleTypeKVCacheManager.apply_kivo_demotion_command(...)`

The current coordinator path is intentionally conservative:

- if there is not exactly one `single_type_manager`, it fails closed with
  `ambiguous_single_type_manager_count`

This keeps multi-group ambiguity out of scope for now.

## Does Scheduler runtime transport exist yet?

Not yet.

S5.15 adds the core-owned command API, but it does not add worker-to-scheduler
transport for real runtime use.

That means the current state is:

- core command API: yes
- scheduler runtime transport: no
- free-to-pool: no

## Exact next step for S5.16

S5.16 should connect the worker-side filtered-row summary to this core command
API through an explicit scheduler/core transport path, or design that transport
if it cannot be added safely yet.

Only after that transport exists should we revisit any remove-from-
`req_to_blocks` or free-to-pool step.
