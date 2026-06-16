# Phase S5.14: Manager Handoff Discovery

S5.13 established that the worker-side demotion-mark adapter works when a safe
core KV manager object is explicitly provided.

The remaining question was whether the V1 runtime already exposes that manager
reference to the worker/model-runner path.

## Where the KV manager is created

The active KV manager is created in:

- `vllm/v1/core/sched/scheduler.py`

The scheduler constructs:

- `KVCacheManager`

inside its own core/scheduler path.

## Where the KV manager is owned

Ownership chain:

- `Scheduler` owns `KVCacheManager`
- `KVCacheManager` owns `KVCacheCoordinator`
- `KVCacheCoordinator` owns `single_type_managers`
- each element in `single_type_managers` is a `SingleTypeKVCacheManager`

That is where Kivo-side ownership bookkeeping lives.

## Whether worker/model runner has a reference

The worker runtime path does **not** appear to hold a direct reference to the
active scheduler/core `KVCacheManager`.

Observed source boundary:

- `GPUWorker.execute_model(...)` passes `scheduler_output` into
  `GPUModelRunner.execute_model(...)`
- `GPUModelRunner` uses `InputBatch`, block tables, slot mappings, and
  `SchedulerOutput`
- but it does not appear to own or receive the live core `KVCacheManager`

So the current S5.9 pre-slot-mapping hook has:

- worker-visible row state
- request ids
- slot-mapping rebuild timing

but not the core ownership manager object.

## Whether request ids align

Request ids do appear to align conceptually across both sides:

- worker-side `InputBatch.req_ids`
- core-side `KVCacheManager` / `SingleTypeKVCacheManager.req_to_blocks`

Both use the same request-id namespace.

That means an explicit bridge is plausible, but the reference handoff itself is
missing.

## Safe synchronous handoff status

A safe synchronous handoff is **not** currently present in the inspected worker
runtime path.

The best current conclusion is:

- safe manager reference found: `false`

## S5.14 adapter result

S5.14 adds a pure validation adapter:

- `vllm/v1/worker/kivo_kv_manager_handoff.py`

This adapter can verify whether a provided object:

- exists
- exposes `mark_kivo_demoted_blocks_if_safe(...)`
- appears request-id compatible

It does not mutate anything by itself.

## Recommended S5.15 path

S5.15 should not try to guess or reconstruct the active core manager from the
worker side.

Recommended next step:

1. add an explicit core-to-worker handoff path for the active KV manager, or a
   narrower callable bridge owned by core/scheduler
2. keep it opt-in and fail-closed
3. only then wire S5.13 runtime demotion marking into the real pre-slot-mapping
   path

If a direct object handoff is too invasive or process-boundary unsafe, S5.15
should instead design a core-side callback or command path that performs the
mark-demoted step inside the scheduler/core process.
