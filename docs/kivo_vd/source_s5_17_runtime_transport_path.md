# Phase S5.17: Runtime Transport Path

S5.16 established the payload boundary:

- worker can build a fail-closed demotion command candidate
- core can apply a demotion command through
  `KVCacheManager.apply_kivo_demotion_command(...)`

S5.17 answers the remaining runtime question: what object actually returns from
worker/model-runner back to scheduler/core, and can it safely carry a small
Kivo sidecar?

## Actual runtime path found

Inspected path:

- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/gpu_worker.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/outputs.py`

The live return path is:

1. `GPUModelRunner.execute_model(...)`
2. constructs `ModelRunnerOutput`
3. `GPUWorker.execute_model(...)` returns that object
4. `EngineCore.step()` receives it from `model_executor.execute_model(...)`
5. `Scheduler.update_from_output(...)` is the first core-side consumer

That path is real and already used for normal step outputs such as:

- `sampled_token_ids`
- `logprobs`
- `kv_connector_output`
- `routed_experts`

## What was added

Small backward-compatible sidecar:

- `vllm/v1/outputs.py`
  - `ModelRunnerOutput.kivo_demotion_transport_envelopes`

Worker-side attachment:

- `vllm/v1/worker/kivo_runtime_block_table_apply.py`
  - exports `KivoDemotionTransportEnvelope` objects only when:
    - transport is enabled
    - block-table apply succeeded
    - slot-mapping refresh is guaranteed
    - candidate demote ids are non-empty
    - local invariants hold
- `vllm/v1/worker/gpu_model_runner.py`
  - attaches exported envelopes from the last gated pre-slot-mapping Kivo
    summary to the current `ModelRunnerOutput`

Core-side intake:

- `vllm/v1/core/kivo_demotion_transport.py`
  - `apply_kivo_demotion_transport_from_model_runner_output(...)`
- `vllm/v1/core/sched/scheduler.py`
  - calls that helper at the start of `update_from_output(...)`

## Gating and default behavior

Transport stays off by default.

Required env gates:

```bash
KIVO_KV_DEMOTION_TRANSPORT_ENABLE=1
KIVO_KV_DEMOTION_TRANSPORT_ACTION=apply_core_mark_demoted
```

Core demotion application is still separately gated by the existing
core-demotion config.

Default behavior remains unchanged when transport is disabled:

- no worker payload is attached
- no core demotion command is applied
- no runtime output fields are used by default

## What this phase does not do

S5.17 does not:

- remove anything from `req_to_blocks`
- free blocks to pool
- mutate KV tensors
- claim memory reduction
- claim latency improvement

`KVCacheManager.apply_kivo_demotion_command(...)` remains the only core-owned
application point, and its current behavior still preserves:

- `removes_from_req_to_blocks = false`
- `frees_to_pool = false`

## Why this path is safe enough

Why this path is acceptable for a local gated transport phase:

- `ModelRunnerOutput` already crosses the worker-to-core boundary
- the added field is optional and defaults to empty
- the payload is pure Python metadata, not tensors
- async and sync model-runner paths both carry the same output object
- scheduler-side handling reuses the existing fail-closed transport helper

## Remaining boundary

This is a transport phase only.

It proves:

- a real worker-to-core return object exists
- a small optional Kivo sidecar can be carried on that object
- core-side demotion-command intake can be triggered from the live return path

It does not prove:

- live KV freeing
- active memory savings
- quality preservation

## Exact next step for S5.18

S5.18 is now worth a pod sanity pass for the gated transport path only:

- confirm envelopes appear on the live worker-to-core path
- confirm core intake remains fail-closed when disabled
- confirm no `req_to_blocks` removal occurs
- confirm free-to-pool remains disabled
