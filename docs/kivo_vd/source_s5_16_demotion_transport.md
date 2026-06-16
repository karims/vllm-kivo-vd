# Phase S5.16: Demotion Transport

S5.15 established the correct ownership boundary: core owns KV demotion
commands, not the worker.

S5.16 adds the smallest local/testable transport pieces around that boundary.

## What S5.16 adds

Worker-side export:

- `build_kivo_demotion_command_for_runtime_row(...)` in
  `vllm/v1/worker/kivo_runtime_block_table_apply.py`

Core-side intake:

- `vllm/v1/core/kivo_demotion_transport.py`

These let us:

- build a `KivoDemotionCommand` payload from a successful worker filtered-row
  apply state
- wrap it in a transport envelope
- apply it through `KVCacheManager.apply_kivo_demotion_command(...)`

## Does a safe worker-to-core runtime transport exist?

Not in the currently inspected live runtime path.

The source still does not show a small existing channel that carries custom
worker-produced command payloads back into scheduler/core in the same step.

So the current status is:

- worker command export: yes
- core command intake: yes
- live runtime transport: no

## Command payload shape

The worker export produces a core-owned payload only when:

- block-table apply succeeded
- slot mapping refresh is guaranteed
- candidate demote ids are non-empty
- candidate demote ids were visible before
- candidate demote ids are not visible after
- candidate demote ids do not overlap protected ids

Otherwise no command is emitted.

## Why free-to-pool remains disabled

S5.16 only connects payload shape and core intake logic locally.

It does **not**:

- remove from `req_to_blocks`
- free blocks to pool
- mutate KV tensors

## Exact next step for S5.17

S5.17 should either:

- add a real scheduler/core transport path for these envelopes

or

- conclude that the current runtime architecture requires a different core-side
  callback / control channel before demotion commands can be applied from live
  worker state.

Only after that transport exists should we consider any remove-from-
`req_to_blocks` or free-to-pool design.
