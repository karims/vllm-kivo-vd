# Phase S5.26: Block Pool Accounting

Phase S5.25 proved that the first real gated pool-free path could be reached:

- ownership removal had already succeeded
- removed blocks were absent from request ownership
- `BlockPool.free_blocks(...)` was called under gating

Phase S5.26 adds observability around whether that free affected reusable block
capacity inside vLLM's block pool.

## Goal

Answer these questions without making a runtime memory claim:

- Did free-to-pool increase reusable block capacity?
- Can the pool's free-block counters be observed safely?
- Can we record CUDA allocator snapshots for context without overclaiming?

## What S5.26 Adds

Around the real gated `free_blocks` call, Kivo now snapshots:

- `block_pool_free_capacity_before`
- `block_pool_free_capacity_after`
- `block_pool_free_capacity_delta`
- `block_pool_num_free_blocks_before`
- `block_pool_num_free_blocks_after`
- `block_pool_num_free_blocks_delta`
- `block_pool_free_accounting_observed`
- `block_pool_free_accounting_increased`
- `block_pool_free_accounting_rejected`
- `block_pool_free_accounting_blocker_reasons`

The current implementation uses `BlockPool.get_num_free_blocks()` when
available. If that surface is unavailable or fails, accounting is reported as
unavailable with an explicit blocker.

## CUDA Snapshot Note

The runner can also record:

- `cuda_memory_allocated_before`
- `cuda_memory_allocated_after`
- `cuda_memory_reserved_before`
- `cuda_memory_reserved_after`

These are contextual only.

Returning blocks to the vLLM block pool is not the same thing as reducing
CUDA-reserved memory, so:

- `memory_claim_allowed` remains `false`
- `performance_claim_allowed` remains `false`
- `quality_claim_allowed` remains `false`

## Reuse Probe Note

This phase exposes reuse-probe fields but keeps the probe conservative by
default. The main success condition remains:

- generation succeeds
- free-to-pool succeeds
- block-pool accounting is either observed safely or rejected explicitly

## Next Step

- If accounting is observed cleanly, proceed to S5.27 reuse and quality sanity
  probing.
- If accounting is unavailable, proceed to S5.27 block-pool accounting API
  bridge work.
