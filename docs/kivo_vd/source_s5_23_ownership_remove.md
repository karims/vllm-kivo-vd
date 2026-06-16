# Phase S5.23: Ownership Remove

Phase S5.23 adds the first tightly gated ownership-side removal step after the
existing core `mark_demoted_only` path succeeds.

## Goal

Remove already-marked Kivo demoted block ids from
`SingleTypeKVCacheManager.req_to_blocks` without:

- freeing blocks back to the pool
- mutating KV tensors
- changing default behavior when the feature is disabled

## Gating

This phase remains opt-in and fail-closed behind:

```bash
KIVO_KV_CORE_DEMOTION_ENABLE=1
KIVO_KV_CORE_DEMOTION_ACTION=mark_demoted_only
KIVO_KV_OWNERSHIP_REMOVE_ENABLE=1
KIVO_KV_OWNERSHIP_REMOVE_ACTION=remove_marked_demoted_only
```

If any requirement is missing, the manager records blocker reasons and leaves
ownership unchanged.

## What Changed

Primary implementation path:

- `/Users/ksnaik/StudioProjects/vllm-kivo-vd/vllm/v1/core/single_type_kv_cache_manager.py`
- `/Users/ksnaik/StudioProjects/vllm-kivo-vd/vllm/v1/core/kivo_demotion_command.py`
- `/Users/ksnaik/StudioProjects/vllm-kivo-vd/vllm/v1/core/kivo_demotion_counters.py`
- `/Users/ksnaik/StudioProjects/vllm-kivo-vd/scripts/kivo_vd/run_source_s5_19_demotable_transport_probe.py`
- `/Users/ksnaik/StudioProjects/vllm-kivo-vd/scripts/kivo_vd/validate_source_s5_19_demotable_transport_probe.py`

New manager behavior:

1. Core demotion command still marks demoted ids first.
2. Only if ownership removal is enabled, the manager tries to remove those
   already-marked ids from `req_to_blocks`.
3. Removal succeeds only when:
   - the request ownership mapping exists
   - marked demoted ids are still owned by that request
   - at least one owned non-null block remains after removal
4. No `BlockPool.free_blocks(...)` call is made.

## Fail-Closed Cases

Ownership removal is rejected when:

- the request mapping is missing
- no marked demoted ids exist
- marked demoted ids are stale and no longer present in `req_to_blocks`
- removal would empty the owned non-null block set
- the action string is invalid or the feature is disabled

## Observability

The probe summary now records ownership-removal counters and samples:

- `ownership_remove_attempted`
- `ownership_remove_succeeded`
- `ownership_remove_rejected`
- `ownership_removed_blocks`
- `ownership_remaining_blocks_last`
- `last_removed_block_ids_sample`
- `last_remaining_block_ids_sample`
- `req_to_blocks_removed`
- `free_to_pool_calls`

`free_to_pool_calls` must remain `0` in this phase.

## Interpretation

S5.23 is an ownership bookkeeping experiment only.

It does **not** prove:

- KV memory reduction
- pool free safety
- latency improvement
- quality preservation
- production-ready selected attention

If pod validation shows `demoted_blocks_marked > 0`,
`ownership_remove_succeeded > 0`, and `free_to_pool_calls == 0`, then the next
phase can inspect whether later runtime metadata still behaves coherently after
ownership-only removal.
