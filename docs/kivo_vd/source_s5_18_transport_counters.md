# Phase S5.18: Transport Counters

S5.17 added a real gated worker-to-core demotion transport path.

S5.18 adds lightweight counters so a pod sanity run can tell whether that path
actually observed live envelopes and core demotion-command application.

## What the counters measure

New module:

- `vllm/v1/core/kivo_demotion_counters.py`

Counters are disabled by default and enabled only with:

```bash
KIVO_KV_DEMOTION_COUNTERS_ENABLE=1
```

Tracked stages:

- worker command build
- worker output attachment
- scheduler-side receipt
- core transport batch intake
- core command attempt / accept / reject
- manager mark-demoted attempt / success / reject
- total demoted blocks marked

Safety counters that should remain zero:

- `req_to_blocks_removed`
- `free_to_pool_calls`

## What the counters do not prove

These counters do not prove:

- memory reduction
- latency improvement
- quality preservation
- live free-to-pool

They only show whether the gated metadata/control path was observed.

## Pod sanity interpretation

Interpret the result conservatively:

- generation works + counters object exists:
  runtime survived the gated path
- scheduler/core counters > 0:
  live transport was observed
- zero envelopes:
  not necessarily a failure on tiny prompts; there may have been no demotable
  rows or no emission under the chosen policy

## Tiny pod command

```bash
VLLM_TARGET_DEVICE=cuda \
KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ENABLE=1 \
KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ACTION=apply_block_table_only \
KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_POLICY=recent_only \
KIVO_KV_RUNTIME_BLOCK_TABLE_KEEP_RECENT_BLOCKS=4 \
KIVO_KV_RUNTIME_BLOCK_TABLE_MAX_FULL_BLOCKS=64 \
KIVO_KV_DEMOTION_TRANSPORT_ENABLE=1 \
KIVO_KV_DEMOTION_TRANSPORT_ACTION=apply_core_mark_demoted \
KIVO_KV_DEMOTION_COUNTERS_ENABLE=1 \
KIVO_KV_CORE_DEMOTION_ENABLE=1 \
KIVO_KV_CORE_DEMOTION_ACTION=mark_demoted_only \
PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.run_source_s5_18_transport_counters \
  --output /tmp/source_s5_18_transport_counters.json
```

Validate:

```bash
python -m scripts.kivo_vd.validate_source_s5_18_transport_counters \
  --input /tmp/source_s5_18_transport_counters.json
```

## Next step for S5.19

S5.19 should use these counters to distinguish:

- no transport because no demotable rows existed
- versus transport observed end-to-end on a prompt/policy combination that
  actually emits envelopes

That still does not authorize free-to-pool or memory claims.
