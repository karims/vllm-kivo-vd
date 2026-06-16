# Phase S5.19: Demotable Transport Probe

S5.18 added lightweight transport counters and validated that a tiny pod run
did not crash. All counters stayed at zero.

That is not a code failure. It most likely means:

- the tiny prompts did not span enough blocks to produce a demotable row, or
- the runtime policy did not emit a demotion envelope under those settings

S5.19 adds a longer-prompt probe to make the existing gated transport path more
observable without changing runtime behavior.

## What changed

New runner:

- `scripts/kivo_vd/run_source_s5_19_demotable_transport_probe.py`

New validator:

- `scripts/kivo_vd/validate_source_s5_19_demotable_transport_probe.py`

The runner uses:

- `facebook/opt-125m`
- long repeated prompts
- eager execution
- conservative GPU utilization
- existing env-gated Kivo runtime block-table + transport + core demotion flags

## Why longer prompts

The goal is to increase the chance of:

- multiple visible KV blocks
- a non-empty removable suffix under `recent_only`
- worker envelope emission
- scheduler/core demotion transport observation

Recommended pod setting:

```bash
KIVO_KV_RUNTIME_BLOCK_TABLE_KEEP_RECENT_BLOCKS=1
```

That is intentionally aggressive for observability, not for quality.

## What counts as observed transport

The probe reports:

- `worker_envelope_observed`
- `scheduler_envelope_observed`
- `core_command_observed`
- `manager_mark_demoted_observed`

If any of those are true, then:

- `transport_observed=true`

## What still is not claimed

S5.19 still does **not** claim:

- memory reduction
- free-to-pool
- `req_to_blocks` removal
- quality preservation
- latency improvement

Safety must remain:

- `req_to_blocks_removed == 0`
- `free_to_pool_calls == 0`

## Pod command

```bash
VLLM_TARGET_DEVICE=cuda \
KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ENABLE=1 \
KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ACTION=apply_block_table_only \
KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_POLICY=recent_only \
KIVO_KV_RUNTIME_BLOCK_TABLE_KEEP_RECENT_BLOCKS=1 \
KIVO_KV_RUNTIME_BLOCK_TABLE_MAX_FULL_BLOCKS=64 \
KIVO_KV_DEMOTION_TRANSPORT_ENABLE=1 \
KIVO_KV_DEMOTION_TRANSPORT_ACTION=apply_core_mark_demoted \
KIVO_KV_DEMOTION_COUNTERS_ENABLE=1 \
KIVO_KV_CORE_DEMOTION_ENABLE=1 \
KIVO_KV_CORE_DEMOTION_ACTION=mark_demoted_only \
PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.run_source_s5_19_demotable_transport_probe \
  --output /tmp/source_s5_19_demotable_transport_probe.json
```

Validate:

```bash
PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.validate_source_s5_19_demotable_transport_probe \
  --input /tmp/source_s5_19_demotable_transport_probe.json
```

## Next step

- if observed:
  `S5.20_gated_req_to_blocks_removal_design`
- if not observed:
  inspect policy emission and block-count thresholds before touching ownership
  or freeing behavior
