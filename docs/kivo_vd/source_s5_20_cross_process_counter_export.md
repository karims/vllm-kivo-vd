# Phase S5.20: Cross-Process Counter Export

S5.19 used longer prompts and more aggressive `recent_only` settings, but the
reported counters still remained zero on the pod.

The most likely explanation is process isolation:

- vLLM EngineCore work happens in a child process
- Kivo demotion counters were module-level state
- the parent runner read its own counters, not the child-process counters

S5.20 adds a small gated export file so child-process counters can be observed
from the parent runner.

## What changed

New behavior in `vllm/v1/core/kivo_demotion_counters.py`:

- `export_kivo_demotion_counters_snapshot_if_enabled(...)`

It is enabled only when both are true:

```bash
KIVO_KV_DEMOTION_COUNTERS_ENABLE=1
KIVO_KV_DEMOTION_COUNTERS_EXPORT_FILE=/tmp/kivo_demotion_counters.json
```

The exporter:

- writes a compact JSON snapshot
- includes `pid`
- uses atomic replace via `*.tmp.<pid>`
- overwrites the latest snapshot in place

## Why this solves the observability gap

The child EngineCore process can now export its own counter snapshot at
low-frequency hook points:

- worker envelope build / attach
- scheduler receive path
- core transport batch apply
- manager mark-demoted result

The parent runner then reads:

- `parent_counters`
- `exported_counters`

and prefers `exported_counters` when available.

## Extra diagnostic counters

S5.20 also adds:

- `block_table_apply_attempted`
- `block_table_apply_succeeded`
- `block_table_apply_rejected`
- `demotion_command_export_attempted`
- `demotion_command_export_rejected`

These help distinguish:

- hook not called
- hook called but no filtered row
- filtered row but command rejected
- command exported but not transported
- transported but core rejected

## What is still not claimed

S5.20 still does **not** claim:

- memory reduction
- free-to-pool
- `req_to_blocks` removal
- latency improvement
- quality preservation

Safety must remain:

- `req_to_blocks_removed == 0`
- `free_to_pool_calls == 0`

## Pod command

```bash
rm -f /tmp/kivo_demotion_counters.json /tmp/source_s5_19_demotable_transport_probe.json

VLLM_TARGET_DEVICE=cuda \
KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ENABLE=1 \
KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_ACTION=apply_block_table_only \
KIVO_KV_RUNTIME_BLOCK_TABLE_APPLY_POLICY=recent_only \
KIVO_KV_RUNTIME_BLOCK_TABLE_KEEP_RECENT_BLOCKS=1 \
KIVO_KV_RUNTIME_BLOCK_TABLE_MAX_FULL_BLOCKS=64 \
KIVO_KV_DEMOTION_TRANSPORT_ENABLE=1 \
KIVO_KV_DEMOTION_TRANSPORT_ACTION=apply_core_mark_demoted \
KIVO_KV_DEMOTION_COUNTERS_ENABLE=1 \
KIVO_KV_DEMOTION_COUNTERS_EXPORT_FILE=/tmp/kivo_demotion_counters.json \
KIVO_KV_CORE_DEMOTION_ENABLE=1 \
KIVO_KV_CORE_DEMOTION_ACTION=mark_demoted_only \
PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.run_source_s5_19_demotable_transport_probe \
  --prompt-repeats 50 \
  --output /tmp/source_s5_19_demotable_transport_probe.json
```

Validate:

```bash
PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.validate_source_s5_19_demotable_transport_probe \
  --input /tmp/source_s5_19_demotable_transport_probe.json
```

## Next step

- if transport is observed:
  `S5.21_gated_req_to_blocks_removal_design`
- if still not observed:
  inspect block-table apply and command-export reasons before touching
  ownership-removal behavior
