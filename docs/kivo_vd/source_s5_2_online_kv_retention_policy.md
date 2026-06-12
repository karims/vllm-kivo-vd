# Phase S5.2: Online KV Retention Policy

Phase S5.2 adds a lightweight score bridge and a retention decision at the real
KV ownership boundary.

## Why this phase exists

Late freeing alone does not reduce peak memory. The decision about which
physical blocks should remain full KV must happen near allocation and
ownership, not only near attention metadata.

CountSketch is a selection policy, not the memory mechanism itself. It can
rank older blocks, but the actual memory effect only appears once the KV owner
decides which blocks stay full, which blocks are demoted, and which blocks can
be freed safely.

## What was added

- A bounded process-local block score store:
  - `vllm/v1/core/kivo_kv_block_score_store.py`
- A pure retention planner:
  - `vllm/v1/core/kivo_kv_retention_policy.py`
- A minimal manager hook:
  - `vllm/v1/core/single_type_kv_cache_manager.py::remove_skipped_blocks(...)`

## Policy behavior

- Default is disabled.
- Default action is `plan_only`.
- `recent_only` keeps recent blocks under a configured budget.
- `countsketch_online` keeps recent blocks and then uses scalar block scores
  for older blocks when available.
- Missing scores are handled conservatively.
- Unsupported direct mutation actions fail closed.

## Score bridge

When the existing S3.3B/S3.3C tensor observer is enabled and
`KIVO_KV_RETENTION_POLICY=countsketch_online`, scalar per-block scores can be
bridged into the process-local score store. This bridge stores only block ids
and float scores. It does not add JSONL-heavy logging, full tensor dumps, or
large CPU copies.

## What this still does not prove

- No measured memory reduction yet.
- No quality preservation claim.
- No active selected attention.
- No direct free/demotion of live blocks outside the existing removable path.

## Next step

S5.3 should validate whether a real free-or-demotion action can be applied
safely from this ownership boundary, with correctness checks around request
ownership and block-table consistency.

