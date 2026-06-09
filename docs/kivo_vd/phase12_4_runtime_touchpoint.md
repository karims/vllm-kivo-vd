# Phase 12.4: Runtime Touchpoint Helper

## Purpose

Phase 12.4 adds a no-op-by-default helper module that is safe for future vLLM
runtime code to call with copied metadata. It is still not wired into scheduler,
GPUModelRunner, block-table, slot-mapping, attention metadata, or kernels.

The helper gives Phase 12 a concrete runtime-facing API while preserving the
current behavior boundary:

- disabled unless `KIVO_PHASE12_SHADOW_ENABLED` is explicitly true;
- fail-closed on malformed metadata or hook initialization errors;
- no GPU tensors or vLLM runtime imports;
- no mutation of caller-owned block IDs or metadata;
- emitted events remain validator-compatible shadow events.

## Added Helper

The [`phase12_vllm_runtime_touchpoint.py`][touchpoint] module exposes:

- `is_phase12_shadow_enabled`;
- `get_phase12_shadow_hook`;
- `observe_phase12_decode_shadow_metadata`;
- `observe_phase12_block_table_shadow_metadata`.

When disabled, both observe helpers return a structured no-op result and write
no event file. When enabled, they delegate to the Phase 12.3 shadow hook and
ultimately to the passive observer.

## Runtime Files Modified

No core vLLM runtime files were modified in Phase 12.4.

The existing Kivo scheduler observer remains the only current vLLM-side Kivo
runtime integration. Phase 12.4 intentionally does not add another automatic
call until a specific call site is reviewed with runtime tests.

## Future Candidate Touchpoint

The safest future place is a bounded debug/observer call after request-visible
block metadata is known and before attention metadata is mutated. A future
call would pass copied logical block IDs and scalar metadata only:

```python
observe_phase12_block_table_shadow_metadata(
    request_id=request_id,
    layer_idx=layer_idx,
    context_token_count=context_token_count,
    total_context_blocks=total_context_blocks,
    block_ids=list(logical_block_ids),
)
```

This helper must not be called from an attention kernel path and must not alter
the normal block table, slot mapping, scheduling, or attention metadata.

## Smoke Workflow

Disabled:

```bash
.venv/bin/python scripts/kivo_vd/run_phase12_runtime_touchpoint_smoke.py \
  --num-events 4 \
  --output-jsonl \
    outputs/kivo_vd/phase12_runtime_touchpoint_disabled_events.jsonl \
  --output-md outputs/kivo_vd/phase12_runtime_touchpoint_disabled.md
```

Enabled:

```bash
.venv/bin/python scripts/kivo_vd/run_phase12_runtime_touchpoint_smoke.py \
  --enabled \
  --num-events 4 \
  --output-jsonl \
    outputs/kivo_vd/phase12_runtime_touchpoint_enabled_events.jsonl \
  --output-md outputs/kivo_vd/phase12_runtime_touchpoint_enabled.md
```

Validate enabled events:

```bash
.venv/bin/python scripts/kivo_vd/validate_phase12_shadow_event.py \
  --input outputs/kivo_vd/phase12_runtime_touchpoint_enabled_events.jsonl \
  --output-json outputs/kivo_vd/phase12_runtime_touchpoint_validation.json \
  --output-md outputs/kivo_vd/phase12_runtime_touchpoint_validation.md
```

## Safety Properties

- `shadow_only=true`;
- `active_routing=false`;
- `measured_runtime_reduction=false`;
- disabled mode writes no JSONL events;
- invalid metadata returns an error result instead of raising;
- no vLLM runtime import is required.

## Caveats

- Smoke inputs are synthetic.
- Logical block IDs are not physical KV block identities.
- No active selected-attention path exists.
- Full KV allocation and normal attention remain unchanged.
- No measured memory, latency, or generation-quality claim is made.

## Next Phase

Phase 12.5 can identify one reviewed runtime call site and define exact
metadata bounds and test gates. Active selected attention remains Phase 13.

[touchpoint]: ../../scripts/kivo_vd/phase12_vllm_runtime_touchpoint.py
