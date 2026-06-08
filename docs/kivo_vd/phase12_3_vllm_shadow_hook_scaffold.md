# Phase 12.3: vLLM Shadow Hook Scaffold

## Purpose

Phase 12.3 adds a tiny manual API between vLLM-shaped decode metadata and the
passive Phase 12 observer. The hook is disabled by default, imports no vLLM
runtime modules, and is not called automatically from any runtime path.

This scaffold makes a future call boundary explicit without changing
scheduling, KV allocation, block tables, slot mappings, attention metadata,
attention kernels, or model output.

## Environment Configuration

| variable | default |
| --- | --- |
| `KIVO_PHASE12_SHADOW_ENABLED` | `false` |
| `KIVO_PHASE12_SHADOW_OUTPUT` | `outputs/kivo_vd/phase12_vllm_shadow_events.jsonl` |
| `KIVO_PHASE12_RATIO_POLICY` | `balanced=0:0.60,5:0.45,8:0.45,11:0.60` |
| `KIVO_PHASE12_SELECTOR_POLICY` | `query_key_block_score` |
| `KIVO_PHASE12_BLOCK_SIZE` | `16` |
| `KIVO_PHASE12_MIN_BUDGET` | `1` |
| `KIVO_PHASE12_MAX_BUDGET` | unset |
| `KIVO_PHASE12_PREVIEW_ONLY` | `true` |

Boolean values accept `1/0`, `true/false`, `yes/no`, and `on/off`.
Malformed environment values produce a disabled, fail-closed hook.

## Hook API

[`phase12_vllm_shadow_hook.py`](../../scripts/kivo_vd/phase12_vllm_shadow_hook.py)
provides:

- `Phase12VllmShadowHookConfig`;
- `Phase12VllmShadowHook`;
- `build_config_from_env`;
- `maybe_get_shadow_hook_from_env`.

A possible future reviewed call site would look like:

```python
hook = maybe_get_shadow_hook_from_env()
result = hook.observe_decode_metadata(
    request_id=request_id,
    layer_idx=layer_idx,
    context_token_count=context_token_count,
    total_context_blocks=total_context_blocks,
    block_ids=logical_block_ids,
)
```

This example is not wired into vLLM. The return value reports whether the hook
was enabled, whether an event was written, warnings, errors, and a compact
event summary.

Score mappings are normalized against copied logical block IDs. Score-ranked
IDs remain separate from ascending sequence-order gather IDs.

## Fail-Closed Behavior

- Disabled hooks return `event_written=false` and write no file.
- Invalid environment configuration returns a disabled hook.
- Invalid observations are counted and return an error result.
- Hook calls do not raise into the caller by default.
- Caller-owned block IDs, scores, and metadata are not mutated.
- Every emitted event keeps `shadow_only=true`.
- Every emitted event keeps `active_routing=false`.
- Every emitted event keeps `measured_runtime_reduction=false`.

## Smoke Workflow

Disabled:

```bash
.venv/bin/python scripts/kivo_vd/run_phase12_vllm_shadow_hook_smoke.py \
  --num-events 4 \
  --output-jsonl \
    outputs/kivo_vd/phase12_vllm_shadow_hook_disabled_events.jsonl \
  --output-md outputs/kivo_vd/phase12_vllm_shadow_hook_disabled.md
```

Enabled:

```bash
.venv/bin/python scripts/kivo_vd/run_phase12_vllm_shadow_hook_smoke.py \
  --enabled \
  --num-events 4 \
  --output-jsonl \
    outputs/kivo_vd/phase12_vllm_shadow_hook_enabled_events.jsonl \
  --output-md outputs/kivo_vd/phase12_vllm_shadow_hook_enabled.md
```

Validate enabled events:

```bash
.venv/bin/python scripts/kivo_vd/validate_phase12_shadow_event.py \
  --input outputs/kivo_vd/phase12_vllm_shadow_hook_enabled_events.jsonl \
  --output-json outputs/kivo_vd/phase12_vllm_shadow_hook_validation.json \
  --output-md outputs/kivo_vd/phase12_vllm_shadow_hook_validation.md
```

## Caveats

- Smoke inputs are synthetic vLLM-like metadata.
- Logical IDs are not physical KV block identities.
- No automatic runtime hook exists.
- Full KV allocation and normal attention remain unchanged.
- No measured memory, latency, or generation-quality claim is made.

## Next Phase

Phase 12.4 may inspect and propose one bounded, opt-in runtime call site. Any
actual core-file edit requires a separate review and must preserve the normal
path exactly.
