# Phase 12.2: Passive Shadow Observer

## Purpose

Phase 12.2 adds a pure-Python observer boundary that a future reviewed vLLM
hook could call with bounded scalar and block metadata. It does not install an
automatic hook and does not import vLLM runtime modules.

The observer computes a shadow selection, writes a validator-compatible JSONL
event when enabled, and discards the result. It cannot alter scheduling, KV
allocation, block tables, slot mappings, attention metadata, or kernels.

## Observer Configuration

[`phase12_shadow_observer.py`](../../scripts/kivo_vd/phase12_shadow_observer.py)
defines:

- `Phase12ShadowObserverConfig`;
- `Phase12ShadowObservation`;
- `Phase12ShadowObserver`.

The observer is disabled by default. Configuration includes output path,
layer-ratio policy, selector label, block size, budget bounds, and preview
behavior. Unsafe values are rejected:

- `shadow_only` must be `true`;
- `active_routing` must be `false`;
- `measured_runtime_reduction` must be `false`.

## Passive Inputs

A future hook may provide:

- request and sequence identity;
- layer and decode-step identity;
- context token and logical block counts;
- bounded logical block IDs;
- optional scalar block scores.

This scaffold treats supplied IDs as logical sequence-addressable IDs. It does
not model physical KV block reuse or read the real KV cache.

When scores are present, the observer selects the highest-scored IDs. When
scores are absent, it creates a deterministic placeholder ordering and marks
the event `preview_only=true`.

## Ordering Invariant

`selected_block_ids_by_score` retains selector rank. The event builder creates
`selected_block_ids_for_gather` in ascending logical sequence order. The two
lists must contain the same unique IDs.

Score order is never an attention or materialization order.

## Smoke Run

```bash
.venv/bin/python scripts/kivo_vd/run_phase12_shadow_observer_smoke.py \
  --num-events 8 \
  --layers 0,5,8,11 \
  --context-blocks 32,64 \
  --output-jsonl \
    outputs/kivo_vd/phase12_shadow_observer_smoke_events.jsonl \
  --output-md outputs/kivo_vd/phase12_shadow_observer_smoke.md
```

Validate the emitted events independently:

```bash
.venv/bin/python scripts/kivo_vd/validate_phase12_shadow_event.py \
  --input outputs/kivo_vd/phase12_shadow_observer_smoke_events.jsonl \
  --output-json outputs/kivo_vd/phase12_shadow_observer_validation.json \
  --output-md outputs/kivo_vd/phase12_shadow_observer_validation.md
```

## Future Hook Points

Potential later passive call sites remain:

- after request-visible block metadata is known;
- at a model-runner observation point where layer and query metadata exist;
- immediately before normal attention invocation for comparison only.

Any future hook must be separately reviewed, bounded, optional, and unable to
modify normal attention inputs.

## Caveats

- Smoke observations and scores are synthetic.
- No automatic vLLM runtime execution exists.
- Full KV allocation and normal attention remain unchanged.
- No measured memory, latency, or generation-quality claim is made.

## Next Phase

Phase 12.3 can design a narrowly scoped adapter that translates reviewed
runtime metadata into `Phase12ShadowObservation`. It must remain passive and
must not enable active routing.
