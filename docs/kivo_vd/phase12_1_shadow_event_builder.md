# Phase 12.1: Shadow Event Builder

## Purpose

Phase 12.1 adds reusable, standalone construction utilities for the Phase 12
shadow-event contract and a deterministic synthetic generator. Later observer
work can reuse one builder instead of assembling event dictionaries ad hoc.

This phase does not import vLLM runtime modules and does not alter inference,
scheduling, KV allocation, block tables, slot mappings, or attention.

## Builder Utilities

[`phase12_shadow_events.py`](../../scripts/kivo_vd/phase12_shadow_events.py)
defines:

- `ShadowRatioPolicy`, a validated layer-to-ratio policy;
- `ShadowSelectionResult`, the two orderings and derived selection metrics;
- `Phase12ShadowEvent`, the serializable event model;
- ratio-policy parsing and layer-budget derivation;
- selected-ratio and theoretical reduction helpers;
- score summaries;
- ordering validation;
- `build_phase12_shadow_event`.

The default policy syntax is:

```text
balanced=0:0.60,5:0.45,8:0.45,11:0.60
```

Budgets use `ceil(total_context_blocks * layer_ratio)`, then apply configured
minimum, maximum, and available-block bounds.

## Ordering Invariant

Selector scores determine ranking:

```text
selected_block_ids_by_score = [19, 3, 11, 7]
```

Synthetic gather simulation restores original sequence order:

```text
selected_block_ids_for_gather = [3, 7, 11, 19]
```

The lists must contain the same unique IDs. Score order must never be treated
as K/V gather order.

## Synthetic Generator

Generate eight deterministic events:

```bash
.venv/bin/python \
  scripts/kivo_vd/generate_phase12_synthetic_shadow_events.py \
  --num-events 8 \
  --layers 0,5,8,11 \
  --context-blocks 32,64 \
  --output-jsonl \
    outputs/kivo_vd/phase12_synthetic_shadow_events.jsonl \
  --output-md outputs/kivo_vd/phase12_synthetic_shadow_events.md
```

The generator creates deterministic scalar scores, ranks all visible block
IDs, derives a ratio-based budget, and exports both score and gather ordering.
It does not read model tensors or a real KV cache.

## Validation

```bash
.venv/bin/python scripts/kivo_vd/validate_phase12_shadow_event.py \
  --input outputs/kivo_vd/phase12_synthetic_shadow_events.jsonl \
  --output-json \
    outputs/kivo_vd/phase12_synthetic_shadow_event_validation.json \
  --output-md \
    outputs/kivo_vd/phase12_synthetic_shadow_event_validation.md
```

Generated events retain the required safety statements:

- `shadow_only=true`;
- `active_routing=false`;
- `measured_runtime_reduction=false`.

## Caveats

- Scores and block choices are synthetic.
- Candidate reduction is a theoretical block ratio.
- Full vLLM KV allocation and normal attention remain unchanged.
- No runtime memory, latency, or generation-quality claim is made.

## Next Phase

Phase 12.2 can adapt these builders to bounded metadata from a passive runtime
observer. It must continue to discard every shadow decision and preserve the
normal attention path.
