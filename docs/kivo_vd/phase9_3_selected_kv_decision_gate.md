# Kivo-VD Phase 9.3: Selected-KV Decision Gate

Phase 9 tests whether dry-run-selected KV subsets can be materialized into
temporary synthetic tensors. It remains entirely outside attention.

Phase 9 does not reduce actual runtime memory, access the real vLLM KV cache,
or change model output.

## Required Artifacts

Before Phase 10, the Phase 9 evidence bundle should contain:

- `selected_kv_materialization.json`
- `selected_kv_materialization.md`
- `selected_kv_materialization_comparison.json`
- `selected_kv_materialization_comparison.md`
- `pipeline_summary.json`
- the Phase 7 event estimate JSON;
- Phase 8 event-aware sketch accounting when available.

## Decision Criteria

The gate requires:

- a successful, non-dry-run Phase 9.2 pipeline;
- all Phase 9.2 stages succeeded;
- processed materialization events;
- nonzero average selected-block count;
- observed copy time;
- an average materialization ratio below `1.0`;
- no warning that selected block IDs are missing;
- preserved synthetic, outside-attention, full-KV, no-routing,
  no-measured-reduction, and quality-unmeasured caveats.

Preview-only block IDs remain a warning because they undercount the complete
candidate payload. The gate now fails closed when any materialization row is
preview-only. Complete IDs are required before Phase 10 planning.

## Materialization Ratio Heuristics

| average materialization ratio | classification |
| --- | --- |
| `0.80` or above | weak signal |
| `0.50` to below `0.80` | moderate signal |
| `0.25` to below `0.50` | promising |
| below `0.25` | strong materialization compression signal |

Copy time has no hard pass/fail threshold. It is classified as observed only.
Repeated-run validation is required before drawing performance conclusions.

## Run The Gate

```bash
RUN_DIR=outputs/kivo_vd/runs/phase9_gpt2_selected_kv_materialization
.venv/bin/python scripts/kivo_vd/check_phase9_readiness.py \
  --pipeline-summary "$RUN_DIR/pipeline_summary.json" \
  --materialization "$RUN_DIR/selected_kv_materialization.json" \
  --comparison \
    "$RUN_DIR/selected_kv_materialization_comparison.json" \
  --event-estimate \
    outputs/kivo_vd/runs/phase7_gpt2_medium_memory_accounting/\
kivo_event_memory_estimate.json \
  --sketch-accounting \
    outputs/kivo_vd/runs/phase8_gpt2_sketch_buffer_accounting/\
event_aware_sketch_buffer_accounting.json \
  --output-json "$RUN_DIR/phase9_readiness.json" \
  --output-md "$RUN_DIR/phase9_readiness.md"
```

The helper can infer Phase 7 and Phase 8 paths recorded in
`pipeline_summary.json` when explicit flags are omitted.

Before running the gate, regenerate Phase 7 with
`--export-full-block-ids`, then rerun the Phase 9.2 pipeline against those
events. A clean readiness result requires
`preview_only_event_count: 0`.

## RunPod Readiness Result

The full-ID L40S evidence bundle passed the gate:

| field | result |
| --- | --- |
| Phase 10 ready | `true` |
| Materialization classification | `promising` |
| Full-ID event count | `32` |
| Preview-only event count | `0` |
| Warnings | none |
| Synthetic KV | `true` |
| Outside attention path | `true` |
| Full KV still allocated | `true` |
| Active routing | `false` |
| Measured runtime reduction | `false` |
| Quality measured | `false` |

The allowed scope is limited to standalone selected-KV torch
reference-attention experiments on synthetic tensors outside vLLM. Passing
this gate does not authorize real vLLM selected attention, block-table or
slot-mapping changes, real KV deallocation, or active routing.

## Allowed Phase 10 Scope

A passing gate authorizes only:

1. a tiny standalone attention-equivalence prototype using synthetic Q/K/V;
2. selected-KV torch reference attention outside vLLM;
3. selected-versus-full synthetic-attention output comparison;
4. only later, consideration of an isolated vLLM-adjacent prototype.

## Initially Out Of Scope

- Block-table or slot-mapping mutation.
- Scheduler behavior changes.
- Production attention-kernel changes.
- Selected-KV attention inside real vLLM.
- Measured memory-reduction, latency, or quality claims.

## Current Claim

Kivo-VD can currently dry-run selected block decisions, estimate active-KV
savings, account for compact sketch-buffer overhead, and materialize selected
KV subsets into synthetic temporary buffers outside attention. It has not
demonstrated active attention routing, measured memory reduction, or quality
preservation.

Phase 9 closes with this conservative gate. Passing it changes only the
allowed standalone research experiment, not vLLM runtime behavior.

The validated full-ID result completes Phase 9. No measured runtime memory
reduction, latency improvement, or quality preservation has been
demonstrated.
