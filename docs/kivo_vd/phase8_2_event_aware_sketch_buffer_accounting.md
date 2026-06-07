# Kivo-VD Phase 8.2: Event-Aware Sketch-Buffer Accounting

Phase 8.2 models compact sketch-buffer overhead under several accounting scopes.
It follows Phase 8.1, whose pool-sized-buffer versus average-event comparison
was intentionally conservative.

This phase is accounting and reporting only. Full KV remains allocated,
attention is unchanged, and active routing remains out of scope.

## Accounting Models

### Global Pool

Compares the sketch allocation for the configured physical-block pool with the
theoretical full K+V payload for that pool.

### Average Per Event

Compares one global sketch allocation with average skipped-KV bytes in one
routing event. This is conservative and can look pessimistic.

### Cumulative Request

Compares one sketch allocation with skipped-KV opportunity accumulated across
all routing events. Exact per-event rows are summed when the event artifact is
complete. Otherwise, the script falls back to average skipped bytes multiplied
by the event count and emits a warning.

### Break Even

Reports:

```text
break_even_events =
    ceil(sketch_pool_bytes / average_skipped_kv_bytes)

break_even_skipped_blocks =
    ceil(sketch_pool_bytes / bytes_per_block)
```

These values estimate how many theoretical opportunities are needed to match
the additional sketch payload. They are not realized runtime savings.

## Run The Model

```bash
RUN_DIR=outputs/kivo_vd/runs/phase7_gpt2_medium_memory_accounting
.venv/bin/python scripts/kivo_vd/model_sketch_buffer_accounting.py \
  --event-estimate "$RUN_DIR/kivo_event_memory_estimate.json" \
  --sketch-overhead \
    outputs/kivo_vd/phase8_0_gpt2_sketch_buffer_overhead.json \
  --memory-comparison "$RUN_DIR/memory_comparison.json" \
  --output-json outputs/kivo_vd/phase8_2_gpt2_sketch_buffer_accounting.json \
  --output-md outputs/kivo_vd/phase8_2_gpt2_sketch_buffer_accounting.md
```

## Classifications

Cumulative overhead:

| ratio | classification |
| --- | --- |
| 5% or less | excellent |
| above 5% through 15% | acceptable |
| above 15% through 30% | questionable |
| above 30% | poor |

Break-even events:

| events | classification |
| --- | --- |
| 1 or less | immediate |
| 2 through 4 | fast |
| 5 through 16 | moderate |
| above 16 | slow |

These thresholds are research heuristics only.

## Recommendations

Prefer configurations with low cumulative overhead and fast break even:

- CountSketch dims `16` and `32`;
- Random Projection dims `16` and `32`;
- `bidiagonal_sign_subsample` dims `16` and `32` as experimental options.

Treat dim `64` as high-overhead/reference unless later evidence changes the
tradeoff. This recommendation does not authorize active routing.

## Caveats

- All savings accounting is theoretical only.
- Full KV is still allocated.
- No active routing is implemented.
- No measured runtime memory reduction is claimed.
- No latency or quality claim follows from this report.

## Next Steps

Run Phase 8.0 on CUDA, prefer exact uncapped per-event accounting where
available, and use the resulting overhead models to decide whether further
runtime accounting is justified. Keep attention and KV allocation unchanged.
