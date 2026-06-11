# Kivo-VD Phase S4.1 Low-Overhead Measurement

Phase S4.1 separates two costs that were mixed together in earlier source-level
measurements:

- verbose JSONL recording overhead
- counters-only active-path overhead

It compares five modes:

- `baseline`
- `recent_window_verbose`
- `recent_window_counters`
- `sketch_active_verbose`
- `sketch_active_counters`

The source hooks use `KIVO_SOURCE_RECORD_MODE`:

- `events` writes verbose JSONL records
- `counters` skips JSONL serialization and keeps process-local aggregate
  counters in memory
- `off` disables recording

The counters path still exercises the active source behavior. It only removes
the verbose event payloads and sketch samples from the JSONL path.

This phase does not claim memory savings, quality preservation, or selected
attention behavior.

## What the harness reports

The runner writes a JSON summary with:

- `passed`
- `total_prompts`
- `repeats`
- `warmup`
- `modes`
- `per_mode`
- verbose/counters latency ratios
- logging-overhead improvement estimates
- `measured_runtime_reduction`
- claim flags, all kept conservative

Each `per_mode` entry includes:

- success and failure counts
- mean, median, min, and max latency
- mean tokens per second
- generated token total
- output drift vs baseline
- `verbose_event_record_count`
- `counter_event_count`
- verbose event summaries
- counter summaries
- blocker reasons and maxima fields

Verbose recent-window mode summarizes the S3.2B JSONL records. Verbose
sketch-active mode summarizes both the S3.3C sketch-plan records and the S3.3C
metadata-alias records. Counters modes keep the same active path but omit the
verbose JSONL writes and sketch sample payloads.

## Suggested command

```bash
python scripts/kivo_vd/run_source_s4_1_low_overhead_measurement.py \
  --model gpt2 \
  --max-tokens 32 \
  --gpu-memory-utilization 0.10 \
  --max-model-len 768 \
  --max-num-batched-tokens 768 \
  --max-num-seqs 1 \
  --seed 123 \
  --repeats 3 \
  --warmup 1 \
  --output-json outputs/kivo_vd/runs/source_s4_1_low_overhead_measurement.json \
  --output-md outputs/kivo_vd/runs/source_s4_1_low_overhead_measurement.md \
  --events-jsonl outputs/kivo_vd/runs/source_s4_1_low_overhead_measurement_events.jsonl
```

The runner defaults to in-process engine-core execution so the counters remain
visible to the same Python process. That is important because the counters are
process-local.

## Validation

Validate a run with:

```bash
python scripts/kivo_vd/validate_source_s4_1_low_overhead_measurement.py \
  --input-json outputs/kivo_vd/runs/source_s4_1_low_overhead_measurement.json \
  --output-json outputs/kivo_vd/runs/source_s4_1_low_overhead_measurement_validation.json \
  --output-md outputs/kivo_vd/runs/source_s4_1_low_overhead_measurement_validation.md
```

## Interpretation

- If verbose mode is slower than counters mode, the difference is likely JSONL
  serialization and sample capture overhead.
- If counters mode is still slow, the bottleneck is probably the source-level
  sketch/control work itself rather than verbose logging.
- If counters mode is only slightly faster, then the active-path bookkeeping is
  already a meaningful cost and future work should focus on GPU-native
  integration.
- `measured_runtime_reduction` only becomes true when all runs succeed and both
  counters modes are at least 5% faster than baseline. Even then, this phase
  still does not claim memory reduction or production-ready selected attention.

## Boundary

- No scheduler behavior is changed.
- No attention kernels are changed.
- No KV cache allocation is changed.
- No memory reduction is claimed.
- No quality preservation is claimed.
- No selected-attention behavior is claimed.

