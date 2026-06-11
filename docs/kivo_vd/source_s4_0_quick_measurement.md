# Kivo-VD Phase S4.0 Quick Measurement

Phase S4.0 is a small source-level measurement harness that compares:

- `baseline`
- `active_recent_window_attention_metadata`
- `active_sketch_kv_metadata_alias`

It measures wall-clock latency, tokens per second, output drift vs baseline,
and Kivo event counts from the source-level JSONL files already produced by the
earlier phases.

This phase does not claim memory savings, quality preservation, or final
selected-attention behavior.

## What the harness reports

The runner writes a JSON summary with:

- `passed`
- `total_prompts`
- `repeats`
- `warmup`
- `modes`
- `per_mode`
- `run_records`
- latency ratios between the three modes
- total output drift vs baseline
- `measured_runtime_reduction`
- the claim flags, all set conservatively to `false`

Each `per_mode` entry includes:

- success and failure counts
- mean, median, min, and max latency
- mean tokens per second
- generated token total
- output drift vs baseline
- event counts and blocker reasons

The recent-window mode summarizes S3.2B events. The sketch mode summarizes
both the S3.3C sketch plan events and the S3.3C metadata-alias events.

## Suggested command

```bash
python scripts/kivo_vd/run_source_s4_0_quick_measurement.py \
  --model gpt2 \
  --max-tokens 32 \
  --gpu-memory-utilization 0.10 \
  --max-model-len 512 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 1 \
  --seed 123 \
  --repeats 3 \
  --warmup 1 \
  --output-json outputs/kivo_vd/runs/source_s4_0_quick_measurement.json \
  --output-md outputs/kivo_vd/runs/source_s4_0_quick_measurement.md \
  --events-jsonl outputs/kivo_vd/runs/source_s4_0_quick_measurement_events.jsonl
```

## Validation

Validate a run with:

```bash
python scripts/kivo_vd/validate_source_s4_0_quick_measurement.py \
  --input-json outputs/kivo_vd/runs/source_s4_0_quick_measurement.json \
  --events-jsonl outputs/kivo_vd/runs/source_s4_0_quick_measurement_events.jsonl \
  --output-json outputs/kivo_vd/runs/source_s4_0_quick_measurement_validation.json \
  --output-md outputs/kivo_vd/runs/source_s4_0_quick_measurement_validation.md
```

## Interpretation

- If the baseline, recent-window, and sketch-active runs all succeed, the
  harness confirms that the source-level control path is healthy.
- If the active modes emit events, the harness confirms that the source hooks
  are still observable.
- If one active mode is faster than baseline, that is only a preliminary timing
  signal. It is not a memory-saving claim.
- `measured_runtime_reduction` only becomes true when all runs succeed and both
  active modes are at least 5% faster than baseline. Even then, this phase still
  does not claim memory reduction or production-ready selected attention.

## Boundary

- No scheduler behavior is changed.
- No attention kernels are changed.
- No KV cache allocation is changed.
- No memory reduction is claimed.
- No quality preservation is claimed.
- No selected-attention behavior is claimed.
