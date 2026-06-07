# Kivo-VD Phase 8.1: Sketch Overhead Vs Theoretical Savings

Phase 8.1 compares compact sketch-buffer payload from Phase 8.0 with
theoretical skipped-KV bytes from Phase 7 dry-run events.

This is comparison and reporting only. Full KV remains allocated, attention is
unchanged, and no active routing exists.

## Purpose

Phase 8.0 answers how much additional memory a proposed per-block sketch buffer
requires. Phase 7.1 estimates how many KV payload bytes correspond to blocks
skipped by dry-run candidate decisions.

Phase 8.1 asks:

> How large is the proposed sketch-buffer overhead relative to the average
> theoretical skipped-KV payload?

It does not claim that skipped bytes were removed from GPU memory.

## Run The Comparison

```bash
RUN_DIR=outputs/kivo_vd/runs/phase7_gpt2_medium_memory_accounting
.venv/bin/python scripts/kivo_vd/compare_sketch_overhead_to_savings.py \
  --event-estimate "$RUN_DIR/kivo_event_memory_estimate.json" \
  --sketch-overhead \
    outputs/kivo_vd/phase8_0_gpt2_sketch_buffer_overhead.json \
  --memory-comparison "$RUN_DIR/memory_comparison.json" \
  --output-json \
    outputs/kivo_vd/phase8_1_gpt2_sketch_overhead_vs_savings.json \
  --output-md \
    outputs/kivo_vd/phase8_1_gpt2_sketch_overhead_vs_savings.md
```

## Comparison Metrics

For each sketch type and dimension:

```text
overhead_vs_avg_skipped_kv_ratio =
    theoretical_sketch_bytes / average_skipped_kv_bytes

net_theoretical_savings_bytes =
    average_skipped_kv_bytes - theoretical_sketch_bytes
```

The net value remains theoretical because the current runtime does not remove
or avoid full KV allocations.

## Affordability Heuristics

These thresholds are planning heuristics, not memory or quality claims:

| overhead relative to skipped KV | classification |
| --- | --- |
| 5% or less | excellent |
| above 5% through 15% | acceptable |
| above 15% through 30% | questionable |
| above 30% | poor |

The sketch buffer can describe a configured physical-block pool while skipped
KV is averaged per routing event. The resulting ratio is useful for planning,
but it is not an accounting identity.

## Recommendation Policy

For the first GPT-2 overhead work:

- use CountSketch dim `32` as a simple baseline;
- use Random Projection dim `32` as a second baseline;
- retain `bidiagonal_sign_subsample` dim `32` as an experimental structured
  candidate;
- keep SRHT reference/experimental only when explicitly present.

This policy does not authorize active routing.

## Caveats

- The comparison is theoretical only.
- Phase 8.0 measures additional overhead only.
- Sketch buffers do not replace full KV.
- No active routing is implemented.
- No measured runtime memory reduction is claimed.
- No latency or quality result follows from this report.

## Next Steps

Run Phase 8.0 on CUDA, compare measured allocator deltas with tensor payload,
and use this report to choose configurations for further overhead accounting.
Do not alter attention or KV allocation from this evidence alone.
