# Kivo-VD Phase 9.1: Selected-KV Materialization Comparison

Phase 9.0 measures synthetic selected-KV gather/copy payload and timing outside
attention. Phase 9.1 places that microbenchmark beside the Phase 7 theoretical
KV opportunity and optional Phase 8 sketch-buffer overhead.

This is comparison and reporting only. It does not access real vLLM KV or
change allocation, scheduling, block tables, or attention.

## Compared Artifacts

The report combines:

1. Phase 9.0 synthetic selected-KV materialization;
2. Phase 7 event-based active/skipped KV accounting;
3. optional Phase 8 event-aware sketch-buffer accounting.

It reports:

- selected payload versus full considered KV;
- selected payload versus skipped-KV opportunity;
- cumulative selected payload versus cumulative skipped KV;
- rough synchronized copy throughput;
- selected payload plus each named sketch-buffer configuration.

## Run The Comparison

```bash
.venv/bin/python \
  scripts/kivo_vd/compare_selected_kv_materialization.py \
  --materialization \
    outputs/kivo_vd/phase9_0_gpt2_selected_kv_materialization.json \
  --event-estimate \
    outputs/kivo_vd/runs/phase7_gpt2_medium_memory_accounting/\
kivo_event_memory_estimate.json \
  --sketch-accounting \
    outputs/kivo_vd/runs/phase8_gpt2_sketch_buffer_accounting/\
event_aware_sketch_buffer_accounting.json \
  --output-json \
    outputs/kivo_vd/\
phase9_1_gpt2_selected_kv_materialization_comparison.json \
  --output-md \
    outputs/kivo_vd/\
phase9_1_gpt2_selected_kv_materialization_comparison.md
```

The sketch artifact is optional. Without it, materialization and Phase 7
comparisons still run.

## Ratio Interpretation

`selected_vs_full_considered_ratio` compares average copied selected bytes with
average selected-plus-skipped bytes represented by Phase 9.0 rows.

`selected_vs_skipped_ratio` compares average copied selected bytes with the
Phase 7 average theoretical skipped-KV opportunity.

`cumulative_selected_vs_cumulative_skipped_ratio` compares all selected bytes
copied by Phase 9.0 with cumulative Phase 7 skipped bytes. Exact Phase 7
per-event rows are summed when complete; otherwise the report uses average
skipped bytes times the routing-event count and warns.

The sketch table adds one configured global sketch-pool payload to average
selected bytes. It is configuration-specific and remains additional overhead.

## Copy Timing

Rough throughput is:

```text
average selected KV bytes / average copy time
```

This is a synchronized synthetic gather microbenchmark. It is not an
end-to-end latency measurement and does not predict candidate-attention speed.

## Preview-Only Limitation

Current observer exports may contain only eight preview block IDs. Phase 9.0
marks those rows as preview-only and copies only exported IDs.

Phase 9.1 carries that warning forward and will not recommend a strong Phase
9.2 repeated-run conclusion until complete selected block IDs are available.
It never treats preview payload as the complete selected set.

## Recommendation Boundary

When complete IDs yield a low materialization ratio and copy timing is
available, the report may recommend Phase 9.2 repeated-run validation.

It never recommends:

- active attention routing;
- replacing or freeing full KV;
- a measured memory-reduction claim;
- a latency or quality claim.

## Caveats

- KV tensors are synthetic.
- Materialization occurs outside the attention path.
- Full KV is still allocated.
- No active routing is implemented.
- No measured runtime memory reduction is claimed.
- Quality is not measured.
- No latency improvement is claimed.
