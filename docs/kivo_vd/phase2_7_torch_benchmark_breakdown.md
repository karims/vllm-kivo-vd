# Kivo-VD Phase 2.7: Torch Benchmark Timing Breakdown

Phase 2.7 splits the offline torch sketch benchmark into separate timing
components.

## Timing fields

Each JSONL row now includes:

- `key_sketch_build_ms`
- `block_aggregation_ms`
- `query_sketch_ms`
- `block_scoring_ms`
- `ranking_ms`
- `total_time_ms`

The script also supports:

- `--topk-blocks`
- `--block-score-mode max|mean`

## Interpretation

Key sketch build and block aggregation are paid when keys or blocks are created
or updated. Query sketch, block scoring, and ranking are paid during decode.

Runtime feasibility depends especially on decode-time costs because those are on
the latency path for each generated token. A one-time key sketch build cost can
still be acceptable if it is amortized over many decode steps, but scoring and
ranking must remain very small or move into a fused/backend path.

CPU and MPS results are useful development signals only. NVIDIA GPU measurements
are still required before making runtime design decisions.
