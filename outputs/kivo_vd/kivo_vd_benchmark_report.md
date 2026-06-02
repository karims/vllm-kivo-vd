# Kivo-VD Offline Benchmark Report

## Status

This report summarizes offline HuggingFace Q/K sketch retrieval and active-KV policy simulation. It is not a measured vLLM runtime memory reduction, latency result, or quality benchmark.

## Executive Summary

- Sketch-based candidate retrieval works well in these offline GPT-2-style Q/K tests.
- Conservative policy estimate: estimated active-KV reduction was about 38.9%; exact-top-block recall was about 99.7%.
- Aggressive policy estimates can show higher reduction, but they need runtime validation and quality checks before being treated as safe.
- No model architecture, tokenizer, training, or weight changes are part of these results.

## Retrieval Benchmark Summary

| sketch_type | sketch_dim | avg block top-k recall | avg recall@2x | avg recall@4x | avg block score corr | count |
| --- | --- | --- | --- | --- | --- | --- |
| count_sketch | 32 | 0.723 | 0.895 | 0.961 | 0.852 | 64 |
| count_sketch | 64 | 0.785 | 0.934 | 0.992 | 0.912 | 64 |
| count_sketch | 128 | 0.883 | 0.980 | 1.000 | 0.983 | 64 |
| random_projection | 32 | 0.738 | 0.875 | 0.965 | 0.908 | 64 |
| random_projection | 64 | 0.773 | 0.938 | 0.984 | 0.939 | 64 |
| random_projection | 128 | 0.801 | 0.941 | 0.992 | 0.964 | 64 |

## Active KV Policy Simulation Summary

| sketch_type | sketch_dim | recent | candidates | avg active ratio | avg estimated reduction | avg exact-top recall | count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| count_sketch | 64 | 4 | 8 | 0.334 | 0.666 | 0.961 | 64 |
| count_sketch | 64 | 8 | 16 | 0.610 | 0.390 | 0.992 | 64 |
| count_sketch | 128 | 4 | 8 | 0.328 | 0.672 | 0.980 | 64 |
| count_sketch | 128 | 8 | 16 | 0.608 | 0.392 | 1.000 | 64 |
| random_projection | 64 | 4 | 8 | 0.335 | 0.665 | 0.977 | 64 |
| random_projection | 64 | 8 | 16 | 0.613 | 0.387 | 0.996 | 64 |
| random_projection | 128 | 4 | 8 | 0.332 | 0.668 | 0.977 | 64 |
| random_projection | 128 | 8 | 16 | 0.612 | 0.388 | 1.000 | 64 |

## Conservative Recommended Policy

Recommended starting policy for future dry-run/runtime experiments:

- `sketch_type`: `count_sketch` dim 64, with `random_projection` dim 64 retained as a baseline.
- `recent_window_blocks`: 8
- `candidate_budget_blocks`: 16

This policy is intentionally conservative: it aims for meaningful but not extreme active-KV reduction while keeping exact-top-block recall near the safest observed range.

## Aggressive Policy Notes

A stretch policy uses `recent_window_blocks=4` and `candidate_budget_blocks=8`.

- estimated active-KV reduction was about 66.8%.
- exact-top-block recall was about 97.4%.

Treat this as a research signal, not a product or runtime claim. It needs quality, latency, and real memory validation.

## What Is Proven vs Not Proven

Proven/offline in these experiments:

- Sketch candidate retrieval on GPT-2-style Q/K tensors.
- Active-KV policy simulation from ranked candidate blocks.

Not proven yet:

- Real vLLM runtime memory reduction.
- Benchmark quality preservation.
- Latency improvement.
- Behavior on modern RoPE/GQA models.
- Book-inspired variation-diminishing or bidiagonal sketches.

## Next Experiments

- Runtime dry-run on real vLLM inference.
- Quality benchmarks with conservative and aggressive policies.
- Real measured GPU memory experiments.
- Modern model support, especially RoPE and GQA/MQA models.
- Implement book-inspired sketch variants as experimental backends.
