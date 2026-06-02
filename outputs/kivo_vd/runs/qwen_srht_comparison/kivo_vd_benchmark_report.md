# Kivo-VD Offline Benchmark Report

## Status

This report summarizes offline HuggingFace Q/K sketch retrieval and active-KV policy simulation. It is not a measured vLLM runtime memory reduction, latency result, or quality benchmark.

## Executive Summary

- Sketch-based candidate retrieval works well in these offline GPT-2-style Q/K tests.
- Conservative policy estimate: estimated active-KV reduction was about 35.2%; exact-top-block recall was about 77.9%.
- Aggressive policy estimates can show higher reduction, but they need runtime validation and quality checks before being treated as safe.
- No model architecture, tokenizer, training, or weight changes are part of these results.

## Model and Extraction Metadata

| model_name | extraction_mode | qk_space | num_query_heads | num_key_value_heads | head_dim | effective_sketch_dim | sketch_compression_ratio | is_full_dimensional_sketch | count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-0.5B | separate_qk_proj | pre_rope_projection | 14 | 2 | 64 | 32 | 0.5 | False | 192 |
| Qwen/Qwen2.5-0.5B | separate_qk_proj | pre_rope_projection | 14 | 2 | 64 | 64 | 1.0 | True | 192 |

Note: at least one row uses `qk_space=pre_rope_projection`. Those results are based on Q/K after linear projection but before RoPE is applied. Runtime post-RoPE attention behavior may differ, so these numbers are not final vLLM runtime claims.

## Retrieval Benchmark Summary

| model_name | extraction_mode | qk_space | sketch_type | sketch_dim | head_dim | effective_sketch_dim | compression ratio | full-dim | avg block top-k recall | avg recall@2x | avg recall@4x | avg block score corr | count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen/Qwen2.5-0.5B | separate_qk_proj | pre_rope_projection | count_sketch | 32 | 64 | 32 | 0.500 | False | 0.555 | 0.594 | 0.820 | 0.668 | 64 |
| Qwen/Qwen2.5-0.5B | separate_qk_proj | pre_rope_projection | count_sketch | 64 | 64 | 64 | 1.000 | True | 0.406 | 0.480 | 0.664 | 0.550 | 64 |
| Qwen/Qwen2.5-0.5B | separate_qk_proj | pre_rope_projection | random_projection | 32 | 64 | 32 | 0.500 | False | 0.371 | 0.438 | 0.637 | 0.408 | 64 |
| Qwen/Qwen2.5-0.5B | separate_qk_proj | pre_rope_projection | random_projection | 64 | 64 | 64 | 1.000 | True | 0.555 | 0.645 | 0.824 | 0.597 | 64 |
| Qwen/Qwen2.5-0.5B | separate_qk_proj | pre_rope_projection | srht | 32 | 64 | 32 | 0.500 | False | 0.520 | 0.559 | 0.734 | 0.735 | 64 |
| Qwen/Qwen2.5-0.5B | separate_qk_proj | pre_rope_projection | srht | 64 | 64 | 64 | 1.000 | True | 1.000 | 0.875 | 0.992 | 1.000 | 64 |

Note: `srht` rows are experimental. SRHT should be compared against CountSketch and Random Projection before being used as a default, and these offline rows do not imply runtime memory reduction.

## Full-Dimensional Sketch Caveat

Rows with `is_full_dimensional_sketch=True` should not be treated as compressed KV sketches. For example, SRHT dim 64 on GPT-2 head_dim 64 is useful as a correctness/reference result, but it is not evidence of sketch compression.

## Active KV Policy Simulation Summary

| sketch_type | sketch_dim | recent | candidates | avg active ratio | avg estimated reduction | avg exact-top recall | count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| count_sketch | 64 | 4 | 8 | 0.380 | 0.620 | 0.523 | 64 |
| count_sketch | 64 | 8 | 16 | 0.643 | 0.357 | 0.695 | 64 |
| random_projection | 64 | 4 | 8 | 0.381 | 0.619 | 0.691 | 64 |
| random_projection | 64 | 8 | 16 | 0.653 | 0.347 | 0.863 | 64 |
| srht | 64 | 4 | 8 | 0.383 | 0.617 | 0.875 | 64 |
| srht | 64 | 8 | 16 | 0.651 | 0.349 | 0.992 | 64 |

## Conservative Recommended Policy

Recommended starting policy for future dry-run/runtime experiments:

- `sketch_type`: `count_sketch` dim 64, with `random_projection` dim 64 retained as a baseline.
- `srht` is experimental and should be compared offline before any runtime policy uses it.
- `recent_window_blocks`: 8
- `candidate_budget_blocks`: 16

This policy is intentionally conservative: it aims for meaningful but not extreme active-KV reduction while keeping exact-top-block recall near the safest observed range.

## Aggressive Policy Notes

A stretch policy uses `recent_window_blocks=4` and `candidate_budget_blocks=8`.

- estimated active-KV reduction was about 61.9%.
- exact-top-block recall was about 60.7%.

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
