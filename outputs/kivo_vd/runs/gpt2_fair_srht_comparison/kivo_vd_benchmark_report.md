# Kivo-VD Offline Benchmark Report

## Status

This report summarizes offline HuggingFace Q/K sketch retrieval and active-KV policy simulation. It is not a measured vLLM runtime memory reduction, latency result, or quality benchmark.

## Executive Summary

- Sketch-based candidate retrieval works well in these offline GPT-2-style Q/K tests.
- Conservative policy estimate: estimated active-KV reduction was about 40.6%; exact-top-block recall was about 98.6%.
- Aggressive policy estimates can show higher reduction, but they need runtime validation and quality checks before being treated as safe.
- No model architecture, tokenizer, training, or weight changes are part of these results.

## Model and Extraction Metadata

| model_name | extraction_mode | qk_space | num_query_heads | num_key_value_heads | head_dim | effective_sketch_dim | sketch_compression_ratio | is_full_dimensional_sketch | count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt2 | gpt2_fused_c_attn | gpt2_projection | 12 | 12 | 64 | 16 | 0.25 | False | 192 |
| gpt2 | gpt2_fused_c_attn | gpt2_projection | 12 | 12 | 64 | 32 | 0.5 | False | 192 |
| gpt2 | gpt2_fused_c_attn | gpt2_projection | 12 | 12 | 64 | 64 | 1.0 | True | 192 |

## Retrieval Benchmark Summary

| model_name | extraction_mode | qk_space | sketch_type | sketch_dim | head_dim | effective_sketch_dim | compression ratio | full-dim | avg block top-k recall | avg recall@2x | avg recall@4x | avg block score corr | count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt2 | gpt2_fused_c_attn | gpt2_projection | count_sketch | 16 | 64 | 16 | 0.250 | False | 0.578 | 0.727 | 0.883 | 0.810 | 64 |
| gpt2 | gpt2_fused_c_attn | gpt2_projection | count_sketch | 32 | 64 | 32 | 0.500 | False | 0.691 | 0.863 | 0.953 | 0.904 | 64 |
| gpt2 | gpt2_fused_c_attn | gpt2_projection | count_sketch | 64 | 64 | 64 | 1.000 | True | 0.766 | 0.910 | 0.977 | 0.923 | 64 |
| gpt2 | gpt2_fused_c_attn | gpt2_projection | random_projection | 16 | 64 | 16 | 0.250 | False | 0.598 | 0.766 | 0.914 | 0.793 | 64 |
| gpt2 | gpt2_fused_c_attn | gpt2_projection | random_projection | 32 | 64 | 32 | 0.500 | False | 0.730 | 0.863 | 0.953 | 0.905 | 64 |
| gpt2 | gpt2_fused_c_attn | gpt2_projection | random_projection | 64 | 64 | 64 | 1.000 | True | 0.730 | 0.902 | 0.973 | 0.936 | 64 |
| gpt2 | gpt2_fused_c_attn | gpt2_projection | srht | 16 | 64 | 16 | 0.250 | False | 0.598 | 0.758 | 0.883 | 0.770 | 64 |
| gpt2 | gpt2_fused_c_attn | gpt2_projection | srht | 32 | 64 | 32 | 0.500 | False | 0.820 | 0.934 | 0.984 | 0.933 | 64 |
| gpt2 | gpt2_fused_c_attn | gpt2_projection | srht | 64 | 64 | 64 | 1.000 | True | 1.000 | 1.000 | 1.000 | 1.000 | 64 |

Note: `srht` rows are experimental. SRHT should be compared against CountSketch and Random Projection before being used as a default, and these offline rows do not imply runtime memory reduction.

## Full-Dimensional Sketch Caveat

Rows with `is_full_dimensional_sketch=True` should not be treated as compressed KV sketches. For example, SRHT dim 64 on GPT-2 head_dim 64 is useful as a correctness/reference result, but it is not evidence of sketch compression.

## Active KV Policy Simulation Summary

| sketch_type | sketch_dim | recent | candidates | avg active ratio | avg estimated reduction | avg exact-top recall | count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| count_sketch | 64 | 4 | 8 | 0.314 | 0.686 | 0.941 | 64 |
| count_sketch | 64 | 8 | 16 | 0.590 | 0.410 | 0.988 | 64 |
| random_projection | 64 | 4 | 8 | 0.321 | 0.679 | 0.957 | 64 |
| random_projection | 64 | 8 | 16 | 0.597 | 0.403 | 0.984 | 64 |
| srht | 64 | 4 | 8 | 0.312 | 0.688 | 1.000 | 64 |
| srht | 64 | 8 | 16 | 0.590 | 0.410 | 1.000 | 64 |

## Conservative Recommended Policy

Recommended starting policy for future dry-run/runtime experiments:

- `sketch_type`: `count_sketch` dim 64, with `random_projection` dim 64 retained as a baseline.
- `srht` is experimental and should be compared offline before any runtime policy uses it.
- `recent_window_blocks`: 8
- `candidate_budget_blocks`: 16

This policy is intentionally conservative: it aims for meaningful but not extreme active-KV reduction while keeping exact-top-block recall near the safest observed range.

## Aggressive Policy Notes

A stretch policy uses `recent_window_blocks=4` and `candidate_budget_blocks=8`.

- estimated active-KV reduction was about 68.3%.
- exact-top-block recall was about 94.9%.

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
