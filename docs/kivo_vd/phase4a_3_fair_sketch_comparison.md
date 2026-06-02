# Kivo-VD Phase 4A.3: Fair Normalized Sketch Comparison

Phase 4A.3 adds compression metadata to offline HF Q/K sweep rows, active-KV
policy simulation rows, comparison summaries, and benchmark reports.

This is offline-only. It does not change vLLM runtime behavior.

## Why This Matters

SRHT dim 64 looked very strong on GPT-2, but GPT-2 has `head_dim=64`. In that
case SRHT dim 64 is full-dimensional: it uses all Hadamard coordinates after
padding/subsampling constraints. That is useful as a correctness/reference
result, but it is not evidence of compressed KV sketching.

Fair comparisons should separate:

- compressed sketches, where `effective_sketch_dim < head_dim`;
- full-dimensional sketches, where `effective_sketch_dim >= head_dim`;
- expanded random projections, where requested `sketch_dim` can exceed
  `head_dim` but the effective compressed information is not smaller than the
  input dimension.

## New Metadata Fields

HF Q/K eval and head sweep rows now include:

- `head_dim`
- `effective_input_dim`
- `effective_sketch_dim`
- `sketch_compression_ratio`
- `is_full_dimensional_sketch`

Definitions:

- `head_dim`: actual Q/K vector dimension.
- `effective_input_dim`: original head dimension before any padding.
- `effective_sketch_dim`: capped effective dimension used for fair comparison.
- `sketch_compression_ratio`: `effective_sketch_dim / head_dim`.
- `is_full_dimensional_sketch`: true when `effective_sketch_dim >= head_dim`.

## SRHT Interpretation

For SRHT:

- `sketch_dim < head_dim` is the fair compression regime.
- `sketch_dim == head_dim` is full-dimensional and should not be described as
  compressed.
- `sketch_dim > head_dim` remains invalid for unique-coordinate SRHT and is
  skipped by the HF head sweep when required.

## Suggested Fair GPT-2 Comparison

GPT-2 has `head_dim=64`, so focus on dimensions below 64:

```bash
.venv/bin/python scripts/kivo_vd/run_offline_benchmark_pipeline.py \
  --model-name gpt2 \
  --prompt-mode blue_orchid \
  --sketch-types count_sketch,random_projection,srht \
  --sketch-dims 16,32,64 \
  --layers 0,1,2,3 \
  --heads 0,1,2,3 \
  --max-tokens 900 \
  --run-name gpt2_fair_sketch_comparison
```

Interpret `sketch_dim=64` as a full-dimensional reference row, not a compressed
result.

## Suggested Modern Model Comparison

For models with `head_dim=128`, compare:

- dim 32, compression ratio 0.25;
- dim 64, compression ratio 0.50;
- dim 128, full-dimensional reference only.

Example:

```bash
.venv/bin/python scripts/kivo_vd/run_offline_benchmark_pipeline.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --extraction-mode auto \
  --sketch-types count_sketch,random_projection,srht \
  --sketch-dims 32,64,128 \
  --layers 0,1,2,3 \
  --heads 0,1,2,3 \
  --max-tokens 512 \
  --run-name qwen_fair_sketch_comparison
```

## Report Caveat

Benchmark reports now include a full-dimensional sketch caveat when applicable:

> Full-dimensional sketches should not be treated as compressed KV sketches.

This keeps SRHT comparisons honest and avoids interpreting correctness/reference
rows as memory-reduction evidence.
