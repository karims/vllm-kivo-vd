# Kivo-VD Phase 4A.4: Fair SRHT Empirical Results

Phase 4A.4 reran the offline benchmark pipeline with CountSketch, Random
Projection, and experimental SRHT at dimensions 16, 32, and 64.

This is an offline HuggingFace Q/K and active-KV policy simulation result. It
does not measure vLLM runtime memory reduction, latency, or quality.

## Run Configuration

- Model: `gpt2`
- Prompt mode: `blue_orchid`
- Layers: `0,1,2,3`
- Heads: `0,1,2,3`
- Max tokens: `900`
- Sketches:
  - `count_sketch`
  - `random_projection`
  - `srht`
- Sketch dims:
  - `16`
  - `32`
  - `64`

Command:

```bash
.venv/bin/python scripts/kivo_vd/run_offline_benchmark_pipeline.py \
  --model-name gpt2 \
  --prompt-mode blue_orchid \
  --sketch-types count_sketch,random_projection,srht \
  --sketch-dims 16,32,64 \
  --layers 0,1,2,3 \
  --heads 0,1,2,3 \
  --max-tokens 900 \
  --run-name gpt2_fair_srht_comparison
```

Output directory:

```text
outputs/kivo_vd/runs/gpt2_fair_srht_comparison/
```

## Compression Interpretation

GPT-2 has `head_dim=64`.

- Dim 16 is compressed: compression ratio `0.25`.
- Dim 32 is compressed: compression ratio `0.50`.
- Dim 64 is full-dimensional/reference: compression ratio `1.00`.

Dim 64 should not be interpreted as a compressed KV sketch result for GPT-2.

## Retrieval Summary

| sketch_type | sketch_dim | compression ratio | full-dimensional | avg block top-k recall | avg recall@2x | avg recall@4x | avg block score corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| count_sketch | 16 | 0.25 | false | 0.578 | 0.727 | 0.883 | 0.810 |
| random_projection | 16 | 0.25 | false | 0.598 | 0.766 | 0.914 | 0.793 |
| srht | 16 | 0.25 | false | 0.598 | 0.758 | 0.883 | 0.770 |
| count_sketch | 32 | 0.50 | false | 0.691 | 0.863 | 0.953 | 0.904 |
| random_projection | 32 | 0.50 | false | 0.730 | 0.863 | 0.953 | 0.905 |
| srht | 32 | 0.50 | false | 0.820 | 0.934 | 0.984 | 0.933 |
| count_sketch | 64 | 1.00 | true | 0.766 | 0.910 | 0.977 | 0.923 |
| random_projection | 64 | 1.00 | true | 0.730 | 0.902 | 0.973 | 0.936 |
| srht | 64 | 1.00 | true | 1.000 | 1.000 | 1.000 | 1.000 |

## Conservative Active KV Policy

Policy:

- `recent_window_blocks=8`
- `candidate_budget_blocks=16`

| sketch_type | sketch_dim | compression ratio | full-dimensional | avg estimated KV reduction | avg exact-top recall |
| --- | --- | --- | --- | --- | --- |
| count_sketch | 16 | 0.25 | false | 0.394 | 0.961 |
| random_projection | 16 | 0.25 | false | 0.405 | 0.957 |
| srht | 16 | 0.25 | false | 0.393 | 0.957 |
| count_sketch | 32 | 0.50 | false | 0.408 | 0.973 |
| random_projection | 32 | 0.50 | false | 0.395 | 0.980 |
| srht | 32 | 0.50 | false | 0.407 | 1.000 |
| count_sketch | 64 | 1.00 | true | 0.410 | 0.988 |
| random_projection | 64 | 1.00 | true | 0.403 | 0.984 |
| srht | 64 | 1.00 | true | 0.410 | 1.000 |

## Conclusion

SRHT compressed dimensions are competitive in this GPT-2 offline run, but the
story is dimension-dependent:

- At dim 16, SRHT is roughly comparable to Random Projection and CountSketch on
  strict top-k recall, but its recall@4x and score correlation are weaker than
  Random Projection.
- At dim 32, SRHT is the strongest compressed backend in this run, with higher
  strict recall, recall@2x, recall@4x, and block score correlation than both
  CountSketch and Random Projection.
- At dim 64, SRHT is perfect, but this is full-dimensional for GPT-2 and should
  be treated only as a reference/correctness result, not compression evidence.

CountSketch and Random Projection should remain the practical defaults for now.
SRHT deserves more testing, especially on modern models with `head_dim=128`
where dims 32 and 64 are both genuinely compressed.

## What Is Still Not Proven

- Real vLLM runtime memory reduction.
- Latency improvement.
- Quality preservation under behavior-changing candidate attention.
- SRHT behavior on post-RoPE Q/K tensors.
- SRHT behavior on modern GQA/MQA model families.
