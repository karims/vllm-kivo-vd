# Kivo-VD Phase 6.0: Structured Linear Sketches

Phase 6.0 begins offline exploration of structured linear-algebra sketch
variants for Kivo-VD candidate KV-block retrieval.

This phase is offline only. It does not change vLLM scheduler behavior,
GPUModelRunner, attention metadata, block tables, slot mapping, CUDA/Triton
kernels, active KV routing, model architecture, training, or tokenizer behavior.

## Motivation

Phase 5 established two complementary signals:

- real Linux/NVIDIA vLLM GPU dry-run instrumentation works and preserves greedy
  GPT-2 output in tested dry-run runs;
- offline sketch benchmarks show that candidate-block retrieval can work well
  with CountSketch, Random Projection, and experimental SRHT baselines.

Those baselines are useful, but they are not final methods. Phase 6 starts a
conservative search over structured transforms that may better preserve local,
smooth, or dominant-subspace structure in Q/K vectors.

## Current Baselines

Current offline baselines remain:

- `count_sketch`
- `random_projection`
- `srht`

`srht` is already experimental: it can show strong retrieval quality in some
GPT-2 rows, but the current torch implementation is much slower than
CountSketch and Random Projection.

## New Experimental Backend: `bidiagonal_sign`

Phase 6.0 adds `bidiagonal_sign`, a lightweight structured transform inspired
by bidiagonal mixing and variation-preserving ideas. This is a research
baseline, not a method with claimed theoretical guarantees in Kivo-VD.

For an input vector `x` of dimension `d`:

1. Draw deterministic signs `s_i` from the seed.
2. Apply sign flips and bidiagonal neighbor mixing:

```text
y_0 = s_0 * x_0
y_i = s_i * x_i + alpha * s_{i-1} * x_{i-1}, for i > 0
```

3. Select deterministic sketch coordinates from `y`.
4. Scale by `sqrt(input_dim / sketch_dim)` to keep magnitudes roughly stable.

Phase 6.0 uses an internal `alpha=0.5`. This can become configurable later if
benchmark results justify another degree of freedom.

## Expected Benefits And Risks

Potential benefits:

- cheaper than dense Random Projection;
- more structured than CountSketch;
- deterministic, local neighbor mixing may preserve smooth/local vector
  patterns better than pure coordinate sampling;
- supports non-power-of-two input dimensions without padding.

Risks:

- may underperform on unstructured or highly rotated Q/K spaces;
- coordinate subsampling can miss important dimensions;
- no Kivo-specific theory has been proven yet;
- retrieval quality may be model/layer/head dependent;
- runtime usefulness depends on future torch/GPU timing and quality benchmarks.

## Implementation Scope

Implemented in this phase:

- NumPy offline sketch math support;
- NumPy Phase 2.1 sketch backend support;
- optional torch offline backend support;
- CLI acceptance in synthetic sweeps, HF Q/K eval, HF head sweep, offline
  pipeline, torch benchmark, and report/comparison tooling;
- tests for determinism, shape, error handling, CLI parsing, and reporting.

Not implemented:

- real runtime K tensor sketch capture;
- real query-time runtime scoring;
- candidate-routed attention;
- active KV residency changes;
- measured runtime memory reduction.

## Comparison Command

Run a small offline comparison against the established baselines:

```bash
python scripts/kivo_vd/run_offline_benchmark_pipeline.py \
  --model-name gpt2 \
  --sketch-types count_sketch,random_projection,srht,bidiagonal_sign \
  --sketch-dims 16,32,64 \
  --layers 0,1,2,3 \
  --heads 0,1,2,3 \
  --max-tokens 512 \
  --extraction-mode auto \
  --run-name phase6_gpt2_bidiagonal_sign_compare \
  --run-torch-benchmark
```

For a faster synthetic smoke test:

```bash
python scripts/kivo_vd/run_sketch_sweep.py \
  --quick \
  --sketch-types count_sketch,random_projection,srht,bidiagonal_sign
```

For torch timing only:

```bash
python scripts/kivo_vd/benchmark_torch_sketch_backend.py \
  --sketch-types bidiagonal_sign \
  --sketch-dims 16,32,64 \
  --device cpu \
  --num-tokens 4096 \
  --head-dim 128
```

## Interpretation Rules

Use the same conservative interpretation as prior phases:

- compare `bidiagonal_sign` against CountSketch, Random Projection, and SRHT;
- focus on compressed dimensions, not full-dimensional reference rows;
- treat high recall as offline candidate-retrieval evidence only;
- do not claim measured vLLM runtime KV memory reduction;
- do not claim quality preservation under candidate-routed attention;
- do not make `bidiagonal_sign` a default backend until broader benchmarks
  justify it.

## Next Steps

If `bidiagonal_sign` is competitive offline, evaluate it on:

- larger GPT-2 sweeps;
- Qwen/Llama-style pre-RoPE Q/K extraction;
- torch CPU/GPU timing;
- candidate-budget policy simulation;
- eventually dry-run runtime scoring once real K/Q tensor sketching exists.

Future structured variants can include variation-diminishing/totally-positive
inspired transforms, Krylov/Lanczos-style block summaries, and other
book-inspired structured linear-algebra sketches.
