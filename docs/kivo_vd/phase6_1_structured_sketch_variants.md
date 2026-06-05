# Kivo-VD Phase 6.1: Structured Sketch Variants

Phase 6.1 expands the offline structured linear-algebra sketch family for
Kivo-VD candidate KV-block retrieval.

This phase is offline only. It does not change vLLM scheduler behavior,
GPUModelRunner, attention metadata, block tables, slot mapping, CUDA/Triton
kernels, active KV routing, model architecture, training, or tokenizer behavior.

## Phase 6.0 Result Summary

Phase 6.0 introduced `bidiagonal_sign`, a deterministic sign-flip plus
bidiagonal neighbor-mixing transform.

RunPod GPT-2 comparison showed that `bidiagonal_sign` did not collapse. At
compressed dimension `32`, it was retrieval-competitive:

| metric | value |
| --- | ---: |
| top-k recall | about `0.766` |
| recall@2x | about `0.926` |
| recall@4x | about `0.980` |
| block score correlation | about `0.827` |

However, the current torch timing was too slow compared with CountSketch and
Random Projection:

| sketch_type | dim 32 timing |
| --- | ---: |
| count_sketch | about `0.36 ms` |
| random_projection | about `0.32 ms` |
| srht | about `5.03 ms` |
| bidiagonal_sign | about `3.30 ms` |

Interpretation: `bidiagonal_sign` is a useful research signal, not a practical
runtime candidate yet.

## New Variants

Phase 6.1 adds two experimental variants:

- `bidiagonal_sign_subsample`
- `tridiagonal_sign`

Both are deterministic from seed, support non-power-of-two input dimensions,
require `sketch_dim <= input_dim`, and remain opt-in only.

## `bidiagonal_sign_subsample`

This variant uses the same mathematical sketch as `bidiagonal_sign`:

```text
y_0 = s_0 * x_0
y_i = s_i * x_i + alpha * s_{i-1} * x_{i-1}, for i > 0
```

with `alpha=0.5`, then selects deterministic sketch coordinates.

The implementation differs by computing only selected mixed coordinates instead
of materializing the full mixed vector. This is intended to test whether the
same structured signal can be retained with lower offline torch timing when
`sketch_dim` is small.

## `tridiagonal_sign`

This variant mixes each selected coordinate with both neighbors:

```text
y_i = s_i * x_i
    + alpha_left  * s_{i-1} * x_{i-1}
    + alpha_right * s_{i+1} * x_{i+1}
```

Boundary coordinates simply omit missing neighbors. Phase 6.1 uses
`alpha_left=0.25` and `alpha_right=0.25`.

The goal is to test whether a slightly wider local stencil better preserves
smooth/local Q/K structure than one-sided bidiagonal mixing.

## Structured Linear-Algebra Framing

These variants are inspired by structured transforms, local stencil operators,
and variation-preserving intuition. They are not claimed to have proven Kivo-VD
quality or memory guarantees.

Use them as experimental baselines for:

- candidate-block retrieval quality;
- candidate-budget recall;
- block score correlation;
- torch sketch timing;
- policy simulation inputs.

## Comparison Command

Run a full offline comparison against the established baselines:

```bash
python scripts/kivo_vd/run_offline_benchmark_pipeline.py \
  --model-name gpt2 \
  --sketch-types count_sketch,random_projection,srht,bidiagonal_sign,bidiagonal_sign_subsample,tridiagonal_sign \
  --sketch-dims 16,32,64 \
  --layers 0,1,2,3 \
  --heads 0,1,2,3 \
  --max-tokens 512 \
  --extraction-mode auto \
  --run-name phase6_1_gpt2_structured_compare \
  --run-torch-benchmark
```

For a faster synthetic smoke test:

```bash
python scripts/kivo_vd/run_sketch_sweep.py \
  --quick \
  --sketch-types count_sketch,random_projection,srht,bidiagonal_sign,bidiagonal_sign_subsample,tridiagonal_sign
```

For torch timing only:

```bash
python scripts/kivo_vd/benchmark_torch_sketch_backend.py \
  --sketch-types bidiagonal_sign,bidiagonal_sign_subsample,tridiagonal_sign \
  --sketch-dims 16,32,64 \
  --device cpu \
  --num-tokens 4096 \
  --head-dim 128
```

## Caveats

- These sketches are experimental structured baselines only.
- CountSketch and Random Projection remain baseline references, not final
  methods.
- SRHT remains experimental.
- `bidiagonal_sign`, `bidiagonal_sign_subsample`, and `tridiagonal_sign` remain
  experimental.
- No measured runtime KV memory reduction is claimed.
- No active KV routing is implemented.
- No quality preservation under candidate-routed/compressed attention is proven.
- No theoretical guarantee is claimed for these variants yet.

## Next Questions

The key Phase 6.1 questions are:

- Does selected-coordinate bidiagonal computation improve torch timing?
- Does tridiagonal local mixing improve retrieval quality or score correlation?
- Are either variant competitive with CountSketch/RP at dim 16 or 32?
- Do results transfer beyond GPT-2 to Qwen/Llama-style Q/K extraction?

Only after offline quality/timing results look promising should these variants
be considered for runtime dry-run tensor scoring.
