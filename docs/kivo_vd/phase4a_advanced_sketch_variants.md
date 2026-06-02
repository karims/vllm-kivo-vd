# Kivo-VD Phase 4A.0: Advanced Sketch Variants

Phase 4A starts a careful offline-only exploration of additional sketch
families. The goal is to compare mathematical candidates before any runtime
behavior changes.

## Scope

Implemented in this phase:

- SRHT / Hadamard-style structured projection in the NumPy offline harness.
- SRHT / Hadamard-style structured projection in the optional torch benchmark
  backend.
- CLI support for `sketch_type=srht` in offline synthetic and HF Q/K scripts.

Not implemented in this phase:

- No vLLM runtime attention changes.
- No scheduler behavior changes.
- No GPUModelRunner changes.
- No block table, slot mapping, or attention metadata changes.
- No CUDA/Triton kernels.
- No KV memory reduction claim.

## Current Baseline Backends

### CountSketch

CountSketch hashes each input dimension into a smaller sketch coordinate with a
random sign. It is sparse, deterministic from a seed, and cheap to apply.

Intended preservation target:

- approximate inner products;
- approximate block ranking;
- low memory overhead for key-block summaries.

Current status:

- implemented in NumPy offline math;
- implemented in the torch benchmark backend;
- strongest practical default so far in prior GPT-2-style offline sweeps.

### Random Projection

Random Projection uses a dense normalized Gaussian projection matrix.

Intended preservation target:

- Johnson-Lindenstrauss-style distance/inner-product preservation;
- baseline comparison against CountSketch.

Current status:

- implemented in NumPy offline math;
- implemented in the torch benchmark backend;
- retained as the main baseline.

## New Candidate: SRHT / Hadamard Structured Projection

SRHT stands for subsampled randomized Hadamard transform. It applies:

1. zero padding to the next power-of-two dimension if needed;
2. a deterministic random sign flip;
3. a Fast Walsh-Hadamard Transform;
4. deterministic coordinate subsampling;
5. normalization.

Intended preservation target:

- approximate inner products with more structure than a fully dense random
  projection;
- lower multiplication cost than dense random projection at larger dimensions;
- useful block-ranking behavior if Q/K structure is compatible.

Current implementation:

- pure NumPy SRHT helpers in `kivo_vd_sketch_math.py`;
- NumPy backend abstraction support in `kivo_vd_sketch_backend.py`;
- optional torch backend support in `kivo_vd_torch_sketch_backend.py`;
- accepted by offline synthetic, HF Q/K, head-sweep, and benchmark scripts via
  `sketch_type=srht`.

Important caveats:

- SRHT is experimental in Kivo-VD.
- It has not replaced CountSketch or Random Projection as the recommended
  default.
- It has not been validated in vLLM runtime.
- It does not imply any measured KV memory reduction.

## Future Experimental Variants

These variants are research hypotheses only. They are documented to guide
future experiments, not because they are validated Kivo-VD methods.

### Bidiagonal Sketch

A bidiagonal sketch would apply a structured transform with only diagonal and
near-diagonal interactions.

Possible preservation target:

- smooth/local structure in key vectors;
- adjacent-dimension relationships;
- low-cost structured summaries.

Risk:

- may not preserve arbitrary inner products well;
- could bias ranking toward local coordinate structure that is not meaningful
  for transformer Q/K tensors.

### Variation-Diminishing / Totally-Positive Inspired Sketch

Variation-diminishing transforms are motivated by matrices that reduce sign
variation or preserve certain ordered structures.

Possible preservation target:

- monotone or smooth trends;
- ordered/local structure across feature dimensions;
- robust block ranking when Q/K activations have structured geometry.

Risk:

- this is not a standard transformer-attention sketching method;
- relevance to Q/K inner-product ranking is unproven;
- needs offline validation before any runtime consideration.

### Krylov/Lanczos-Style Block Summaries

Krylov or Lanczos-inspired summaries could approximate dominant subspaces of
key blocks.

Possible preservation target:

- dominant block-level subspace;
- principal directions relevant to repeated query patterns;
- block summaries richer than a single random sketch.

Risk:

- higher update cost;
- batching and per-layer/head bookkeeping complexity;
- unclear benefit relative to simpler sketches.

## How To Test SRHT Offline

Synthetic quick sweep:

```bash
.venv/bin/python scripts/kivo_vd/run_sketch_sweep.py --quick
```

Single synthetic run:

```bash
.venv/bin/python scripts/kivo_vd/run_offline_sketch_eval.py \
  --sketch-type srht \
  --input-dim 128 \
  --sketch-dim 64 \
  --num-tokens 1024 \
  --block-size 16 \
  --mode mixed
```

Optional torch benchmark:

```bash
.venv/bin/python scripts/kivo_vd/benchmark_torch_sketch_backend.py \
  --sketch-types srht \
  --device cpu \
  --num-tokens 1024 \
  --head-dim 128 \
  --iters 3 \
  --warmup 1
```

HF Q/K head sweep, if optional HF dependencies and model downloads are
available:

```bash
.venv/bin/python scripts/kivo_vd/run_hf_qk_head_sweep.py \
  --model-name gpt2 \
  --sketch-types srht,count_sketch,random_projection \
  --sketch-dims 32,64,128 \
  --layers 0,1 \
  --heads 0,1 \
  --max-tokens 512 \
  --include-ranked-blocks
```

For a full CountSketch vs Random Projection vs SRHT comparison pipeline, see
[Phase 4A.1: SRHT Comparison](phase4a_1_srht_comparison.md).

## Interpretation

SRHT should be judged against the same conservative metrics as the baseline
backends:

- block top-k recall;
- block recall at 2x and 4x candidate budgets;
- block score correlation;
- active-KV policy simulation recall, if ranked HF rows are available.

Good SRHT results would justify further offline comparisons. They would not, by
themselves, prove runtime memory reduction, latency improvement, or quality
preservation.
