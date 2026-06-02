# Kivo-VD Phase 2.6: Offline Torch Sketch Backend Benchmark

Phase 2.6 adds an optional torch-based sketch backend and benchmark script for
measuring CountSketch and Random Projection feasibility on torch tensors.

## What this phase adds

- `vllm/v1/core/kivo_vd_torch_sketch_backend.py`
  - `TorchKivoSketchBackend`
  - `TorchCountSketchBackend`
  - `TorchRandomProjectionBackend`
  - `make_torch_sketch_backend(...)`
- `scripts/kivo_vd/benchmark_torch_sketch_backend.py`
  - synthetic key/query tensors
  - key sketch timing
  - block aggregation timing
  - query sketch timing
  - block scoring timing
  - ranking/top-k timing
  - full-K vs sketched-K memory estimate

## Offline only

This phase does not connect to vLLM runtime. It does not touch the scheduler,
GPUModelRunner, block tables, slot mappings, attention metadata, kernels, model
weights, tokenizer, training, or model architecture.

## Why NumPy is not enough for runtime

The NumPy backend is useful for synthetic and HuggingFace offline validation,
but real runtime K/Q tensors live as torch tensors, usually on GPU. Moving them
to CPU for NumPy sketching would introduce synchronization and transfer overhead.
Runtime sketching will need torch/GPU tensor operations, and potentially custom
kernels later if generic torch ops are too slow.

## Benchmark command

CPU:

```bash
.venv/bin/python scripts/kivo_vd/benchmark_torch_sketch_backend.py \
  --device cpu \
  --num-tokens 4096 \
  --head-dim 128
```

MPS on Apple Silicon, if available:

```bash
.venv/bin/python scripts/kivo_vd/benchmark_torch_sketch_backend.py \
  --device mps \
  --num-tokens 4096 \
  --head-dim 128
```

Output defaults to:

```text
outputs/kivo_vd/torch_sketch_benchmark.jsonl
```

## Metrics

Each JSONL row includes:

- `key_sketch_build_time_ms`
- `key_sketch_build_ms`
- `block_aggregation_ms`
- `query_sketch_ms`
- `block_scoring_ms`
- `ranking_ms`
- `total_time_ms`
- `full_k_memory_bytes`
- `sketch_memory_bytes`
- `sketch_memory_ratio`

## How this informs runtime integration

The benchmark separates sketch update cost, query sketch cost, and scoring cost.
Those numbers guide whether Phase 2.8 runtime tensor capture should start with
generic torch ops or skip directly to a backend-specific implementation.
