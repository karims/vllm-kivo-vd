# Kivo-VD Phase 8.0: Compact Sketch-Buffer Overhead

Phase 8.0 measures the standalone memory overhead of compact per-block sketch
buffers. It follows the Phase 7 readiness gate, which authorized only this
overhead experiment.

No vLLM runtime path is modified. Full KV remains allocated, normal attention
remains active, and candidate routing remains out of scope.

## What A Compact Sketch Buffer Represents

The first layout stores one vector per layer, KV head, and physical block:

```text
(num_layers, num_kv_heads, num_blocks, sketch_dim)
```

The corresponding payload formula is:

```text
sketch_bytes =
    num_layers * num_kv_heads * sketch_dim * num_blocks * dtype_bytes
```

For comparison, full K+V payload is:

```text
full_kv_bytes =
    2 * num_layers * num_kv_heads * head_dim
      * block_size * num_blocks * dtype_bytes
```

This assumes per-block sketches. A future per-token layout would require a
different formula.

## Why Memory May Increase

Phase 8.0 adds synthetic sketch buffers without replacing KV. Measured process
memory should therefore increase by the sketch payload plus possible allocator
overhead. This is expected and is the measurement target.

## CPU Smoke

```bash
.venv/bin/python scripts/kivo_vd/measure_sketch_buffer_overhead.py \
  --device cpu \
  --num-blocks 64 \
  --sketch-dims 16,32
```

CPU mode validates shapes, tensor payload, formulas, and reports. It does not
produce CUDA allocator deltas.

## RunPod CUDA Measurement

```bash
.venv/bin/python scripts/kivo_vd/measure_sketch_buffer_overhead.py \
  --model gpt2 \
  --num-layers 12 \
  --num-kv-heads 12 \
  --head-dim 64 \
  --block-size 16 \
  --num-blocks 256 \
  --dtype-bytes 2 \
  --sketch-types count_sketch,random_projection,bidiagonal_sign_subsample \
  --sketch-dims 16,32,64 \
  --device cuda \
  --output-json outputs/kivo_vd/phase8_0_gpt2_sketch_buffer_overhead.json \
  --output-md outputs/kivo_vd/phase8_0_gpt2_sketch_buffer_overhead.md
```

Each configuration is allocated and cleaned up separately. CUDA output records
before-allocation, after-allocation, and after-cleanup allocator checkpoints.

## Interpreting Overhead Ratios

The overhead ratio is:

```text
theoretical_sketch_bytes / theoretical_full_kv_bytes
```

It describes additional compact-buffer payload relative to full K+V payload
for the same number of blocks. It is not a memory-savings ratio. CUDA allocator
deltas can exceed payload bytes because of allocation granularity and caching.
Any reported smallest configuration is selected by payload size only and does
not rank sketch retrieval quality.

## Caveats

- This is overhead measurement only.
- Sketch buffers do not replace full KV.
- No active routing is implemented.
- No measured runtime memory reduction is claimed.
- No latency or quality conclusion follows from this phase.
- The script is standalone and does not allocate buffers inside vLLM.

## Next Steps

Run the CUDA experiment on the validated RunPod environment, compare payload
bytes with allocator deltas, and decide whether the overhead is small enough to
justify Phase 8.1 runtime selected-block accounting. Attention and KV allocation
must remain unchanged.
