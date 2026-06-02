# Kivo-VD Phase 2.5: Real Runtime Sketch Feasibility

Phase 2.5 is an analysis/design phase only. It inspects where real Kivo-VD key
sketches and query scoring could attach to vLLM runtime, without implementing
tensor sketching or changing attention behavior.

## Files inspected

- `vllm/v1/worker/gpu/model_runner.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/block_table.py`
- `vllm/v1/worker/gpu/block_table.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/core/kv_cache_manager.py`
- `vllm/v1/core/block_pool.py`
- `vllm/v1/worker/gpu/attn_utils.py`
- `vllm/v1/worker/gpu/model_states/default.py`
- `vllm/v1/attention/backend.py`
- `vllm/v1/attention/backends/flash_attn.py`
- `vllm/v1/attention/backends/triton_attn.py`

## A. Key sketch construction

Where K tensors are produced:

- Attention layers produce `query`, `key`, and `value` tensors before calling
  the selected attention backend.
- Backend interfaces expose these tensors in `AttentionImpl.forward(...)`.
- FlashAttention and Triton attention both receive:
  - `query: [num_tokens, num_heads, head_size]`
  - `key: [num_tokens, num_kv_heads, head_size]`
  - `value: [num_tokens, num_kv_heads, head_size]`
  - `kv_cache`
  - backend-specific attention metadata

Where K tensors are written into KV cache:

- FlashAttention path:
  - `FlashAttentionImpl.do_kv_cache_update(...)`
  - writes through `reshape_and_cache_flash(...)`
  - uses `slot_mapping`
- Triton path:
  - `TritonAttentionImpl.do_kv_cache_update(...)`
  - writes through Triton reshape/cache ops
  - uses `slot_mapping`
- Some backends support fused RoPE + KV cache update through
  `do_rope_and_kv_cache_update(...)`, which receives `query`, `key`, `value`,
  `positions`, `kv_cache`, and `layer_slot_mapping`.

Can we access per-layer/per-head/per-block K data there?

- Per-layer: yes, backend invocation is layer-specific.
- Per-head: yes, `key` is shaped by KV heads.
- Per-block: indirectly. The new key tensor is token-major, while block
  placement is determined by `slot_mapping` and block table metadata.
- To aggregate by block, Kivo-VD must map token positions/slot ids to physical
  block ids and block offsets.

Would sketching K require GPU tensor ops?

- For real runtime, yes. Keys are torch tensors on device. Moving them to CPU for
  NumPy sketching would add synchronization and transfer overhead that defeats
  the purpose.
- The current NumPy backends are useful for offline validation and policy tests,
  but insufficient for production runtime sketching.

What would a minimal torch/GPU implementation need?

- Torch-side sketch backend with deterministic parameters on device.
- CountSketch:
  - bucket index tensor
  - sign tensor
  - scatter/add or indexed accumulation over `head_size`
- Random projection:
  - projection matrix `[head_size, sketch_dim]`
  - matmul for `key @ projection`
- Block aggregation:
  - map token sketches into `(request, layer, kv_head, block_id)`
  - reduce within each block, likely mean/max/last-token summary
- Storage:
  - compact per-block sketch tensor, probably keyed by physical block id plus
    layer/group/head metadata.

## B. Query scoring

Where current query tensor is available during decode:

- The attention backend `forward(...)` receives `query` for each attention layer.
- This is the cleanest point where layer/head-specific query vectors are
  available before attention computation.

Can we sketch the query before attention?

- Yes, in principle. Query sketching can happen inside or immediately before
  backend `forward(...)`.
- For dry-run only, this could compute scores and log decisions while still
  passing the original full metadata to attention.

Can candidate block scoring happen before attention backend call?

- Yes for dry-run if implemented as torch ops in the backend or a wrapper around
  backend invocation.
- Python-side scoring at every layer/head/token risks severe overhead.
- Practical scoring should stay on device and be batched.

Shape/layout issues:

- Decode usually has small query length, but speculative decode can schedule
  multiple query tokens.
- Prefill/chunked prefill has many query tokens per request; scoring every token
  may be too expensive.
- MQA/GQA have fewer KV heads than query heads. Candidate policy may need to be
  per-KV-head, shared across query heads, or reduced across query heads.
- Layer behavior differs. Offline sweeps already suggest per-layer/head policy
  may matter.

## C. Candidate block routing

Where selected candidate block IDs could be represented:

- Side metadata during dry-run:
  - Kivo observer/export events
  - per-request/layer/head candidate summaries
- Behavior-changing path:
  - attention metadata field carrying selected block ids or a candidate mask
  - backend-specific metadata object derived from `CommonAttentionMetadata`

Would this require modifying block tables?

- For a real sparse/candidate path, possibly but it is not the first choice.
- Mutating canonical block tables is risky because they are also used for slot
  mapping, CUDAGraph padding, prefix cache, and request state.

Would this require modifying attention metadata?

- Likely yes. Candidate block ids need to reach attention execution without
  corrupting canonical full block tables.
- Metadata is the natural place for optional candidate structures.

Would this require custom attention backend/kernel support?

- For real speedup, yes. Existing FlashAttention/PagedAttention paths consume
  standard block tables and sequence lengths. Merely computing candidates does
  not skip work.
- Candidate-aware attention needs backend/kernel support to restrict exact
  attention to selected blocks plus required recent/local blocks.

Can we do dry-run comparison without altering attention?

- Yes. Phase 2.2/2.3 already computes metadata-only dry-run decisions and logs
  them while full attention continues unchanged.
- A later dry-run can compute real query/key sketches and compare candidate
  blocks against full block tables without changing the backend output.

## D. Runtime feasibility

What can be done as Python-side dry-run:

- Allocation/free lifecycle tracing.
- Metadata-only candidate selection.
- JSONL export and debugging utilities.
- Offline NumPy/HF experiments.
- Limited runtime logging of selected block counts and candidate ids.

What requires torch ops:

- Real K sketch computation from runtime key tensors.
- Query sketch computation from runtime query tensors.
- Block-level sketch aggregation on GPU tensors.
- Device-side score computation against stored block sketches.

What requires CUDA/Triton/kernel changes:

- Actually skipping non-candidate KV blocks during attention.
- Efficient candidate-aware paged attention.
- Fixed-shape candidate buffers compatible with CUDAGraph replay.
- Fused sketch update if standalone torch ops are too expensive.

What is safe for Mac/local testing:

- Pure-Python/NumPy offline validation.
- HF CPU extraction and torch CPU experiments.
- Observer/index/selector/export tests.
- API/interface tests for torch sketch backend without CUDA, if kept CPU-safe.

What likely requires Linux/NVIDIA:

- vLLM GPU runtime execution.
- Real CUDA graph compatibility checks.
- FlashAttention/PagedAttention candidate-path prototyping.
- Meaningful latency/throughput benchmarks.

## E. Recommended next implementation sequence

Phase 2.6: real sketch storage design / interfaces

- Define storage keys and tensor shapes for per-layer/per-KV-head/per-block
  sketches.
- Separate metadata index from tensor storage.
- Specify ownership cleanup on free/preempt/prefix-cache eviction.

Phase 2.7: offline torch sketch backend benchmark

- Implement torch CPU/GPU sketch kernels outside vLLM runtime first.
- Benchmark CountSketch dim 64 and Random Projection dim 64.
- Measure sketch update and query scoring overhead separately.

Phase 2.8: runtime tensor capture dry-run if feasible

- Add gated instrumentation at attention backend `forward(...)` or
  `do_kv_cache_update(...)`.
- Compute/log real sketches and real query scores, but ignore decisions.
- Export compact diagnostics after generation.

Phase 3.0: candidate-block attention path prototype

- Add candidate metadata to attention path.
- Implement backend/kernel prototype that attends over selected blocks plus
  mandatory recent/local window.
- Keep full-attention fallback for unsupported modes.

## Risk analysis

Overhead of sketching K:

- Updating sketches during every KV write can add per-token/per-layer overhead.
- Fused or batched GPU ops may be required for acceptable cost.

Overhead of scoring blocks per query:

- Per-query scoring across many blocks, layers, and heads can be expensive.
- Candidate budgets, layer/head policies, and batching are necessary.

Batching complications:

- Each request has different sequence length, block count, and query count.
- Candidate metadata must be padded or represented in fixed-capacity buffers for
  CUDA graph compatibility.

Prefix cache interaction:

- Prefix-cached blocks may be shared and reused.
- Sketch storage must respect block ownership/refcounts and avoid stale sketches
  when physical blocks are recycled.

Sliding-window attention:

- Sliding window imposes mandatory local attention semantics.
- Candidate selection must always include required recent/window blocks.

MQA/GQA:

- Query heads may outnumber KV heads.
- Policy must decide whether to score per query head, per KV head, or via a
  reduced group score.

FlashAttention/PagedAttention compatibility:

- Existing kernels expect standard metadata and block-table semantics.
- Candidate-aware sparse block execution likely needs backend-specific support.

Memory overhead of sketches:

- Sketch storage scales with layers, KV heads, physical blocks, and sketch_dim.
- CountSketch dim 64 is compact, but full model/block scale still needs memory
  accounting.

## Explicit non-goals

- No model architecture changes.
- No training or finetuning.
- No tokenizer changes.
- No quality claims.
- No KV memory reduction claims yet.
- No kernel changes in this phase.
- No attention behavior change in this phase.
