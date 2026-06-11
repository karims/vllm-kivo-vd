# Phase S3.0A: Attention Metadata Path Discovery

This note traces the current vLLM path from block-table / slot-mapping
construction to attention metadata packaging and finally to backend attention
invocation.

This phase is discovery only. It does not mutate runtime behavior, does not
change CUDA/C++/CMake files, and does not attempt selected attention yet.

## Files And Functions Inspected

### Block table and slot mapping

- `vllm/v1/worker/block_table.py`
  - `BlockTable.compute_slot_mapping(...)`
  - `MultiGroupBlockTable.compute_slot_mapping(...)`
  - `_compute_slot_mapping_kernel(...)`
  - `maybe_observe_compute_slot_mapping(...)`

### GPU runner and metadata packaging

- `vllm/v1/worker/gpu_model_runner.py`
  - `_get_slot_mappings(...)`
  - `_build_attention_metadata(...)`
  - the `set_forward_context(...)` call around model execution

- `vllm/v1/worker/gpu/attn_utils.py`
  - `build_attn_metadata(...)`
  - `build_slot_mappings_by_layer(...)`

- `vllm/forward_context.py`
  - `ForwardContext`
  - `set_forward_context(...)`
  - `get_forward_context(...)`

### Attention entrypoint and backend builders

- `vllm/model_executor/layers/attention/attention.py`
  - `Attention.forward(...)`
  - `unified_kv_cache_update(...)`
  - `unified_attention_with_output(...)`
  - `get_attention_context(...)`

- `vllm/v1/attention/backend.py`
  - `CommonAttentionMetadata`
  - `AttentionMetadataBuilder`
  - `AttentionMetadataBuilder.build(...)`
  - `AttentionMetadataBuilder.update_block_table(...)`

- Backend-specific metadata and kernels
  - `vllm/v1/attention/backends/flash_attn.py`
  - `vllm/v1/attention/backends/triton_attn.py`
  - `vllm/v1/attention/backends/cpu_attn.py`
  - `vllm/v1/attention/backends/flashinfer.py`
  - `vllm/v1/attention/backends/mla/indexer.py`
  - `vllm/v1/attention/backends/utils.py`

## Likely Path From Block Metadata To Attention Backend

The observed path is:

1. `BlockTable.compute_slot_mapping(...)` materializes slot IDs from the block
   table and token positions.
2. `gpu_model_runner._get_slot_mappings(...)` reads the per-group slot mapping
   tensors for the current step.
3. `gpu_model_runner._build_attention_metadata(...)` packages the per-step
   data into `CommonAttentionMetadata`, including:
   - `block_table_tensor`
   - `slot_mapping`
   - `query_start_loc`
   - `seq_lens`
   - `positions`
   - `max_query_len` / `max_seq_len`
4. `build_attn_metadata(...)` in `vllm/v1/worker/gpu/attn_utils.py` converts
   the common batch metadata into per-group metadata dictionaries.
5. `set_forward_context(...)` stores the attention metadata and per-layer slot
   mappings in `ForwardContext`.
6. `vllm/model_executor/layers/attention/attention.py::Attention.forward(...)`
   retrieves the current layer’s metadata from `ForwardContext` via
   `get_attention_context(...)`.
7. `unified_attention_with_output(...)` calls the backend-specific
   `AttentionImpl.forward(...)`.
8. Backend implementations (`flash_attn`, `triton_attn`, `cpu_attn`,
   `flashinfer`, MLA variants) consume the metadata and launch the actual
   attention ops / kernels.

## Candidate Interception Points

### 1. Safest observer-only point

**File:** `vllm/v1/worker/block_table.py`  
**Function:** `BlockTable.compute_slot_mapping(...)`  
**Why it is safe:** This is the first point where slot IDs are computed from
request positions and block-table state. The existing
`maybe_observe_compute_slot_mapping(...)` hook already lets us inspect the
resulting mapping without changing behavior.

**Data available there:**
- `query_start_loc` (`torch.Tensor`, GPU-visible input to the Triton kernel)
- `positions` (`torch.Tensor`)
- `self.block_table` / `self.slot_mapping` (`CpuGpuBuffer` with CPU and GPU
  views)

**Mutation risk:** High if changed; this is still the authoritative slot map
that downstream attention relies on. Observer-only use is appropriate.

### 2. Safest future selected-attention point

**File:** `vllm/v1/worker/gpu_model_runner.py`  
**Functions:** `_get_slot_mappings(...)`, `_build_attention_metadata(...)`  
**Related helper:** `vllm/v1/worker/gpu/attn_utils.py::build_attn_metadata(...)`

This is the best backend-agnostic place to introduce future selected-attention
experiments because the runner already has:
- `block_tables` / `block_table_tensor`
- per-group `slot_mappings`
- `query_start_loc`
- `seq_lens`
- `positions`

At this stage the data is still in Python/PyTorch metadata form, before it is
frozen into `ForwardContext` and before backend-specific kernels run.

**Data available there:**
- `block_table_tensor` (`torch.Tensor`, GPU)
- `slot_mapping` (`torch.Tensor`, GPU)
- `query_start_loc` (`torch.Tensor`, GPU + CPU mirror)
- `seq_lens` (`torch.Tensor`)
- `positions` (`torch.Tensor`)
- per-kv-group mappings (`dict[int, torch.Tensor]`)

**Why it is suitable for future filtering:** A future observer or selector can
derive a filtered block-table / slot-mapping view here and hand that forward to
backend metadata builders without touching the low-level attention kernels yet.

**Caveat:** Some backends rewrite metadata further during
`AttentionMetadataBuilder.build(...)` or `update_block_table(...)`, so a
backend-specific override may still be needed for architectures with special
compression or decode layouts.

### 3. Backend-specific metadata adaptation point

**File:** `vllm/v1/attention/backend.py`  
**Class:** `AttentionMetadataBuilder`  
**Methods:** `build(...)`, `update_block_table(...)`

This is the shared abstraction that backend builders use to turn
`CommonAttentionMetadata` into backend-specific metadata objects such as:
- `FlashAttentionMetadata`
- `TritonAttentionMetadata`
- `CPUAttentionMetadata`
- `FlashInferMetadata`
- MLA / indexer metadata classes

**Data available there:**
- `CommonAttentionMetadata.block_table_tensor`
- `CommonAttentionMetadata.slot_mapping`
- `CommonAttentionMetadata.query_start_loc`
- `CommonAttentionMetadata.seq_lens`
- backend-specific knobs (sliding window, cascade, quantization, etc.)

**Why it matters:** This is the most natural place to adapt selected blocks for
individual attention backends if the future design needs backend-specific
metadata rewriting.

### 4. Risky / avoid points for now

**File:** `vllm/model_executor/layers/attention/attention.py`  
**Functions:** `Attention.forward(...)`, `unified_attention_with_output(...)`

**File:** backend `forward(...)` methods  
**Functions:** `FlashAttentionImpl.forward(...)`, `TritonAttentionImpl.forward(...)`,
`CPUAttentionBackendImpl.forward(...)`, `FlashInferBackend.forward(...)`

These points are too close to the compiled attention kernels. By the time code
reaches them, metadata is already committed to backend-specific execution.
Mutating here would be harder to reason about and may require kernel or custom
op changes.

**Do not mutate yet.**

## Attention Metadata Objects And Tensors That Matter

- `block_table` / `block_table_tensor`
  - Source: `BlockTable` / `CommonAttentionMetadata`
  - Type: CPU+GPU buffer in the worker, then `torch.Tensor`
  - Role: per-request page/block list consumed by attention backends

- `slot_mapping`
  - Source: `BlockTable.compute_slot_mapping(...)` and runner helpers
  - Type: CPU+GPU buffer in the worker, then `torch.Tensor`
  - Role: token-to-KV-slot map used by cache update and some backends

- `seq_lens`
  - Source: `gpu_model_runner._build_attention_metadata(...)`
  - Type: `torch.Tensor` with CPU mirrors in some flows
  - Role: per-request context length used by backend scheduling and kernels

- `query_start_loc`
  - Source: `gpu_model_runner._build_attention_metadata(...)`
  - Type: `torch.Tensor` plus CPU mirror
  - Role: query token offsets per request

- `paged attention metadata`
  - Source: backend builders such as FlashAttention, Triton, CPU, FlashInfer,
    and MLA builders
  - Type: backend-specific dataclasses holding tensors and scalar kernel knobs
  - Role: final bridge into the actual attention kernel call

## Conclusion

**Best next hook candidate:** `gpu_model_runner._build_attention_metadata(...)`
and the shared `build_attn_metadata(...)` helper, because they are the last
backend-agnostic Python points where block tables, slot mappings, sequence
lengths, and query offsets are all visible before the metadata is committed to
the model forward context.

**Observer-only hook:** `BlockTable.compute_slot_mapping(...)` is already the
right place to observe how slot mappings are formed.

**Do not mutate yet.** This phase only identifies the attention metadata path
for the next observer or selection hook.
