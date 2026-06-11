# Phase S3.3A Attention Tensor Sketch Observer

## Purpose

Phase S3.3A identifies a viable Python-level source for future real Kivo-VD
sketch construction. It observes tensor metadata only. It does not compute
sketches, select blocks, or mutate runtime behavior.

The hook is in
`vllm/model_executor/layers/attention/attention.py` inside
`unified_attention_with_output(...)`, after `get_attention_context(...)` and
before the backend `forward(...)` call. At this boundary the runtime exposes:

- reshaped query, key, and value tensors;
- the bound per-layer KV-cache tensor;
- backend-specific attention metadata;
- the per-layer slot mapping;
- the attention layer name and backend.

This is preferable to instrumenting one backend because GPT-2 may use
FlashAttention while other configurations may use Triton or FlashInfer.

## Observation Policy

Enable the observer with:

```bash
KIVO_SOURCE_ENABLE=1
KIVO_SOURCE_POLICY=observe_attention_tensors_for_sketch
KIVO_SOURCE_OBSERVE_PATH=/path/to/events.jsonl
```

Events use:

```text
schema_version = kivo_source_s3_3a_attention_tensor_sketch_observer_v1
policy_name = observe_attention_tensors_for_sketch
```

The observer records shapes, dtypes, devices, dimensions, and element counts.
It does not copy tensor contents to CPU and does not record tensor values.
Failures are swallowed so observation cannot interrupt attention.

## Sketch-Source Interpretation

The query tensor is the likely source for a future per-step query sketch. The
key tensor contains keys produced for tokens in the current forward pass and
is a possible source for incremental sketch updates.

The KV-cache tensor is the most direct candidate for existing per-block key
summaries when its backend-specific block layout is visible. Using it safely
will require understanding layout, quantization, layer/head structure, and
the cost of GPU-side sketch operations. Merely observing the tensor does not
establish that reading full cache blocks is efficient.

The event field `recommended_sketch_source` is therefore a feasibility hint,
not a runtime policy:

- `kv_cache` when a block-shaped cache tensor is visible;
- `key` when current-forward keys are visible but a usable cache is not;
- `query` or `value` as weaker fallbacks;
- `metadata_proxy_only` or `unknown` when tensor sources are unavailable.

## RunPod Probe

```bash
cd /workspace/vllm-kivo-vd

PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.run_source_s3_3a_attention_tensor_sketch_observer \
  --model gpt2 \
  --max-tokens 8 \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_3a_attention_tensor_sketch_observer.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_3a_attention_tensor_sketch_observer.md \
  --events-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_3a_attention_tensor_sketch_observer_events.jsonl \
  --continue-on-error
```

Validate the artifacts:

```bash
PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.validate_source_s3_3a_attention_tensor_sketch_observer \
  --input-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_3a_attention_tensor_sketch_observer.json \
  --events-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_3a_attention_tensor_sketch_observer_events.jsonl \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_3a_attention_tensor_sketch_observer_validation.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_3a_attention_tensor_sketch_observer_validation.md
```

The validator filters mixed JSONL files by the S3.3A schema and ignores
unrelated Kivo event schemas.

## Success Criteria

S3.3A passes when:

- all baseline and observer generations succeed;
- observer outputs match baseline outputs;
- at least one S3.3A event is written;
- at least one query, key, value, or KV-cache tensor is visible;
- every mutation and runtime-change flag remains false.

## Boundary

S3.3A does not implement final sketches. It does not prove selected attention,
KV-memory reduction, latency reduction, or quality preservation. If useful
Q/K/V or KV-cache tensors are observed, S3.3B can evaluate bounded GPU-side
per-block sketch construction. If they are not visible, the next experiment
must be backend-specific or lower-level.
