# Phase 12.6A RunPod Validation Summary

## What Passed

The marker-only `kivo_shadow` plugin was discovered through the official
`vllm.general_plugins` entry point on an RTX 4090 RunPod environment using:

- vLLM `0.22.1`;
- PyTorch `2.11.0+cu130`;
- CUDA 13.0;
- installed-wheel vLLM from
  `/usr/local/lib/python3.12/dist-packages/vllm/__init__.py`.

The environment-only probe wrote a valid marker. The GPT-2 generation probe
also succeeded and produced output containing `The first`. Both reported
`phase12_6b_plugin_shadow_hook_candidate=true` and
`active_routing=false`.

## What Failed Initially

Plugin loading also worked with PyTorch `2.8.0+cu128` and vLLM `0.10.2`, but
GPT-2/OPT generation failed with:

```text
GPT2Tokenizer has no attribute all_special_tokens_extended
```

That older stack is excluded from Phase 12.6B.

## Phase 12.6B Environment

Use vLLM `0.22.1` with PyTorch `2.11.0+cu130`. Install only
`plugins/kivo_vllm_shadow_plugin`, run copied Kivo scripts from `/tmp`, and
keep the repository root out of `PYTHONPATH`. Do not install the repository
itself as vLLM or import its unbuilt local `vllm/` source.

## Safety Boundary

This validation proves plugin discovery and invocation only. It does not
prove access to runtime block tables, KV cache, scheduler metadata, attention
metadata, or decode tensors. Phase 12.6B may explore only an opt-in, passive,
fail-closed no-op observation surface.

No monkeypatch, active routing, KV mutation, block-table mutation, measured
memory reduction, or latency improvement is present or claimed.
