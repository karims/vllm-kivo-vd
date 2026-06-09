# Phase 12.5 RunPod Validation Summary

## What Passed

Phase 12.5 passed on an NVIDIA RTX 4090 using:

- Torch `2.11.0+cu130`;
- CUDA `13.0`;
- installed vLLM `0.22.1`;
- compiled `vllm._C`, stable-libtorch, and FlashAttention extensions.

Baseline generation and shadow-enabled generation both completed. The
post-generation Phase 12 shadow path wrote four events, and all four passed
the independent validator with no errors or warnings.

The readiness report returned `phase12_6_runtime_hook_ready=true`.

## What Failed Initially

Commands launched with the repository root on Python's import path selected
the repo-local unbuilt `vllm` package rather than the installed wheel. That
package could not import `vllm._C`.

Building the local source checkout was deferred because the extension build
expected missing Torch headers, including:

```text
torch/headeronly/util/Exception.h
```

## Working Isolation Method

Only `scripts/kivo_vd` was copied to `/tmp/kivo_phase12`. The workflow then
ran from `/tmp` with:

```text
PYTHONPATH=/tmp/kivo_phase12:/tmp/kivo_phase12/kivo_vd
```

This kept `/workspace/vllm-kivo-vd` out of the import path while allowing Kivo
script imports and writing reports back into the repository.

## Boundary Before Phase 12.6

The installed wheel is valid evidence for Phase 12.5 because this phase does
not patch vLLM runtime files. It is not sufficient evidence for a future
repo-local runtime hook.

Phase 12.6 may proceed to design or separately review one opt-in shadow hook,
but executing modified runtime code requires either:

- a compatible local vLLM source build; or
- a reviewed patch/overlay mechanism that proves the modified code is running.

No active routing, measured memory reduction, latency improvement, or
generation-quality preservation has been demonstrated.
