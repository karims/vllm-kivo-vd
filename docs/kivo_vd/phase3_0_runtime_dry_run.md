# Kivo-VD Phase 3.0: Runtime Dry-Run Entry Point

Phase 3.0 adds a small script for running a real vLLM offline generation path
with Kivo-VD dry-run enabled.

This phase remains dry-run only. It does not alter attention, block tables,
slot mapping, attention metadata, kernels, model weights, tokenizer behavior,
or model architecture.

## Script

```bash
.venv/bin/python scripts/kivo_vd/run_vllm_kivo_dry_run.py \
  --model sshleifer/tiny-gpt2 \
  --max-tokens 8 \
  --enable-kivo-vd \
  --gpu-memory-utilization 0.05 \
  --max-model-len 128 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 1
```

Defaults:

- Model: `sshleifer/tiny-gpt2`
- Prompt: a short Kivo-VD dry-run prompt
- GPU memory utilization: `0.05`
- Max model length: `128`
- Max batched tokens: `128`
- Max sequences: `1`
- Event output: `outputs/kivo_vd/vllm_kivo_dry_run_events.jsonl`
- Baseline comparison: enabled
- In-process V1 engine core: enabled by setting
  `VLLM_ENABLE_V1_MULTIPROCESSING=0` before importing vLLM

The runtime limits are intentionally conservative. They are meant to prevent
tiny validation runs, such as `sshleifer/tiny-gpt2` on an RTX 4090, from
planning a very large KV cache. This script favors correctness/debugging over
throughput.

## What The Script Does

The script:

1. Runs a baseline generation with Kivo-VD disabled.
2. Runs a second generation with Kivo-VD enabled when `--enable-kivo-vd` is set.
3. Uses greedy sampling (`temperature=0.0`) for a stable comparison.
4. Attempts to access the in-process scheduler observer.
5. Calls `KivoVDObserver.export_events(...)` if the observer is reachable.
6. Prints compact JSON with:
   - model
   - prompt token length if available
   - GPU/runtime safety limits
   - baseline text
   - Kivo text
   - whether outputs match exactly
   - event export path
   - number of events exported
   - observer counters if available

## Config Path

The current Kivo fields live on `VllmConfig`:

- `enable_kivo_vd`
- `kivo_vd_event_export_path`
- `kivo_vd_export_event_limit`

`LLM(..., **kwargs)` currently flows through `EngineArgs`, and `EngineArgs`
does not expose these Kivo fields as public constructor arguments yet. To avoid
adding broad public API surface in this dry-run phase, the script locally wraps
`EngineArgs.create_engine_config(...)` and sets the Kivo fields on the resulting
`VllmConfig`.

Future cleanup can add a proper public config path once runtime validation
confirms the desired UX.

## Observer Accessibility

For local debug export, the script defaults to an in-process engine core. In
that mode, the observer is reachable through:

```text
llm.llm_engine.engine_core.engine_core.scheduler.kivo_vd_observer
```

If V1 multiprocessing is enabled, the scheduler lives in another process and
the observer is not directly reachable from this script. In that case, future
work should add an explicit engine-core utility RPC for Kivo event export.

## What Counts As Success

For Phase 3.0, success means:

- vLLM inference completes locally.
- The Kivo-enabled dry-run path produces the same greedy output as baseline.
- Dry-run lifecycle/routing events are exported when the observer is accessible.
- If export is not accessible, the script reports that limitation clearly.

## Mac/Local Limitations

Local Mac runs may fail if the current environment cannot load vLLM runtime
dependencies, if the selected model is not cached and networking is unavailable,
or if the platform lacks the backend needed for the selected model. Those are
runtime environment limitations, not evidence that Kivo dry-run routing changes
attention behavior.

In an unbuilt local source tree, CPU runtime initialization may fail with a
missing compiled extension error such as:

```text
AttributeError: '_OpNamespace' '_C' object has no attribute 'init_cpu_memory_env'
```

That means vLLM's compiled CPU extension is not available in the current
environment. Build/install the repo-supported runtime before using this script
as a real inference validation.

## Future Work

- Add a public Kivo runtime config path.
- Add an engine-core utility RPC for event export in multiprocessing mode.
- Compute real K tensor sketches.
- Score real query vectors at runtime.
- Prototype candidate-block attention.
- Measure real GPU KV memory, latency, and quality.
