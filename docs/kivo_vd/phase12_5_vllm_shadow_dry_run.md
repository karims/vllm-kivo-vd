# Phase 12.5: vLLM Shadow Dry-Run

## Purpose

Phase 12.5 combines three checks in one report:

1. vLLM, Torch, CUDA, and compiled-extension import status;
2. an optional conservative baseline vLLM generation;
3. validator-compatible Phase 12 events emitted after generation through the
   standalone runtime touchpoint.

This is not an active runtime hook. Generation uses normal vLLM behavior, and
the runtime-adjacent shadow events are produced separately afterward.

## Safety Boundary

The workflow does not modify:

- scheduler decisions;
- GPUModelRunner;
- attention kernels or metadata;
- KV cache allocation or contents;
- block tables or slot mappings;
- generated output.

Every report states:

- `dry_run_only=true`;
- `shadow_only=true`;
- `active_routing=false`;
- `measured_runtime_reduction=false`;
- `no_attention_kernel_change=true`;
- `no_kv_cache_mutation=true`;
- `no_scheduler_change=true`.

## Environment-Only Check

```bash
.venv/bin/python scripts/kivo_vd/run_phase12_vllm_shadow_dry_run.py \
  --model gpt2 \
  --skip-vllm-generation \
  --output-json outputs/kivo_vd/runs/phase12_5_env_check.json \
  --output-md outputs/kivo_vd/runs/phase12_5_env_check.md \
  --continue-on-error
```

The report records Python, Torch, CUDA, GPU, vLLM, `vllm._C`,
`vllm._C_stable_libtorch`, and `vllm.vllm_flash_attn` import status. Optional
extension failures are reported rather than hidden.

## Installed Wheel Versus Repo-Local vLLM Source

Phase 12.5 can validate against an installed vLLM wheel because it does not
install a runtime hook or depend on modified repo-local vLLM code. On a pod
where the source checkout has no compiled CUDA extensions, use:

```bash
PYTHONPATH=/workspace/vllm-kivo-vd/scripts:\
/workspace/vllm-kivo-vd/scripts/kivo_vd \
python -m kivo_vd.run_phase12_vllm_shadow_dry_run \
  --prefer-installed-vllm \
  --model gpt2 \
  --skip-vllm-generation \
  --output-json \
    /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_5_env_check.json \
  --output-md \
    /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_5_env_check.md \
  --continue-on-error
```

`--prefer-installed-vllm` removes only import entries that resolve to the
repository root. It preserves `scripts/`, `scripts/kivo_vd/`, and
site-packages. Reports include:

- whether import-path sanitization was requested and performed;
- removed `sys.path` entries;
- the imported vLLM path;
- whether that path is repo-local;
- whether that path is under site-packages or dist-packages.

If installed-wheel mode still imports repo-local vLLM, environment readiness
is false. This prevents an unbuilt source package from being mistaken for a
working wheel.

For Phase 12.6 and later, an installed wheel is insufficient if repo-local
runtime files are modified. Those phases require either a compatible local
source build or a separately reviewed patch/overlay strategy.

## RunPod Validation Result

Phase 12.5 passed on a RunPod RTX 4090 environment using an installed vLLM
wheel and a clean script-copy import boundary.

### Environment

| item | validated value |
| --- | --- |
| GPU | NVIDIA RTX 4090 |
| Torch | `2.11.0+cu130` |
| Torch CUDA | `13.0` |
| vLLM | `0.22.1` |
| vLLM source | `/usr/local/lib/python3.12/dist-packages/vllm/__init__.py` |
| `vllm._C` | imported successfully |
| `vllm._C_stable_libtorch` | imported successfully |
| `vllm.vllm_flash_attn` | imported successfully |

Baseline generation completed successfully with the conservative runtime
limits. Shadow-enabled generation also completed successfully. The shadow
path wrote four events, and the independent validator reported:

- total events: `4`;
- valid events: `4`;
- invalid events: `0`;
- errors: none;
- warnings: none.

The final report returned:

- `phase12_6_runtime_hook_ready=true`;
- `active_routing=false`;
- `measured_runtime_reduction=false`;
- no attention-kernel change;
- no KV-cache or block-table mutation;
- no scheduler behavior change.

This result proves that the Phase 12.5 environment, baseline-generation, and
post-generation shadow-event workflow can run together on the target GPU
environment. It does not prove an in-runtime hook, active routing, memory
reduction, latency improvement, or generation-quality preservation.

## Import Isolation Lesson

Running from `/workspace/vllm-kivo-vd`, or otherwise retaining the repository
root on Python's import path, selected the repo-local unbuilt `vllm` package.
That source package could not load `vllm._C`. A local source build was also
blocked because the extension build expected unavailable Torch headers,
including:

```text
torch/headeronly/util/Exception.h
```

The installed vLLM wheel was healthy. The clean Phase 12.5 workaround copied
only the Kivo scripts to `/tmp`, kept the repository root out of the import
path, and wrote resulting artifacts back into the repository:

```bash
rm -rf /tmp/kivo_phase12
mkdir -p /tmp/kivo_phase12
cp -r /workspace/vllm-kivo-vd/scripts/kivo_vd /tmp/kivo_phase12/

cd /tmp

PYTHONPATH=/tmp/kivo_phase12:/tmp/kivo_phase12/kivo_vd \
python -m kivo_vd.run_phase12_vllm_shadow_dry_run \
  --prefer-installed-vllm \
  --model gpt2 \
  --prompt "Kivo Phase 12 vLLM shadow smoke." \
  --max-tokens 8 \
  --gpu-memory-utilization 0.05 \
  --max-model-len 128 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 1 \
  --enable-shadow \
  --shadow-output-jsonl \
    /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_5_shadow_events_tmpcopy.jsonl \
  --output-json \
    /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_5_vllm_generation_with_shadow_tmpcopy.json \
  --output-md \
    /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_5_vllm_generation_with_shadow_tmpcopy.md \
  --continue-on-error
```

This installed-wheel technique is acceptable for Phase 12.5 because the
workflow installs no runtime hook and modifies no vLLM runtime code. Phase
12.6 needs a different strategy if it modifies repo-local runtime files:

- build the source checkout against compatible Torch/CUDA headers; or
- use a separately reviewed patch or source-overlay strategy that guarantees
  the executed runtime includes the intended changes.

An installed wheel that does not contain future runtime changes cannot
validate those changes.

## Baseline Generation

```bash
.venv/bin/python scripts/kivo_vd/run_phase12_vllm_shadow_dry_run.py \
  --prefer-installed-vllm \
  --model gpt2 \
  --prompt "Kivo Phase 12 vLLM baseline smoke." \
  --max-tokens 8 \
  --gpu-memory-utilization 0.05 \
  --max-model-len 128 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 1 \
  --output-json \
    outputs/kivo_vd/runs/phase12_5_vllm_generation_no_shadow.json \
  --output-md \
    outputs/kivo_vd/runs/phase12_5_vllm_generation_no_shadow.md
```

No Phase 12 event file is created when shadow mode is disabled.

## Generation With Shadow Events

```bash
.venv/bin/python scripts/kivo_vd/run_phase12_vllm_shadow_dry_run.py \
  --prefer-installed-vllm \
  --model gpt2 \
  --prompt "Kivo Phase 12 vLLM shadow smoke." \
  --max-tokens 8 \
  --gpu-memory-utilization 0.05 \
  --max-model-len 128 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 1 \
  --enable-shadow \
  --shadow-output-jsonl \
    outputs/kivo_vd/runs/phase12_5_shadow_events.jsonl \
  --output-json \
    outputs/kivo_vd/runs/phase12_5_vllm_generation_with_shadow.json \
  --output-md \
    outputs/kivo_vd/runs/phase12_5_vllm_generation_with_shadow.md
```

The script uses actual prompt token count when vLLM exposes it. It then emits
four runtime-adjacent events for research layers `0`, `5`, `8`, and `11`.
Block scores remain synthetic; no query, key, value, or KV-cache tensor is
read.

Validate independently:

```bash
.venv/bin/python scripts/kivo_vd/validate_phase12_shadow_event.py \
  --input outputs/kivo_vd/runs/phase12_5_shadow_events.jsonl \
  --output-json \
    outputs/kivo_vd/runs/phase12_5_shadow_event_validation.json \
  --output-md \
    outputs/kivo_vd/runs/phase12_5_shadow_event_validation.md
```

## Readiness Interpretation

`phase12_6_runtime_hook_ready=true` requires all of:

- vLLM import succeeds;
- baseline generation succeeds;
- shadow mode is explicitly enabled;
- all emitted events pass validation.

Environment-only or synthetic-event-only success is insufficient. A passing
report permits consideration of one separately reviewed opt-in runtime hook;
it does not authorize active routing.

## Caveats

- Shadow events are emitted after generation, not from a real runtime hook.
- Scores are synthetic.
- No runtime memory reduction or latency improvement is measured.
- Generation success is not a quality-preservation claim.
- Active selected attention remains outside Phase 12.
