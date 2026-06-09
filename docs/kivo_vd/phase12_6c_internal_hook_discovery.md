# Phase 12.6C: Plugin-Based Internal Hook Discovery

## Purpose

Phase 12.6C inventories installed-wheel vLLM `0.22.1` runtime surfaces before
any internal hook is proposed. Discovery comes first because method names such
as `schedule`, `execute_model`, or `allocate_slots` may expose valuable
metadata while also sitting on correctness- and latency-critical paths.

This phase imports modules, inspects known classes and callables, records
signatures and provenance, and ranks candidates. It installs no wrapper and
changes no runtime behavior.

## Inspected Areas

The fixed catalog covers:

- public `vllm.LLM.generate`;
- `LLMEngine` and `EngineCore` step boundaries;
- current and legacy scheduler import paths;
- `GPUModelRunner.execute_model`;
- private attention metadata and slot-mapping builders;
- `KVCacheManager` allocation, metadata, and free methods;
- KV-cache utility free-block metadata;
- worker block-table construction, CPU views, and slot mapping;
- FlashAttention metadata and forward methods;
- KV-cache metrics callbacks.

Missing modules and methods are report data, not fatal errors. This makes the
same catalog useful across nearby vLLM versions without pretending their
internal layouts are stable.

## Candidate Metadata

Each row records:

- module, class, and method names;
- module and callable availability;
- callable signature;
- source file and bounded docstring preview;
- optional bounded source preview;
- category signals such as scheduler, model execution, KV allocation,
  block-table construction, slot mapping, or attention backend;
- risk and usefulness levels;
- a conservative reason.

Every candidate sets:

```text
safe_to_patch_in_phase12_6c=false
discovery_only=true
```

## Ranking Rules

Candidates are sorted by usefulness first and lower risk second. This is an
inspection order, not an authorization order.

- `low` risk: previously validated public boundary.
- `medium` risk: read-oriented metadata or metrics surfaces that still need
  call-semantics review.
- `high` risk: scheduler, model execution, KV mutation, block-table, slot
  mapping, attention metadata, or attention-forward paths.

High usefulness never implies that a method is safe to patch. High-risk
methods are inventory-only in Phase 12.6C.

## Installed-Wheel Provenance

The report records `vllm.__file__` and classifies it as:

- installed wheel when it resolves under `site-packages` or `dist-packages`;
- repository-local source otherwise.

Phase 12.6D review readiness requires an installed-wheel path and at least one
callable catalog candidate. A repository-local import produces a
`needs_attention` result and cannot pass the readiness check.

## RunPod Command

Use the validated environment:

- vLLM `0.22.1`;
- PyTorch `2.11.0+cu130`;
- CUDA 13.0;
- RTX 4090;
- installed-wheel vLLM.

Install only the plugin package:

```bash
cd /workspace/vllm-kivo-vd
git checkout chore/sync-upstream-main
git pull origin chore/sync-upstream-main
uv pip install --system -e plugins/kivo_vllm_shadow_plugin
```

Do not run `uv pip install -e .`.

Copy the scripts and run from `/tmp`:

```bash
rm -rf /tmp/kivo_phase12
mkdir -p /tmp/kivo_phase12
cp -r /workspace/vllm-kivo-vd/scripts/kivo_vd /tmp/kivo_phase12/

cd /tmp

PYTHONPATH=/tmp/kivo_phase12:/tmp/kivo_phase12/kivo_vd \
python -m kivo_vd.run_phase12_vllm_internal_hook_discovery \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_6c_internal_hook_discovery.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_6c_internal_hook_discovery.md \
  --continue-on-error
```

Optional source previews can be added with
`--include-source-previews`. They are disabled by default.

## Success Criteria

- vLLM imports from `site-packages` or `dist-packages`;
- discovery completes;
- at least one catalog callable is available;
- missing modules are recorded without aborting;
- ranked candidates and recommendations are present;
- `patch_installed=false`;
- `runtime_behavior_changed=false`;
- `active_routing=false`;
- `measured_runtime_reduction=false`.

## Caveats

Inspection can show that a method exists and reveal its Python signature. It
cannot prove that wrapping the method is safe under multiprocessing,
compilation, batching, or distributed execution.

Scheduler, attention, KV-cache, block-table, and slot-mapping methods remain
higher risk even when their signatures look convenient. No source build,
internal monkeypatch, active routing, selected attention, measured memory
reduction, latency improvement, or quality claim is part of this phase.

## Next Phase Options

If RunPod discovery succeeds, Phase 12.6D may review one candidate area for a
copied-metadata, passive observation design. That review must answer:

1. whether the surface is stable enough for installed vLLM `0.22.1`;
2. whether observing it can be fail-closed and opt-in;
3. whether inputs can be copied without mutation or synchronization hazards;
4. whether a public or metrics surface avoids a high-risk execution path.

Discovery alone does not authorize installation of an internal hook.
