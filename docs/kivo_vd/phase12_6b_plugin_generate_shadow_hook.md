# Phase 12.6B: Plugin-Owned Generate Shadow Hook

## Purpose

Phase 12.6B adds an opt-in, plugin-owned wrapper around the public
`vllm.LLM.generate` method. This is the least invasive runtime-adjacent
boundary available after Phase 12.6A proved that installed-wheel vLLM can
discover the separately installed `kivo_shadow` plugin.

The wrapper proves that a plugin can observe a completed public generation
call and emit Phase 12-compatible preview events without modifying any
repository-local vLLM file.

It does not provide scheduler, block-table, KV-cache, attention metadata, or
decode-step access.

## Safety Properties

The wrapper is disabled by default and enabled only with:

```text
KIVO_SHADOW_PLUGIN_PATCH_GENERATE=1
```

When enabled, it:

1. calls the original `LLM.generate` method with the original arguments;
2. stores the exact returned object;
3. attempts preview event emission after generation;
4. records an emission warning in the marker if shadow logic fails;
5. returns the exact original result object.

It never changes prompts, sampling parameters, generated outputs, scheduler
state, KV cache, block tables, attention metadata, or kernels. Installation
is re-entrant and wraps `LLM.generate` at most once per process.

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `KIVO_SHADOW_PLUGIN_PATCH_GENERATE` | disabled | Enables the wrapper |
| `KIVO_SHADOW_PLUGIN_EVENTS` | unset | Preview event JSONL path |
| `KIVO_SHADOW_PLUGIN_MARKER` | unset | Plugin marker JSON path |
| `KIVO_SHADOW_PLUGIN_LAYERS` | `0,5,8,11` | Preview event layers |
| `KIVO_SHADOW_PLUGIN_BLOCK_SIZE` | `16` | Preview block size |
| `KIVO_SHADOW_PLUGIN_RATIO_POLICY` | `balanced=0:0.60,5:0.45,8:0.45,11:0.60` | Per-layer preview ratios |

## Preview Event Semantics

The public generate boundary has no real block-table view. Events therefore
set:

```text
selector_policy=plugin_generate_boundary_preview
preview_only=true
shadow_only=true
active_routing=false
measured_runtime_reduction=false
```

Prompt token count comes from public output `prompt_token_ids` when available.
Otherwise, the plugin uses a conservative prompt-length estimate. Total block
count is derived from that count and the configured block size.

Selected IDs are deterministic synthetic previews. Score order is descending,
while gather order is ascending over the same set. These IDs exist only to
exercise the Phase 12 event and validation plumbing. They are never passed
back into vLLM.

## Validated RunPod Environment

Use the Phase 12.6A working stack:

- RTX 4090;
- vLLM `0.22.1`;
- PyTorch `2.11.0+cu130`;
- CUDA 13.0;
- installed-wheel vLLM from `site-packages`.

Install only the plugin package:

```bash
cd /workspace/vllm-kivo-vd
git checkout chore/sync-upstream-main
git pull origin chore/sync-upstream-main
uv pip install --system -e plugins/kivo_vllm_shadow_plugin
```

Do not run `uv pip install -e .` and do not add the repository root to
`PYTHONPATH`.

Copy only the scripts:

```bash
rm -rf /tmp/kivo_phase12
mkdir -p /tmp/kivo_phase12
cp -r /workspace/vllm-kivo-vd/scripts/kivo_vd /tmp/kivo_phase12/
```

Run the probe from `/tmp`:

```bash
cd /tmp

PYTHONPATH=/tmp/kivo_phase12:/tmp/kivo_phase12/kivo_vd \
python -m kivo_vd.run_phase12_vllm_plugin_probe \
  --model gpt2 \
  --prompt "Kivo Phase 12 plugin generate hook probe." \
  --max-tokens 4 \
  --enable-generate-hook \
  --events-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_6b_plugin_generate_events.jsonl \
  --marker-path /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_6b_plugin_generate_marker.json \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_6b_plugin_generate_probe.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_6b_plugin_generate_probe.md \
  --continue-on-error
```

Validate the event file independently:

```bash
PYTHONPATH=/tmp/kivo_phase12:/tmp/kivo_phase12/kivo_vd \
python -m kivo_vd.validate_phase12_shadow_event \
  --input /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_6b_plugin_generate_events.jsonl \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_6b_plugin_generate_validation.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_6b_plugin_generate_validation.md
```

## Success Criteria

- `plugin_loaded=true`;
- `plugin_marker_written=true`;
- `patch_generate_requested=true`;
- `patch_generate_installed=true`;
- `generation_status=succeeded`;
- `events_written > 0`;
- `validation_passed=true`;
- `active_routing=false`;
- `measured_runtime_reduction=false`;
- generated output returns normally.

`phase12_6c_internal_hook_candidate=true` means only that plugin discovery,
the public wrapper, generation, event emission, and validation all passed.

## Boundaries And Next Step

This phase uses a deliberate public-method monkeypatch owned entirely by the
separate plugin package. It does not patch scheduler or attention internals.
Preview events are not evidence of real KV-block selection.

Only after this phase passes may Phase 12.6C investigate whether a documented
or separately reviewed passive internal metadata surface exists. Any such
work must remain opt-in, shadow-only, fail-closed, and free of active routing.

No measured memory reduction, latency improvement, quality preservation, or
active selected attention is claimed.
