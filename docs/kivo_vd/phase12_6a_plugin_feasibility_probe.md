# Phase 12.6A: Plugin Feasibility Probe

## Purpose

Phase 12.6A tests whether installed-wheel vLLM can discover and invoke a
separately installed Kivo package through the official Python entry-point
plugin system.

Plugin style is preferred before source-building or forking vLLM because it
keeps the working installed wheel intact and gives Kivo an out-of-tree
integration boundary. This phase tests discovery only. It does not assume the
general plugin API exposes block tables, decode metadata, or attention hooks.

## vLLM Plugin Contract

vLLM general plugins use:

```text
vllm.general_plugins
```

The Kivo package registers:

```toml
[project.entry-points."vllm.general_plugins"]
kivo_shadow = "kivo_vllm_shadow_plugin.plugin:register"
```

`VLLM_PLUGINS=kivo_shadow` filters loading to this plugin. vLLM may load
general plugins in multiple processes, so `register()` is re-entrant and only
writes an atomic marker file when `KIVO_SHADOW_PLUGIN_MARKER` is configured.

## Package Layout

```text
plugins/kivo_vllm_shadow_plugin/
  pyproject.toml
  README.md
  kivo_vllm_shadow_plugin/
    __init__.py
    plugin.py
```

The package declares no dependency on the repo-local `vllm` source tree. Its
marker includes:

- plugin name and load timestamp;
- Python executable, process ID, and current directory;
- a bounded `sys.path` preview;
- imported vLLM version and file path when available;
- explicit no-mutation caveats.

The plugin does not monkeypatch, register a model, inspect KV tensors, or
change scheduler or attention behavior.

## RunPod Installation

Install only the plugin package into the environment containing the working
vLLM wheel:

```bash
cd /workspace/vllm-kivo-vd
git pull origin chore/sync-upstream-main

uv pip install --system -e plugins/kivo_vllm_shadow_plugin
```

Do not install the repository itself as vLLM and do not add the repository
root to `PYTHONPATH`.

## Isolated Environment Probe

Copy only the Kivo scripts:

```bash
rm -rf /tmp/kivo_phase12
mkdir -p /tmp/kivo_phase12
cp -r /workspace/vllm-kivo-vd/scripts/kivo_vd /tmp/kivo_phase12/

cd /tmp

PYTHONPATH=/tmp/kivo_phase12:/tmp/kivo_phase12/kivo_vd \
python -m kivo_vd.run_phase12_vllm_plugin_probe \
  --skip-generation \
  --marker-path \
    /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_6a_plugin_marker_env.json \
  --output-json \
    /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_6a_plugin_probe_env.json \
  --output-md \
    /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_6a_plugin_probe_env.md \
  --continue-on-error
```

The probe sets:

```text
VLLM_PLUGINS=kivo_shadow
KIVO_SHADOW_PLUGIN_MARKER=<marker path>
```

It imports installed vLLM, calls the official general-plugin loader, and
checks that the marker reports `loaded=true` and
`plugin_name=kivo_shadow`.

## Generation Probe

After environment-only discovery succeeds:

```bash
cd /tmp

PYTHONPATH=/tmp/kivo_phase12:/tmp/kivo_phase12/kivo_vd \
python -m kivo_vd.run_phase12_vllm_plugin_probe \
  --model gpt2 \
  --prompt "Kivo Phase 12 plugin generation probe." \
  --max-tokens 4 \
  --marker-path \
    /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_6a_plugin_marker_generation.json \
  --output-json \
    /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_6a_plugin_probe_generation.json \
  --output-md \
    /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_6a_plugin_probe_generation.md \
  --continue-on-error
```

`phase12_6b_plugin_shadow_hook_candidate=true` means plugin discovery,
invocation, and the requested generation mode completed. It does not mean the
plugin can observe useful internal metadata.

## Caveats

- Plugin loading can occur in multiple vLLM processes.
- A marker proves entry-point invocation only.
- General plugins are commonly used for out-of-tree registration, not as a
  guaranteed scheduler or attention callback API.
- No active routing, KV mutation, block-table mutation, or monkeypatch exists.
- No measured memory, latency, or quality claim is made.

## Next Decision

If the marker and generation probe pass, Phase 12.6B can inspect whether a
plugin can obtain useful passive metadata through a documented extension
surface. If not, the project may need a compatible source build or a
separately reviewed source-overlay strategy. Active selected attention remains
outside Phase 12.
