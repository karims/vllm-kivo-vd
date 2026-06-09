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

With Phase 12.6A settings, the plugin does not install a wrapper, register a
model, inspect KV tensors, or change scheduler or attention behavior. Phase
12.6B adds a separately documented, explicitly enabled public
`LLM.generate` wrapper.

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

## RunPod Validation Result

Phase 12.6A passed on RunPod with this installed-wheel environment:

| Component | Validated value |
| --- | --- |
| GPU | NVIDIA RTX 4090 |
| Driver CUDA support | 13.0 |
| PyTorch | `2.11.0+cu130` |
| PyTorch CUDA | `13.0` |
| vLLM | `0.22.1` |
| vLLM import path | `/usr/local/lib/python3.12/dist-packages/vllm/__init__.py` |
| `vllm._C` | Import succeeded |
| `vllm._C_stable_libtorch` | Import succeeded |
| `vllm.vllm_flash_attn` | Import succeeded |

The environment-only probe reported:

```text
plugin_loaded=true
plugin_marker_written=true
generation_status=skipped
phase12_6b_plugin_shadow_hook_candidate=true
active_routing=false
```

The generation probe used `gpt2`. Generation succeeded, with output text that
included `The first`, and reported:

```text
plugin_loaded=true
plugin_marker_written=true
phase12_6b_plugin_shadow_hook_candidate=true
active_routing=false
```

Both probes imported vLLM from `site-packages`, not from the unbuilt
repository-local source tree. This establishes that plugin-style discovery
and invocation are feasible enough to continue to Phase 12.6B. It does not
establish access to scheduler, block-table, KV-cache, attention, or
decode-step metadata.

## Known Bad / Avoided Environment

An earlier environment used:

- PyTorch `2.8.0+cu128`;
- vLLM `0.10.2`.

The plugin loaded and wrote its marker, so entry-point discovery itself
worked. GPT-2/OPT generation then failed because that stack exposed an
incompatible tokenizer API:

```text
GPT2Tokenizer has no attribute all_special_tokens_extended
```

This failure does not invalidate plugin feasibility, but the environment
should not be used for Phase 12.6B. The validated target is vLLM `0.22.1`
with PyTorch `2.11.0+cu130`.

## Recommended Phase 12.6B Environment

Use the validated installed-wheel stack:

- vLLM `0.22.1`;
- PyTorch `2.11.0+cu130`;
- execution from `/tmp`;
- only the Kivo plugin package installed into that environment.

Install the plugin from the repository without installing the repository as
vLLM:

```bash
cd /workspace/vllm-kivo-vd
uv pip install --system -e plugins/kivo_vllm_shadow_plugin
```

Then return to `/tmp` and use the isolated script-copy pattern shown above.
Do not run `uv pip install -e .`, add `/workspace/vllm-kivo-vd` to
`PYTHONPATH`, or otherwise import the repository-local `vllm/` package unless
a compatible source build is separately completed and validated.

## Phase 12.6A Conclusion

The official `vllm.general_plugins` entry point successfully discovered and
invoked the marker-only `kivo_shadow` plugin in the working installed-wheel
environment. Baseline generation through that environment also succeeded
while the plugin was enabled.

Phase 12.6A proves plugin loading only. It does not prove that a general
plugin can observe block tables, KV cache state, scheduler decisions,
attention metadata, or decode-step tensors. No runtime monkeypatch was
applied, and no scheduler, attention, KV-cache, block-table, or model-output
behavior was changed.

Phase 12.6B may attempt discovery of a plugin-owned no-op observation surface.
That work must remain:

- explicitly opt-in;
- passive and shadow-only;
- fail-closed when metadata is unavailable;
- free of active routing and runtime mutation.

Active selected attention remains outside Phase 12. No measured memory,
latency, or quality claim follows from this validation.
