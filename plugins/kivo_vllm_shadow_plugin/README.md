# Kivo vLLM Shadow Plugin

This package is a Phase 12.6A feasibility probe for vLLM's general plugin
entry-point mechanism.

The `kivo_shadow` plugin writes an optional JSON load marker when
`KIVO_SHADOW_PLUGIN_MARKER` is set. It does not monkeypatch vLLM, register a
model, alter scheduling, inspect KV tensors, or change attention behavior.

Enable only this plugin with:

```bash
export VLLM_PLUGINS=kivo_shadow
export KIVO_SHADOW_PLUGIN_MARKER=/tmp/kivo_shadow_plugin_marker.json
```

The marker is evidence that vLLM discovered and invoked the entry point. It is
not evidence that the plugin can access internal block tables or decode
metadata.
