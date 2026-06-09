# Kivo vLLM Shadow Plugin

This package supports the Phase 12.6A entry-point feasibility probe and the
Phase 12.6B opt-in public `LLM.generate` shadow hook.

The `kivo_shadow` plugin writes an optional JSON load marker when
`KIVO_SHADOW_PLUGIN_MARKER` is set. By default it does nothing else.

Setting `KIVO_SHADOW_PLUGIN_PATCH_GENERATE=1` installs a fail-closed wrapper
around the public `vllm.LLM.generate` method. The wrapper returns the original
result object unchanged and can append preview-only events to
`KIVO_SHADOW_PLUGIN_EVENTS`.

Enable only this plugin with:

```bash
export VLLM_PLUGINS=kivo_shadow
export KIVO_SHADOW_PLUGIN_MARKER=/tmp/kivo_shadow_plugin_marker.json
```

The marker is evidence that vLLM discovered and invoked the entry point. It is
not evidence that the plugin can access internal block tables or decode
metadata.

The optional wrapper does not alter prompts, sampling parameters, outputs,
scheduler state, KV cache, block tables, or attention. Its block IDs are
deterministic synthetic previews and are never used for routing.

The package also provides `internal_discovery`, a read-only catalog scanner
for installed-wheel vLLM. It records callable signatures and conservative
risk/usefulness rankings. It does not install an internal hook.

Phase 12.6D can independently enable a fail-closed observation wrapper around
`KVCacheManager.get_block_ids`:

```bash
export KIVO_SHADOW_PLUGIN_PATCH_KV_GET_BLOCK_IDS=1
export KIVO_SHADOW_PLUGIN_KV_OBS=/tmp/kivo_kv_observations.jsonl
```

The wrapper appends bounded copied metadata after the original method returns,
then returns the exact original result. It does not mutate KV state, block
tables, slot mappings, scheduler decisions, or attention behavior.
