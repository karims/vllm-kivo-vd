# Phase 12.6D: KV Block-ID Observation

## Purpose

Phase 12.6D adds an opt-in plugin-owned wrapper around:

```text
vllm.v1.core.kv_cache_manager.KVCacheManager.get_block_ids
```

Phase 12.6C ranked this method as the leading internal candidate on installed
vLLM `0.22.1`:

- risk: `medium`;
- usefulness: `high`;
- category: `kv_cache_metadata`.

It was selected because it is a read-oriented accessor that returns request
block IDs after normal vLLM logic has produced them. Scheduler steps, model
execution, allocation, slot mapping, block-table mutation, and attention
methods remained high risk and are not patched.

## Safety Contract

The wrapper is disabled by default. When explicitly enabled, it:

1. calls the original `get_block_ids` with unchanged arguments;
2. stores the exact returned object;
3. copies a bounded metadata observation after the call;
4. catches all observation failures;
5. returns the exact original object.

It does not mutate block IDs, KV cache, scheduler state, block tables, slot
mapping, attention metadata, kernels, prompts, or model outputs.

The observation record always states:

```text
active_routing=false
measured_runtime_reduction=false
runtime_behavior_changed=false
mutation=false
```

## Environment Variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `KIVO_SHADOW_PLUGIN_PATCH_KV_GET_BLOCK_IDS` | disabled | Enables the observation wrapper |
| `KIVO_SHADOW_PLUGIN_KV_OBS` | unset | Observation JSONL path |
| `KIVO_SHADOW_PLUGIN_PATCH_GENERATE` | disabled | Independently enables the Phase 12.6B public wrapper |
| `KIVO_SHADOW_PLUGIN_MARKER` | unset | Records installation status and warnings |

## Observation Schema

Each JSONL record includes:

- schema version, timestamp, process ID, and thread ID;
- hook, class, method, and result type;
- bounded result representation;
- top-level result length when available;
- bounded flattened integer block-ID preview;
- extracted block-ID count and min/max;
- bounded argument type and keyword summaries;
- selected scalar instance attributes only;
- explicit no-routing, no-reduction, no-behavior-change, and no-mutation flags.

Lists and tuples are handled directly. Tensor- or array-like values are
converted only when their reported size is bounded. Large arrays are not
serialized. All extraction occurs inside the fail-closed observer path.

## RunPod Environment

Use the validated installed-wheel environment:

- RTX 4090;
- vLLM `0.22.1`;
- PyTorch `2.11.0+cu130`;
- CUDA 13.0;
- vLLM imported from `site-packages`.

Install only the plugin package:

```bash
cd /workspace/vllm-kivo-vd
git checkout chore/sync-upstream-main
git pull origin chore/sync-upstream-main
uv pip install --system -e plugins/kivo_vllm_shadow_plugin
```

Do not run `uv pip install -e .` and do not import the repository-local
`vllm/` source.

Copy scripts:

```bash
rm -rf /tmp/kivo_phase12
mkdir -p /tmp/kivo_phase12
cp -r /workspace/vllm-kivo-vd/scripts/kivo_vd /tmp/kivo_phase12/
```

Run the combined public-generation and KV-observation probe:

```bash
cd /tmp

PYTHONPATH=/tmp/kivo_phase12:/tmp/kivo_phase12/kivo_vd \
python -m kivo_vd.run_phase12_vllm_plugin_probe \
  --model gpt2 \
  --prompt "Kivo Phase 12 KV block observation probe." \
  --max-tokens 4 \
  --enable-generate-hook \
  --enable-kv-get-block-ids-hook \
  --events-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_6d_generate_events.jsonl \
  --kv-observations-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_6d_kv_get_block_ids_observations.jsonl \
  --marker-path /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_6d_kv_observation_marker.json \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_6d_kv_observation_probe.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_6d_kv_observation_probe.md \
  --continue-on-error
```

Validate observations independently:

```bash
PYTHONPATH=/tmp/kivo_phase12:/tmp/kivo_phase12/kivo_vd \
python -m kivo_vd.validate_phase12_6d_kv_observation \
  --input /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_6d_kv_get_block_ids_observations.jsonl \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_6d_kv_observation_validation.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_6d_kv_observation_validation.md
```

## Success Criteria

- `generation_status=succeeded`;
- `patch_kv_get_block_ids_requested=true`;
- `patch_kv_get_block_ids_installed=true`;
- `kv_observations_written > 0`;
- standalone validation passes;
- generated output returns normally;
- `active_routing=false`;
- `measured_runtime_reduction=false`;
- `runtime_behavior_changed=false`.

`phase12_7_active_experiment_candidate=true` means only that a future
experiment may be reviewed. It does not authorize active routing or selected
attention.

## Failure Decision

If the wrapper installs but `kv_observations_written == 0` during real
generation, stop adding plugin scaffold. That result means the plugin process
did not intercept the relevant runtime call path. The next integration review
should move to a compatible source/fork strategy rather than layering more
public-boundary wrappers.

## What This Does Not Prove

Successful observations prove that the plugin can copy real request block IDs
from calls that reach this method. They do not prove:

- active selected-block attention;
- KV allocation reduction;
- measured GPU memory reduction;
- latency improvement;
- quality preservation;
- safe mutation of scheduler, block table, slot mapping, or attention state.
