# Phase 12.7: Installed vLLM Runtime Patch

## Why The Plugin Path Stopped

Phase 12.6D installed a plugin-owned wrapper around
`KVCacheManager.get_block_ids`, but real generation wrote zero KV
observations. The likely cause is vLLM EngineCore process and import
boundaries: the plugin loaded in one process while the relevant method ran in
another process with its own class definition.

Phase 12.7 therefore stops adding plugin wrappers. It uses a generated,
reversible patch against the working installed vLLM `0.22.1` wheel so each
runtime process imports the observation code with the target module.

Repository-local `vllm/` files remain untouched.

## Safety Model

The patch manager:

1. imports vLLM only to locate its installed package;
2. refuses paths outside `site-packages` or `dist-packages`;
3. selects one reviewed target;
4. writes an exact backup and checksum manifest before editing;
5. transforms the target with marked wrapper/helper code;
6. parses the transformed source before installation;
7. replaces the target atomically;
8. restores the exact backup atomically after testing.

Patch markers are:

```text
# KIVO_PHASE12_7_BEGIN
# KIVO_PHASE12_7_END
```

The manifest and backups live under:

```text
outputs/kivo_vd/phase12_7_backups/
```

When an absolute report path under a repository `outputs/` directory is used,
the backup directory is derived beside that directory. It can also be set
explicitly with `--backup-dir`.

## Target Selection

`--target auto` uses the first available target in this order:

1. `GPUModelRunner._get_slot_mappings`;
2. `BlockTable.compute_slot_mapping`;
3. `GPUModelRunner._build_attention_metadata`;
4. `BlockTable.get_cpu_tensor`;
5. `KVCacheManager.get_block_ids`;
6. `Scheduler.schedule`.

The scheduler target is a final observation fallback only. No target is
allowed to mutate scheduler decisions, KV tensors, block tables, slot
mappings, attention metadata, kernels, or outputs.

## Phase 12.7A Observation

With `KIVO_PHASE12_7_ENABLE=1`, the wrapper calls the original method, records
bounded summaries after it returns, and returns the exact original result.

The JSONL record includes:

- process/thread provenance and hook location;
- bounded argument and result summaries;
- metadata, block, slot, attention, and KV-like field names;
- explicit no-mutation and no-routing flags.

Observation mode always records:

```text
mutation_attempted=false
mutation_applied=false
runtime_behavior_changed=false
active_routing=false
measured_runtime_reduction=false
```

## Phase 12.7B Guarded Active Decision

`KIVO_PHASE12_7_ACTIVE=1` computes only a side-channel
`would_select_blocks` preview when a bounded result length is available.

It deliberately records:

```text
mutation_attempted=true
mutation_applied=false
active_experiment_blocked=true
```

The blocker explains that mutation of runtime-consumed metadata is not proven
safe. This is a real decision computation inside the runtime path, but it is
not active selected attention.

## RunPod Environment

Use the validated environment:

- Python `3.12.3`;
- PyTorch `2.11.0+cu130`;
- CUDA 13.0;
- vLLM `0.22.1`;
- RTX 4090;
- installed vLLM under `site-packages`.

Copy only the scripts:

```bash
cd /workspace/vllm-kivo-vd
git checkout chore/sync-upstream-main
git pull origin chore/sync-upstream-main

rm -rf /tmp/kivo_phase12
mkdir -p /tmp/kivo_phase12
cp -r /workspace/vllm-kivo-vd/scripts/kivo_vd /tmp/kivo_phase12/
cd /tmp
```

Check status:

```bash
PYTHONPATH=/tmp/kivo_phase12:/tmp/kivo_phase12/kivo_vd \
python -m kivo_vd.run_phase12_7_installed_vllm_patch \
  --status \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_7_patch_status_before.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_7_patch_status_before.md
```

Install:

```bash
PYTHONPATH=/tmp/kivo_phase12:/tmp/kivo_phase12/kivo_vd \
python -m kivo_vd.run_phase12_7_installed_vllm_patch \
  --install-patch \
  --target auto \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_7_patch_install.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_7_patch_install.md \
  --continue-on-error
```

Run observation-only generation:

```bash
PYTHONPATH=/tmp/kivo_phase12:/tmp/kivo_phase12/kivo_vd \
python -m kivo_vd.run_phase12_7_runtime_generation_probe \
  --model gpt2 \
  --prompt "Kivo Phase 12.7 runtime observation probe." \
  --max-tokens 4 \
  --observations-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_7_runtime_observations.jsonl \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_7_runtime_generation_probe.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_7_runtime_generation_probe.md \
  --continue-on-error
```

Validate:

```bash
PYTHONPATH=/tmp/kivo_phase12:/tmp/kivo_phase12/kivo_vd \
python -m kivo_vd.validate_phase12_7_runtime_observation \
  --input /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_7_runtime_observations.jsonl \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_7_runtime_observation_validation.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_7_runtime_observation_validation.md
```

If observations are nonempty and validation passes, run guarded active mode:

```bash
PYTHONPATH=/tmp/kivo_phase12:/tmp/kivo_phase12/kivo_vd \
python -m kivo_vd.run_phase12_7_runtime_generation_probe \
  --model gpt2 \
  --prompt "Kivo Phase 12.7 guarded active runtime probe." \
  --max-tokens 4 \
  --active \
  --observations-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_7_runtime_active_observations.jsonl \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_7_runtime_active_probe.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_7_runtime_active_probe.md \
  --continue-on-error
```

Always restore after testing:

```bash
PYTHONPATH=/tmp/kivo_phase12:/tmp/kivo_phase12/kivo_vd \
python -m kivo_vd.run_phase12_7_installed_vllm_patch \
  --restore \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_7_patch_restore.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_7_patch_restore.md
```

## Success And Blocker Decisions

Phase 12.7A succeeds when generation completes, observation records are
nonempty, validation passes, and the generated output returns normally.

Phase 12.7B succeeds as a guarded experiment when the runtime computes a
side-channel decision and records a blocker without applying mutation.

If a selected target writes zero observations, restore and review the next
target from the ranked list. If generation crashes, restore immediately and
record the exact file and method blocker.

`phase12_8_active_selected_attention_candidate=true` means design review only.
It does not authorize active selected attention.

No measured memory reduction, latency improvement, quality preservation, or
production routing claim is made.
