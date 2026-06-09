# Phase 12.8/12.9: Active Mutation Ladder

## Scope

This is the first deliberately behavior-changing Kivo-VD runtime experiment.
It patches only the installed vLLM wheel and targets
`GPUModelRunner._get_slot_mappings`.

The escalation ladder is:

1. baseline generation with mutation disabled;
2. shallow-copy the returned attention-metadata dictionary and remove one key;
3. only if stage 2 succeeds, shallow-copy one direct Python slot list/tuple and
   remove one item.

This is an invariant-discovery experiment, not production selected attention.
A crash is a useful result when its traceback is preserved.

## Safety Boundary

- The target must resolve under `site-packages` or `dist-packages`.
- Repository-local `vllm/` is never edited.
- The patch uses independent `KIVO_PHASE12_8_9` markers and environment flags.
- Mutation is disabled by default and capped by
  `KIVO_PHASE12_8_9_MAX_MUTATIONS`.
- Metadata and mapping containers are shallow-copied before mutation.
- Tensors and nested runtime objects are never mutated.
- Any injected-helper exception returns the original result.
- Patch restoration is mandatory.
- No memory, latency, quality, or production-routing claim is made.

## Possible Outcomes

| result | interpretation |
| --- | --- |
| metadata generation crashes | removing a layer metadata key violates required invariants |
| metadata succeeds, selected slot blocks | no safe direct Python slot structure exists at this hook |
| selected slot applies, generation crashes | the traceback identifies a required routing invariant |
| selected slot applies and generation succeeds | Phase 13 design review may begin |

`phase13_selected_attention_candidate=true` requires both an applied
selected-slot mutation and successful generation. It does not itself prove
quality, correctness, or memory reduction.

## RunPod Procedure

Use the validated installed-wheel environment: Python 3.12.3, PyTorch
2.11.0+cu130, CUDA 13.0, vLLM 0.22.1, and RTX 4090.

```bash
cd /workspace/vllm-kivo-vd
git checkout chore/sync-upstream-main
git pull origin chore/sync-upstream-main

rm -rf /tmp/kivo_phase12
mkdir -p /tmp/kivo_phase12
cp -r /workspace/vllm-kivo-vd/scripts/kivo_vd /tmp/kivo_phase12/
cd /tmp
```

Install the active-ladder patch:

```bash
PYTHONPATH=/tmp/kivo_phase12:/tmp/kivo_phase12/kivo_vd \
python -m kivo_vd.run_phase12_7_installed_vllm_patch \
  --install-patch \
  --target slot_mappings_active_ladder \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_8_9_patch_install.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_8_9_patch_install.md \
  --continue-on-error
```

Run the ladder:

```bash
PYTHONPATH=/tmp/kivo_phase12:/tmp/kivo_phase12/kivo_vd \
python -m kivo_vd.run_phase12_8_9_active_ladder_probe \
  --model gpt2 \
  --prompt "Kivo Phase 12.8/12.9 active ladder probe." \
  --max-tokens 4 \
  --baseline-obs-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_8_9_baseline_observations.jsonl \
  --metadata-obs-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_8_9_metadata_observations.jsonl \
  --selected-slot-obs-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_8_9_selected_slot_observations.jsonl \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_8_9_active_ladder_probe.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_8_9_active_ladder_probe.md \
  --continue-on-error
```

Validate the emitted records:

```bash
PYTHONPATH=/tmp/kivo_phase12:/tmp/kivo_phase12/kivo_vd \
python -m kivo_vd.validate_phase12_8_9_active_ladder \
  --baseline-input /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_8_9_baseline_observations.jsonl \
  --metadata-input /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_8_9_metadata_observations.jsonl \
  --selected-slot-input /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_8_9_selected_slot_observations.jsonl \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_8_9_active_ladder_validation.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_8_9_active_ladder_validation.md
```

## Mandatory Restore

Restore even when installation or generation reports a failure:

```bash
PYTHONPATH=/tmp/kivo_phase12:/tmp/kivo_phase12/kivo_vd \
python -m kivo_vd.run_phase12_7_installed_vllm_patch \
  --restore \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_8_9_patch_restore.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_8_9_patch_restore.md
```

Confirm the restore report contains `restored_exactly=true`.
