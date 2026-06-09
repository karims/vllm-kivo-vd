# Phase 12.10: BlockTable Slot-Mapping Mutation

## Why This Exists

Phase 12.8/12.9 showed that `GPUModelRunner._get_slot_mappings` is reachable
and that a shallow-copied metadata dictionary can be changed without an
immediate crash. That same hook did not expose a safe direct Python slot or
block structure for selected-slot mutation.

Phase 12.10 therefore drops one level lower and patches
`BlockTable.compute_slot_mapping` inside the installed vLLM wheel.

## What Counts As Success

- baseline generation succeeds and writes observations;
- active generation writes observations;
- if the result is a copied Python list or tuple of slot IDs, one final entry
  is removed from the copy and generation still succeeds.

If the result is tensor-like, the experiment should block with:
`tensor-like slot mapping requires tensor-safe mutation design`.

If the result is unsupported or `None`, the experiment should block with:
`no safe Python-level slot mapping result found`.

Any crash is still useful if its traceback is preserved exactly.

## Safety Boundary

- installed-wheel patch only, never repository-local `vllm/`;
- exact backup before editing and exact restore after testing;
- mutation disabled by default;
- no KV tensor mutation;
- no scheduler mutation;
- no attention-kernel mutation;
- no measured memory or latency claim.

## RunPod Commands

```bash
cd /workspace/vllm-kivo-vd
git checkout chore/sync-upstream-main
git pull origin chore/sync-upstream-main

rm -rf /tmp/kivo_phase12
mkdir -p /tmp/kivo_phase12
cp -r /workspace/vllm-kivo-vd/scripts/kivo_vd /tmp/kivo_phase12/
cd /tmp
```

Install the patch:

```bash
PYTHONPATH=/tmp/kivo_phase12:/tmp/kivo_phase12/kivo_vd \
python -m kivo_vd.run_phase12_7_installed_vllm_patch \
  --install-patch \
  --target block_table_compute_slot_mapping_active \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_10_patch_install.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_10_patch_install.md \
  --continue-on-error
```

Run the probe:

```bash
PYTHONPATH=/tmp/kivo_phase12:/tmp/kivo_phase12/kivo_vd \
python -m kivo_vd.run_phase12_10_block_table_mutation_probe \
  --model gpt2 \
  --prompt "Kivo Phase 12.10 block table mutation probe." \
  --max-tokens 4 \
  --baseline-obs-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_10_baseline_observations.jsonl \
  --active-obs-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_10_active_observations.jsonl \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_10_block_table_mutation_probe.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_10_block_table_mutation_probe.md \
  --continue-on-error
```

Validate:

```bash
PYTHONPATH=/tmp/kivo_phase12:/tmp/kivo_phase12/kivo_vd \
python -m kivo_vd.validate_phase12_10_block_table_mutation \
  --baseline-input /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_10_baseline_observations.jsonl \
  --active-input /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_10_active_observations.jsonl \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_10_block_table_mutation_validation.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_10_block_table_mutation_validation.md
```

Restore immediately afterward:

```bash
PYTHONPATH=/tmp/kivo_phase12:/tmp/kivo_phase12/kivo_vd \
python -m kivo_vd.run_phase12_7_installed_vllm_patch \
  --restore \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_10_patch_restore.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/phase12_10_patch_restore.md
```
