# Phase S2.1: Active Block Mask

## Purpose

Phase S2.0 observed visible KV blocks and computed a shadow selected-block set
without changing runtime behavior. Phase S2.1 keeps that same selection logic
but actively remaps slot visibility for older, unselected blocks.

This makes the source hook behave like a crude block-visibility controller. It
is still experimental and still does not free KV memory, alter the scheduler,
or change attention kernels.

## Policy

With `KIVO_SOURCE_POLICY=active_mask_unselected_blocks`, the source hook:

1. Computes visible blocks from valid slot IDs.
2. Keeps the most recent visible block window.
3. Uses the same deterministic placeholder score and budget ratio as S2.0 to
   select a subset of older blocks.
4. Remaps slots that belong to older unselected blocks onto selected blocks
   when that can be done safely.

The remap preserves intra-block offset when possible. The hook fails closed on
any uncertainty.

## Recorded Boundary

Every S2.1 record uses schema
`kivo_source_s2_1_active_block_mask_v1` and keeps these claims false:

- `measured_runtime_reduction`
- `selected_attention_claim_allowed`
- `performance_claim_allowed`

The active path is real source-level control, but it does not prove memory
reduction or latency improvement. It also does not free KV cache entries.

## RunPod Command

```bash
cd /workspace/vllm-kivo-vd
git pull origin chore/sync-upstream-main

PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.run_source_s2_1_active_block_mask \
  --model gpt2 \
  --max-tokens 8 \
  --budget-ratio 0.5 \
  --keep-recent-blocks 1 \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s2_1_active_block_mask.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s2_1_active_block_mask.md \
  --continue-on-error
```

Validate the output:

```bash
PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.validate_source_s2_1_active_block_mask \
  --input-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s2_1_active_block_mask.json \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s2_1_active_block_mask_validation.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s2_1_active_block_mask_validation.md
```

## Interpretation

If `mutation_applied_count > 0`, the source hook is remapping older block
visibility in the runtime. That is a real control signal, but it still does
not demonstrate:

- measured KV memory reduction,
- latency improvement,
- quality preservation under a production policy, or
- selected attention.

If output changes sharply, the policy is too crude. That is still useful, but
it means the next step should be a more careful block-selection strategy, not
a claim of success.

Prompts with only one visible block, or with no unselected older blocks, are
valid no-op cases. They should pass when baseline and active generations both
succeed, with remap counts staying at zero. S2.1 passes if at least one
eligible prompt applies remapping.
