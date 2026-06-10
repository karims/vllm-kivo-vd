# Phase S2.0: Block Visibility and Shadow Selection

## Purpose

Phase S1 proved source-level runtime control by mutating valid slot-mapping
entries. That was useful for verifying that the repo-local Python hook reaches
real generation, but the mutation was intentionally destructive and is not a
selected-attention policy.

Phase S2.0 stops mutation. It observes the completed slot mapping and block
table, derives the KV blocks visible to the current batch, and computes a
deterministic shadow selected-block set. The selected set is recorded and then
ignored.

## Shadow Policy

With `KIVO_SOURCE_POLICY=sketch_shadow_blocks`, the source hook:

1. Ignores padding and negative slot IDs.
2. Computes `block_id = slot_id // block_size`.
3. Keeps the most recent visible block, or the requested recent window.
4. Uses a deterministic placeholder score to fill the remaining block budget.
5. Writes an S2.0 JSONL record without changing the slot mapping or block
   table.

The default budget ratio is `0.5`, and the default recent window is one block.
The placeholder score does not inspect key, value, query, or model tensors.

## Recorded Boundary

Every S2.0 record uses schema
`kivo_source_s2_0_block_visibility_shadow_v1` and keeps these fields false:

- `mutation_attempted`
- `mutation_applied`
- `runtime_behavior_changed`
- `active_routing`
- `measured_runtime_reduction`
- `selected_attention_claim_allowed`
- `performance_claim_allowed`

`theoretical_visible_block_reduction` and its ratio count visible blocks
excluded by the shadow policy. They are not measured KV memory savings.

## RunPod Command

The existing repo-local source build can run this phase without rebuilding:

```bash
cd /workspace/vllm-kivo-vd
git pull origin chore/sync-upstream-main

PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.run_source_s2_0_block_visibility_shadow \
  --model gpt2 \
  --max-tokens 8 \
  --budget-ratio 0.5 \
  --keep-recent-blocks 1 \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s2_0_block_visibility_shadow.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s2_0_block_visibility_shadow.md \
  --continue-on-error
```

Validate the result:

```bash
PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.validate_source_s2_0_block_visibility_shadow \
  --input-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s2_0_block_visibility_shadow.json \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s2_0_block_visibility_shadow_validation.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s2_0_block_visibility_shadow_validation.md
```

## Interpretation

If every prompt reports only one visible block, increase prompt length and
`--max-model-len`. The current context is too short to test block selection.

If visible block counts grow and selected block counts are smaller, the trace
shows theoretical room for block selection. It still does not demonstrate:

- selected attention,
- active KV routing,
- measured KV memory reduction,
- latency improvement, or
- quality preservation under a behavior-changing policy.
