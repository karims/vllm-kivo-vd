# Kivo-VD Phase S3.2A Active Selected-Attention Metadata

Phase S3.2A is the first active metadata-level feasibility experiment. It
reuses the S3.1B `slot_coverage_recency_proxy` plan and applies a cloned
block-table view before attention metadata builders consume it.

## Active Filter

The initial active filter mode is:

```text
alias_excluded_blocks_to_recent_selected
```

For each safe plan:

- the per-group metadata block table is cloned
- valid logical entries for excluded physical blocks are replaced with the
  most recent selected physical block ID
- the original scheduler-owned block table remains unchanged
- `slot_mapping` and KV cache allocation remain unchanged
- invalid or negative block IDs are never introduced

If no safe selected block or valid excluded entry exists, the policy records a
blocker and leaves metadata unchanged.

## Interpretation

Passing S3.2A means active control of the block-table view consumed by
attention metadata is possible without crashing the tested run.

It does not mean:

- KV memory was reduced
- attention work was reduced
- latency improved
- quality was preserved
- final selected attention was implemented

Output changes are allowed and reported because this phase intentionally
changes the metadata view. Output preservation is not a pass condition.

## RunPod

```bash
cd /workspace/vllm-kivo-vd

PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.run_source_s3_2a_active_selected_attention_metadata \
  --model gpt2 \
  --max-tokens 8 \
  --budget-ratio 0.5 \
  --keep-recent-blocks 1 \
  --coverage-weight 0.6 \
  --recency-weight 0.4 \
  --active-filter-mode alias_excluded_blocks_to_recent_selected \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_2a_active_selected_attention_metadata.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_2a_active_selected_attention_metadata.md \
  --events-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_2a_active_selected_attention_metadata_events.jsonl \
  --continue-on-error
```

Validate:

```bash
PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.validate_source_s3_2a_active_selected_attention_metadata \
  --input-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_2a_active_selected_attention_metadata.json \
  --events-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_2a_active_selected_attention_metadata_events.jsonl \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_2a_active_selected_attention_metadata_validation.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_2a_active_selected_attention_metadata_validation.md
```
