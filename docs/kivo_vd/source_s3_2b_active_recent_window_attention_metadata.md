# Kivo-VD Phase S3.2B Active Recent-Window Attention Metadata

Phase S3.2B is the first active reduced-context metadata experiment. Instead of
aliasing excluded blocks to another selected block, it tries to represent a
smaller contiguous recent window directly in cloned metadata.

## What It Does

The active mode is:

```text
compact_to_recent_window
```

For each eligible metadata view:

- clone the per-group metadata block table
- clone `seq_lens`
- keep only the most recent contiguous block window
- rewrite the cloned block table so the kept physical blocks sit in the visible
  prefix
- reduce the cloned `seq_lens` to the selected recent-token length

Scheduler-owned block tables, slot mappings, and KV allocation remain
unchanged.

## Why This Phase Matters

Arbitrary sparse selected attention usually wants backend or kernel support.
Contiguous recent-window compaction is the quickest way to test whether the
existing vLLM metadata path can express a truly smaller attention-visible
context without custom kernels.

Passing this phase means metadata-level context reduction is possible without
crashing the tested run. Failing it suggests the current path likely needs
deeper backend integration.

## Boundaries

- This does not reduce KV allocation.
- This does not prove latency improvement.
- This does not prove quality preservation.
- This is still a feasibility experiment, not a final selected-attention path.

## RunPod

```bash
cd /workspace/vllm-kivo-vd

PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.run_source_s3_2b_active_recent_window_attention_metadata \
  --model gpt2 \
  --max-tokens 8 \
  --keep-recent-blocks 1 \
  --active-filter-mode compact_to_recent_window \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_2b_active_recent_window_attention_metadata.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_2b_active_recent_window_attention_metadata.md \
  --events-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s3_2b_active_recent_window_attention_metadata_events.jsonl \
  --continue-on-error
```
