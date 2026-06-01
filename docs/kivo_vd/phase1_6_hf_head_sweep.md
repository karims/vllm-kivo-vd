# Kivo-VD Phase 1.6: HF Layer/Head Sweep

Phase 1.6 adds an optional offline script to evaluate sketch candidate retrieval
across many layers, heads, and query positions for GPT-2 style models.

## What this phase adds

- `scripts/kivo_vd/run_hf_qk_head_sweep.py`
  - Sweeps layers (`--layers`) and heads (`--heads`), with `all` support.
  - Sweeps query positions (`--query-positions`), including `sweep` mode
    (25%, 50%, 75%, last).
  - Computes the same metrics as Phase 1.5 for each layer/head/position.
  - Writes one JSONL row per combination.
  - Prints aggregate summary grouped by `sketch_type`, `sketch_dim`, and `layer`,
    plus overall summary.

## Why this matters

Single-head results can be misleading. This sweep checks whether Kivo-VD sketch
retrieval is consistent across model internals.

Interpretation:
- High `block_recall_at_2x_budget` / `block_recall_at_4x_budget` across many
  heads supports candidate-retrieval + exact-rerank viability.
- Low recall in some heads/layers suggests future runtime logic may need
  per-layer/head policies.

## Example command

```bash
.venv/bin/python scripts/kivo_vd/run_hf_qk_head_sweep.py \
  --model-name distilgpt2 \
  --sketch-type count_sketch \
  --sketch-dim 64 \
  --layers 0,1 \
  --heads 0,1 \
  --max-tokens 512
```

Default output:
- `outputs/kivo_vd/hf_qk_head_sweep.jsonl`

## Scope limits

- Optional offline script only.
- Requires `torch` + `transformers`.
- No vLLM runtime integration changes.
- No scheduler/GPUModelRunner/attention/kernel modifications.
