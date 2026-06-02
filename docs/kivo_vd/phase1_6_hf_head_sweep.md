# Kivo-VD Phase 1.6: HF Layer/Head Sweep

Phase 1.6 adds an optional offline script to evaluate sketch candidate retrieval
across many layers, heads, and query positions for GPT-2 style models.

## What this phase adds

- `scripts/kivo_vd/run_hf_qk_head_sweep.py`
  - Sweeps layers (`--layers`) and heads (`--heads`), with `all` support.
  - Sweeps query positions (`--query-positions`), including `sweep` mode
    (25%, 50%, 75%, last).
  - Supports single or multiple sketch configs:
    - `--sketch-type` / `--sketch-dim` (single, backward compatible)
    - `--sketch-types` / `--sketch-dims` (comma-separated sweep)
  - Computes the same metrics as Phase 1.5 for each layer/head/position.
  - Can optionally emit `approx_ranked_block_ids` with
    `--include-ranked-blocks` for downstream policy simulation.
  - Writes one JSONL row per combination.
  - Prints aggregate summary for:
    - overall
    - `sketch_type + sketch_dim`
    - `sketch_type + sketch_dim + layer`

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

Multi-sketch + multi-dim comparison:

```bash
.venv/bin/python scripts/kivo_vd/run_hf_qk_head_sweep.py \
  --model-name gpt2 \
  --sketch-types count_sketch,random_projection \
  --sketch-dims 16,32,64,128 \
  --layers 0,1,2,3 \
  --heads 0,1,2,3 \
  --max-tokens 900
```

Default output:
- `outputs/kivo_vd/hf_qk_head_sweep.jsonl`

For Phase 2.8 active-KV policy simulation, include full approximate block
rankings:

```bash
.venv/bin/python scripts/kivo_vd/run_hf_qk_head_sweep.py \
  --model-name distilgpt2 \
  --sketch-types count_sketch,random_projection \
  --sketch-dims 32,64,128 \
  --layers 0,1 \
  --heads 0,1 \
  --max-tokens 512 \
  --include-ranked-blocks
```

## Scope limits

- Optional offline script only.
- Requires `torch` + `transformers`.
- No vLLM runtime integration changes.
- No scheduler/GPUModelRunner/attention/kernel modifications.
