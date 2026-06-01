# Kivo-VD Phase 1.2: Offline Sketch Evaluation Harness

Phase 1.2 adds a pure-Python/NumPy harness to evaluate whether low-dimensional
sketches preserve top-attended token/block rankings in offline experiments.

## What this phase adds

- `vllm/v1/core/kivo_vd_sketch_math.py`
  - random projection and count-sketch construction/application
  - exact vs sketched score computation
  - top-k recall helpers
  - block-level score/recall helpers
- `scripts/kivo_vd/run_offline_sketch_eval.py`
  - synthetic data generation
  - exact and sketched ranking comparison
  - compact JSON metric output
- isolated tests in `tests_kivo/test_kivo_vd_sketch_math.py`

## What this phase explicitly does NOT do

- No integration with vLLM runtime scheduling.
- No integration with GPUModelRunner or attention kernels.
- No CUDA/Triton/torch usage.
- No KV tensor access from runtime.
- No routing/scheduling behavior changes.

## Purpose

The goal is to validate sketch quality offline before runtime integration:

- token-level top-k recall
- block-level top-k recall

This phase optimizes for ranking fidelity checks, not memory reduction or
runtime speed.

## Example command

```bash
.venv/bin/python scripts/kivo_vd/run_offline_sketch_eval.py \
  --sketch-type random_projection \
  --input-dim 256 \
  --sketch-dim 64 \
  --num-tokens 1024 \
  --block-size 16 \
  --topk 32 \
  --seed 1234
```

## Sweep runner

Phase 1.3 adds `scripts/kivo_vd/run_sketch_sweep.py` to run many offline
configurations and write JSONL results.

Quick smoke run:

```bash
.venv/bin/python scripts/kivo_vd/run_sketch_sweep.py --quick
```

Full sweep:

```bash
.venv/bin/python scripts/kivo_vd/run_sketch_sweep.py \
  --output outputs/kivo_vd/sketch_sweep.jsonl
```

The script prints aggregate summaries grouped by:
- `mode`
- `sketch_type`
- `sketch_dim`
- `topk_blocks`

Key interpretation:
- `block_topk_recall` near 1.0 means sketch ranking preserves exact top blocks
  well.
- `block_topk_recall` near 0.0 means heavy ranking distortion.
- `block_recall_at_2x_budget` / `block_recall_at_4x_budget` measure candidate
  retrieval quality when sketch can return a larger set for later exact rerank.
- `block_mrr` tracks how early exact top blocks appear in the approximate ranking.
- `token_score_correlation` / `block_score_correlation` provide score-shape
  agreement diagnostics.

## Synthetic modes (Phase 1.4)

- `gaussian`: pure random baseline (often hard/worst-case).
- `clustered`: topic-like clustered keys.
- `smooth_sequence`: gradual temporal evolution.
- `needle_blocks`: mostly noise with a few strongly aligned blocks.
- `mixed`: clustered base + smooth/noise + needle blocks.

Guidance:
- Treat `gaussian` as stress baseline.
- Judge viability mainly by `block_topk_recall` under structured modes
  (`clustered`, `smooth_sequence`, `needle_blocks`, `mixed`), especially
  `mixed`.
- Strict top-k recall is intentionally harsh; candidate-budget metrics
  (`recall@2x`, `recall@4x`) can still indicate viable retrieval for a future
  exact-attention reranking stage.

## Next expected step

Use the same metrics on real captured query/key arrays (outside runtime first),
then decide how to attach real sketch computation in later phases.

## Optional real-model check (Phase 1.5)

Phase 1.5 adds `scripts/kivo_vd/run_hf_qk_sketch_eval.py` for optional offline
validation on real HuggingFace GPT-2 style Q/K tensors.

Example:

```bash
.venv/bin/python scripts/kivo_vd/run_hf_qk_sketch_eval.py \
  --model-name sshleifer/tiny-gpt2 \
  --sketch-type random_projection \
  --sketch-dim 64 \
  --topk-blocks 4
```

This remains offline-only and does not connect to vLLM runtime.
