# Kivo-VD Phase 4A.1: SRHT Comparison Benchmark Support

Phase 4A.1 makes SRHT easy to compare against the current offline baselines:
CountSketch and Random Projection.

This is still offline-only. SRHT is experimental and is not the default sketch
backend.

## What This Phase Supports

- Synthetic quick sweeps including `srht`.
- HF Q/K head sweeps with `count_sketch`, `random_projection`, and `srht`.
- Active KV policy simulation from ranked HF sweep rows.
- Benchmark report generation that includes SRHT rows when present.
- Compact JSONL comparison through `compare_sketch_backends.py`.

## What This Phase Does Not Prove

- No measured vLLM runtime KV memory reduction.
- No output-quality preservation.
- No latency improvement.
- No attention behavior change.
- No runtime use of SRHT.

## Synthetic Quick Sweep

```bash
.venv/bin/python scripts/kivo_vd/run_sketch_sweep.py --quick
```

This is the fastest way to confirm that SRHT is wired into the NumPy offline
harness. Treat these synthetic results as development signals only.

## One-Command Offline Pipeline

Recommended SRHT comparison command:

```bash
.venv/bin/python scripts/kivo_vd/run_offline_benchmark_pipeline.py \
  --model-name gpt2 \
  --prompt-mode blue_orchid \
  --sketch-types count_sketch,random_projection,srht \
  --sketch-dims 32,64,128 \
  --layers 0,1,2,3 \
  --heads 0,1,2,3 \
  --max-tokens 900 \
  --run-name gpt2_srht_comparison
```

Dry-run the pipeline first:

```bash
.venv/bin/python scripts/kivo_vd/run_offline_benchmark_pipeline.py \
  --dry-run \
  --sketch-types count_sketch,random_projection,srht \
  --run-name srht_comparison_dry_run
```

Expected files under `outputs/kivo_vd/runs/<run-name>/`:

- `hf_qk_head_sweep_ranked.jsonl`
- `active_kv_policy_simulation.jsonl`
- `kivo_vd_benchmark_report.md`
- `pipeline_summary.json`

## Manual HF Q/K Head Sweep

If running stages manually:

```bash
.venv/bin/python scripts/kivo_vd/run_hf_qk_head_sweep.py \
  --model-name gpt2 \
  --sketch-types count_sketch,random_projection,srht \
  --sketch-dims 32,64,128 \
  --layers 0,1,2,3 \
  --heads 0,1,2,3 \
  --max-tokens 900 \
  --include-ranked-blocks \
  --output outputs/kivo_vd/hf_qk_head_sweep_srht_ranked.jsonl
```

## Active KV Policy Simulation

```bash
.venv/bin/python scripts/kivo_vd/simulate_active_kv_policy.py \
  --input outputs/kivo_vd/hf_qk_head_sweep_srht_ranked.jsonl \
  --output outputs/kivo_vd/active_kv_policy_simulation_srht.jsonl
```

The simulator automatically preserves rows with `sketch_type=srht`.

## Benchmark Report Generation

```bash
.venv/bin/python scripts/kivo_vd/generate_kivo_benchmark_report.py \
  --hf-sweep outputs/kivo_vd/hf_qk_head_sweep_srht_ranked.jsonl \
  --policy-sim outputs/kivo_vd/active_kv_policy_simulation_srht.jsonl \
  --output outputs/kivo_vd/kivo_vd_srht_comparison_report.md
```

The report includes SRHT rows in retrieval and policy tables when present. It
also states that SRHT remains experimental and should be compared against
CountSketch and Random Projection before any default-policy discussion.

## Compact Backend Comparison

```bash
.venv/bin/python scripts/kivo_vd/compare_sketch_backends.py \
  --input outputs/kivo_vd/hf_qk_head_sweep_srht_ranked.jsonl
```

The helper prints JSON grouped by `sketch_type` and `sketch_dim` with:

- average block top-k recall;
- average recall@2x;
- average recall@4x;
- average block score correlation.

## Interpretation

SRHT is competitive only if it is close to or better than CountSketch and Random
Projection on the same model, prompt, layers, heads, query positions, and sketch
dimensions.

Useful signs:

- high `avg_block_recall_at_2x_budget`;
- high `avg_block_recall_at_4x_budget`;
- reasonable `avg_block_score_correlation`;
- policy simulation retains high exact-top-block recall at conservative active
  block ratios.

Even strong SRHT offline results would remain evidence for candidate retrieval,
not a measured runtime memory-reduction or quality claim.
