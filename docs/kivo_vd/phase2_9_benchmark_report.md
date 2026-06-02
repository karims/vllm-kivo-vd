# Kivo-VD Phase 2.9: Benchmark Report Generator

Phase 2.9 adds an offline Markdown report generator for Kivo-VD sketch
retrieval and active-KV policy simulation results.

The report is deliberately conservative. It summarizes offline HuggingFace Q/K
experiments and policy simulation rows, but it does not claim measured vLLM
runtime KV memory reduction, latency improvement, or quality preservation.

## Generate Ranked HF Q/K Sweep Rows

The policy simulator needs full approximate block rankings:

```bash
.venv/bin/python scripts/kivo_vd/run_hf_qk_head_sweep.py \
  --model-name gpt2 \
  --sketch-types count_sketch,random_projection \
  --sketch-dims 32,64,128 \
  --layers 0,1,2,3 \
  --heads 0,1,2,3 \
  --max-tokens 900 \
  --include-ranked-blocks \
  --output outputs/kivo_vd/hf_qk_head_sweep_ranked.jsonl
```

Experimental SRHT rows can be added by including `srht` in `--sketch-types`.
Treat SRHT report entries as exploratory until they pass the same offline,
runtime dry-run, quality, and memory validations as the baseline backends.
For the full SRHT comparison flow, see
[Phase 4A.1: SRHT Comparison](phase4a_1_srht_comparison.md).

## Run Active KV Policy Simulation

```bash
.venv/bin/python scripts/kivo_vd/simulate_active_kv_policy.py \
  --input outputs/kivo_vd/hf_qk_head_sweep_ranked.jsonl \
  --output outputs/kivo_vd/active_kv_policy_simulation.jsonl
```

## Generate Report

```bash
.venv/bin/python scripts/kivo_vd/generate_kivo_benchmark_report.py
```

The full offline evidence bundle can also be regenerated through the Phase 3.4
pipeline runner:

```bash
.venv/bin/python scripts/kivo_vd/run_offline_benchmark_pipeline.py \
  --model-name gpt2 \
  --prompt-mode blue_orchid
```

Defaults:

- HF sweep input: `outputs/kivo_vd/hf_qk_head_sweep_ranked.jsonl`
- Policy simulation input: `outputs/kivo_vd/active_kv_policy_simulation.jsonl`
- Report output: `outputs/kivo_vd/kivo_vd_benchmark_report.md`

## Report Contents

The generated Markdown includes:

- Status and scope caveats.
- Executive summary.
- Retrieval summary grouped by `sketch_type` and `sketch_dim`.
- Active-KV policy summary for selected policies.
- Conservative recommended policy.
- Aggressive policy notes.
- What is proven vs not proven.
- Next experiments.

## Conservative vs Aggressive Policies

The conservative policy is:

- `recent_window_blocks=8`
- `candidate_budget_blocks=16`
- CountSketch dim 64 as the tentative default.
- Random Projection dim 64 as a baseline.

This is the policy to discuss first because it targets high exact-top-block
recall without chasing extreme active-KV reduction.

The aggressive stretch policy is:

- `recent_window_blocks=4`
- `candidate_budget_blocks=8`

It can estimate larger active-KV reduction, but it should not be treated as a
runtime or quality claim until measured inside vLLM.

## Interpretation

High sketch retrieval recall and low simulated active ratio are useful evidence
for candidate-block retrieval. They are not proof of:

- Real vLLM KV memory reduction.
- Latency improvement.
- Output-quality preservation.
- Behavior on modern RoPE/GQA models.
- Effectiveness of future variation-diminishing or bidiagonal sketch variants.

Future phases should validate runtime dry-run behavior, benchmark quality,
measure actual GPU memory, and test modern model families.
