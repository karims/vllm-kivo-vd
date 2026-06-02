# Kivo-VD Phase 3.4: Offline Benchmark Pipeline

Phase 3.4 adds a one-command offline benchmark pipeline runner.

The pipeline reproduces the offline Kivo-VD evidence bundle:

1. HuggingFace Q/K layer/head sweep with ranked approximate blocks.
2. Active KV policy simulation.
3. Conservative Markdown benchmark report.
4. Optional torch sketch backend benchmark.

This remains offline analysis only. It does not run vLLM inference, does not
measure real runtime KV memory, and does not change scheduler, GPUModelRunner,
attention metadata, block tables, slot mapping, kernels, model architecture, or
training.

## Dry Run

Use `--dry-run` first to inspect commands without executing heavy stages:

```bash
.venv/bin/python scripts/kivo_vd/run_offline_benchmark_pipeline.py --dry-run
```

The dry run creates a timestamped run directory and writes
`pipeline_summary.json` with all stages marked `planned`.

## Full GPT-2 BLUE ORCHID Run

```bash
.venv/bin/python scripts/kivo_vd/run_offline_benchmark_pipeline.py \
  --model-name gpt2 \
  --prompt-mode blue_orchid \
  --sketch-types count_sketch,random_projection \
  --sketch-dims 32,64,128 \
  --layers 0,1,2,3 \
  --heads 0,1,2,3 \
  --max-tokens 900
```

Add the optional torch benchmark:

```bash
.venv/bin/python scripts/kivo_vd/run_offline_benchmark_pipeline.py \
  --run-torch-benchmark
```

## Output Layout

Each run writes into:

```text
outputs/kivo_vd/runs/<run-name>/
```

Files:

- `hf_qk_head_sweep_ranked.jsonl`
- `active_kv_policy_simulation.jsonl`
- `kivo_vd_benchmark_report.md`
- `torch_sketch_benchmark.jsonl`, if requested
- `pipeline_summary.json`

## Pipeline Summary

`pipeline_summary.json` records:

- run name;
- model name;
- parameters;
- output file paths;
- per-stage commands;
- start/end timestamps;
- return codes;
- success/failure status.

## Interpreting Results

The pipeline report summarizes offline Q/K retrieval and simulated active KV
policy behavior. It is evidence for candidate retrieval quality, not proof of:

- measured vLLM runtime memory reduction;
- latency improvement;
- output quality preservation;
- candidate-block attention behavior.

Those claims require future Linux/NVIDIA runtime validation and quality
benchmarks.
