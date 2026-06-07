# Kivo-VD Phase 7.3: Memory Accounting Pipeline

Phase 7.3 adds a one-command orchestrator for the complete Phase 7 memory
accounting workflow.

This is orchestration and reporting only. Kivo remains dry-run only, vLLM still
allocates its normal/full KV cache, and attention behavior is unchanged.

## Pipeline Stages

The pipeline runs:

1. Phase 7.0 baseline CUDA memory measurement with Kivo disabled.
2. Phase 7.0 CUDA memory measurement with Kivo dry-run enabled.
3. Phase 7.1 theoretical active-KV estimation from exported routing events.
4. Phase 7.2 measured-baseline versus theoretical-estimate comparison.

Each stage records its command, timestamps, return code, status, stdout/stderr
previews, and expected output paths in `pipeline_summary.json`.

## Run On RunPod

From the repository root in the prepared Linux/NVIDIA environment:

```bash
.venv/bin/python scripts/kivo_vd/run_memory_accounting_pipeline.py \
  --model gpt2 \
  --max-tokens 32 \
  --gpu-memory-utilization 0.05 \
  --max-model-len 256 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 1 \
  --num-layers 12 \
  --num-kv-heads 12 \
  --head-dim 64 \
  --block-size 16 \
  --dtype-bytes 2 \
  --run-name phase7_gpt2_memory_accounting
```

The conservative runtime limits favor validation over throughput. Explicit KV
metadata keeps the theoretical byte calculation auditable.

## RunPod Medium-Context Validation

The full pipeline completed successfully on the RunPod GPU runtime with the
vLLM source overlay active.

Configuration:

| field | value |
| --- | ---: |
| model | `gpt2` |
| prompt tokens | `632` |
| generated tokens | `32` |
| max model length | `768` |
| max batched tokens | `768` |
| max sequences | `1` |
| block size | `16` |
| dtype bytes | `2` |
| layers | `12` |
| KV heads | `12` |
| head dimension | `64` |

Command:

```bash
.venv/bin/python scripts/kivo_vd/run_memory_accounting_pipeline.py \
  --model gpt2 \
  --prompt "$LONG_PROMPT" \
  --max-tokens 32 \
  --gpu-memory-utilization 0.05 \
  --max-model-len 768 \
  --max-num-batched-tokens 768 \
  --max-num-seqs 1 \
  --num-layers 12 \
  --num-kv-heads 12 \
  --head-dim 64 \
  --block-size 16 \
  --dtype-bytes 2 \
  --run-name phase7_gpt2_medium_memory_accounting
```

All four stages succeeded:

- `baseline_memory_measurement`
- `kivo_dry_run_memory_measurement`
- `event_memory_estimate`
- `memory_comparison_report`

Measured baseline and Kivo dry-run memory were identical:

| metric | baseline | Kivo dry-run | difference |
| --- | ---: | ---: | ---: |
| model/init allocated delta | `1,147,972,096` | `1,147,972,096` | `0` |
| model/init reserved delta | `1,174,405,120` | `1,174,405,120` | `0` |
| generation allocated delta | `512` | `512` | `0` |
| generation reserved delta | `6,291,456` | `6,291,456` | `0` |
| peak allocated bytes | `1,161,792,512` | `1,161,792,512` | `0` |
| peak reserved bytes | `1,180,696,576` | `1,180,696,576` | `0` |

This identical measured result is expected: dry-run Kivo records candidate
decisions but does not alter KV allocation or attention.

The event estimator processed `32` routing events:

| theoretical metric | value |
| --- | ---: |
| bytes per KV block | `589,824` |
| average selected blocks | `16.0` |
| average skipped blocks | `24.9375` |
| average active KV bytes | `9,437,184` |
| average skipped KV bytes | `14,708,736` |
| average estimated reduction ratio | `0.609045` |

The `0.609045` ratio is theoretical active-KV accounting only. The pipeline
reported `savings_are_theoretical_only: true` and
`measured_runtime_reduction: false`.

## Dry-Run Planning

Dry-run mode creates the run directory and `pipeline_summary.json`, but does
not start vLLM or execute any stage:

```bash
.venv/bin/python scripts/kivo_vd/run_memory_accounting_pipeline.py \
  --model gpt2 \
  --run-name phase7_gpt2_memory_accounting_dry_run \
  --dry-run
```

Use `--continue-on-error` to attempt later stages after a failure for diagnostic
purposes. Without it, dependent stages are recorded as skipped.

## Expected Outputs

The run directory contains:

- `baseline_memory.json`
- `kivo_dry_run_memory.json`
- `kivo_dry_run_events.jsonl`
- `kivo_event_memory_estimate.json`
- `kivo_event_memory_estimate.md`
- `memory_comparison.json`
- `memory_comparison.md`
- `pipeline_summary.json`

Unless `--output-dir` supplies an exact directory, runs are written under:

```text
outputs/kivo_vd/runs/<run-name>/
```

## Interpretation

The baseline and Kivo memory files contain measured CUDA allocator checkpoints.
The event estimate describes counterfactual selected/skipped KV payload bytes.
The comparison report places both in one artifact without treating theoretical
savings as memory released by the current runtime.

Measured memory should not drop because Kivo does not yet alter KV allocation
or attention. A lower value in one run is a measurement observation, not a
Kivo-caused reduction claim.

## Caveats

- Savings are theoretical only.
- `measured_runtime_reduction` remains `false`.
- No active routing, candidate-block attention, or KV residency mechanism
  exists yet.
- No latency or quality claim follows from this pipeline.
