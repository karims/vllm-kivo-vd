# Kivo-VD Phase 8.3: Sketch-Buffer Accounting Pipeline

Phase 8.3 provides one command for the complete Phase 8.0 through Phase 8.2
sketch-buffer accounting workflow.

This is orchestration and reporting only. Full KV remains allocated, attention
continues to use normal KV, and active routing is not implemented.

## Relationship To Phase 7

The pipeline consumes the Phase 7.1 event memory estimate and can optionally
consume the Phase 7.2 measured-memory comparison. It then models the additional
sketch-buffer payload against theoretical skipped-KV opportunities.

The Phase 7 event estimate remains counterfactual: it describes what selected
and skipped blocks could imply if a future runtime changed KV residency.

## Pipeline Stages

1. Measure or estimate compact sketch-buffer overhead with Phase 8.0.
2. Compare overhead with Phase 7 theoretical savings using Phase 8.1.
3. Model global, per-event, cumulative, and break-even accounting with
   Phase 8.2.

Every stage records its command, return code, timestamps, status, stdout/stderr
previews, and expected output paths in `pipeline_summary.json`.

## Run On CPU

```bash
.venv/bin/python \
  scripts/kivo_vd/run_sketch_buffer_accounting_pipeline.py \
  --event-estimate \
    outputs/kivo_vd/runs/phase7_gpt2_medium_memory_accounting/\
kivo_event_memory_estimate.json \
  --memory-comparison \
    outputs/kivo_vd/runs/phase7_gpt2_medium_memory_accounting/\
memory_comparison.json \
  --device cpu \
  --run-name phase8_gpt2_sketch_buffer_accounting_cpu
```

CPU mode validates payload formulas and report plumbing. It does not produce
CUDA allocator measurements.

## Run On RunPod CUDA

```bash
.venv/bin/python \
  scripts/kivo_vd/run_sketch_buffer_accounting_pipeline.py \
  --event-estimate \
    outputs/kivo_vd/runs/phase7_gpt2_medium_memory_accounting/\
kivo_event_memory_estimate.json \
  --memory-comparison \
    outputs/kivo_vd/runs/phase7_gpt2_medium_memory_accounting/\
memory_comparison.json \
  --model gpt2 \
  --num-layers 12 \
  --num-kv-heads 12 \
  --head-dim 64 \
  --block-size 16 \
  --num-blocks 256 \
  --dtype-bytes 2 \
  --sketch-types \
    count_sketch,random_projection,bidiagonal_sign_subsample \
  --sketch-dims 16,32,64 \
  --device cuda \
  --run-name phase8_gpt2_sketch_buffer_accounting
```

## RunPod Validation Result

The command above completed successfully on RunPod after regenerating the
Phase 7 medium-context GPT-2 artifacts.

Phase 7 input context:

| field | value |
| --- | ---: |
| prompt tokens | `632` |
| generated tokens | `32` |
| routing events | `32` |
| bytes per KV block | `589,824` |
| average selected blocks | `16.0` |
| average skipped blocks | `24.9375` |
| average skipped KV bytes | `14,708,736` |
| cumulative skipped KV bytes | `470,679,552` |
| theoretical active-KV reduction | `60.9045%` |

All Phase 8.3 stages succeeded:

- `sketch_buffer_overhead_measurement`
- `overhead_vs_savings_comparison`
- `event_aware_sketch_buffer_accounting`

The pipeline preserved:

- `savings_are_theoretical_only: true`
- `measured_runtime_reduction: false`
- `active_routing: false`

### Global Pool Accounting

The modeled full KV pool was `150,994,944` bytes.

| sketch dim | sketch bytes | ratio versus full KV pool |
| ---: | ---: | ---: |
| `16` | `1,179,648` | `0.7812%` |
| `32` | `2,359,296` | `1.5625%` |
| `64` | `4,718,592` | `3.1250%` |

### Per-Event And Cumulative Accounting

| dim | overhead / average skipped KV | overhead / cumulative skipped KV |
| ---: | ---: | ---: |
| `16` | `8.0201%` | `0.2506%` |
| `32` | `16.0401%` | `0.5013%` |
| `64` | `32.0802%` | `1.0025%` |

All three dimensions were classified `excellent` under cumulative accounting.
Their break-even event count was `1`, classified `immediate`. Break-even
skipped-block counts were `2`, `4`, and `8` for dims `16`, `32`, and `64`.

The cumulative net theoretical bytes were:

| dim | net theoretical bytes |
| ---: | ---: |
| `16` | `469,499,904` |
| `32` | `468,320,256` |
| `64` | `465,960,960` |

CountSketch, Random Projection, and `bidiagonal_sign_subsample` have identical
buffer sizes at the same dimension in this phase. This experiment measures
buffer shape and payload, not sketch retrieval quality.

## Dry-Run Planning

Dry-run mode creates the run directory and summary without executing torch or
reading the Phase 7 artifact:

```bash
.venv/bin/python \
  scripts/kivo_vd/run_sketch_buffer_accounting_pipeline.py \
  --event-estimate \
    outputs/kivo_vd/runs/phase7_gpt2_medium_memory_accounting/\
kivo_event_memory_estimate.json \
  --run-name phase8_gpt2_sketch_buffer_accounting_dry_run \
  --dry-run
```

Use `--continue-on-error` to attempt later stages for diagnostics after an
earlier failure. Dependent stages normally remain skipped.

## Expected Outputs

The run directory contains:

- `sketch_buffer_overhead.json`
- `sketch_buffer_overhead.md`
- `sketch_overhead_vs_savings.json`
- `sketch_overhead_vs_savings.md`
- `event_aware_sketch_buffer_accounting.json`
- `event_aware_sketch_buffer_accounting.md`
- `pipeline_summary.json`

Unless `--output-dir` provides an exact directory, outputs are written under:

```text
outputs/kivo_vd/runs/<run-name>/
```

## Interpretation

The Phase 8.0 artifact describes an additional sketch allocation. Phase 8.1
compares that overhead with average theoretical skipped-KV bytes. Phase 8.2
adds cumulative and break-even views.

A favorable ratio only says that the modeled sketch payload is small relative
to a theoretical skipped-KV opportunity. It does not mean vLLM released GPU
memory.

The RunPod result therefore validates compact-buffer accounting, not a
memory-saving runtime. The sketch buffers remain additional allocations while
the complete KV cache remains allocated and used by attention.

## Caveats

- Sketch buffers are additional overhead only.
- Full KV is still allocated.
- Savings are theoretical only.
- `measured_runtime_reduction` remains `false`.
- Active routing remains `false`.
- No latency or quality claim follows from this pipeline.
