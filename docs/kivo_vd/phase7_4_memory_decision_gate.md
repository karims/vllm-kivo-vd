# Kivo-VD Phase 7.4: Memory Decision Gate

Phase 7.4 closes memory accounting with a conservative decision gate before any
Phase 8 runtime memory experiment.

## Phase 7 Purpose Recap

Phase 7 does not reduce memory. It:

- measures baseline and Kivo dry-run CUDA memory checkpoints;
- estimates theoretical active-KV payload bytes from dry-run selected/skipped
  block events;
- compares measured runtime values with counterfactual event accounting;
- packages the workflow into a reproducible RunPod pipeline.

Phase 7 exists to decide whether a carefully scoped runtime memory experiment
is justified. vLLM still allocates and attends over the normal/full KV cache.

## Required Artifacts Before Phase 8

A complete evidence bundle contains:

- Phase 7.0 `baseline_memory.json`;
- Phase 7.0 `kivo_dry_run_memory.json`;
- Phase 7.1 event estimate JSON and Markdown;
- Phase 7.2 comparison JSON and Markdown;
- Phase 7.3 `pipeline_summary.json`;
- the exported Kivo event JSONL used by the estimator.

## Decision Criteria

Before Phase 8.0:

- runtime dry-run completes successfully;
- baseline and Kivo outputs match under greedy decoding;
- observer events are exported;
- selected and skipped block statistics are both nonzero;
- theoretical active-KV reduction is meaningful enough to justify more work;
- measured runtime reduction remains expected to be false in dry-run mode;
- pipeline stages and event analysis contain no unresolved warnings;
- model layers, KV heads, head dimension, block size, and dtype bytes are
  explicit and reproducible.

Passing this gate means ready for Phase 8.0 overhead measurement only. It does
not authorize active routing.

## Heuristic Thresholds

These thresholds are research heuristics, not memory or quality claims:

| theoretical active-KV reduction | interpretation |
| --- | --- |
| below 10% | probably not worth runtime work |
| 10% to below 25% | weak signal; consider more offline testing |
| 25% to below 40% | promising for a first runtime overhead experiment |
| 40% or above | strong research signal only if quality risk is controlled |

## Readiness Helper

```bash
RUN_DIR=outputs/kivo_vd/runs/phase7_gpt2_memory_accounting
.venv/bin/python scripts/kivo_vd/check_phase7_readiness.py \
  --pipeline-summary "$RUN_DIR/pipeline_summary.json" \
  --memory-comparison "$RUN_DIR/memory_comparison.json" \
  --event-estimate "$RUN_DIR/kivo_event_memory_estimate.json" \
  --output-json "$RUN_DIR/phase7_readiness.json" \
  --output-md "$RUN_DIR/phase7_readiness.md"
```

The helper reports artifact presence, pipeline and output checks, theoretical
reduction classification, warnings, and a recommended next step. Missing
optional artifacts become warnings rather than crashes.

## RunPod Decision-Gate Result

The GPT-2 medium-context RunPod pipeline passed the Phase 7 gate:

| gate field | result |
| --- | --- |
| pipeline success | `true` |
| all four stages succeeded | `true` |
| prompt tokens | `632` |
| routing events estimated | `32` |
| average selected blocks | `16.0` |
| average skipped blocks | `24.9375` |
| theoretical reduction | `0.609045` |
| classification | `above_40_percent_strong_research_signal` |
| measured runtime reduction | `false` |
| Phase 8.0 ready | `true` |

Baseline and Kivo dry-run CUDA measurements were identical. Kivo-minus-baseline
initialization, generation, peak allocated, and peak reserved differences were
all `0`; no peak drop was observed. This is expected because the dry-run path
does not change KV allocation or attention.

The theoretical estimate was:

- `589,824` bytes per KV block;
- `9,437,184` average active KV bytes;
- `14,708,736` average skipped KV bytes;
- `0.609045` average estimated reduction ratio.

This is a strong theoretical active-KV research signal, not measured runtime
memory reduction. The gate recommendation is:

> Proceed only to Phase 8.0 compact sketch-buffer allocation and overhead
> measurement on GPT-2. Do not enable active routing.

## Phase 7 Completion

Phase 7 is complete. It established:

- reproducible measured CUDA baselines;
- identical measured baseline and Kivo dry-run memory in the validated run;
- theoretical event-based active-KV accounting;
- a measured-versus-theoretical comparison;
- a conservative decision gate.

Phase 8.0 is authorized only for compact sketch-buffer overhead measurement.
No active routing has been implemented or authorized.

## Recommended First Phase 8 Target

Start with `gpt2`, where Linux/NVIDIA dry-run validation is stable. Use a
conservative policy and a simple CountSketch or Random Projection baseline.
Only add `bidiagonal_sign_subsample` after the baseline path works. Keep SRHT
out of the first runtime experiment because its current implementation is slow.

The safest first experiment is auxiliary sketch-buffer allocation and measured
overhead. It should not route attention.

## What Phase 8 Should Not Do Initially

- Do not modify attention kernels first.
- Do not deeply alter block tables or slot mapping first.
- Do not implement eviction or offload first.
- Do not start with Qwen runtime routing.
- Do not claim quality or memory reduction before direct measurement.

## Staged Phase 8 Plan

- Phase 8.0: allocate compact sketch buffers and measure overhead.
- Phase 8.1: add runtime selected-block accounting with real model metadata.
- Phase 8.2: prototype selected-KV materialization outside the attention path.
- Phase 8.3: compare measured overhead with savings potential.
- Phase 8.4: decide whether candidate-routed attention is justified.

## Current Claim

Kivo-VD currently has validated dry-run runtime instrumentation and theoretical
active-KV memory accounting. It has not yet demonstrated measured runtime KV
memory reduction. Phase 7 is complete, and Phase 8.0 is limited to measuring
compact sketch-buffer overhead on GPT-2 without active routing.
