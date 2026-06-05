# Kivo-VD Phase 6.4: Structured Sketch Summary And Pruning Plan

Phase 6 explored structured linear-algebra sketch variants for Kivo-VD offline
candidate KV-block retrieval. This document closes the phase by summarizing the
evidence, identifying what remains worth testing, and freezing lower-priority
variants unless future evidence changes the picture.

This is a documentation summary only. It does not change vLLM runtime behavior,
scheduler behavior, GPUModelRunner, attention kernels, block tables, slot
mapping, active routing, model architecture, training, or tokenizer behavior.

## Phase 6 Purpose

Phase 6 asked whether structured sign-mixing transforms could become better
candidate-block retrieval sketches than the existing baselines:

- CountSketch
- Random Projection
- SRHT

The purpose was not to implement active KV routing or memory reduction. The
purpose was to create reproducible offline comparisons for structured sketches
before any runtime tensor path or behavior-changing attention path is attempted.

## Implemented Structured Sketch Families

Implemented experimental structured variants:

- `bidiagonal_sign`
- `bidiagonal_sign_subsample`
- `tridiagonal_sign`

Implemented sweep mechanisms:

- structured alpha sweep;
- coordinate strategy sweep;
- GPT-2 structured comparison;
- Qwen/Qwen2.5-0.5B modern-model smoke-check workflow.

Coordinate strategies supported:

- `uniform`
- `stride`
- `low`
- `high`
- `alternating`

Alpha values used for sweeps:

- `0.0`
- `0.25`
- `0.5`
- `0.75`
- `1.0`

## GPT-2 Evidence Summary

Known GPT-2 Phase 6.0/6.1 evidence:

- `bidiagonal_sign_subsample` matched `bidiagonal_sign` retrieval while being
  somewhat faster.
- At compressed GPT-2 sketch dim `32`, `bidiagonal_sign_subsample` achieved
  roughly:
  - top-k recall: `0.766`
  - recall@2x: `0.926`
  - recall@4x: `0.980`
  - block score correlation: `0.827`
- CountSketch and Random Projection remain strong baselines, but they are still
  baselines rather than final methods.
- SRHT can show high retrieval quality, but it remains slow in the current
  implementation.
- `tridiagonal_sign` did not clearly beat the bidiagonal variants.

Interpretation: `bidiagonal_sign_subsample` is the most promising structured
sign-mixing variant so far, but it is not yet a practical runtime winner.

## Qwen Smoke Check

Phase 6.3 added the Qwen/Qwen2.5-0.5B modern-model structured smoke-check
workflow. This is important because GPT-2 is not enough: modern models use
separate Q/K projections, RoPE-oriented attention, and sometimes GQA/MQA-style
query-head to KV-head mapping.

No Phase 6.3 Qwen structured-smoke result artifacts were present in this local
checkout at the time this summary was written. Fill this section after RunPod
execution of:

```bash
python scripts/kivo_vd/run_structured_sketch_param_sweep.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --sketch-types bidiagonal_sign_subsample,bidiagonal_sign,tridiagonal_sign \
  --sketch-dims 16,32 \
  --alphas 0.0,0.25,0.5,0.75,1.0 \
  --coordinate-strategies uniform,stride,low,high,alternating \
  --layers 0,1 \
  --heads 0,1,2,3 \
  --max-tokens 256 \
  --output outputs/kivo_vd/runs/phase6_3_qwen_structured_smoke/structured_param_sweep.jsonl
```

When results are available, summarize:

- best `bidiagonal_sign_subsample` alpha/coordinate strategy;
- whether Qwen pre-RoPE retrieval agrees with GPT-2 signals;
- whether GQA/MQA head mapping creates head-specific behavior;
- whether compressed dims `16` and `32` remain competitive;
- whether `tridiagonal_sign` shows any new advantage.

Caveat: Qwen rows are currently pre-RoPE projected Q/K retrieval results, not
post-RoPE vLLM runtime attention behavior.

## Practical Interpretation

Current interpretation:

- CountSketch and Random Projection are baselines, not final Kivo-VD methods.
- SRHT remains a high-recall but slow reference.
- `bidiagonal_sign_subsample` is the most promising structured sign-mixing
  variant so far.
- `tridiagonal_sign` should be frozen unless later sweeps reveal a clear
  advantage.
- Full-dimensional rows are reference/correctness rows, not compression
  evidence.
- None of the Phase 6 results prove runtime memory reduction.

## Keep And Prune Recommendation

Keep for next offline tests:

- `bidiagonal_sign_subsample`
- best alpha/coordinate strategy from Phase 6.2, once Qwen/GPT-2 sweep summaries
  identify one clearly
- CountSketch as baseline
- Random Projection as baseline

Keep as reference only:

- SRHT
- full-dimensional sketch rows

Freeze unless new evidence appears:

- `tridiagonal_sign`
- unoptimized `bidiagonal_sign`

This pruning keeps the research surface manageable while preserving enough
baselines to make future claims credible.

## What Is Proven

Phase 6 proves:

- structured sketch variants can be evaluated reproducibly offline;
- structured variants can produce non-collapsed candidate-block retrieval;
- `bidiagonal_sign_subsample` can match `bidiagonal_sign` retrieval with better
  offline timing;
- the benchmark tooling can sweep alpha and coordinate strategy parameters;
- the modern-model Qwen workflow is prepared for RunPod execution.

## What Is Not Proven

Phase 6 does not prove:

- real vLLM runtime KV memory reduction;
- active KV routing;
- quality preservation under selected/candidate KV attention;
- latency improvement in real inference;
- post-RoPE modern-model behavior;
- production suitability of any structured variant.

## Recommended Next Phase

Recommended next work should move from sketch discovery toward memory accounting
without yet changing attention behavior:

- Phase 7.0: memory accounting baseline.
- Phase 7.1: dry-run event based memory estimator.
- Later: active KV routing experiments only after memory accounting and quality
  baselines are in place.

The next proof point should be a conservative memory-accounting story, not
active routing. Kivo-VD should continue to avoid memory-reduction claims until a
runtime mechanism exists and is measured directly.
