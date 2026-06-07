# Kivo-VD Documentation Index

This directory tracks Kivo-VD design, offline validation, runtime dry-run, and
roadmap notes.

Recommended reading order:

## Status

- [Phase 3.7: Status and Next Steps](phase3_7_status_and_next_steps.md)
- [Phase 5.1: Linux Runtime Validation Result](phase5_1_linux_runtime_validation_result.md)
- [Phase 5.2: RunPod Benchmark Results](phase5_2_runpod_benchmark_results.md)

## Current Evidence Snapshot

| evidence | status |
| --- | --- |
| Real vLLM GPU dry-run | `gpt2` RunPod validation passed with matching greedy output |
| Dry-run event export | 97 lifecycle/routing events exported and analyzed |
| Offline active-KV policy estimate | conservative estimate about `17.7%` active-KV reduction at about `99.8%` exact-top-block recall |
| Phase 7 medium-context estimate | `60.9045%` theoretical active-KV reduction across 32 routing events |
| Phase 8 cumulative buffer overhead | dim 16: `0.2506%`; dim 32: `0.5013%`; dim 64: `1.0025%` |
| Phase 8 readiness | complete; `phase9_ready: true` for temporary selected-KV materialization only |
| Baseline vs Kivo measured memory | identical CUDA measurements in the validated dry-run |
| Runtime memory reduction | not measured or claimed yet |

## Current Research Status

- Runtime dry-run is validated on Linux/NVIDIA for a GPT-2 run.
- Offline sketch baselines are established for CountSketch, Random Projection,
  and SRHT.
- Phase 6 offline structured exploration is complete. The strongest current
  research candidate is `bidiagonal_sign_subsample` at dim `32` with `stride`;
  alpha `0.25` and `0.5` remain the next settings to compare.
- Structured results are offline retrieval evidence, not a proven runtime
  winner, active-routing result, or measured memory reduction.
- Phase 7 is complete. A 632-token GPT-2 RunPod pipeline passed all stages and
  produced a `0.609045` theoretical active-KV reduction estimate.
- Baseline and Kivo dry-run CUDA memory measurements were identical, as
  expected because allocation and attention behavior remain unchanged.
- Phase 8 is complete. Its RunPod accounting found compact cumulative buffer
  overhead relative to the Phase 7 theoretical skipped-KV opportunity.
- The Phase 8 gate reports `phase9_ready: true`, authorizing only selected-KV
  gather/copy into temporary measurement buffers outside attention.
- No measured runtime memory reduction or active routing has been demonstrated.
  Full KV remains allocated and used by attention.

## vLLM KV Runtime Map And Integration Plan

- [Phase 0: vLLM KV Map](phase0_vllm_kv_map.md)
- [Phase 2 Runtime Integration Plan](phase2_runtime_integration_plan.md)
- [Phase 2.5: Real Sketch Feasibility](phase2_5_real_sketch_feasibility.md)

## Sketch Data Model And Offline Validation

- [Phase 1: Sketch Index](phase1_sketch_index.md)
- [Phase 1.2: Offline Sketch Eval](phase1_2_offline_sketch_eval.md)
- [Phase 1.5: HF Q/K Eval](phase1_5_hf_qk_eval.md)
- [Phase 1.6: HF Layer/Head Sweep](phase1_6_hf_head_sweep.md)

## Backend, Policy, And Reports

- [Phase 2.1: Sketch Backend And Selector](phase2_1_sketch_backend_and_selector.md)
- [Phase 2.2: Runtime Dry-Run](phase2_2_runtime_dry_run.md)
- [Phase 2.3: Dry-Run Export](phase2_3_dry_run_export.md)
- [Phase 2.4: Debug Export](phase2_4_debug_export.md)
- [Phase 2.6: Torch Sketch Backend](phase2_6_torch_sketch_backend.md)
- [Phase 2.7: Torch Benchmark Breakdown](phase2_7_torch_benchmark_breakdown.md)
- [Phase 2.8: Active KV Policy Simulation](phase2_8_active_kv_policy_simulation.md)
- [Phase 2.9: Benchmark Report](phase2_9_benchmark_report.md)
- [Phase 4A: Advanced Sketch Variants](phase4a_advanced_sketch_variants.md)
- [Phase 4A.1: SRHT Comparison](phase4a_1_srht_comparison.md)
- [Phase 4A.2: SRHT Empirical Summary](phase4a_2_srht_empirical_summary.md)
- [Phase 4A.3: Fair Sketch Comparison](phase4a_3_fair_sketch_comparison.md)
- [Phase 4A.4: Fair SRHT Results](phase4a_4_fair_srht_results.md)
- [Phase 4A.5: Qwen SRHT Attempt](phase4a_5_qwen_srht_results.md)
- [Phase 6.0: Structured Linear Sketches](phase6_0_structured_linear_sketches.md)
- [Phase 6.1: Structured Sketch Variants](phase6_1_structured_sketch_variants.md)
- [Phase 6.2: Structured Parameter Sweep](phase6_2_structured_parameter_sweep.md)
- [Phase 6.3: Modern Model Structured Check](phase6_3_modern_model_structured_check.md)
- [Phase 6.4: Structured Sketch Summary](phase6_4_structured_sketch_summary.md)

## Phase 7 Memory Accounting

- [Phase 7.0: Runtime Memory Baseline](phase7_0_runtime_memory_baseline.md)
- [Phase 7.1: Dry-Run Event Memory Estimator](phase7_1_dry_run_event_memory_estimator.md)
- [Phase 7.2: Memory Baseline Vs Estimate](phase7_2_memory_baseline_vs_estimate.md)
- [Phase 7.3: Memory Accounting Pipeline](phase7_3_memory_accounting_pipeline.md)
- [Phase 7.4: Memory Decision Gate](phase7_4_memory_decision_gate.md)

Phase 7 closes with a conservative decision gate before Phase 8 overhead
experiments. Passing the gate does not authorize active routing.

## Phase 8 Runtime Memory Experiments

- [Phase 8.0: Compact Sketch-Buffer Overhead](phase8_0_sketch_buffer_overhead.md)
- [Phase 8.1: Sketch Overhead Vs Savings](phase8_1_sketch_overhead_vs_savings.md)
- [Phase 8.2: Event-Aware Buffer Accounting](phase8_2_event_aware_sketch_buffer_accounting.md)
- [Phase 8.3: Sketch-Buffer Accounting Pipeline](phase8_3_sketch_buffer_accounting_pipeline.md)
- [Phase 8.4: Sketch-Buffer Decision Gate](phase8_4_sketch_buffer_decision_gate.md)

Phase 8.0 measures additional sketch-buffer memory only. It does not replace
full KV, change attention, or enable active routing. Phase 8.1 compares that
overhead with theoretical skipped-KV bytes without claiming realized savings.
Phase 8.2 separates pool, per-event, cumulative, and break-even accounting.
Phase 8.3 reproduces the complete overhead workflow with one command while
preserving explicit theoretical-only and no-active-routing caveats.
Phase 8 closes with a decision gate before any Phase 9 selected-KV
materialization experiment. A passing gate does not authorize active routing.
The validated gate passed for temporary gather/copy measurement only. Phase 8
buffer overhead is additive and is not a memory-saving result by itself.

## Phase 9 Selected-KV Measurement

- [Phase 9.0: Selected-KV Materialization](phase9_0_selected_kv_materialization.md)
- [Phase 9.1: Materialization Comparison](phase9_1_selected_kv_materialization_comparison.md)
- [Phase 9.2: Materialization Pipeline](phase9_2_selected_kv_materialization_pipeline.md)
- [Phase 9.3: Selected-KV Decision Gate](phase9_3_selected_kv_decision_gate.md)

Phase 9.0 gathers synthetic selected KV blocks into temporary tensors outside
attention. It does not access real vLLM KV, replace full KV, or enable active
routing.
Phase 9.1 compares that microbenchmark with theoretical Phase 7 skipped-KV
opportunity and optional Phase 8 sketch-buffer overhead.
Phase 9.2 reproduces both steps with one command and preserves the synthetic,
outside-attention, no-routing boundary.
Phase 9 closes with a conservative gate before standalone Phase 10
reference-attention experiments. Passing does not authorize real vLLM routing.
Complete selected block IDs are now available through explicit
`--export-full-block-ids` opt-in. Preview-only exports remain the default and
cannot pass the Phase 10 readiness gate.

### Phase 9 Final Result

Phase 9 is complete. The RunPod L40S validation exported complete selected
block IDs for all 32 routing events, materialized all 16 requested selected
blocks per event on average, and passed the Phase 9 readiness gate with
`phase10_ready: true` and no warnings.

The corrected full-ID materialization ratio was about `0.391`, replacing the
artificially optimistic preview-only ratio near `0.195`. The authorized next
step is limited to standalone selected-attention experiments on synthetic
tensors outside vLLM.

No measured runtime memory reduction, active routing, latency improvement, or
quality preservation has been demonstrated. Full KV remains allocated.

## Phase 10 Standalone Selected Attention

- [Phase 10.0: Selected-Attention Equivalence](phase10_0_selected_attention_equivalence.md)
- [Phase 10.1: Real-QKV Selected-Attention Eval](phase10_1_real_qkv_selected_attention_eval.md)

Phase 10.0 starts the correctness path authorized by the Phase 9 gate. It
compares full versus selected attention on synthetic PyTorch Q/K/V tensors
outside vLLM. Oracle top-k selection is included only as an upper-bound
diagnostic. This phase does not use real vLLM KV, mutate block tables, change
attention kernels, enable active routing, or prove real model quality.

Phase 10.1 extracts real GPT-2 Q/K/V projections and compares full versus
selected attention output outside vLLM. It is still not generation quality,
logits evaluation, active routing, or measured runtime memory reduction.

The RunPod RTX A6000 evaluation found a strong oracle-top-k upper bound across
GPT-2 layers 0, 5, and 11 with a four-block budget. Recent-only selection
failed badly at layer 5 (`0.763696` average cosine and `0.935313` average
relative L2 error), while oracle top-k remained strong. Candidate selection is
therefore the next bottleneck. No logits, generation quality, vLLM attention
changes, active routing, or measured memory reduction have been evaluated.

## Phase 3 Runtime Dry-Run And Quality Prep

- [Phase 3.0: Runtime Dry-Run](phase3_0_runtime_dry_run.md)
- [Phase 3.1: Linux Runtime Validation](phase3_1_linux_runtime_validation.md)
- [Phase 3.2: Dry-Run Event Analysis](phase3_2_dry_run_event_analysis.md)
- [Phase 3.3: Quality Benchmark Plan](phase3_3_quality_benchmark_plan.md)
- [Phase 3.4: Offline Benchmark Pipeline](phase3_4_offline_benchmark_pipeline.md)
- [Phase 3.5: Modern HF Q/K Extraction](phase3_5_modern_hf_qk_extraction.md)
- [Phase 3.6: Modern Model Reporting](phase3_6_modern_model_reporting.md)

## Local Setup

- [Local Test Setup](local_test_setup.md)
