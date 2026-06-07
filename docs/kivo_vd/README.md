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
- Phase 7.0 adds runtime memory baseline measurement without changing KV
  allocation or attention behavior.
- Next recommended work is memory accounting, not active routing.

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
