# Kivo-VD Documentation Index

This directory tracks Kivo-VD design, offline validation, runtime dry-run, and
roadmap notes.

Recommended reading order:

## Status

- [Phase 3.7: Status and Next Steps](phase3_7_status_and_next_steps.md)
- [Phase S5.0: KV Memory Intervention Map](source_s5_0_kv_memory_intervention_map.md)
- [Phase S5.1: KV Ownership Intervention](source_s5_1_kv_ownership_intervention.md)
- [Phase S5.2: Online KV Retention Policy](source_s5_2_online_kv_retention_policy.md)
- [Phase S5.3: KV Free Candidates](source_s5_3_kv_free_candidates.md)
- [Phase S5.4: Live Block Demotion Plan](source_s5_4_live_block_demotion_plan.md)
- [Phase S5.5: Block Table Sync Plan](source_s5_5_block_table_sync_plan.md)
- [Phase S5.6: Apply Filtered Block Table](source_s5_6_apply_filtered_block_table.md)
- [Phase S5.7: KV Sync Apply](source_s5_7_kv_sync_apply.md)
- [Phase S5.8: Runtime Block Table Apply](source_s5_8_runtime_block_table_apply.md)
- [Phase S5.9: Prepare Inputs Block Table Hook](source_s5_9_prepare_inputs_block_table_hook.md)
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
- [Phase S3.0A: Attention Metadata Path Discovery](source_s3_0_attention_metadata_path_discovery.md)
- [Phase S3.0B: Attention Metadata Observer](source_s3_0b_attention_metadata_observer.md)
- [Phase S3.1A: Shadow Selected-Attention Metadata](source_s3_1a_shadow_selected_attention_metadata.md)
- [Phase S3.1B: Shadow Sketch Selected-Attention Metadata](source_s3_1b_shadow_sketch_selected_attention_metadata.md)
- [Phase S3.2A: Active Selected-Attention Metadata](source_s3_2a_active_selected_attention_metadata.md)
- [Phase S3.2B: Active Recent-Window Attention Metadata](source_s3_2b_active_recent_window_attention_metadata.md)
- [Phase S3.3A: Attention Tensor Sketch Observer](source_s3_3a_attention_tensor_sketch_observer.md)
- [Phase S3.3B: Shadow KV Block Sketch](source_s3_3b_shadow_kv_block_sketch.md)
- [Phase S3.3C: Active Sketch KV Metadata Alias](source_s3_3c_active_sketch_kv_metadata_alias.md)
- [Phase S4.0: Quick Measurement Harness](source_s4_0_quick_measurement.md)
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
- [Phase 10.2: Real-QKV Policy Sweep](phase10_2_real_qkv_policy_sweep.md)
- [Phase 10.3: Sketch-Based Real-QKV Selectors](phase10_3_sketch_based_real_qkv_selectors.md)
- [Phase 10.4: Practical-Budget Decision Gate](phase10_4_practical_budget_decision_gate.md)

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

Phase 10.2 provides a reusable sweep across policies, layers, candidate
budgets, block sizes, and prompts. It reports oracle gaps and research failure
flags so candidate selection can be evaluated before any vLLM attention
integration.

The Phase 10.2 A6000 sweep completed 60 GPT-2 real-Q/K/V runs with no
failures. Oracle top-k was strongest overall and improved smoothly with
budget. Recent and random were unreliable; layer 5 showed that selected
attention can be strong with oracle blocks while heuristic selection fails.
Candidate selection is now the bottleneck. There is still no vLLM integration,
logits or generation-quality result, active routing, latency claim, or
measured memory reduction.

Phase 10.3 adds direct Q/K block scoring and deterministic CountSketch, random
projection, and experimental bidiagonal sign-subsample selectors. These
selectors operate on real GPT-2 projected Q/K tensors without using full
attention probabilities. The sweep compares them with oracle top-k and ranks
the best non-oracle selector, but it remains standalone and does not establish
logits, generation quality, active routing, latency, or runtime memory
reduction.

Phase 10.4 closes the selector-evidence stage with a practical-budget gate.
Selected attention is viable when block selection is good, and
`query_key_block_score` is the current best deployable baseline. Sketch
selectors are not yet safe enough as-is because their small-budget worst cases
remain severe. Budgets `2` and `4` are stress tests; practical work should
focus on `8,16,32,64`.

The allowed Phase 11 scope is logits and generation-quality evaluation outside
vLLM using `query_key_block_score` and practical budgets. No vLLM integration,
active routing, latency result, measured runtime memory reduction, or
generation-quality preservation claim exists yet.

## Phase 11 Standalone Quality Sensitivity

- [Phase 11.0: Selected-Attention Logit Sensitivity](phase11_0_selected_attention_logit_sensitivity.md)
- [Phase 11.1: Logit-Sensitivity Sweep](phase11_1_logit_sensitivity_sweep.md)
- [Phase 11.2: Selected-Attention Generation Eval](phase11_2_selected_attention_generation_eval.md)
- [Phase 11.3: Multi-Layer Generation Eval](phase11_3_multilayer_generation_eval.md)
- [Phase 11.4: Adaptive Multi-Layer Generation Sweep](phase11_4_adaptive_multilayer_generation_sweep.md)
- [Phase 11.5: Long-Context Adaptive Generation Sweep](phase11_5_long_context_adaptive_generation_sweep.md)
- [Phase 11.6: Ratio-Scaled Long-Context Sweep](phase11_6_ratio_scaled_long_context_sweep.md)
- [Phase 11.7: Longer-Context Model Probe](phase11_7_long_context_model_probe.md)

Phase 11 starts logits-level evaluation outside vLLM. Phase 11.0 patches only
one GPT-2 layer's last-token attention contribution, continues the remaining
model computation, and compares next-token logits with the unmodified model.
It is not full generation-quality evaluation, does not use real vLLM KV, and
does not authorize vLLM attention integration or active routing.

The Phase 11.0 RunPod check preserved the top-1 next-token prediction across
layers `0,5,8,11` for both `query_key_block_score` and oracle top-k at budget
`16`, with very small KL divergence. Phase 11.1 expands that signal into a
reproducible practical-budget sweep with oracle gaps and a conservative
generation-test recommendation. This remains outside vLLM and is not a claim
of generation-quality preservation.

Phase 11.2 adds a standalone greedy-generation probe. At every decode step it
patches one GPT-2 layer's last-token attention contribution and compares the
resulting continuation with unmodified greedy generation. Free-running and
teacher-forced context modes are supported. This remains outside vLLM and
does not prove production generation quality, runtime memory reduction, or
latency improvement.

The Phase 11.2 RunPod sweep found budget `16` clean across tested single
layers `0`, `5`, `8`, and `11` for both `query_key_block_score` and
`oracle_topk`. Budget `8` was clean for layers `5`, `8`, and `11`, but layer
`0` diverged for both query-key and oracle selection. A targeted layer-0
budget-12 run recovered cleanly over 32 generated tokens. The current
recommendation is adaptive and layer-aware: layer 0 budget `12` or `16`,
middle/later layers budget `8` or `16`, and no vLLM integration yet.

Phase 11.3 applies selected-attention patches to multiple GPT-2 layers during
the same standalone greedy generation run. The conservative progression is
layers `5,8` first, then `5,8,11`, then the adaptive map
`0:12,5:8,8:8,11:12`. This remains an offline quality probe and does not
authorize vLLM integration, active routing, memory-reduction claims, or
latency claims.

The RunPod results refined the adaptive map to `0:12,5:8,8:8,11:12`.
Layers `5,8` passed at budget 8. Adding layer 11 at budget 8 caused
`query_key_block_score` to diverge while oracle remained clean; raising layer
11 to budget 12 fixed the failure. The full adaptive map passed 32-token greedy
generation for both query-key and oracle selection with a selected-block ratio
near `0.39-0.40`. That ratio is theoretical standalone evidence, not measured
runtime memory reduction. No vLLM integration is authorized.

Phase 11.4 turns the adaptive map into a reproducible matrix across policies,
generation lengths, prompt sets, and optional safer maps. It reports strict
failure flags, oracle gaps, worst cases, and the best non-oracle
configuration. The sweep remains outside vLLM and deliberately keeps
`phase12_ready` false.

The Phase 11.4 RunPod sweeps exactly matched baseline greedy GPT-2
continuations for the adaptive map `0:12,5:8,8:8,11:12` across five default
prompts and twelve extended prompts, with generation lengths up to 64 tokens.
The selected-block ratio varied from about `0.394` on the default set to about
`0.74` on the extended set, demonstrating that the theoretical opportunity is
strongly context dependent. These are standalone quality and block-ratio
observations, not measured runtime memory savings.

Phase 11.5 extends the same standalone evaluator to controlled 768- and
896-token contexts. It records actual tokenizer lengths, quality metrics,
selected-block ratios, and their theoretical complements. Phase 12 remains
unauthorized; a clean result supports more prompt coverage or a larger model,
not vLLM integration.

The Phase 11.5 RunPod results showed that the short-context map
`0:12,5:8,8:8,11:12` fails at long context for both query-key and oracle
selection. The context-scaled map `0:32,5:24,8:24,11:32` passed at about 734
and 917 prompt tokens. Near GPT-2's context limit it matched the baseline in
the tested greedy continuations with a selected ratio of about `0.4807`, whose
complement is a `51.9%` theoretical active-block reduction estimate. This is
not measured runtime memory reduction, and no vLLM integration or active
routing exists.

Phase 11.6 replaces manually chosen long-context maps with ratio/context-scaled
maps derived from estimated context block count. It compares aggressive,
balanced, and safer layer-specific ratios while preserving the standalone,
outside-vLLM boundary. Passing Phase 11.6 rows can suggest Phase 11.7 broader
coverage, but Phase 12 remains unauthorized.

The Phase 11.6 RunPod sweep found that aggressive query-key ratios failed,
balanced query-key ratios passed at target 960 but failed at target 768, and
the safer ratio policy passed across the tested lengths. The best tested
quality/savings tradeoff was balanced at target 960 with map
`0:35,5:27,8:27,11:35`, selected ratio `0.527726`, and theoretical estimated
reduction `0.472274`. The safest passing config was safer at target 960 with
map `0:41,5:32,8:32,11:41`, selected ratio `0.626644`, and theoretical
estimated reduction `0.373356`. These results remain outside vLLM and do not
show measured runtime memory reduction.

Phase 11.7 adds a capability-first probe for small 2K+ context HuggingFace
models. It inspects model/tokenizer metadata and reports whether a reviewed
selected-attention adapter exists. GPT-2 remains supported; GPTNeoX/Pythia and
OPT are probe-only until architecture-specific adapters handle their
projection, positional, normalization, and residual semantics correctly.
This phase does not authorize vLLM integration.

## Phase 12 vLLM Shadow Integration

- [Phase 12.0: vLLM Shadow Integration Design](phase12_0_vllm_shadow_integration_design.md)
- [Phase 12 Shadow Event Contract](phase12_shadow_event_schema.json)
- [Phase 12.1: Shadow Event Builder](phase12_1_shadow_event_builder.md)
- [Phase 12.2: Passive Shadow Observer](phase12_2_shadow_observer.md)
- [Phase 12.3: vLLM Shadow Hook Scaffold](phase12_3_vllm_shadow_hook_scaffold.md)
- [Phase 12.4: Runtime Touchpoint Helper](phase12_4_runtime_touchpoint.md)
- [Phase 12.5: vLLM Shadow Dry-Run](phase12_5_vllm_shadow_dry_run.md)
- [Phase 12.5 RunPod Validation](phase12_5_runpod_validation_summary.md)
- [Phase 12.6A: Plugin Feasibility Probe](phase12_6a_plugin_feasibility_probe.md)
- [Phase 12.6A RunPod Validation](phase12_6a_runpod_validation_summary.md)
- [Phase 12.6B: Plugin Generate Shadow Hook](phase12_6b_plugin_generate_shadow_hook.md)
- [Phase 12.6C: Internal Hook Discovery](phase12_6c_internal_hook_discovery.md)
- [Phase 12.6D: KV Block-ID Observation](phase12_6d_kv_get_block_ids_observation.md)
- [Phase 12.7: Installed vLLM Runtime Patch](phase12_7_installed_vllm_runtime_patch.md)
- [Phase 12.8/12.9: Active Mutation Ladder](phase12_8_9_active_ladder.md)
- [Phase 12.10: BlockTable Slot-Mapping Mutation](phase12_10_block_table_slot_mapping_mutation.md)
- [Phase S1: Source-Level Selected-Block State Mutation](source_s1_selected_block_state_mutation.md)
- [Phase S1.3: Source-Level Policy Drift](source_s1_3_policy_drift.md)
- [Phase S2.0: Block Visibility and Shadow Selection](source_s2_0_block_visibility_shadow.md)
- [Phase S2.1: Active Block Mask](source_s2_1_active_block_mask.md)
- [Phase S2: Source-Built vLLM Smoke Run](source_s2_runpod_source_build_smoke.md)

Phase 12 starts a shadow-only vLLM integration design. The event contract
separates score-ranked block IDs from sequence-ordered gather IDs and requires
ordering, causal, and no-active-routing invariants. A standalone validator and
valid example trace are included. No vLLM runtime behavior, scheduler, block
table, attention metadata, KV allocation, or kernel path changes in Phase
12.0. Active integration is deferred to Phase 13 and requires a later
readiness gate.

Phase 12.1 adds a reusable standard-library event builder and deterministic
synthetic generator. Its validator-compatible events preserve score and gather
ordering separately and remain fully standalone from vLLM runtime code.

Phase 12.2 adds a disabled-by-default passive observer scaffold and synthetic
observer smoke runner. No automatic vLLM hook or behavior change is included.

Phase 12.3 adds an environment-configurable manual hook API. It remains
disabled by default, fail-closed, and unwired from all vLLM runtime paths.

Phase 12.4 adds a no-op runtime-facing helper that future vLLM code can call
with copied metadata. It is still not wired into core runtime files and does
not change model execution.

Phase 12.5 adds a real-environment and optional baseline-generation workflow,
then emits separately validated runtime-adjacent shadow events. A readiness
result requires environment, generation, and event validation to pass
together; no automatic runtime hook is added. Installed-wheel mode can
sanitize repo-root import entries and reports vLLM import provenance so an
unbuilt local source package cannot masquerade as the working wheel.

Phase 12.5 passed on RunPod with an RTX 4090 and installed vLLM `0.22.1`.
Baseline and shadow-enabled generation succeeded, and all four emitted shadow
events validated without errors or warnings. The readiness report returned
`phase12_6_runtime_hook_ready=true`. This authorizes consideration of a
separately reviewed opt-in shadow hook only; active routing remains absent,
and no measured memory or latency improvement is claimed.

Phase 12.6A adds a separately installable, marker-only
`vllm.general_plugins` package and probe runner. It tests installed-wheel
plugin discovery without monkeypatching scheduler, attention, KV cache, or
block tables. Plugin loading alone does not establish access to useful runtime
metadata.

Phase 12.6A passed on RunPod using installed vLLM `0.22.1`, PyTorch
`2.11.0+cu130`, and an RTX 4090. The environment-only marker probe and GPT-2
generation probe both succeeded, so
`phase12_6b_plugin_shadow_hook_candidate=true`. This authorizes only a
passive, opt-in, fail-closed hook-discovery experiment. No active routing,
runtime monkeypatch, scheduler or attention change, KV or block-table
mutation, or measured memory or latency improvement is present.

Phase 12.6B adds a disabled-by-default plugin-owned wrapper around the public
`vllm.LLM.generate` boundary. When explicitly enabled, it emits
validator-compatible, preview-only synthetic block-selection events after
generation and returns the exact original result object. It does not access
or modify scheduler state, attention, KV cache, block tables, or generated
outputs. Active routing and measured runtime reduction remain false.

Phase 12.6C adds discovery-only inspection of an installed vLLM wheel. It
records import availability, signatures, source provenance, and conservative
risk/usefulness rankings for public, engine, scheduler, model-runner,
KV-cache, block-table, slot-mapping, metrics, and attention surfaces. It
installs no internal patch and changes no runtime behavior. High-risk methods
are inventory targets only, not approved hooks.

Phase 12.6D adds a separately enabled, plugin-owned observation wrapper around
`KVCacheManager.get_block_ids`. It calls the original method first, copies
only bounded metadata, catches observer failures, and returns the exact
original result object. A nonempty validated observation file would prove
that the plugin reaches real block-ID metadata, but it would not authorize
active routing, KV mutation, block-table changes, or memory/latency claims.

Phase 12.7 adds tooling for a reversible patch against an installed vLLM
wheel after the plugin-only internal wrapper produced zero runtime
observations. The patch manager refuses repository-local source, backs up and
checksums the target, inserts a fail-closed observation wrapper, and restores
the exact original bytes. Guarded active mode computes only a side-channel
`would_select_blocks` preview and explicitly blocks runtime mutation.

Phase 12.8/12.9 adds the first guarded active escalation against the installed
wheel. It first removes one key from a shallow-copied attention-metadata
dictionary, then attempts to shorten one shallow-copied direct Python
slot/block list only if the metadata stage succeeds. Tensors are never mutated
in place, exceptions return the original result, and patch restore remains
mandatory. This is an invariant-discovery experiment, not production selected
attention or evidence of memory, latency, or quality improvement.

Phase 12.10 moves one level lower to
`BlockTable.compute_slot_mapping` because `_get_slot_mappings` did not expose a
safe direct Python slot structure for selected-slot mutation. The active policy
removes exactly one trailing entry from a copied Python list or tuple result,
blocks tensor-like results explicitly, and preserves the same installed-wheel
backup/restore boundary. This is still an invariant-discovery experiment only.

Phase S1 shifts the experiment into repo-local source so the live
`BlockTable.compute_slot_mapping` state can be observed and, if safe, mutated
in the source-built runtime. It mutates only the last slot-mapping entry under
explicit env flags and remains a fail-closed, no-default-change experiment.

Phase S1.1 refines that active policy to target the last valid non-padding
slot instead of the padded tail. It remains source-only, fail-closed, and
non-production.

Phase S1.2 adds a small multi-prompt quality sanity runner for the valid-slot
mutation. It checks that baseline and active generations complete across a few
prompts and aggregates output-change and blocker statistics, but it still does
not claim selected attention, quality preservation, memory reduction, or
latency improvement.

Phase S1.3 compares safer valid-slot mutation policies, including oldest and
middle slot targets, and records the resulting output drift. It is still only
a source-level control experiment and does not claim selected attention or
runtime improvement.

Phase S2.0 stops slot mutation and observes the real valid slots and visible KV
block IDs after normal slot mapping completes. It computes a deterministic
shadow selected-block set and records theoretical visible-block reduction, but
does not apply the set or claim memory, latency, quality, or selected-attention
improvement.

Phase S2.1 keeps the same selected-block logic but actively remaps older,
unselected block visibility in the slot mapping. It is a real source-level
control path, but it still does not free KV memory or prove latency or quality
improvement.

Phase S2 is the source-built smoke runbook for S1. It checks whether the
runtime is importing the repo-local source, verifies the compiled extensions,
and then exercises the GPT-2 S1 probe inside generation. No performance,
memory, latency, or production-selected-attention claim is made.

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
