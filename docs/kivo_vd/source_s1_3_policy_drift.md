# Phase S1.3: Source-Level Policy Drift

## Why This Phase Exists

Phase S1.2 proved that the source-level valid-slot mutation machinery works in
a source-built vLLM runtime and can complete real GPT-2 generation. The first
policy was intentionally aggressive because it mutates the newest valid slot.

Phase S1.3 asks a narrower question:

- can we mutate older or middle valid slots instead?
- do those policies still complete generation?
- how much output drift do they cause relative to the S1.2 last-slot policy?

This is still a source-level control experiment, not selected attention, not a
quality-preservation claim, and not a memory or latency result.

## Policies Compared

- `mask_oldest_valid_slot`
- `mask_middle_valid_slot`
- `mask_last_valid_slot`
- optional `noop_valid_slot_shadow`

The goal is to compare drift, not to claim semantic correctness.

## What It Tests

- Baseline generation with Kivo disabled.
- Active generation for multiple mutation policies.
- Prompt-level output drift comparison.
- Prompt-level blocker reasons and valid-slot counts.
- Best drift policy selection using the lowest `output_changed_count` among
  policies that actually mutated.

## What It Does Not Test

- Selected attention correctness.
- Quality preservation under a real routing policy.
- Measured runtime memory reduction.
- Latency improvement.
- Any production routing path.

## Run Commands

Use the same source-built pod that already validated S1.2.

```bash
cd /workspace/vllm-kivo-vd
git pull origin chore/sync-upstream-main

PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.run_source_s1_2_quality_sanity \
  --model gpt2 \
  --max-tokens 8 \
  --policy mask_oldest_valid_slot \
  --policy mask_middle_valid_slot \
  --policy mask_last_valid_slot \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s1_3_policy_drift.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s1_3_policy_drift.md \
  --continue-on-error
```

Validate the report:

```bash
PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.validate_source_s1_3_policy_drift \
  --input-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s1_3_policy_drift.json \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s1_3_policy_drift_validation.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s1_3_policy_drift_validation.md
```

## Expected Interpretation

- Lower `output_changed_count` is only lower drift, not quality preserved.
- `best_drift_policy` identifies the least disruptive policy among the
  policies that actually mutated.
- `selected_attention_claim_allowed` must remain `false`.
- `performance_claim_allowed` must remain `false`.
- `measured_runtime_reduction` must remain `false`.
