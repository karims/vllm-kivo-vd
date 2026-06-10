# Phase S1.2: Source-Level Valid-Slot Mutation Quality Sanity

## Why This Phase Exists

Phase S1.1 proved that the repo-local source hook can mutate a valid,
non-padding slot during real GPT-2 generation in a source-built vLLM runtime.
That is an important boundary, but it is only a state-mutation proof.

Phase S1.2 asks a narrower question:

- does the same source-level valid-slot mutation behave consistently across a
  few prompts?
- do baseline and active generations still complete?
- do we observe any obvious output changes?

This phase is still not a selected-attention claim, not a quality-preservation
claim, and not a memory/latency result.

## What It Tests

- Multiple prompts against the same source-built GPT-2 runtime.
- Baseline generation with Kivo disabled.
- Active generation with `mask_last_valid_slot`.
- Prompt-level aggregation of:
  - mutation attempts
  - mutation applications
  - blocker reasons
  - valid-slot counts
  - output changes

## What It Does Not Test

- Selected attention correctness.
- Quality preservation under a real routing policy.
- Measured runtime memory reduction.
- Latency improvement.
- Any production routing path.

## Run Commands

Use the source-built pod that already validated S1.1.

```bash
cd /workspace/vllm-kivo-vd
git pull origin chore/sync-upstream-main

PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.run_source_s1_2_quality_sanity \
  --model gpt2 \
  --max-tokens 8 \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s1_2_quality_sanity.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s1_2_quality_sanity.md \
  --continue-on-error
```

Validate the report:

```bash
PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.validate_source_s1_2_quality_sanity \
  --input-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s1_2_quality_sanity.json \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s1_2_quality_sanity_validation.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s1_2_quality_sanity_validation.md
```

## Expected Interpretation

- `quality_sanity_passed=true` means the multi-prompt source-level mutation
  completed cleanly across the prompt set.
- That is still only a source-level state-mutation sanity check.
- It does not authorize selected attention or runtime optimization claims.
- `measured_runtime_reduction` must remain `false`.
