# Phase S2: Source-Built vLLM Smoke Run For S1

## Purpose

Phase S1 added a repo-local source hook at `BlockTable.compute_slot_mapping`.
Phase S2 is the smallest possible smoke run to prove that a source-built vLLM
environment can import that hook and execute the S1 probe inside GPT-2
generation.

This phase is only a smoke-run readiness check. It does not claim measured
memory reduction, latency improvement, or production selected attention.

## Assumptions

- Python 3.12 is available on the RunPod image.
- Torch and CUDA are installed in the working GPU environment.
- vLLM source import or editable install is available.
- The repo-local `vllm/` tree is meant to be imported only after the source
  build is working.

## Recommended RunPod Flow

1. Pull the latest repo state.
2. Confirm whether the runtime is importing the repo-local source or the
   installed wheel.
3. If needed, perform one source-build attempt.
4. Run the S1 GPT-2 probe.
5. Validate the generated JSONL observations.

## Runtime Inspection Command

```bash
cd /workspace/vllm-kivo-vd

PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.check_source_vllm_runtime
```

## Source-Build Attempt

If the repo-local source is not importable yet, try one editable install/build
attempt:

```bash
cd /workspace/vllm-kivo-vd
uv pip install -e . --system --no-build-isolation
```

If `uv` is unavailable, a single fallback attempt is:

```bash
cd /workspace/vllm-kivo-vd
pip install -e . --no-build-isolation
```

## S1 GPT-2 Probe

```bash
cd /workspace/vllm-kivo-vd

PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.run_source_s1_gpt2_probe \
  --model gpt2 \
  --prompt "Kivo source S1 active slot mapping probe." \
  --max-tokens 4 \
  --baseline-obs-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s1_baseline.jsonl \
  --observation-obs-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s1_observations.jsonl \
  --active-obs-jsonl /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s1_active_observations.jsonl \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s1_gpt2_probe.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s1_gpt2_probe.md \
  --continue-on-error
```

## Validation Command

```bash
PYTHONPATH=/workspace/vllm-kivo-vd:/workspace/vllm-kivo-vd/scripts \
python -m scripts.kivo_vd.validate_source_s1_observations \
  --baseline-input /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s1_baseline.jsonl \
  --observation-input /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s1_observations.jsonl \
  --active-input /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s1_active_observations.jsonl \
  --output-json /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s1_validation.json \
  --output-md /workspace/vllm-kivo-vd/outputs/kivo_vd/runs/source_s1_validation.md
```

## Success Criteria

- The runtime inspection reports the repo-local source path when the source
  build is active.
- The compiled extensions import successfully.
- GPT-2 generation succeeds.
- The observation file is written.
- The active file is written.
- `mutation_attempted=true` is recorded for the active run.
- Either the mutation is applied and generation succeeds, or the blocker is
  explicit and actionable.

## Failure Criteria

- Repo-local source cannot be imported and the helper still sees only the
  installed wheel.
- Compiled extensions are missing.
- The GPT-2 probe crashes before writing the observation files.
- The active record is missing blocker details when mutation is not applied.

Phase S2 does not claim performance, memory reduction, or production-selected
attention.
