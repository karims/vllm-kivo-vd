# Kivo-VD Phase 4A.5: Qwen SRHT Comparison Attempt

Phase 4A.5 prepared and attempted a modern-model offline SRHT comparison using
`Qwen/Qwen2.5-0.5B`.

This phase remains offline-only. It does not change vLLM runtime behavior and
does not make any measured memory, latency, or quality claims.

## Target Run

- Model: `Qwen/Qwen2.5-0.5B`
- Extraction mode: `auto`
- Expected extraction path: separate `q_proj` / `k_proj`
- Prompt mode: `blue_orchid`
- Sketches:
  - `count_sketch`
  - `random_projection`
  - `srht`
- Sketch dims:
  - `32`
  - `64`
- Layers: `0,1,2,3`
- Heads: `0,1,2,3`
- Max tokens: `900`

## Dry Run

Command:

```bash
.venv/bin/python scripts/kivo_vd/run_offline_benchmark_pipeline.py \
  --dry-run \
  --model-name Qwen/Qwen2.5-0.5B \
  --extraction-mode auto \
  --prompt-mode blue_orchid \
  --sketch-types count_sketch,random_projection,srht \
  --sketch-dims 32,64 \
  --layers 0,1,2,3 \
  --heads 0,1,2,3 \
  --max-tokens 900 \
  --run-name qwen_srht_comparison_dry_run
```

Result: succeeded.

Dry-run directory:

```text
outputs/kivo_vd/runs/qwen_srht_comparison_dry_run/
```

## Real Run Attempt

Command:

```bash
.venv/bin/python scripts/kivo_vd/run_offline_benchmark_pipeline.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --extraction-mode auto \
  --prompt-mode blue_orchid \
  --sketch-types count_sketch,random_projection,srht \
  --sketch-dims 32,64 \
  --layers 0,1,2,3 \
  --heads 0,1,2,3 \
  --max-tokens 900 \
  --run-name qwen_srht_comparison
```

Real-run directory:

```text
outputs/kivo_vd/runs/qwen_srht_comparison/
```

## Failure / Blocker Summary

The first real attempt reached model loading but failed during Q/K extraction
because Qwen tensors were `bfloat16` and NumPy conversion rejected that dtype.

Failure category:

- runtime error in optional offline HF extraction

Error snippet:

```text
TypeError: Got unsupported ScalarType BFloat16
```

Minimal fix applied:

- cast extracted Q/K tensors to float32 before CPU/NumPy conversion in
  `scripts/kivo_vd/run_hf_qk_sketch_eval.py`.

After the fix, Kivo-only tests passed and the retry progressed into the Qwen HF
head sweep. The retry produced partial HF rows but did not complete the full
pipeline in a reasonable local Mac window.

Observed partial progress:

```text
outputs/kivo_vd/runs/qwen_srht_comparison/hf_qk_head_sweep_ranked.jsonl
56 rows
```

Because the full sweep did not complete, no Qwen SRHT competitiveness claim is
made in this phase.

## What Was Validated

- The pipeline command is correctly constructed for Qwen.
- `--extraction-mode auto` resolves to the modern separate q/k projection path.
- Qwen model loading can proceed when the model is available locally or network
  access is allowed.
- The bfloat16 extraction issue is fixed for offline Q/K extraction.

## Tiny Qwen Smoke Sweep Result

A tiny Qwen head sweep was run after the bfloat16 extraction fix:

- Model: `Qwen/Qwen2.5-0.5B`
- Extraction mode: `auto`
- Q/K space: `pre_rope_projection`
- Layer: `0`
- Heads: `0,1,2,3`
- Max tokens: `256`
- Sketch dim: `32`
- Sketches:
  - `count_sketch`
  - `random_projection`
  - `srht`

| sketch_type | sketch_dim | avg block top-k recall | avg recall@2x | avg recall@4x | avg block score corr |
| --- | --- | --- | --- | --- | --- |
| count_sketch | 32 | 0.734375 | 0.937500 | 1.000000 | 0.545078 |
| random_projection | 32 | 0.750000 | 0.875000 | 1.000000 | 0.309832 |
| srht | 32 | 0.500000 | 0.796875 | 1.000000 | -0.026410 |

Conservative interpretation:

- The modern extraction path works for this tiny Qwen smoke sweep.
- The bfloat16 conversion issue is fixed.
- These rows use pre-RoPE projected Q/K, not post-RoPE attention tensors.
- This is not a full benchmark.
- SRHT does not look better than CountSketch or Random Projection in this tiny
  Qwen smoke test.
- CountSketch and Random Projection remain safer defaults.
- SRHT remains experimental and model-dependent.

## What Was Not Validated

- Full Qwen layer/head sweep completion.
- Qwen active-KV policy simulation.
- Qwen benchmark report metrics.
- SRHT competitiveness on Qwen.
- Post-RoPE Q/K behavior.
- Real vLLM runtime memory, latency, or quality behavior.

## Recommended Next Environment

Run the same command on a machine better suited for modern HF model sweeps:

- Linux workstation or cloud instance;
- enough RAM to hold `Qwen/Qwen2.5-0.5B`;
- GPU preferred, but CPU is acceptable if runtime is not a concern;
- HuggingFace network access and optionally `HF_TOKEN` for smoother downloads.

Recommended rerun:

```bash
.venv/bin/python scripts/kivo_vd/run_offline_benchmark_pipeline.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --extraction-mode auto \
  --prompt-mode blue_orchid \
  --sketch-types count_sketch,random_projection,srht \
  --sketch-dims 32,64 \
  --layers 0,1,2,3 \
  --heads 0,1,2,3 \
  --max-tokens 900 \
  --run-name qwen_srht_comparison
```

If local runtime remains too slow, use a smaller smoke sweep first:

```bash
.venv/bin/python scripts/kivo_vd/run_offline_benchmark_pipeline.py \
  --model-name Qwen/Qwen2.5-0.5B \
  --extraction-mode auto \
  --prompt-mode blue_orchid \
  --sketch-types count_sketch,random_projection,srht \
  --sketch-dims 32,64 \
  --layers 0 \
  --heads 0 \
  --max-tokens 512 \
  --run-name qwen_srht_smoke
```

## Conservative Conclusion

Phase 4A.5 does not prove whether SRHT remains competitive on Qwen. It only
confirms that the modern-model command path is prepared, the separate q/k
projection extractor can progress past bfloat16 tensors, and the full Qwen
comparison should be rerun in a stronger environment.
