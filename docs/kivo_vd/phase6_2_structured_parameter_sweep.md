# Kivo-VD Phase 6.2: Structured Parameter Sweep

Phase 6.2 adds offline parameter sweeps for structured sign-mixing sketches.

This phase is offline only. It does not change vLLM scheduler behavior,
GPUModelRunner, attention metadata, block tables, slot mapping, CUDA/Triton
kernels, active KV routing, model architecture, training, or tokenizer behavior.

## Motivation

Phase 6.1 showed that `bidiagonal_sign_subsample` preserved the retrieval
metrics of `bidiagonal_sign` while improving torch timing, but it remained
slower than CountSketch and Random Projection.

RunPod GPT-2 dim 32 summary:

| sketch | retrieval/timing signal |
| --- | --- |
| `bidiagonal_sign_subsample` | top-k about `0.766`, recall@2x about `0.926`, recall@4x about `0.980`, score corr about `0.827`, torch total time about `1.78 ms` |
| `bidiagonal_sign` | similar retrieval, slower than subsample |
| `tridiagonal_sign` | no clear retrieval improvement and slower |
| CountSketch/RP | still fastest baselines |
| SRHT | experimental and slow in current implementation |

Phase 6.2 asks whether structured variants improve under different alpha values
and coordinate selection strategies.

## Parameters

Structured sketch types:

- `bidiagonal_sign`
- `bidiagonal_sign_subsample`
- `tridiagonal_sign`

Alpha values:

- `0.0`
- `0.25`
- `0.5`
- `0.75`
- `1.0`

Coordinate strategies:

- `uniform`: deterministic random coordinate selection, preserving prior default
  behavior.
- `stride`: evenly spaced coordinates across the input dimension.
- `low`: first `sketch_dim` coordinates.
- `high`: last `sketch_dim` coordinates.
- `alternating`: alternates low/high coordinates until filled.

Defaults remain compatible with prior structured sketch usage:

- bidiagonal alpha: `0.5`
- tridiagonal left/right alpha: `0.25` each unless overridden
- coordinate strategy: `uniform`

## Run The Sweep

```bash
python scripts/kivo_vd/run_structured_sketch_param_sweep.py \
  --model-name gpt2 \
  --sketch-types bidiagonal_sign_subsample,bidiagonal_sign,tridiagonal_sign \
  --sketch-dims 16,24,32,48 \
  --alphas 0.0,0.25,0.5,0.75,1.0 \
  --coordinate-strategies uniform,stride,low,high,alternating \
  --layers 0,1,2,3 \
  --heads 0,1,2,3 \
  --max-tokens 512 \
  --output outputs/kivo_vd/runs/phase6_2_structured_param_sweep/structured_param_sweep.jsonl
```

Each JSONL row includes:

- `sketch_type`
- `sketch_dim`
- `structured_alpha`
- `structured_coordinate_strategy`
- `model_name`
- `layer`
- `head`
- block recall and score-correlation metrics
- exact/approx top block IDs

## Summarize Results

```bash
python scripts/kivo_vd/summarize_structured_sketch_param_sweep.py \
  --input outputs/kivo_vd/runs/phase6_2_structured_param_sweep/structured_param_sweep.jsonl \
  --json-output outputs/kivo_vd/runs/phase6_2_structured_param_sweep/structured_param_sweep_summary.json \
  --markdown-output outputs/kivo_vd/runs/phase6_2_structured_param_sweep/structured_param_sweep_summary.md
```

The summary groups by:

- `sketch_type`
- `sketch_dim`
- `structured_alpha`
- `structured_coordinate_strategy`

It sorts by:

1. average recall@2x descending;
2. average strict block top-k recall descending;
3. average block score correlation descending.

## RunPod Partial GPT-2 Results

The structured sweep was run on a RunPod PyTorch development pod using the
repository source overlay. The observed vLLM version was `0.22.1`; Kivo-only
tests and syntax compilation passed in that environment.

Partial large-sweep artifacts:

- input:
  `outputs/kivo_vd/runs/phase6_2_structured_param_sweep/structured_param_sweep.jsonl`
- summary:
  `outputs/kivo_vd/runs/phase6_2_structured_param_sweep/structured_param_summary_partial.md`
- input rows: `5,814`
- grouped summary rows: `91`

Among rows where average strict top-k recall, recall@2x, and recall@4x all
reached `1.0`, block score correlation distinguished the leading settings:

| sketch | dim | alpha | coordinates | avg block score correlation |
| --- | ---: | ---: | --- | ---: |
| `bidiagonal_sign_subsample` | 48 | 0.25 | `uniform` | about `0.6994` |
| `bidiagonal_sign_subsample` | 32 | 0.50 | `stride` | about `0.6765` |
| `bidiagonal_sign_subsample` | 48 | 0.00 | `uniform` | about `0.6729` |

The focused GPT-2 sweep produced `1,440` input rows and `45` grouped summary
rows. It reached a similar conclusion: dim `32` with `stride` was a strong
balanced setting, while dim `24` and dim `48` with `uniform` also produced good
offline retrieval signals.

GPT-2 has head dimension `64`. Dim `48` retains `75%` of the original vector,
so it is less interesting as a compression-oriented setting even when its
correlation is highest. The current balanced GPT-2 candidate is therefore:

```text
bidiagonal_sign_subsample, sketch_dim=32, alpha=0.5, coordinates=stride
```

This is a parameter-selection signal from a partial offline sweep, not evidence
of runtime memory reduction. Recall saturation also means that the ranking
between these settings depends heavily on score correlation and should not be
treated as definitive.

## Existing Script Knobs

Single synthetic/HF and torch benchmark scripts also accept:

```bash
--structured-alpha 0.75
--structured-coordinate-strategy stride
```

Defaults remain unchanged, and structured sketches are not default benchmark
backends.

## Conservative Interpretation

These results are offline retrieval evidence only.

Do not interpret them as:

- measured vLLM runtime KV memory reduction;
- active KV routing;
- candidate-routed attention quality;
- latency improvement in real inference;
- proof that a structured variant is production-ready.

A useful Phase 6.2 result would identify parameter settings that improve
candidate retrieval or timing enough to justify further offline torch/GPU
benchmarking.
