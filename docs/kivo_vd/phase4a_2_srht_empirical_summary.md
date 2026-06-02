# Kivo-VD Phase 4A.2: SRHT Empirical Comparison Summary

Phase 4A.2 ran the offline benchmark pipeline comparing:

- CountSketch
- Random Projection
- experimental SRHT

This is an offline HuggingFace Q/K and active-KV policy simulation result. It
is not a measured vLLM runtime memory reduction, latency result, or quality
claim.

## Command

```bash
.venv/bin/python scripts/kivo_vd/run_offline_benchmark_pipeline.py \
  --model-name gpt2 \
  --prompt-mode blue_orchid \
  --sketch-types count_sketch,random_projection,srht \
  --sketch-dims 32,64,128 \
  --layers 0,1,2,3 \
  --heads 0,1,2,3 \
  --max-tokens 900 \
  --run-name gpt2_srht_comparison
```

Output directory:

```text
outputs/kivo_vd/runs/gpt2_srht_comparison/
```

Generated files:

- `hf_qk_head_sweep_ranked.jsonl`
- `active_kv_policy_simulation.jsonl`
- `kivo_vd_benchmark_report.md`
- `sketch_backend_comparison.json`
- `pipeline_summary.json`

## Run Note

GPT-2 has `head_dim=64`. SRHT uses unique subsampled Hadamard coordinates, so
`srht` at `sketch_dim=128` is invalid for this model/head dimension and was
skipped by the HF head sweep. CountSketch and Random Projection still ran at
32, 64, and 128.

The tokenizer emitted a warning that the untruncated prompt had 1227 tokens,
above GPT-2's 1024-token limit. The script still applied `--max-tokens 900`
before model evaluation.

## Retrieval Summary

| sketch_type | sketch_dim | count | avg block top-k recall | avg recall@2x | avg recall@4x | avg block score corr |
| --- | --- | --- | --- | --- | --- | --- |
| count_sketch | 32 | 64 | 0.691 | 0.863 | 0.953 | 0.904 |
| count_sketch | 64 | 64 | 0.766 | 0.910 | 0.977 | 0.923 |
| count_sketch | 128 | 64 | 0.852 | 0.977 | 0.996 | 0.973 |
| random_projection | 32 | 64 | 0.730 | 0.863 | 0.953 | 0.905 |
| random_projection | 64 | 64 | 0.730 | 0.902 | 0.973 | 0.936 |
| random_projection | 128 | 64 | 0.840 | 0.953 | 1.000 | 0.950 |
| srht | 32 | 64 | 0.820 | 0.934 | 0.984 | 0.933 |
| srht | 64 | 64 | 1.000 | 1.000 | 1.000 | 1.000 |

## Conservative Policy Summary

Policy: `recent_window_blocks=8`, `candidate_budget_blocks=16`.

| sketch_type | sketch_dim | avg active ratio | estimated reduction | exact-top recall |
| --- | --- | --- | --- | --- |
| count_sketch | 32 | 0.592 | 0.408 | 0.973 |
| count_sketch | 64 | 0.590 | 0.410 | 0.988 |
| count_sketch | 128 | 0.591 | 0.409 | 0.996 |
| random_projection | 32 | 0.605 | 0.395 | 0.980 |
| random_projection | 64 | 0.597 | 0.403 | 0.984 |
| random_projection | 128 | 0.590 | 0.410 | 1.000 |
| srht | 32 | 0.593 | 0.407 | 1.000 |
| srht | 64 | 0.590 | 0.410 | 1.000 |

## Aggressive Policy Summary

Policy: `recent_window_blocks=4`, `candidate_budget_blocks=8`.

| sketch_type | sketch_dim | avg active ratio | estimated reduction | exact-top recall |
| --- | --- | --- | --- | --- |
| count_sketch | 32 | 0.315 | 0.685 | 0.926 |
| count_sketch | 64 | 0.314 | 0.686 | 0.941 |
| count_sketch | 128 | 0.311 | 0.689 | 0.980 |
| random_projection | 32 | 0.325 | 0.675 | 0.938 |
| random_projection | 64 | 0.321 | 0.679 | 0.957 |
| random_projection | 128 | 0.313 | 0.687 | 0.980 |
| srht | 32 | 0.315 | 0.685 | 0.977 |
| srht | 64 | 0.312 | 0.688 | 1.000 |

## Interpretation

SRHT is competitive in this offline GPT-2 BLUE ORCHID comparison.

Most notably:

- SRHT dim 32 outperformed CountSketch and Random Projection dim 32 on strict
  block top-k recall, recall@2x, recall@4x, and block score correlation.
- SRHT dim 64 was perfect in this run, but this is expected to be an especially
  favorable case because GPT-2 head dimension is 64 and SRHT dim 64 preserves
  all padded Hadamard coordinates.
- Conservative policy estimates for SRHT match the prior target zone: roughly
  41% estimated active-KV reduction with 100% exact-top-block recall in this
  offline simulation.

## Conservative Conclusion

SRHT is worth keeping as an experimental backend for further offline tests.

Do not promote SRHT to the runtime default yet:

- this is one GPT-2-style offline Q/K run;
- SRHT dim 64 is a special full-coordinate case for GPT-2 head dimension 64;
- no real vLLM runtime attention or memory behavior changed;
- no quality benchmark has been run for behavior-changing candidate attention;
- no modern post-RoPE/GQA runtime validation has been performed.

The practical next step is to repeat this comparison on modern Qwen/Llama-style
models and inspect whether SRHT remains competitive when head dimensions,
RoPE behavior, and GQA/MQA mapping differ.
