# Kivo-VD Phase 5.2: RunPod Benchmark Results

Phase 5.2 records the RunPod runtime dry-run and offline benchmark results that
connect the real vLLM GPU validation path with the existing sketch-retrieval
benchmark evidence.

This phase is documentation only. It does not change scheduler behavior,
GPUModelRunner, attention metadata, block tables, kernels, slot mapping, model
architecture, training, tokenizer behavior, or runtime routing.

## Environment Summary

The successful RunPod setup used a PyTorch development pod rather than the
RunPod vLLM serving template.

Environment details:

- Platform: RunPod PyTorch development pod
- GPU: RTX 4090
- Installed vLLM: `0.22.0`
- Initial PyTorch image torch: `2.8.0+cu128`
- Runtime torch after `python -m pip install vllm`: `2.11.0+cu130`
- CUDA runtime via torch: `13.0`
- vLLM source build: not used
- Kivo source overlay: `PYTHONPATH=/workspace/vllm-kivo-vd:$PYTHONPATH`
- Compiled vLLM extensions: linked into the repo source tree with
  `scripts/kivo_vd/setup_runtime_source_overlay.py`

This validates the practical source-overlay workflow documented in
[Phase 5.1](phase5_1_linux_runtime_validation_result.md): use prebuilt vLLM
wheel extensions, overlay the Kivo source tree, and keep the runtime dry-run
limits conservative.

## Runtime Dry-Run Command

The successful runtime validation command was:

```bash
python scripts/kivo_vd/run_vllm_kivo_dry_run.py \
  --model gpt2 \
  --prompt "We are testing Kivo-VD inside vLLM runtime. The system should preserve output while recording dry-run KV routing events." \
  --max-tokens 32 \
  --enable-kivo-vd \
  --gpu-memory-utilization 0.05 \
  --max-model-len 256 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 1
```

`gpt2` was used for runtime validation because `sshleifer/tiny-gpt2` can hit a
backend attention embedding-dimension limitation in this vLLM path.

## Runtime Dry-Run Result Summary

```json
{
  "model": "gpt2",
  "prompt_token_length": 28,
  "kivo_enabled": true,
  "gpu_memory_utilization": 0.05,
  "max_model_len": 256,
  "max_num_batched_tokens": 256,
  "max_num_seqs": 1,
  "baseline_text": "\n\nKivo-VD is a new feature in the vLLM runtime. It allows you to use the VLLM runtime to run your VLL",
  "kivo_text": "\n\nKivo-VD is a new feature in the vLLM runtime. It allows you to use the VLLM runtime to run your VLL",
  "outputs_match": true,
  "event_output": "outputs/kivo_vd/vllm_kivo_dry_run_events.jsonl",
  "num_events_exported": 97,
  "observer_counters": {
    "num_before_allocate_calls": 32,
    "num_after_allocate_calls": 32,
    "num_free_request_calls": 1,
    "num_dry_run_select_calls": 32,
    "num_events": 97
  },
  "observer_note": null,
  "dry_run_only": true
}
```

The important runtime result is that greedy baseline output and Kivo-enabled
output matched while the observer exported lifecycle and dry-run routing events.
The routing decisions were computed and recorded, but ignored by the runtime.

## Analyzer Result Summary

```json
{
  "input": "outputs/kivo_vd/vllm_kivo_dry_run_events.jsonl",
  "total_events": 97,
  "malformed_rows": 0,
  "event_counts": {
    "after_allocate_slots": 32,
    "before_allocate_slots": 32,
    "dry_run_routing_decision": 32,
    "free_request": 1
  },
  "num_dry_run_routing_decision_events": 32,
  "avg_selected_block_count": 3.1875,
  "avg_recent_block_count": 3.1875,
  "avg_skipped_block_count": 0.0,
  "candidate_budget_blocks": [16],
  "recent_window_blocks": [8],
  "selected_block_preview": [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2]],
  "recent_block_preview": [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2]],
  "skipped_block_preview": [[], [], [], [], []],
  "request_ids_seen": ["0-a0cdab34"],
  "sources_seen": ["free_blocks", "running", "waiting"],
  "warnings": []
}
```

The event analyzer confirms that the runtime dry-run generated all expected
observer event types:

- `before_allocate_slots`
- `after_allocate_slots`
- `dry_run_routing_decision`
- `free_request`

No malformed rows or analyzer warnings were observed.

## Offline Sketch Benchmark Command

The offline RunPod sketch benchmark used the established pipeline:

```bash
python scripts/kivo_vd/run_offline_benchmark_pipeline.py \
  --model-name gpt2 \
  --sketch-types count_sketch,random_projection,srht \
  --sketch-dims 16,32,64 \
  --layers 0,1,2,3 \
  --heads 0,1,2,3 \
  --max-tokens 512 \
  --extraction-mode auto \
  --run-name runpod_gpt2_sketch_compare \
  --run-torch-benchmark
```

Offline report path:

```text
outputs/kivo_vd/runs/runpod_gpt2_sketch_compare/kivo_vd_benchmark_report.md
```

These are offline Q/K extraction and policy simulation results, not measured
runtime memory-reduction results.

## Offline Policy Simulation Summary

The generated report summarized two useful policy points:

| policy | estimated active-KV reduction | exact-top-block recall | interpretation |
| --- | ---: | ---: | --- |
| conservative | about `17.7%` | about `99.8%` | safer candidate policy estimate |
| aggressive | about `44.7%` | about `99.2%` | stretch policy requiring quality/runtime validation |

These numbers estimate active KV block residency from offline sketch retrieval
results. They do not measure actual vLLM runtime GPU memory reduction.

## Retrieval Summary

Compressed sketch dimensions on GPT-2 head dimension `64`:

| sketch_type | dim | compression ratio | avg top-k recall | recall@2x | recall@4x | score corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| count_sketch | 16 | 0.25 | 0.664 | 0.852 | 0.984 | 0.711 |
| count_sketch | 32 | 0.50 | 0.750 | 0.941 | 0.988 | 0.835 |
| random_projection | 16 | 0.25 | 0.699 | 0.922 | 0.992 | 0.803 |
| random_projection | 32 | 0.50 | 0.742 | 0.910 | 0.992 | 0.869 |
| srht | 16 | 0.25 | 0.707 | 0.859 | 0.965 | 0.716 |
| srht | 32 | 0.50 | 0.844 | 0.980 | 0.992 | 0.865 |

Dimension `64` rows are full-dimensional for GPT-2 (`head_dim=64`) and should
be treated only as correctness/reference rows, not compression evidence.

## Torch Sketch Timing Summary

The same RunPod pipeline also ran the offline torch sketch benchmark:

| sketch_type | avg total time |
| --- | ---: |
| count_sketch | about `0.46 ms` |
| random_projection | about `0.36 ms` |
| srht | about `7.09 ms` |

SRHT had strong offline retrieval quality at dimension `32`, but the current
implementation is much slower than CountSketch and Random Projection. SRHT
therefore remains experimental and should not be promoted to the default path
without further implementation work and broader model validation.

## Baselines, Not Final Methods

CountSketch, Random Projection, and SRHT are current benchmark baselines. They
are useful because they establish reproducible comparisons for candidate-block
retrieval quality, policy simulation, and sketch-backend timing.

They are not final claims about the best Kivo-VD sketch. In particular:

- CountSketch and Random Projection remain practical baseline defaults for now.
- SRHT is promising on some offline retrieval metrics but currently slower.
- The planned structured linear-algebra and book-inspired variants have not yet
  been implemented or validated.
- Future variants should be compared against these baselines with the same
  offline pipeline before any runtime claims are made.

## Proven Vs Not Proven

Proven by Phase 5.1 and this RunPod rerun:

- Kivo-VD dry-run observer instrumentation can run inside real vLLM GPU
  inference on Linux/NVIDIA.
- The runtime path can export allocation/free and dry-run routing events.
- The tested `gpt2` greedy output matched baseline when Kivo-VD dry-run was
  enabled.
- Offline GPT-2 Q/K sketch retrieval can recover exact top-attended blocks well
  at candidate budgets such as `2x` and `4x`.
- Offline policy simulation can estimate active-block ratios for candidate
  policies.

Not proven:

- Measured vLLM runtime KV memory reduction.
- Active KV routing.
- Candidate-routed attention.
- Quality preservation under compressed/candidate attention.
- Latency improvement in real inference.
- Production suitability of CountSketch, Random Projection, or SRHT.
- Modern RoPE/GQA runtime behavior.
- The planned structured linear-algebra sketch variants.

## Recommended Next Phase

Phase 6 should implement and evaluate structured linear-algebra sketch variants
against the established baselines.

Recommended Phase 6 criteria:

- Keep evaluation offline first.
- Compare against CountSketch, Random Projection, and SRHT.
- Track retrieval quality, candidate-budget recall, score correlation, and torch
  benchmark timing.
- Avoid runtime memory-reduction claims until a behavior-changing runtime path
  exists and passes quality benchmarks.

A conservative project claim after Phase 5.2 is:

```text
Kivo-VD has real vLLM GPU dry-run validation plus offline evidence that sketch
methods can retrieve candidate KV blocks with high recall. It has not yet
implemented or measured active KV memory reduction.
```
