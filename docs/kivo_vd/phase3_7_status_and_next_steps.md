# Kivo-VD Phase 3.7: Status and Next Steps

This document is a pause-point summary before real Linux/NVIDIA runtime
validation.

## Current Branch And Status

- Branch: `chore/sync-upstream-main`
- Current work is dry-run/offline except for scheduler observer lifecycle hooks.
- No attention behavior changes have been implemented.
- No block table, slot mapping, attention metadata, GPUModelRunner, CUDA,
  Triton, or kernel behavior changes have been implemented.
- No real measured vLLM KV memory reduction exists yet.

## Implemented Components

- Kivo-VD observer and scheduler KV lifecycle hooks.
- Metadata-only sketch index.
- Candidate selector policy.
- Runtime dry-run routing event logging.
- Explicit JSONL event export.
- Synthetic debug export utility.
- Offline HF Q/K sketch evaluation.
- HF layer/head sweep.
- Modern HF Q/K extraction support:
  - GPT-2 fused `c_attn`
  - Qwen/Llama-style `q_proj` / `k_proj`
  - GQA/MQA query-head to KV-head mapping
  - pre-RoPE projection metadata
- Active KV policy simulator.
- Benchmark report generator.
- One-command offline benchmark pipeline.
- Optional torch sketch backend benchmark.
- Real vLLM dry-run script.
- Linux/NVIDIA runtime validation plan and environment checker.
- Dry-run event analyzer.
- Quality benchmark scaffold.

## Evidence So Far

Offline GPT-2-style Q/K traces show strong sketch-guided candidate-block
retrieval.

Conservative policy summary from the current offline benchmark report:

- about 38-40% estimated active-KV residency reduction;
- about 99% exact-top-block recall.

Aggressive policies show higher estimated active-KV reduction, but they should
be treated as research signals only.

The offline torch sketch backend benchmark suggests CountSketch and Random
Projection are feasible enough to continue investigation.

These results are not measured runtime memory reductions. They are offline
evidence for sketch-guided candidate KV-block retrieval.

## What Is Not Proven

- Real vLLM runtime KV memory reduction.
- Quality preservation under behavior-changing candidate attention.
- Latency improvement.
- Modern RoPE/GQA post-RoPE behavior.
- Linux/NVIDIA runtime dry-run success.
- Book-inspired sketch variants such as variation-diminishing or bidiagonal
  sketches.

## Recommended Next Milestones

- M1: Run Linux/NVIDIA runtime dry-run.
- M2: Export and analyze real dry-run routing events.
- M3: Add real K tensor sketch capture dry-run.
- M4: Add real query sketch scoring dry-run.
- M5: Implement candidate-block attention prototype.
- M6: Run quality benchmarks.
- M7: Run real memory/latency benchmarks.
- M8: Add advanced/book-inspired sketch backends.

## Conservative Claims

Safe wording:

- "Kivo-VD currently has offline evidence for sketch-guided candidate KV-block
  retrieval."
- "The current benchmark report estimates active-KV residency reduction, but
  does not measure vLLM runtime memory reduction."
- "The next proof point is Linux/NVIDIA runtime dry-run followed by quality and
  memory benchmarks."

Avoid claiming:

- measured KV memory reduction;
- production readiness;
- quality preservation;
- speedup;
- modern model runtime correctness;
- real sparse/candidate attention behavior.

## Immediate Commands

Kivo-only tests:

```bash
.venv/bin/python -m pytest tests_kivo -q
```

Offline pipeline dry run:

```bash
.venv/bin/python scripts/kivo_vd/run_offline_benchmark_pipeline.py --dry-run
```

Linux/NVIDIA runtime dry-run:

```bash
.venv/bin/python scripts/kivo_vd/run_vllm_kivo_dry_run.py \
  --model sshleifer/tiny-gpt2 \
  --max-tokens 8 \
  --enable-kivo-vd
```

Dry-run event analyzer:

```bash
.venv/bin/python scripts/kivo_vd/analyze_dry_run_events.py \
  --input outputs/kivo_vd/vllm_kivo_dry_run_events.jsonl
```

Benchmark report generation:

```bash
.venv/bin/python scripts/kivo_vd/generate_kivo_benchmark_report.py
```

Environment checker:

```bash
.venv/bin/python scripts/kivo_vd/check_vllm_runtime_env.py
```

## Pause-Point Recommendation

Pause feature work here until Linux/NVIDIA runtime dry-run is validated. The
next meaningful proof point is real vLLM inference with Kivo-VD enabled,
matching greedy baseline output and exporting dry-run routing events.
