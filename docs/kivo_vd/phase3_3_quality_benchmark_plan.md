# Kivo-VD Phase 3.3: Quality Benchmark Scaffold

Phase 3.3 prepares quality benchmark planning for future Kivo-VD variants.

No candidate-block attention path exists yet. This phase does not run real
quality benchmarks, does not load models by default, and does not change vLLM
runtime behavior.

## Why This Exists

Offline sketch retrieval and active-KV policy simulation are useful, but they do
not prove output quality. Before any future behavior-changing candidate
attention implementation can claim memory or latency wins, it must pass quality
benchmarks against a full-attention baseline.

## Scaffold Script

```bash
.venv/bin/python scripts/kivo_vd/run_quality_benchmark_plan.py \
  --benchmark needle_synthetic \
  --output outputs/kivo_vd/quality_benchmark_plan.json
```

Supported scaffold modes:

- `dry_run_equality`
- `needle_synthetic`
- `perplexity_stub`

The script writes a JSON plan/output and prints compact JSON to stdout. It does
not require vLLM runtime, model downloads, or GPU access by default.

## Benchmark Families

### Dry-Run Equality

Dry-run mode should produce identical outputs because Kivo-VD decisions are
computed and ignored.

Success criteria:

- baseline inference completes;
- Kivo-enabled dry-run inference completes;
- greedy generated outputs match exactly;
- dry-run routing events export successfully when the observer is accessible.

### Perplexity On WikiText-Style Text

Future candidate attention should compare perplexity against full attention on
WikiText-style text.

Success criteria:

- same model/tokenizer/config for baseline and Kivo variant;
- bounded perplexity delta;
- no memory-reduction claim unless quality is acceptable.

### Needle-In-A-Haystack Retrieval

The scaffold can generate synthetic prompts with:

- an early needle phrase;
- repeated filler text;
- a final query asking for the needle.

Example:

```bash
.venv/bin/python scripts/kivo_vd/run_quality_benchmark_plan.py \
  --benchmark needle_synthetic \
  --needle "BLUE ORCHID" \
  --num-filler-repeats 64
```

Success criteria for future model evaluation:

- baseline retrieves the needle;
- future Kivo candidate-attention variant retrieves the same needle;
- failures are analyzed by prompt length, layer/head policy, and candidate
  budget.

### Long-Context Synthetic QA

Future tests should include synthetic long-context QA where answer evidence
appears at controlled positions: early, middle, late, and multiple distractor
positions.

### LongBench Subset

If available later, use a small LongBench subset to test realistic long-context
tasks. Keep this optional because datasets and runtime dependencies may be
heavier than the current Kivo-only test path.

### Code/Reasoning Smoke Subset

Add a small code/reasoning smoke subset once runtime candidate attention exists.
This should catch obvious regressions in exact-token dependency and multi-step
reasoning.

## Quality Gate Before Memory Claims

Future Kivo-VD memory reduction claims require all of:

- dry-run equality passing for no-op mode;
- quality benchmark deltas within agreed bounds for candidate attention mode;
- measured runtime GPU memory reduction;
- measured latency/throughput impact;
- clear separation between conservative and aggressive policies.

## Non-Goals

- No scheduler changes.
- No GPUModelRunner changes.
- No attention/kernel changes.
- No block table or slot mapping changes.
- No model architecture or training changes.
- No claim of quality preservation yet.
