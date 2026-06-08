# Phase 11.7: Longer-Context Model Probe

## Status

Phase 11.7 adds a capability-first probe for small HuggingFace causal language
models with 2K or longer context windows.

The probe runs outside vLLM. It does not modify vLLM runtime behavior, use the
vLLM KV cache, implement active routing, or demonstrate measured memory,
latency, or generation-quality preservation.

## Motivation

Phase 11.6 established ratio/context-scaled budgets on GPT-2. The best tested
quality/savings tradeoff used the balanced policy at target 960 with map
`0:35,5:27,8:27,11:35`, selected ratio `0.527726`, and theoretical estimated
active-block reduction `0.472274`.

The safest passing configuration used the safer policy at target 960 with map
`0:41,5:32,8:32,11:41`, selected ratio `0.626644`, and theoretical estimated
reduction `0.373356`.

GPT-2 is now near its useful context limit. Testing 2K-4K+ contexts requires a
different architecture, but the existing last-token selected-attention patch
is GPT-2-specific. Phase 11.7 therefore probes compatibility before attempting
new attention-patching code.

## Initial Candidates

The initial candidate set is:

- `EleutherAI/pythia-160m`
- `EleutherAI/pythia-410m`
- `facebook/opt-125m`
- `facebook/opt-350m`

Only the model passed through `--model` is probed by default. `--models`
enables an explicit comma-separated comparison and does not implicitly
download all candidates.

## Compatibility Checks

For each model, the script reports:

- model class and architecture family;
- estimated maximum context;
- number of layers;
- hidden size;
- attention heads;
- whether the model loads;
- whether controlled long prompts tokenize within context;
- whether a selected-attention adapter is registered;
- the adapter name or a concrete unsupported reason.

Architecture detection initially recognizes:

- GPT-2;
- GPTNeoX/Pythia;
- OPT.

## Adapter Boundary

`Gpt2AttentionPatchAdapter` delegates a supported GPT-2 smoke evaluation to
the existing Phase 11.6 ratio-scaled runner.

GPTNeoX/Pythia and OPT are probe-only in this phase:

- GPTNeoX/Pythia needs a reviewed adapter for fused `query_key_value`
  projection, rotary-position handling, attention output projection, and
  parallel residual semantics.
- OPT needs a reviewed adapter for decoder-layer normalization, separate
  q/k/v projections, attention output projection, and residual/MLP ordering.

The probe reports these architectures as unsupported for selected-attention
evaluation rather than attempting a speculative patch.

## Outputs

The output directory contains:

- `long_context_model_probe_runs.jsonl`
- `long_context_model_probe_summary.json`
- `long_context_model_probe_summary.md`

Supported model rows may include optional smoke metrics:

- exact sequence match;
- token match;
- average per-step KL;
- selected-block ratio;
- theoretical estimated active-block reduction.

Unsupported rows retain the load/tokenization metadata and adapter work
reason, so they remain useful for architecture planning.

## Dry Run

Dry-run plans candidate rows without importing transformers, loading model
weights, or downloading artifacts:

```bash
.venv/bin/python scripts/kivo_vd/run_long_context_model_probe.py \
  --models EleutherAI/pythia-160m,facebook/opt-125m \
  --target-token-lengths 1024 \
  --dry-run
```

## Probe One Small Model

```bash
.venv/bin/python scripts/kivo_vd/run_long_context_model_probe.py \
  --model EleutherAI/pythia-160m \
  --target-token-lengths 1024 \
  --num-prompts-per-length 1 \
  --max-new-tokens 16 \
  --device cuda \
  --output-dir outputs/kivo_vd/runs/phase11_7_pythia160m_probe
```

This is expected to report model and tokenizer compatibility while selected
attention remains unsupported until a GPTNeoX adapter is implemented.

## Probe Multiple Candidates

```bash
.venv/bin/python scripts/kivo_vd/run_long_context_model_probe.py \
  --models EleutherAI/pythia-160m,facebook/opt-125m \
  --target-token-lengths 1024 \
  --num-prompts-per-length 1 \
  --max-new-tokens 16 \
  --device cuda \
  --continue-on-error \
  --output-dir outputs/kivo_vd/runs/phase11_7_candidate_model_probe
```

## Interpreting Results

A useful probe result separates three questions:

1. Can the model and tokenizer load?
2. Can the tokenizer construct the requested long prompt within context?
3. Is a reviewed selected-attention patch adapter available?

Passing the first two checks does not imply selected-attention compatibility.
An unsupported adapter result is expected and informative during this phase.

If no candidate has a supported adapter, the next recommended implementation
target is the smallest suitable architecture, initially
`EleutherAI/pythia-160m`, followed by standalone correctness tests for the new
adapter.

## Caveats

- This probe runs outside vLLM.
- No vLLM integration is implemented.
- No active KV routing is implemented.
- Unsupported architectures are not patched speculatively.
- Any smoke generation uses greedy decoding only.
- No measured runtime memory reduction is claimed.
- No latency claim is made.
- Generation quality preservation is not claimed.
