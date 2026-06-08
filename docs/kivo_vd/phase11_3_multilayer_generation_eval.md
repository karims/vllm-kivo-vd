# Phase 11.3: Multi-Layer Generation Evaluation

## Status

Phase 11.3 extends the standalone GPT-2 generation probe from one patched
layer to multiple patched layers in the same greedy decode run.

It remains outside vLLM. It does not use real vLLM KV, enable active routing,
or establish production generation-quality preservation.

## Motivation

Phase 11.2 found:

- budget `16` clean across tested single layers `0,5,8,11`;
- budget `8` clean for layers `5,8,11`;
- layer `0` unsafe at budget `8`, including under oracle selection;
- layer `0` clean at budget `12` over 32 generated tokens.

That evidence supports a conservative multi-layer progression with
layer-aware budgets.

## Patch Semantics

At each decode step, the script:

1. computes the baseline greedy next token;
2. starts a separate patched forward pass;
3. walks GPT-2 blocks in normal layer order;
4. at each configured layer, recomputes selected attention for the final
   token;
5. applies the normal attention output projection, residual, and MLP;
6. passes the modified hidden state into all later layers;
7. computes patched logits and the patched greedy next token.

Non-final tokens use each block's normal output. Later patched layers see the
hidden state produced by earlier patches.

## Layer-Budget Configuration

Two forms are supported.

One shared or per-layer budget:

```bash
--layers 5,8,11 --budgets 8
--layers 0,5,8,11 --budgets 12,8,8,12
```

Explicit adaptive map:

```bash
--layer-budget-map 0:12,5:8,8:8,11:12
```

The explicit map takes precedence. A single `--budgets` value applies to all
listed layers; otherwise the budget count must match the layer count.

## Metrics

Generation metrics reuse Phase 11.2:

- exact token-sequence match;
- common-prefix length;
- token match rate;
- first mismatch index;
- normalized edit distance;
- average per-step KL divergence;
- average per-step top-1 agreement.

The report additionally records, for every patched layer:

- configured candidate budget;
- average selected block count;
- average selected block ratio.

## Recommended RunPod Sequence

### 1. Conservative Two-Layer Smoke

```bash
python scripts/kivo_vd/run_multilayer_selected_attention_generation_eval.py \
  --model gpt2 \
  --selection-policy query_key_block_score \
  --layers 5,8 \
  --budgets 8 \
  --block-size 16 \
  --max-new-tokens 16 \
  --device cuda \
  --output-json outputs/kivo_vd/phase11_3_qk_layers5_8.json \
  --output-md outputs/kivo_vd/phase11_3_qk_layers5_8.md
```

Repeat with `--selection-policy oracle_topk`.

### 2. Add Layer 11

```bash
python scripts/kivo_vd/run_multilayer_selected_attention_generation_eval.py \
  --model gpt2 \
  --selection-policy query_key_block_score \
  --layers 5,8,11 \
  --budgets 8 \
  --block-size 16 \
  --max-new-tokens 16 \
  --device cuda
```

### 3. Adaptive Layer-Aware Map

```bash
python scripts/kivo_vd/run_multilayer_selected_attention_generation_eval.py \
  --model gpt2 \
  --selection-policy query_key_block_score \
  --layer-budget-map 0:12,5:8,8:8,11:12 \
  --block-size 16 \
  --max-new-tokens 16 \
  --device cuda
```

Run the same configurations with `oracle_topk` as an upper-bound diagnostic.

## Interpretation

- If layers `5,8` remain stable at budget `8`, add layer `11`.
- The naive `5:8,8:8,11:8` map is a recorded stress case, not the current
  recommendation.
- Use layer 11 budget `12` in the adaptive map.
- If oracle diverges, the budget or cumulative patch is too aggressive.
- If oracle remains stable but query-key diverges, selector quality remains
  the bottleneck.

Stable adaptive-map results would justify a Phase 11.4 offline generation
sweep and readiness gate. They would not authorize vLLM integration.

## Caveats

- Evaluation runs outside vLLM.
- No vLLM integration is implemented or authorized.
- Multiple layers are patched only in this standalone experiment.
- Generation uses greedy decoding and batch size one.
- No real vLLM KV cache is used.
- No active routing is implemented.
- No measured runtime memory reduction is claimed.
- No latency improvement is claimed.
- This is a generation-quality probe, not a preservation claim.

## RunPod Results

Phase 11.3 was run with GPT-2, five built-in prompts, greedy decoding, and
standalone HuggingFace/PyTorch. vLLM overlay was not used and no vLLM runtime
behavior changed.

### Layers 5 And 8, Budget 8

The first multi-layer smoke was clean for both policies over 16 generated
tokens.

| policy | exact match | token match | prefix | edit distance | avg KL | step top-1 | selected ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_key_block_score` | `1.0` | `1.0` | `16.0` | `0.0` | `0.001122` | `1.0` | `0.321851` |
| `oracle_topk` | `1.0` | `1.0` | `16.0` | `0.0` | `0.000342` | `1.0` | `0.321851` |

### Layers 5, 8, And 11, Budget 8

Adding layer 11 at budget 8 exposed a query-key failure while oracle remained
clean.

| policy | exact match | token match | prefix | edit distance | avg KL | step top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_key_block_score` | `0.8` | `0.8125` | `12.8` | `0.1625` | `1.208180` | `0.8125` |
| `oracle_topk` | `1.0` | `1.0` | `16.0` | `0.0` | `0.001920` | `1.0` |

Because oracle passed where query-key failed, this is a selector/accumulation
issue rather than evidence that multi-layer selected attention is impossible.

### Layer 11 Budget 12 Recovery

The adaptive map `5:8,8:8,11:12` recovered clean generation for both policies.

| policy | exact match | token match | prefix | edit distance | avg KL | step top-1 | selected ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_key_block_score` | `1.0` | `1.0` | `16.0` | `0.0` | `0.001681` | `1.0` | `0.375492` |
| `oracle_topk` | `1.0` | `1.0` | `16.0` | `0.0` | `0.000901` | `1.0` | `0.375492` |

Increasing only layer 11 from budget 8 to 12 fixed the deployable-selector
failure.

### Full Adaptive Map, 16 Tokens

The map `0:12,5:8,8:8,11:12` passed for both policies.

| policy | exact match | token match | prefix | edit distance | avg KL | step top-1 | selected ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_key_block_score` | `1.0` | `1.0` | `16.0` | `0.0` | `0.003302` | `1.0` | `0.402313` |
| `oracle_topk` | `1.0` | `1.0` | `16.0` | `0.0` | `0.001130` | `1.0` | `0.402313` |

### Full Adaptive Map, 32 Tokens

The same map remained clean when generation length increased to 32 tokens.

| policy | exact match | token match | prefix | edit distance | avg KL | step top-1 | selected ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_key_block_score` | `1.0` | `1.0` | `32.0` | `0.0` | `0.002792` | `1.0` | `0.394099` |
| `oracle_topk` | `1.0` | `1.0` | `32.0` | `0.0` | `0.001101` | `1.0` | `0.394099` |

## Current Interpretation

The current adaptive map is:

```text
0:12,5:8,8:8,11:12
```

It passed 32-token greedy generation for both query-key and oracle selection.
The selected-block ratio across patched layers was about `0.39-0.40`, which
corresponds to roughly `60%` theoretical inactive blocks in this short-context
standalone experiment. This is not measured runtime memory reduction.

The evidence remains limited to GPT-2, five prompts, greedy decoding, short
contexts, and selected layers. It does not establish quality on larger or
modern models, sampling generation, many prompts, or production inference.

## Phase 11.4 Readiness

```bash
.venv/bin/python scripts/kivo_vd/check_phase11_multilayer_readiness.py \
  --inputs outputs/kivo_vd/runs/phase11_3_*.json \
  --output-json outputs/kivo_vd/phase11_4_multilayer_readiness.json \
  --output-md outputs/kivo_vd/phase11_4_multilayer_readiness.md
```

The recommended next phase is:

> Phase 11.4 should run a larger offline generation sweep outside vLLM using
> the adaptive layer-budget map `0:12,5:8,8:8,11:12`, with more prompts and
> `max_new_tokens` 32/64. vLLM integration remains out of scope.
