# Phase 11.2: Selected-Attention Generation Evaluation

## Status

Phase 11.2 is the first generation-level Kivo-VD quality probe. It compares
baseline GPT-2 greedy generation with a standalone selected-attention-patched
generation path.

It remains outside vLLM. It does not use real vLLM KV, enable active routing,
or establish production generation-quality preservation.

## Why Phase 11.2

The Phase 11.1 RunPod sweep completed `30` GPT-2 logit-sensitivity runs over
layers `0,3,5,8,11` and budgets `8,16,32` with no failures.

`query_key_block_score` retained:

- top-1 match rate `1.0`;
- average KL `0.001038`;
- average logits relative L2 `0.006149`;
- average top-5 overlap `4.946667`;
- average top-10 overlap `9.933333`.

That result authorized a narrow generation-level experiment outside vLLM.

## Generation Procedure

For each prompt and decode step:

1. run the normal model on the baseline context;
2. choose the baseline next token with greedy argmax;
3. run the selected-attention patch path on the patched context;
4. replace one layer's final-token attention contribution;
5. continue the remaining GPT-2 layers;
6. choose the patched next token with greedy argmax;
7. append tokens and repeat.

The implementation intentionally recomputes full forwards. It is a
correctness experiment, not a latency or KV-cache optimization.

## Context Modes

The default is free-running generation:

- baseline tokens extend the baseline context;
- patched tokens extend the patched context;
- contexts may diverge after the first mismatch.

With `--teacher-forced-context`, patched logits are evaluated on the current
baseline-generated context at every step. This isolates per-step logit
sensitivity but is not an independently generated patched continuation.

Per-step KL is most directly comparable in teacher-forced mode. In free-running
mode, KL after a mismatch may compare different contexts.

## Metrics

Per prompt:

- baseline and patched generated token IDs and text;
- exact token-sequence match;
- common-prefix length;
- position-wise token match rate;
- first mismatch index;
- normalized token edit distance;
- average per-step KL divergence;
- average per-step top-1 match;
- average selected-block ratio.

Aggregate metrics include exact-match rate, average token match, prefix length,
edit distance, per-step KL, and per-step top-1 agreement.

## RunPod Commands

Fast query-key test:

```bash
python scripts/kivo_vd/run_selected_attention_generation_eval.py \
  --model gpt2 \
  --selection-policy query_key_block_score \
  --candidate-budget-blocks 16 \
  --block-size 16 \
  --layer-idx 0 \
  --max-new-tokens 16 \
  --device cuda \
  --output-json outputs/kivo_vd/runs/phase11_2_gpt2_qk_budget16_layer0_generation.json \
  --output-md outputs/kivo_vd/runs/phase11_2_gpt2_qk_budget16_layer0_generation.md
```

Oracle upper-bound test:

```bash
python scripts/kivo_vd/run_selected_attention_generation_eval.py \
  --model gpt2 \
  --selection-policy oracle_topk \
  --candidate-budget-blocks 16 \
  --block-size 16 \
  --layer-idx 0 \
  --max-new-tokens 16 \
  --device cuda \
  --output-json outputs/kivo_vd/phase11_2_oracle_generation.json \
  --output-md outputs/kivo_vd/phase11_2_oracle_generation.md
```

Controlled teacher-forced comparison:

```bash
python scripts/kivo_vd/run_selected_attention_generation_eval.py \
  --model gpt2 \
  --selection-policy query_key_block_score \
  --candidate-budget-blocks 16 \
  --layer-idx 0 \
  --max-new-tokens 16 \
  --teacher-forced-context \
  --device cuda
```

## Interpretation

- Strong exact and token match with low divergence supports longer and
  cross-layer offline experiments.
- Stable oracle generation with unstable query-key generation means selector
  quality remains the bottleneck.
- Divergent oracle generation means the selected-attention budget or layer may
  itself be unsafe.

Budget `16` is the initial safer baseline. Budget `8` remains an important
practical stress point. Budget `32` may be effectively full attention for
short prompts and therefore less informative for savings.

## Caveats

- Evaluation runs outside vLLM.
- No vLLM integration is implemented or authorized.
- Only one layer's final-token attention output is patched.
- Greedy decoding and batch size one are used.
- No real vLLM KV cache is accessed.
- No active routing is implemented.
- No measured runtime memory reduction is claimed.
- No latency improvement is claimed.
- This is a generation-quality probe, not a preservation claim.

## Next Step

If both query-key and oracle runs remain stable across layers and practical
budgets, Phase 11.3 may test longer prompts or multi-layer patches outside
vLLM. Runtime integration remains out of scope.

## RunPod Results

Phase 11.2 was run on GPT-2 with standalone HuggingFace/PyTorch. vLLM overlay
was not used and no vLLM runtime behavior changed. The experiment used greedy
decoding, five built-in prompts, block size `16`, and single-layer
last-token attention patching at every decode step.

### Budget 16 Layer Sweep

Budget `16` was clean across layers `0`, `5`, `8`, and `11` for both
`query_key_block_score` and `oracle_topk` over `16` generated tokens.

| policy | layer | exact match | token match | prefix | edit distance | avg KL | step top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_key_block_score` | `0` | `1.0` | `1.0` | `16.0` | `0.0` | `0.000337` | `1.0` |
| `query_key_block_score` | `5` | `1.0` | `1.0` | `16.0` | `0.0` | `0.000039` | `1.0` |
| `query_key_block_score` | `8` | `1.0` | `1.0` | `16.0` | `0.0` | `0.000224` | `1.0` |
| `query_key_block_score` | `11` | `1.0` | `1.0` | `16.0` | `0.0` | `0.000238` | `1.0` |
| `oracle_topk` | `0` | `1.0` | `1.0` | `16.0` | `0.0` | `0.000068` | `1.0` |
| `oracle_topk` | `5` | `1.0` | `1.0` | `16.0` | `0.0` | `0.000021` | `1.0` |
| `oracle_topk` | `8` | `1.0` | `1.0` | `16.0` | `0.0` | `0.000041` | `1.0` |
| `oracle_topk` | `11` | `1.0` | `1.0` | `16.0` | `0.0` | `0.000227` | `1.0` |

### Budget 8 Layer Sweep

Budget `8` was clean for layers `5`, `8`, and `11`, but layer `0` diverged
for both `query_key_block_score` and `oracle_topk`.

| policy | layer | exact match | token match | prefix | edit distance | avg KL | step top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_key_block_score` | `0` | `0.8` | `0.9` | `14.2` | `0.05` | `0.490914` | `0.9` |
| `oracle_topk` | `0` | `0.8` | `0.9` | `14.2` | `0.05` | `0.483265` | `0.9` |
| `query_key_block_score` | `5` | `1.0` | `1.0` | `16.0` | `0.0` | `0.000306` | `1.0` |
| `oracle_topk` | `5` | `1.0` | `1.0` | `16.0` | `0.0` | `0.000197` | `1.0` |
| `query_key_block_score` | `8` | `1.0` | `1.0` | `16.0` | `0.0` | `0.001043` | `1.0` |
| `oracle_topk` | `8` | `1.0` | `1.0` | `16.0` | `0.0` | `0.000260` | `1.0` |
| `query_key_block_score` | `11` | `1.0` | `1.0` | `16.0` | `0.0` | `0.001838` | `1.0` |
| `oracle_topk` | `11` | `1.0` | `1.0` | `16.0` | `0.0` | `0.001636` | `1.0` |

Because `oracle_topk` also diverged at layer `0` with budget `8`, this is a
budget/risk signal rather than merely a selector-quality failure.

### Layer 0 Budget 12 Recovery

A targeted layer-0 budget-12 run over `32` generated tokens recovered cleanly.

| policy | exact match | token match | prefix | edit distance | avg KL | step top-1 | selected ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_key_block_score` | `1.0` | `1.0` | `32.0` | `0.0` | `0.001209` | `1.0` | `0.472919` |
| `oracle_topk` | `1.0` | `1.0` | `32.0` | `0.0` | `0.000226` | `1.0` | `0.472919` |

## Adaptive-Budget Interpretation

The Phase 11.2 evidence supports a layer-aware budget policy for the next
offline experiment:

- layer `0`: use budget `12` or `16`, with `16` as the safer default;
- layers `5`, `8`, and `11`: budget `8` may be acceptable, but budget `16`
  remains the conservative baseline;
- use fallback to a larger budget when uncertainty is high.

The recommended next phase is:

> Phase 11.3 should test multi-layer generation patching outside vLLM using
> adaptive layer-aware budgets, starting conservatively: layer 0 budget 12 or
> 16, layers 5/8/11 budget 8 or 16. No vLLM integration yet.

## Readiness Helper

```bash
.venv/bin/python scripts/kivo_vd/check_phase11_generation_readiness.py \
  --inputs outputs/kivo_vd/runs/phase11_2_*.json \
  --output-json outputs/kivo_vd/phase11_generation_readiness.json \
  --output-md outputs/kivo_vd/phase11_generation_readiness.md
```

The helper reads one or more Phase 11.2 JSON files and reports clean
layer/budget pairs, divergent results, and an adaptive budget map.
