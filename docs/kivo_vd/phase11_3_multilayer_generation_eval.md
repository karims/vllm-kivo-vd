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
--layers 0,5,8,11 --budgets 12,8,8,8
```

Explicit adaptive map:

```bash
--layer-budget-map 0:12,5:8,8:8,11:8
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
  --layer-budget-map 0:12,5:8,8:8,11:8 \
  --block-size 16 \
  --max-new-tokens 16 \
  --device cuda
```

Run the same configurations with `oracle_topk` as an upper-bound diagnostic.

## Interpretation

- If layers `5,8` remain stable at budget `8`, add layer `11`.
- If layers `5,8,11` remain stable, test the adaptive map.
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
