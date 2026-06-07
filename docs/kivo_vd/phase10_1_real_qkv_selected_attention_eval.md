# Kivo-VD Phase 10.1: Real-QKV Selected-Attention Evaluation

Phase 10.0 compared full attention with selected-KV attention on synthetic
Q/K/V tensors. Phase 10.1 is the first real-model correctness signal: it
extracts real GPT-2 Q/K/V projections and compares full versus selected
attention output outside vLLM.

This still does not run generation through selected attention. It does not
evaluate logits, does not access the real vLLM KV cache, and does not modify
vLLM runtime behavior.

## What The Script Does

`scripts/kivo_vd/run_real_qkv_selected_attention_eval.py`:

1. Loads a HuggingFace GPT-2-style causal LM and tokenizer.
2. Tokenizes one prompt or one prompt per line from `--prompts-file`.
3. Runs the model with hidden states enabled.
4. Selects the input to a chosen GPT-2 transformer block.
5. Applies the block's `ln_1`, then GPT-2's fused `attn.c_attn` projection.
6. Splits fused Q/K/V and reshapes to:

```text
Q: [batch, heads, tokens, head dim]
K: [batch, heads, tokens, head dim]
V: [batch, heads, tokens, head dim]
```

7. Uses the last query token by default.
8. Computes full scaled dot-product attention over all causal K/V tokens.
9. Selects KV blocks, gathers those blocks, and computes selected attention.
10. Reports output similarity and attention-mass coverage.

The extracted Q/K/V space is `gpt2_projection_after_ln_1`, which matches the
input convention to GPT-2's attention projection. This script currently
targets GPT-2-style fused `c_attn` models. Modern separate q/k/v projection
models should be added later as a separate extension.

## Selection Policies

- `recent`: select the final `candidate_budget_blocks` blocks.
- `first`: select the earliest blocks.
- `random`: select deterministic random blocks from `--seed`.
- `oracle_topk`: select blocks with the largest full-attention mass.

`oracle_topk` is an upper-bound diagnostic and is not deployable because it
uses full attention probabilities to choose blocks. If oracle top-k performs
poorly, selected attention is risky even under best-case block selection. If
oracle top-k performs well but recent/random/first are weak, candidate
selection is the likely bottleneck.

Explicit `--selected-blocks` overrides policy selection and is useful for
debugging known block sets.

## Metrics

Per prompt, the report includes:

- token length;
- layer index;
- block count;
- selected block IDs;
- selected block and token ratios;
- attention mass captured by selected blocks;
- cosine similarity between full and selected attention outputs;
- relative L2 error;
- mean and maximum absolute error;
- full and selected output norms.

Aggregate metrics include the number of prompts, average cosine similarity,
average relative L2 error, average attention mass captured, minimum cosine
similarity, and maximum relative L2 error.

The JSON schema uses stable `config`, `aggregate`, and `per_prompt` objects.
The original `aggregate_metrics` and `per_prompt_rows` names remain as
compatibility aliases for early Phase 10.1 artifacts.

## RunPod Validation

Phase 10.1 was validated with standalone HuggingFace/PyTorch execution on:

- GPU: NVIDIA RTX A6000;
- Python: `3.12.3`;
- torch: `2.8.0+cu128`;
- torch CUDA: `12.8`;
- CUDA available: `true`;
- model: `gpt2`;
- vLLM overlay: not used.

No vLLM runtime behavior was involved or changed.

### Single-Prompt Result

The initial layer-0 run used 402 tokens, block size 16, 26 total blocks, and a
16-block candidate budget.

| policy | selected ratio | attention mass | cosine | relative L2 | mean abs | max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Recent | `0.615385` | `0.930932` | `0.994852` | `0.106698` | `0.005229` | `0.242405` |
| Oracle top-k | `0.615385` | `0.941169` | `0.999240` | `0.040638` | `0.003098` | `0.057238` |

Both policies were strong for this one layer-0 prompt. Oracle top-k produced
the better output match, as expected from its use of full attention mass.

### Four-Prompt Stress Test

The stress test used four longer prompts, block size 16, a four-block
candidate budget, layers 0, 5, and 11, and recent, random, and oracle-top-k
policies.

| policy | layer | avg cosine | avg relative L2 | avg mass | min cosine | max relative L2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Recent | `0` | `0.991743` | `0.110081` | `0.886173` | `0.977193` | `0.225905` |
| Recent | `5` | `0.763696` | `0.935313` | `0.544249` | `0.611168` | `1.419566` |
| Recent | `11` | `0.973831` | `0.264353` | `0.661592` | `0.950873` | `0.411150` |
| Random | `0` | `0.989903` | `0.109097` | `0.832380` | `0.968259` | `0.253555` |
| Random | `5` | `0.995226` | `0.081291` | `0.963442` | `0.988862` | `0.148835` |
| Random | `11` | `0.986174` | `0.139911` | `0.832908` | `0.971246` | `0.239230` |
| Oracle top-k | `0` | `0.992119` | `0.107568` | `0.887626` | `0.979631` | `0.210406` |
| Oracle top-k | `5` | `0.997689` | `0.058560` | `0.975808` | `0.995771` | `0.091875` |
| Oracle top-k | `11` | `0.995155` | `0.081527` | `0.905193` | `0.988072` | `0.155385` |

## Result Interpretation

This is the first real-model Q/K/V correctness signal for Kivo-VD.

Oracle top-k remained strong across all tested layers with only four selected
blocks. That result suggests selected-KV attention is not immediately invalid
on the tested real GPT-2 Q/K/V tensors.

Recent-only selection was not reliable. Layer 5 was the clear failure:

- average cosine similarity: `0.763696`;
- average relative L2 error: `0.935313`;
- average attention mass captured: `0.544249`;
- worst relative L2 error: `1.419566`.

The contrast between recent and oracle at layer 5 indicates that candidate
selection policy, rather than the selected-attention calculation itself, is
the immediate research bottleneck. Random selection also performed strongly
at layer 5, but four prompts are far too few to interpret that as a robust
policy result.

These measurements compare attention output vectors only. They do not measure
logits, generated text, benchmark accuracy, end-to-end latency, or runtime
memory reduction.

## Recent Policy Example

```bash
.venv/bin/python scripts/kivo_vd/run_real_qkv_selected_attention_eval.py \
  --model gpt2 \
  --selection-policy recent \
  --candidate-budget-blocks 16 \
  --block-size 16 \
  --layer-idx 0 \
  --device cuda \
  --output-json outputs/kivo_vd/phase10_1_gpt2_recent_real_qkv_eval.json \
  --output-md outputs/kivo_vd/phase10_1_gpt2_recent_real_qkv_eval.md
```

## Oracle Policy Example

```bash
.venv/bin/python scripts/kivo_vd/run_real_qkv_selected_attention_eval.py \
  --model gpt2 \
  --selection-policy oracle_topk \
  --candidate-budget-blocks 16 \
  --block-size 16 \
  --layer-idx 0 \
  --device cuda \
  --output-json outputs/kivo_vd/phase10_1_gpt2_oracle_real_qkv_eval.json \
  --output-md outputs/kivo_vd/phase10_1_gpt2_oracle_real_qkv_eval.md
```

## Prompt Files

To evaluate multiple prompts:

```bash
.venv/bin/python scripts/kivo_vd/run_real_qkv_selected_attention_eval.py \
  --model gpt2 \
  --prompts-file prompts.txt \
  --selection-policy oracle_topk \
  --candidate-budget-blocks 16 \
  --block-size 16 \
  --device cuda
```

`prompts.txt` should contain one prompt per non-empty line.

## Interpreting Results

High cosine similarity and low relative L2 error indicate that selected
attention output is close to full attention for the extracted last-token Q/K/V
vectors. High attention mass captured indicates the selected blocks cover much
of the full attention distribution.

Useful comparisons:

- Run `oracle_topk` first to estimate an upper-bound signal.
- Compare `recent`, `first`, and `random` against oracle.
- Increase or decrease `candidate_budget_blocks` to find sensitivity.
- Repeat across layers before drawing model-level conclusions.

This is not generation quality. It does not compare logits, sampled text, or
benchmark answers.

## Caveats

- Q/K/V projections come from a real GPT-2-style model.
- Evaluation runs outside vLLM.
- Evaluation runs outside production attention kernels.
- The real vLLM KV cache is not used.
- No block tables or slot mappings are mutated.
- No active routing is implemented.
- No measured runtime memory reduction is claimed.
- No latency improvement is claimed.
- No generation quality or quality preservation is measured.

## Next Steps

Phase 10.2 should be a policy sweep/evaluator over real Q/K/V tensors. It
should compare recent, random, oracle top-k, explicit, and later sketch-based
selectors across layers, budgets, and prompts before any vLLM attention
integration.

Modern-model Q/K/V extraction and logits or generation-quality evaluation
remain later standalone steps. Active vLLM routing is not authorized.
