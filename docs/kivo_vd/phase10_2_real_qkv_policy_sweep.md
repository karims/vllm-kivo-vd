# Kivo-VD Phase 10.2: Real-QKV Policy Sweep

Phase 10.1 established the first real GPT-2 Q/K/V correctness signal. Oracle
top-k selected attention remained strong across tested layers, while a
recent-only four-block policy failed badly at layer 5. Phase 10.2 turns that
observation into a reproducible policy, layer, budget, block-size, and prompt
sweep.

This remains a standalone HuggingFace/PyTorch experiment outside vLLM. It does
not evaluate logits or generation quality and does not change runtime
attention behavior.

## Sweep Runner

`scripts/kivo_vd/run_real_qkv_policy_sweep.py` sweeps:

- policies: recent, first, random, and oracle top-k;
- transformer layers;
- candidate block budgets;
- block sizes;
- a prompt file or five built-in long prompts.

The built-in prompts cover:

1. an early retrieval key;
2. systems and debugging instructions;
3. a code/function sentinel;
4. an important early token followed by distractors;
5. a generic long-context explanation.

They are intentionally long enough for low candidate budgets to be
meaningful while remaining below the configured GPT-2 maximum length.

## Efficient Execution

The runner loads the model once, executes GPT-2 once per prompt, caches hidden
states, and derives Q/K/V once per requested layer. All policy, budget, and
block-size combinations reuse those tensors.

This avoids rerunning the model separately for every sweep row. The attention
comparison itself reuses the Phase 10.1 reference helpers.

## Outputs

The output directory contains:

- `policy_sweep_runs.jsonl`: one row per configuration;
- `policy_sweep_summary.json`: grouped metrics, best/worst rows, oracle gaps,
  thresholds, and caveats;
- `policy_sweep_summary.md`: human-readable comparison report.

Each successful run row records:

- policy, layer, budget, and block size;
- prompt count;
- average and minimum cosine similarity;
- average and maximum relative L2 error;
- average attention mass captured;
- average selected block and token ratios;
- research failure flags.

## Run On CUDA

```bash
.venv/bin/python scripts/kivo_vd/run_real_qkv_policy_sweep.py \
  --model gpt2 \
  --layers 0,5,11 \
  --budgets 4,8,16 \
  --block-sizes 16 \
  --policies recent,random,oracle_topk \
  --device cuda \
  --output-dir outputs/kivo_vd/runs/phase10_2_gpt2_policy_sweep
```

## Faster CUDA Sweep

```bash
.venv/bin/python scripts/kivo_vd/run_real_qkv_policy_sweep.py \
  --model gpt2 \
  --layers 0,5,11 \
  --budgets 4,8 \
  --block-sizes 16 \
  --policies recent,random,oracle_topk \
  --device cuda \
  --output-dir outputs/kivo_vd/runs/phase10_2_gpt2_policy_sweep_fast
```

## CPU Example

Use a smaller plan for local development:

```bash
.venv/bin/python scripts/kivo_vd/run_real_qkv_policy_sweep.py \
  --model gpt2 \
  --layers 0 \
  --budgets 4 \
  --block-sizes 16 \
  --policies recent,oracle_topk \
  --max-length 256 \
  --device cpu \
  --output-dir outputs/kivo_vd/runs/phase10_2_gpt2_policy_sweep_cpu
```

## Download-Free Dry Run

Dry-run builds all combinations and writes planned JSONL and summary artifacts
without importing the evaluator, loading torch/Transformers, or downloading a
model:

```bash
.venv/bin/python scripts/kivo_vd/run_real_qkv_policy_sweep.py \
  --layers 0,5,11 \
  --budgets 4,8,16 \
  --policies recent,random,oracle_topk \
  --dry-run
```

## Oracle Gap

For each matching layer, budget, and block size, the report compares heuristic
policies with oracle top-k:

```text
cosine gap = oracle average cosine - policy average cosine
relative L2 gap = policy average relative L2 - oracle average relative L2
attention mass gap = oracle average mass - policy average mass
```

A large positive gap means the oracle upper bound is substantially better than
the tested policy. That points toward candidate selection as the bottleneck.

If oracle itself performs poorly at a low budget, selected attention may be
risky at that budget even with ideal block selection.

## Research Failure Flags

The runner marks rows when:

- average cosine similarity is below `0.95`;
- minimum cosine similarity is below `0.90`;
- average relative L2 error is above `0.25`;
- maximum relative L2 error is above `0.50`.

These are research heuristics for finding weak configurations. They are not
logits, benchmark-quality, or production acceptance thresholds.

## Interpretation

- Strong oracle plus weak recent/random means candidate selection is the
  immediate problem.
- Weak oracle at low budgets means the selected-attention budget itself may be
  insufficient.
- Random outperforming recent in some layers means recency is not a sufficient
  universal routing policy.
- A promising sweep does not authorize vLLM attention integration.

If oracle remains strong across prompts and budgets, Phase 10.3 should evaluate
sketch-based selectors on the same real Q/K/V tensors before any vLLM
attention work.

## Caveats

- Q/K/V projections come from a real GPT-2-style model.
- Evaluation runs outside vLLM.
- Evaluation runs outside production attention kernels.
- The real vLLM KV cache is not used.
- No block tables or slot mappings are mutated.
- No logits or generation quality is measured.
- No active routing is implemented.
- No measured runtime memory reduction is claimed.
- No latency improvement is claimed.
