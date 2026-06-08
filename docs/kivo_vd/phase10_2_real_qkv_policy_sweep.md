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

The summary JSON preserves a nested `summary` object and also exposes stable
top-level convenience aliases:

- `num_runs`;
- `num_succeeded`;
- `num_failed`;
- `best_policy_by_average_cosine`;
- `worst_policy_layer_budget`;
- `best_by_average_cosine`;
- `worst_by_max_relative_l2`;
- `worst_by_min_cosine`.

## Run On CUDA

```bash
.venv/bin/python scripts/kivo_vd/run_real_qkv_policy_sweep.py \
  --model gpt2 \
  --layers 0,5,11 \
  --budgets 8,16,32,64 \
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
  --budgets 8,16 \
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
  --budgets 8 \
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
  --budgets 8,16,32,64 \
  --policies recent,random,oracle_topk \
  --dry-run
```

Budgets `2` and `4` are still useful for aggressive stress testing and
failure analysis, but they are not recommended practical defaults.

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

## RunPod Results

Phase 10.2 was run on an NVIDIA RTX A6000 with Python `3.12.3`, torch
`2.8.0+cu128`, CUDA available, model `gpt2`, and standalone
HuggingFace/PyTorch execution. vLLM overlay was not used and no vLLM runtime
behavior changed.

### Fast Sweep

Command:

```bash
python scripts/kivo_vd/run_real_qkv_policy_sweep.py \
  --model gpt2 \
  --layers 0,5,11 \
  --budgets 4,8 \
  --block-sizes 16 \
  --policies recent,random,oracle_topk \
  --device cuda \
  --output-dir outputs/kivo_vd/runs/phase10_2_gpt2_policy_sweep_fast
```

Result:

- runs: `18`;
- succeeded: `18`;
- failed: `0`;
- best policy by average cosine: `oracle_topk`;
- worst policy/layer/budget by maximum relative L2: random, layer `5`,
  budget `4`, block size `16`.

| policy | count | avg cosine | min cosine | avg rel L2 | max rel L2 | avg mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle_topk | `6` | `0.981291` | `0.952236` | `0.173167` | `0.320883` | `0.856201` |
| random | `6` | `0.616757` | `0.194487` | `1.216978` | `2.546401` | `0.109252` |
| recent | `6` | `0.779774` | `0.454973` | `0.960722` | `2.088794` | `0.479193` |

### Full Sweep

Command:

```bash
python scripts/kivo_vd/run_real_qkv_policy_sweep.py \
  --model gpt2 \
  --layers 0,3,5,8,11 \
  --budgets 2,4,8,16 \
  --block-sizes 16 \
  --policies recent,random,oracle_topk \
  --device cuda \
  --output-dir outputs/kivo_vd/runs/phase10_2_gpt2_policy_sweep_full
```

Result:

- runs: `60`;
- succeeded: `60`;
- failed: `0`;
- best policy by average cosine: `oracle_topk`;
- worst policy/layer/budget by maximum relative L2: random, layer `5`,
  budget `4`, block size `16`.

Per-policy summary:

| policy | count | avg cosine | min cosine | avg rel L2 | max rel L2 | avg mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle_topk | `20` | `0.968658` | `0.762481` | `0.198869` | `0.648273` | `0.865280` |
| random | `20` | `0.597745` | `0.194487` | `1.247053` | `2.546401` | `0.148426` |
| recent | `20` | `0.813996` | `0.454511` | `0.941830` | `2.101402` | `0.476029` |

Per-layer summary:

| layer | count | avg cosine | min cosine | avg rel L2 | max rel L2 | avg mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `12` | `0.898962` | `0.452218` | `0.380939` | `1.012306` | `0.608847` |
| `3` | `12` | `0.819226` | `0.391587` | `0.655209` | `1.446637` | `0.545629` |
| `5` | `12` | `0.623034` | `0.194487` | `1.390707` | `2.546401` | `0.456542` |
| `8` | `12` | `0.759780` | `0.271645` | `1.018613` | `2.027017` | `0.468687` |
| `11` | `12` | `0.866330` | `0.633367` | `0.534120` | `1.027834` | `0.403188` |

Per-budget summary:

| budget | count | avg cosine | min cosine | avg rel L2 | max rel L2 | avg mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `2` | `15` | `0.745351` | `0.207958` | `0.945989` | `2.536308` | `0.399631` |
| `4` | `15` | `0.777174` | `0.194487` | `0.848361` | `2.546401` | `0.465550` |
| `8` | `15` | `0.801215` | `0.207545` | `0.769785` | `2.541953` | `0.513036` |
| `16` | `15` | `0.850125` | `0.205258` | `0.619534` | `2.446755` | `0.608098` |

The aggregate per-layer and per-budget rows combine all policies, so weak
recent/random rows pull them down. The oracle-only diagnostic below isolates
the best-case selected-block signal.

### Oracle-Only Budget And Layer Diagnostic

| layer | budget | avg cosine | min cosine | avg rel L2 | max rel L2 | mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | `2` | `0.921301` | `0.872230` | `0.420817` | `0.542081` | `0.628074` |
| `0` | `4` | `0.954544` | `0.952236` | `0.314450` | `0.320883` | `0.775595` |
| `0` | `8` | `0.975243` | `0.967358` | `0.225477` | `0.265860` | `0.862210` |
| `0` | `16` | `0.998941` | `0.998408` | `0.045503` | `0.056555` | `0.930712` |
| `3` | `2` | `0.919991` | `0.873350` | `0.374332` | `0.487130` | `0.793582` |
| `3` | `4` | `0.978662` | `0.973003` | `0.203988` | `0.231354` | `0.950096` |
| `3` | `8` | `0.991935` | `0.989499` | `0.127021` | `0.145539` | `0.969194` |
| `3` | `16` | `0.998430` | `0.997612` | `0.055790` | `0.069660` | `0.985758` |
| `5` | `2` | `0.951029` | `0.909793` | `0.285376` | `0.415975` | `0.875193` |
| `5` | `4` | `0.997568` | `0.996321` | `0.070091` | `0.085790` | `0.975644` |
| `5` | `8` | `0.999073` | `0.998225` | `0.042484` | `0.059583` | `0.988071` |
| `5` | `16` | `0.999751` | `0.999478` | `0.021655` | `0.032323` | `0.994056` |
| `8` | `2` | `0.871632` | `0.762481` | `0.458091` | `0.648273` | `0.776109` |
| `8` | `4` | `0.974605` | `0.950780` | `0.224164` | `0.316609` | `0.912712` |
| `8` | `8` | `0.989973` | `0.979373` | `0.138666` | `0.203135` | `0.947000` |
| `8` | `16` | `0.997840` | `0.995014` | `0.062492` | `0.099775` | `0.976405` |
| `11` | `2` | `0.893941` | `0.852821` | `0.448686` | `0.522207` | `0.521691` |
| `11` | `4` | `0.971833` | `0.959823` | `0.239700` | `0.294103` | `0.709712` |
| `11` | `8` | `0.989484` | `0.985771` | `0.146797` | `0.175429` | `0.825976` |
| `11` | `16` | `0.997378` | `0.995454` | `0.071796` | `0.097953` | `0.907822` |

Oracle quality improves smoothly with budget. Budget `2` is risky, especially
for layers 8 and 11. Budget `4` is mostly promising. Budget `8` is strong
across all tested layers, and budget `16` is very strong across all tested
layers.

Layer 5 was not a selected-attention failure: oracle performs very well there
even at budget `4`. The layer-5 failures are selection-policy failures.

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

The RunPod full sweep confirms that oracle top-k is strongest overall.
Recent and random are not reliable enough for routing. Selected attention
remains viable when the correct blocks are chosen, but candidate selection is
now the bottleneck.

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
