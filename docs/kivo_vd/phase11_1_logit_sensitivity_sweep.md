# Phase 11.1: Logit-Sensitivity Sweep

## Status

Phase 11.1 adds a reusable sweep over the Phase 11.0 single-layer,
last-token selected-attention logit experiment.

It is standalone HuggingFace/PyTorch work outside vLLM. It does not use real
vLLM KV, change attention kernels, enable active routing, or establish
generation-quality preservation.

## Phase 11.0 Signal

The initial RunPod GPT-2 evaluation tested layers `0,5,8,11`, five prompts,
block size `16`, budget `16`, and both `query_key_block_score` and
`oracle_topk`.

- Top-1 next-token match was `1.0` for every tested layer and policy.
- Top-5 overlap was `5.0` throughout.
- Top-10 overlap was `10.0` except layer 11, where both policies averaged
  `9.8`.
- Logit cosine similarity was effectively `1.0`.
- KL divergence was very low.

This is the first logits-level green signal. It supports a broader sweep, but
it is not a full generation-quality result and does not authorize vLLM
integration.

## Sweep Dimensions

`scripts/kivo_vd/run_logit_sensitivity_sweep.py` varies:

- transformer layers;
- practical candidate budgets;
- block sizes;
- selection policies;
- sketch dimensions for sketch policies;
- prompts from a file or the five Phase 11.0 built-in prompts.

The model is loaded once and reused across all configurations.

## Outputs

The output directory contains:

- `logit_sensitivity_runs.jsonl`;
- `logit_sensitivity_summary.json`;
- `logit_sensitivity_summary.md`.

Each successful run records:

- policy, layer, budget, block size, and optional sketch dimension;
- prompt count;
- average logit cosine and relative L2;
- average KL divergence;
- top-1 match rate;
- average top-5 and top-10 overlap;
- average attention-output cosine and relative L2;
- research failure flags.

The summary includes per-policy, per-layer, per-budget, and
policy/layer/budget tables, worst cases, oracle gaps, and the best non-oracle
policy.

## Research Heuristics

Rows are flagged when:

- top-1 match rate is below `0.95`;
- average KL is above `0.01`;
- average logits relative L2 is above `0.05`;
- average top-5 overlap is below `4`.

These thresholds organize experiments. They are not model-quality guarantees.

## RunPod Commands

Full practical sweep:

```bash
python scripts/kivo_vd/run_logit_sensitivity_sweep.py \
  --model gpt2 \
  --layers 0,3,5,8,11 \
  --budgets 8,16,32 \
  --block-sizes 16 \
  --policies query_key_block_score,oracle_topk \
  --device cuda \
  --output-dir outputs/kivo_vd/runs/phase11_1_gpt2_logit_sensitivity_sweep
```

Faster sweep:

```bash
python scripts/kivo_vd/run_logit_sensitivity_sweep.py \
  --model gpt2 \
  --layers 0,5,11 \
  --budgets 8,16 \
  --block-sizes 16 \
  --policies query_key_block_score,oracle_topk \
  --device cuda \
  --output-dir outputs/kivo_vd/runs/phase11_1_gpt2_logit_sensitivity_sweep_fast
```

Download-free plan:

```bash
.venv/bin/python scripts/kivo_vd/run_logit_sensitivity_sweep.py \
  --layers 0,5,11 \
  --budgets 8,16 \
  --policies query_key_block_score,oracle_topk \
  --dry-run
```

## Oracle Gaps

For matching layer, budget, and block-size rows, the report computes:

```text
KL gap = policy KL - oracle KL
top-1 gap = oracle top-1 rate - policy top-1 rate
logits L2 gap = policy logits L2 - oracle logits L2
attention L2 gap = policy attention L2 - oracle attention L2
```

A stable oracle and unstable deployable policy indicate a selection problem.
An unstable oracle indicates that the selected-attention budget itself may be
too aggressive.

## Phase 11.2 Recommendation

The report recommends Phase 11.2 only when all successful
`query_key_block_score` rows at practical budgets have:

- top-1 match rate at least `0.95`; and
- average KL no more than `0.01`.

Phase 11.2 would remain a generation-level offline experiment outside vLLM.

## Caveats

- Evaluation runs outside vLLM.
- No vLLM integration is implemented or authorized.
- Each run patches one layer and only the final-token attention output.
- No active routing is implemented.
- No measured runtime memory reduction is claimed.
- No latency improvement is claimed.
- Full generation quality has not been measured or proven.
