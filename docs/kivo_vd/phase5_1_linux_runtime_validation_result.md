# Kivo-VD Phase 5.1: Linux Runtime Validation Result

Phase 5.1 records the first successful real vLLM GPU runtime dry-run
validation for Kivo-VD and documents the reproducible RunPod setup.

For the follow-up RunPod benchmark summary that connects this runtime dry-run
with offline sketch/policy evidence, see
[Phase 5.2: RunPod Benchmark Results](phase5_2_runpod_benchmark_results.md).

This result is still dry-run only. It does not change attention behavior, block
tables, slot mapping, attention metadata, kernels, model architecture, training,
or tokenizer behavior.

## Environment Summary

Successful validation used a RunPod Linux/NVIDIA PyTorch development pod, not
the RunPod vLLM serving template.

Runtime details:

- Base image: RunPod PyTorch development image
- Initial torch in image: `2.8.0+cu128`
- Installed command: `python -m pip install vllm`
- Installed vLLM: `0.22.0`
- Runtime torch after install: `2.11.0+cu130`
- CUDA runtime via torch: `13.0`
- vLLM source build: not used
- Kivo source overlay: `PYTHONPATH=/workspace/vllm-kivo-vd:$PYTHONPATH`

The environment checker passed, and Kivo-only tests passed:

```bash
python scripts/kivo_vd/check_vllm_runtime_env.py
python -m pytest tests_kivo -q
```

## What Failed Before

Several approaches were tried before the stable setup:

- Building vLLM from source on the RunPod PyTorch image failed due
  build/compiler/header issues.
- The RunPod vLLM serving template auto-started Qwen and consumed VRAM before
  Kivo validation could run.
- `bash -lc "sleep infinity"` did not work inside the public vLLM serving
  template because that template's command field is effectively vLLM-server
  oriented.
- `sshleifer/tiny-gpt2` is a poor runtime validation model for this vLLM path
  because its attention embedding dimension can be too small for
  `FLEX_ATTENTION` (`E=1`, while at least `16` is required).
- Without conservative runtime limits, vLLM planned/provisioned huge KV
  capacity and failed with CUDA OOM.

## Final Successful Setup

The stable path was:

1. Start a RunPod PyTorch development pod.
2. Install the prebuilt vLLM wheel.
3. Clone the Kivo repo.
4. Overlay the repo source with `PYTHONPATH`.
5. Symlink compiled wheel artifacts into the repo source tree.
6. Run Kivo-only tests.
7. Run the real vLLM dry-run script with conservative KV/runtime limits.
8. Analyze the exported dry-run event JSONL.

## Source Overlay And Symlink Setup

Install vLLM from the wheel:

```bash
python -m pip install vllm
```

Clone and enter the repo:

```bash
cd /workspace
git clone https://github.com/<your-org-or-user>/vllm-kivo-vd.git
cd /workspace/vllm-kivo-vd
git checkout chore/sync-upstream-main
```

Discover the installed vLLM package path without letting the repo overlay
shadow it:

```bash
env -u PYTHONPATH python - <<'PY'
import pathlib
import vllm

print(pathlib.Path(vllm.__file__).resolve().parent)
PY
```

Export the source overlay:

```bash
export PYTHONPATH=/workspace/vllm-kivo-vd:$PYTHONPATH
```

Then run the helper to symlink installed wheel artifacts into the source tree:

```bash
python scripts/kivo_vd/setup_runtime_source_overlay.py \
  --repo-root /workspace/vllm-kivo-vd
```

The helper:

- discovers installed vLLM with `PYTHONPATH` removed;
- links installed top-level `vllm/*.so` files into the repo `vllm/`;
- links installed `vllm/_version.py`;
- links nested `vllm/vllm_flash_attn/**/*.so` files when present;
- prints a compact JSON summary;
- is idempotent.

Manual equivalent, if needed:

```bash
INSTALLED_VLLM="$(env -u PYTHONPATH python - <<'PY'
import pathlib
import vllm

print(pathlib.Path(vllm.__file__).resolve().parent)
PY
)"

ln -sf "$INSTALLED_VLLM"/*.so /workspace/vllm-kivo-vd/vllm/
ln -sf "$INSTALLED_VLLM/_version.py" /workspace/vllm-kivo-vd/vllm/_version.py
mkdir -p /workspace/vllm-kivo-vd/vllm/vllm_flash_attn
find "$INSTALLED_VLLM/vllm_flash_attn" -name '*.so' -print
```

Prefer the helper script because it preserves nested flash-attention paths and
emits a reproducible JSON summary.

## Successful Dry-Run Command

`gpt2` succeeded where `sshleifer/tiny-gpt2` hit the tiny-attention-dimension
issue.

```bash
python scripts/kivo_vd/run_vllm_kivo_dry_run.py \
  --model gpt2 \
  --prompt "We are testing Kivo-VD inside vLLM runtime. The system should preserve output while recording dry-run KV routing events." \
  --max-tokens 32 \
  --enable-kivo-vd \
  --gpu-memory-utilization 0.05 \
  --max-model-len 256 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 1
```

The conservative runtime limits are important. They keep the validation focused
on correctness and event export rather than throughput or large KV allocation.

## Dry-Run Result Summary

```json
{
  "model": "gpt2",
  "prompt_token_length": 28,
  "kivo_enabled": true,
  "gpu_memory_utilization": 0.05,
  "max_model_len": 256,
  "max_num_batched_tokens": 256,
  "max_num_seqs": 1,
  "outputs_match": true,
  "event_output": "outputs/kivo_vd/vllm_kivo_dry_run_events.jsonl",
  "num_events_exported": 97,
  "observer_counters": {
    "num_before_allocate_calls": 32,
    "num_after_allocate_calls": 32,
    "num_free_request_calls": 1,
    "num_dry_run_select_calls": 32,
    "num_events": 97
  },
  "observer_note": null,
  "dry_run_only": true
}
```

## Analyzer Result Summary

```json
{
  "total_events": 97,
  "malformed_rows": 0,
  "event_counts": {
    "after_allocate_slots": 32,
    "before_allocate_slots": 32,
    "dry_run_routing_decision": 32,
    "free_request": 1
  },
  "num_dry_run_routing_decision_events": 32,
  "avg_selected_block_count": 3.1875,
  "avg_recent_block_count": 3.1875,
  "avg_skipped_block_count": 0.0,
  "candidate_budget_blocks": [16],
  "recent_window_blocks": [8],
  "request_ids_seen": ["0-b128cdee"],
  "sources_seen": ["free_blocks", "running", "waiting"],
  "warnings": []
}
```

## Conservative Interpretation

Proven by this run:

- The Kivo dry-run observer path runs inside real vLLM GPU inference.
- Scheduler lifecycle events are captured in a real generation path.
- Dry-run candidate-selection events are emitted and exported.
- Greedy output matched baseline for the tested `gpt2` run.
- The dry-run path did not route attention through selected blocks.

Not proven by this run:

- Real KV memory reduction.
- Active KV residency management.
- Candidate-routed attention.
- Quality preservation under any behavior-changing compressed/candidate
  attention path.
- Latency improvement.
- Behavior on modern RoPE/GQA models in real vLLM runtime.

The correct claim is:

```text
Kivo-VD dry-run observer/candidate-selection instrumentation has been validated
inside a real Linux/NVIDIA vLLM GPU inference path for one GPT-2 run, with
matching greedy output and exported routing events.
```

It is not yet a claim of KV memory reduction or production inference speedup.

## Next Steps

- Repeat dry-run validation on a modern RoPE/GQA model once runtime constraints
  are understood.
- Add real K tensor sketch capture as a dry-run-only path.
- Add real query sketch scoring as a dry-run-only path.
- Compare dry-run candidate selections against full attention behavior.
- Only after quality and runtime dry-run stability, prototype candidate-block
  attention.
