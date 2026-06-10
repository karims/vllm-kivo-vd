# Phase S1: Source-Level Selected-Block State Mutation

## Why This Phase Exists

Phase 12.8/12.9 and Phase 12.10 showed that installed-wheel patching can reach
the runtime, but the reachable objects were still too shallow or too
tensor-bound for a useful selected-block mutation experiment.

Phase S1 moves the experiment into repo-local vLLM source so we can inspect and
optionally mutate the live `BlockTable.compute_slot_mapping` state in the same
process that owns the real source build.

## What S1 Mutates

- `self.slot_mapping` only.
- Only when `KIVO_SOURCE_ENABLE=1`, `KIVO_SOURCE_ACTIVE=1`, and
  `KIVO_SOURCE_POLICY=mask_last_slot`.
- Only when the slot mapping is tensor-like, integer-typed, and has at least
  two elements.
- Only the last element, by repeating the previous value.

## What S1 Does Not Mutate

- KV cache tensors.
- Attention kernels.
- Scheduler state.
- `self.block_table`.
- Repository-local behavior when `KIVO_SOURCE_ENABLE` is unset.

## Phase S1.1: Valid Non-Padding Slot Mutation

Phase S1 proved that the source-level hook can observe and mutate slot-mapping
state, but the first active policy hit the padded tail of the tensor. Phase
S1.1 keeps the same source hook and changes only the active policy so it
targets the last valid non-padding slot instead.

- New default policy: `mask_last_valid_slot`
- Backward-compatible policy: `mask_last_slot`
- Mutation now requires at least two valid slot entries and a differing
  previous valid value
- The experiment still makes no memory, latency, or production-selected
  attention claim

## Run Commands

S1 requires a source-built vLLM environment. Do not expect it to work from the
installed-wheel-only path.

```bash
python -m pytest tests_kivo -q

python -m py_compile \
  scripts/kivo_vd/run_source_s1_gpt2_probe.py \
  scripts/kivo_vd/validate_source_s1_observations.py

git diff --check
```

The probe is intended for a source-built GPT-2 smoke run after vLLM is built
locally from source.

## Success And Failure

If the active mutation is applied and generation still succeeds, the experiment
has identified a source-level mutable slot-mapping path worth the next review
step.

If the active mutation is blocked, the blocker tells us whether the next target
should be tensor-safe mutation or a different source hook.

Any crash is useful if its traceback is captured. No memory, latency, or
quality claim is made here.
