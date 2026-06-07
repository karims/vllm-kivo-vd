# Kivo-VD Phase 7.2: Memory Baseline Vs Event Estimate

Phase 7.2 combines measured CUDA memory artifacts from Phase 7.0 with
theoretical active-KV estimates from Phase 7.1. It keeps the two evidence types
separate and explains why dry-run inference does not yet produce measured
memory savings.

## Purpose

Phase 7.0 measures actual torch CUDA allocator checkpoints during unchanged
vLLM generation. Phase 7.1 asks how many KV tensor bytes the selected and
skipped dry-run blocks represent.

Phase 7.2 places those results in one report:

- measured model initialization, generation, and peak allocator values;
- optional baseline-versus-Kivo dry-run differences;
- theoretical selected/skipped KV accounting;
- an explicit explanation of the gap between them.

## Why Measured Memory Does Not Drop Yet

Kivo remains dry-run only. vLLM allocates its normal/full KV cache, normal
attention consumes it, and candidate decisions are ignored. Therefore, the
event estimate is counterfactual accounting rather than memory avoided by the
runtime.

If one Kivo dry-run happens to report a lower peak than one baseline run, the
comparison records that observation but does not attribute it to Kivo.
Allocator state and normal run-to-run variation must be controlled with
repeated measurements.

## Run The Comparison

```bash
.venv/bin/python scripts/kivo_vd/compare_memory_baseline_and_estimate.py \
  --baseline-memory outputs/kivo_vd/phase7_0_gpt2_baseline_memory.json \
  --kivo-memory outputs/kivo_vd/phase7_0_gpt2_kivo_dry_run_memory.json \
  --event-estimate outputs/kivo_vd/phase7_1_gpt2_event_memory_estimate.json \
  --output-json outputs/kivo_vd/phase7_2_gpt2_memory_comparison.json \
  --output-md outputs/kivo_vd/phase7_2_gpt2_memory_comparison.md
```

`--kivo-memory` is optional. Without it, the report still combines the measured
baseline with the theoretical event estimate.

## Report Contents

The report includes:

- measured initialization and generation allocated/reserved deltas;
- measured peak allocated and reserved bytes;
- cleanup checkpoint values when present;
- theoretical bytes per block;
- average selected, skipped, active, and skipped KV values;
- average theoretical reduction ratio;
- optional Kivo-minus-baseline measured differences;
- explicit caveats:
  - `estimated_only_for_savings: true`
  - `measured_runtime_reduction: false`

## Interpretation

Measured CUDA values answer what the unchanged runtime allocated. Theoretical
event values answer what selected blocks would represent under a future active
KV mechanism. They should not be subtracted from each other as though the
runtime already implemented that mechanism.

## Proven Vs Not Proven

This phase proves that measured runtime baselines and theoretical dry-run
accounting can be joined into a reproducible, auditable report.

It does not prove:

- measured vLLM KV memory reduction;
- active KV residency or candidate-block attention;
- latency improvement;
- quality preservation.

## Next Steps

Repeat Phase 7.0 baseline and Kivo dry-run measurements under identical
conditions, analyze event-estimate trends over longer requests, and keep active
routing deferred until memory accounting and quality baselines are stable.
