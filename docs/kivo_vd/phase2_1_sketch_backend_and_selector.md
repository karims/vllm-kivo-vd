# Kivo-VD Phase 2.1: Sketch Backend + Candidate Selector (Dry-Run)

Phase 2.1 adds runtime-facing abstractions for sketching/scoring and candidate
selection policy, without changing vLLM attention behavior.

## What this phase adds

- `vllm/v1/core/kivo_vd_sketch_backend.py`
  - `KivoVDSketchBackend` interface
  - `CountSketchBackend`
  - `RandomProjectionBackend`
  - deterministic parameter creation from `input_dim`, `sketch_dim`, `seed`
  - NumPy-only sketching/scoring/ranking helpers

- `vllm/v1/core/kivo_vd_candidate_selector.py`
  - `KivoVDCandidateSelectorConfig`
  - `KivoVDCandidateSelector`
  - dry-run candidate selection using metadata-only `KivoVDSketchIndex`

- `KivoVDObserver` optional dry-run method
  - `dry_run_select_candidates(...)`
  - only runs when explicitly invoked
  - does not modify scheduler or attention behavior

## Scope and non-goals

- No scheduler behavior changes.
- No GPUModelRunner changes.
- No attention/backend/kernel changes.
- No CUDA/Triton changes.
- No torch dependency in new core modules.
- No real Q/K runtime sketching yet.

## Default policy direction (tentative)

Based on offline and HF sweeps:

- tentative default: `CountSketch`, `sketch_dim=64`
- baseline retained: `RandomProjection`, `sketch_dim=64`

These are policy defaults for dry-run experimentation only in Phase 2.1.

## Why this phase exists

This phase separates policy and backend abstractions from runtime execution so
Phase 2.2 can wire observability and tracing with low risk. Candidate sets are
computed in dry-run mode only and are not used to route real attention yet.
