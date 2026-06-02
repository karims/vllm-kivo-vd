# Kivo-VD Phase 2.4: Debug Dry-Run Export Utility

Phase 2.4 adds a small standalone debug utility for validating Kivo-VD event
export format without running real vLLM inference.

## Script

```bash
.venv/bin/python scripts/kivo_vd/debug_export_dry_run_events.py
```

Default output:

```text
outputs/kivo_vd/debug_dry_run_events.jsonl
```

The script creates a synthetic request, adds metadata-only block sketches,
records an allocation event, records a dry-run routing decision, records a free
event, exports JSONL, and prints a compact JSON summary.

## Options

- `--output`
- `--num-blocks`
- `--candidate-budget-blocks`
- `--recent-window-blocks`
- `--request-id`
- `--sketch-type`
- `--sketch-dim`

## Scope

This utility does not start vLLM runtime, does not run model inference, and does
not touch attention behavior. It imports only Kivo-VD core modules and uses
synthetic metadata.

## Purpose

The goal is to validate observer event export and dry-run routing plumbing on
local CPU-only environments. Real runtime export will be validated later on an
environment that can run vLLM inference.
