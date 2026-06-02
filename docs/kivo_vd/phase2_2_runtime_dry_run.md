# Kivo-VD Phase 2.2: Runtime Dry-Run Candidate Logging

Phase 2.2 wires the Phase 2.1 candidate selector into scheduler allocation
lifecycle hooks in dry-run mode only.

## What changed

- After successful KV slot allocation in the scheduler `running` path, the
  observer calls `dry_run_select_candidates(..., source="running")`.
- After successful KV slot allocation in the scheduler `waiting` path, the
  observer calls `dry_run_select_candidates(..., source="waiting")`.
- The returned `KivoVDRoutingDecision` is recorded as a compact observer event.

## Dry-run means ignored

The routing decision is computed and recorded, but it is not used by the
scheduler, block tables, slot mappings, attention metadata, or kernels. vLLM
continues to run full attention exactly as before.

## Event metadata

The observer records `dry_run_routing_decision` events with small fields:

- `request_id`
- `selected_block_count`
- `recent_block_count`
- `skipped_block_count`
- `candidate_budget_blocks`
- `recent_window_blocks`
- `source`
- small preview lists capped to the first few block ids

The observer also tracks `num_dry_run_select_calls`.

## Purpose

This validates that live scheduler/block metadata is sufficient to produce
candidate decisions before any attention behavior changes. Phase 2.3 can build
on this by exporting or summarizing decisions for offline analysis.
