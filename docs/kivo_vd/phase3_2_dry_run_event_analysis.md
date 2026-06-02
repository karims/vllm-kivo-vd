# Kivo-VD Phase 3.2: Dry-Run Event Analysis

Phase 3.2 adds an offline analyzer for Kivo-VD runtime dry-run event exports.

The analyzer does not run inference and does not change vLLM runtime behavior.
It reads JSONL files produced by `KivoVDObserver.export_events(...)` and
summarizes whether dry-run routing events look structurally healthy.

## Script

```bash
.venv/bin/python scripts/kivo_vd/analyze_dry_run_events.py \
  --input outputs/kivo_vd/vllm_kivo_dry_run_events.jsonl
```

Defaults:

- Input: `outputs/kivo_vd/vllm_kivo_dry_run_events.jsonl`
- Output: `outputs/kivo_vd/vllm_kivo_dry_run_summary.json`

## What It Summarizes

- Total events.
- Event counts by `event_type`.
- Number of `dry_run_routing_decision` events.
- Average selected, recent, and skipped block counts.
- Candidate budget and recent-window values.
- Preview fields when present.
- Request IDs seen.
- Sources seen, such as `running` or `waiting`.
- Warnings for missing or malformed data.

Expected event types include:

- `after_allocate_slots`
- `dry_run_routing_decision`
- `free_request`

## Success Criteria

For Linux/NVIDIA runtime dry-run validation, a healthy event file should show:

- at least one `dry_run_routing_decision` event;
- nonzero selected block counts once enough blocks exist;
- request IDs that correspond to real vLLM requests;
- sources such as `running` or `waiting`;
- no implication that attention behavior changed.

The analyzer only validates exported metadata. It does not prove memory
reduction, latency improvement, quality preservation, or candidate-block
attention behavior.

## Warnings

The analyzer warns when:

- the input event file is missing;
- malformed JSONL rows are present;
- no `dry_run_routing_decision` events exist;
- only allocation/free events are present;
- selected block count is always zero.

Warnings are included in the output JSON and do not prevent analysis of valid
rows.

## Output

The script prints compact JSON to stdout and writes a pretty JSON summary to
`--output`.
