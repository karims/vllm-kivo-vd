# Kivo-VD Phase 2.3: Dry-Run Event Export

Phase 2.3 adds optional JSONL export for Kivo-VD lifecycle and dry-run routing
events. Export is disabled by default and does not affect scheduling or
attention.

## Configuration

Kivo-VD adds two optional fields near `enable_kivo_vd`:

- `kivo_vd_event_export_path: str | None = None`
- `kivo_vd_export_event_limit: int = 10000`

When no export path is configured, `export_events()` returns `0` and writes
nothing.

## Explicit export only

This phase intentionally does not write on every observer event. Runtime callers
must explicitly invoke:

```python
observer.export_events()
```

or provide an override path:

```python
observer.export_events(path="outputs/kivo_vd/runtime_events.jsonl", limit=1000)
```

The method writes compact JSONL rows and returns the number of events written.

## Safety

Exported events come from the existing bounded observer event buffer. Tensor-like
or arbitrary objects are sanitized before JSON serialization, and large block
tables are not recorded by the Kivo-VD events.

## Purpose

The export path is for inspection and debugging only. Phase 2.3 still ignores
candidate decisions and does not change vLLM outputs. A future runtime/debug CLI
can call `export_events(...)` after generation to collect dry-run decisions for
offline analysis.
