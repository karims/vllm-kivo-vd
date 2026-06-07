# Kivo-VD Phase 8.4: Sketch-Buffer Decision Gate

Phase 8 established compact sketch-buffer accounting around the Phase 7
theoretical active-KV opportunity. It did not change KV allocation, attention,
or routing.

This gate decides whether the evidence supports a narrowly scoped Phase 9
selected-KV materialization experiment outside the attention path.

## Required Artifacts

Before Phase 9, a completed Phase 8.3 run must contain:

- `sketch_buffer_overhead.json`
- `sketch_buffer_overhead.md`
- `sketch_overhead_vs_savings.json`
- `sketch_overhead_vs_savings.md`
- `event_aware_sketch_buffer_accounting.json`
- `event_aware_sketch_buffer_accounting.md`
- `pipeline_summary.json`

## Decision Criteria

The gate passes only when:

- the Phase 8.3 pipeline reports success and is not a dry-run plan;
- every pipeline stage succeeded;
- all required artifacts exist;
- at least one dim-16 or dim-32 configuration has excellent or acceptable
  cumulative-request overhead;
- recommended configurations and break-even accounting are available;
- all reports preserve the theoretical-only, full-KV, no-routing, and
  no-measured-reduction caveats.

Sketch overhead should also remain small relative to the modeled full KV pool.
The cumulative classification is the conservative machine-readable gate.

## Thresholds

| cumulative overhead | classification |
| --- | --- |
| 5% or less | excellent |
| above 5% through 15% | acceptable |
| above 15% through 30% | questionable |
| above 30% | poor |

| break-even events | classification |
| --- | --- |
| 1 or less | immediate |
| 2 through 4 | fast |
| 5 through 16 | moderate |
| above 16 | slow |

These are research heuristics, not runtime memory or quality guarantees.

## Run The Gate

```bash
RUN_DIR=outputs/kivo_vd/runs/phase8_gpt2_sketch_buffer_accounting
.venv/bin/python scripts/kivo_vd/check_phase8_readiness.py \
  --pipeline-summary "$RUN_DIR/pipeline_summary.json" \
  --event-aware-accounting \
    "$RUN_DIR/event_aware_sketch_buffer_accounting.json" \
  --overhead-vs-savings \
    "$RUN_DIR/sketch_overhead_vs_savings.json" \
  --sketch-overhead "$RUN_DIR/sketch_buffer_overhead.json" \
  --output-json "$RUN_DIR/phase8_readiness.json" \
  --output-md "$RUN_DIR/phase8_readiness.md"
```

## Initial Configurations

Start with:

- CountSketch dims `16` and `32`;
- Random Projection dims `16` and `32`;
- `bidiagonal_sign_subsample` dims `16` and `32` as experimental.

Do not start with SRHT. Treat dim `64` as reference-only unless later evidence
changes the tradeoff.

## RunPod Readiness Result

The gate was run against the successful
`phase8_gpt2_sketch_buffer_accounting` artifacts:

```bash
RUN_DIR=outputs/kivo_vd/runs/phase8_gpt2_sketch_buffer_accounting
.venv/bin/python scripts/kivo_vd/check_phase8_readiness.py \
  --pipeline-summary "$RUN_DIR/pipeline_summary.json" \
  --event-aware-accounting \
    "$RUN_DIR/event_aware_sketch_buffer_accounting.json" \
  --overhead-vs-savings \
    "$RUN_DIR/sketch_overhead_vs_savings.json" \
  --sketch-overhead "$RUN_DIR/sketch_buffer_overhead.json" \
  --output-json "$RUN_DIR/phase8_readiness.json" \
  --output-md "$RUN_DIR/phase8_readiness.md"
```

Result:

| field | value |
| --- | --- |
| Phase 9 ready | `true` |
| best eligible sketch | `bidiagonal_sign_subsample` |
| sketch dim | `16` |
| cumulative overhead ratio | `0.2506%` |
| cumulative classification | `excellent` |
| break-even events | `1` |
| break-even classification | `immediate` |
| break-even skipped blocks | `2` |
| warnings | none |
| theoretical only | `true` |
| measured runtime reduction | `false` |
| active routing | `false` |
| full KV still allocated | `true` |

The broader accounting context was:

| dim | cumulative overhead | Phase 7 theoretical reduction |
| ---: | ---: | ---: |
| `16` | `0.2506%` | `60.9045%` |
| `32` | `0.5013%` | `60.9045%` |
| `64` | `1.0025%` | `60.9045%` |

The gate selected `bidiagonal_sign_subsample` dim `16` as one eligible row.
That is not a sketch-quality conclusion. At a fixed dimension, all three
tested sketch families use the same Phase 8 buffer shape and byte count.
Phase 9 should begin with CountSketch or Random Projection as simple baselines
before testing `bidiagonal_sign_subsample`.

Phase 8 is complete. `phase9_ready: true` authorizes only temporary
selected-KV gather/copy and overhead measurement outside attention.

## Allowed Phase 9 Scope

A passing gate authorizes only:

- gathering or copying selected KV blocks into temporary buffers;
- measuring temporary buffer payload, allocator overhead, and copy cost;
- comparing selected-buffer measurements with existing accounting.

It does not authorize:

- attention kernel changes;
- block-table or slot-mapping mutation;
- scheduler behavior changes;
- active routing or candidate-block attention.

## Current Claim

Kivo-VD has runtime dry-run instrumentation, theoretical active-KV accounting,
and compact sketch-buffer overhead accounting. It has not yet demonstrated
measured runtime KV memory reduction or active attention routing.

Phase 9 must preserve that boundary until temporary selected-KV
materialization has been measured independently of attention.

The Phase 8 buffer result is not itself a memory-saving result. The sketch
buffers are additional overhead, full KV remains allocated, and attention
continues to consume the full KV cache.
