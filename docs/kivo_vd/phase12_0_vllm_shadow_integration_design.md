# Phase 12.0: vLLM Shadow Integration Design

## Purpose

Phase 12 is a vLLM shadow-integration feasibility proof. It introduces a
runtime-neutral event contract, ordering and causal invariants, passive
observation points, and readiness criteria before any selected-attention path
is considered.

Shadow mode computes or records candidate block decisions but never applies
them to attention, block tables, slot mappings, scheduling, or KV allocation.
Output tokens must remain those of the normal vLLM path.

## Non-Goals

Phase 12 does not implement:

- active selected attention;
- an attention-kernel replacement;
- KV cache mutation or compaction;
- scheduler behavior changes;
- block-table or slot-mapping changes;
- production behavior changes;
- measured speed or memory improvements;
- generation-quality preservation claims.

## Why Phase 12 Exists

Phases 10 and 11 showed that selected attention can match controlled GPT-2
attention outputs, logits, and greedy continuations outside vLLM when budgets
and selectors have enough margin. They also showed that fixed tiny budgets
fail as context grows, and that budgets should be functions of layer, context
blocks, and risk policy.

vLLM integration introduces different risks:

- paged KV cache layout and physical block identity;
- block-table and request-lifecycle semantics;
- preserving causal and original sequence order;
- FlashAttention/PagedAttention backend compatibility;
- irregular gather and materialization overhead;
- per-request, per-layer, per-step selector cost;
- batching and mixed context lengths;
- prefix caching and sliding-window interactions;
- realistic memory accounting.

These risks should be measured in shadow mode before changing attention.

## Phase 12 Versus Phase 13

### Phase 12

- passive shadow collector;
- selected-block computation and export;
- ordering and causal invariant validation;
- gather/materialization simulation;
- overhead and byte accounting;
- readiness gate.

### Phase 13

- explicit opt-in experimental active path;
- one supported model/backend/layer first;
- actual selected-KV attention path;
- baseline output comparison;
- measured memory and latency;
- immediate fallback to normal attention on invariant failure.

Phase 13 is not authorized by this document.

## Core Invariants

### Score Order Is Not Gather Order

Blocks may be ranked by selector score:

```text
selected_block_ids_by_score = [31, 5, 18, 2]
```

K/V materialization must restore original sequence/block order:

```text
selected_block_ids_for_gather = [2, 5, 18, 31]
```

Attention must never receive K/V in selector-score order. Reordering keys and
values by score changes positional and causal semantics.

### Causality

- Never select a future token or block relative to the query.
- The final partially filled block requires token-level masking.
- Sliding-window and prefix-cache visibility rules remain authoritative.
- A selected block ID is valid only within the observed request, layer, block
  table, and decode step.

### Identity Preservation

Every decision must preserve:

- request identity;
- sequence identity when available;
- layer identity;
- decode-step identity;
- block-table identity or version when available;
- physical block ID;
- original logical block position.

Physical block IDs alone are insufficient because blocks can be freed and
reused.

### Shadow Isolation

Every event must state:

```text
shadow_only = true
active_routing = false
measured_runtime_reduction = false
```

Shadow events cannot alter model output, cache allocation, scheduling, block
tables, attention metadata, or kernel arguments.

## Proposed Shadow Event Data Model

The example contract is stored in
[`phase12_shadow_event_schema.json`](phase12_shadow_event_schema.json).

Important fields include:

- `event_type` and `version`;
- `request_id` and optional `sequence_id`;
- `layer_idx` and `step_idx`;
- `context_token_count`;
- `block_size` and `total_context_blocks`;
- `ratio_policy`;
- `candidate_budget_blocks`;
- `selected_block_ids_by_score`;
- `selected_block_ids_for_gather`;
- `selected_block_count` and `selected_ratio`;
- `estimated_active_block_reduction_ratio`;
- `selector_policy` and compact `selector_scores_summary`;
- `ordering_valid` and `causal_valid`;
- `preview_only`;
- `shadow_only`;
- `active_routing`;
- `measured_runtime_reduction`;
- explicit caveats.

Large tensors, K/V values, full attention matrices, and unbounded score arrays
must not be serialized.

## Ratio Policy In Shadow Mode

Budget selection is:

```text
budget = f(layer_idx, context_blocks, risk_policy)
```

Example GPT-2 research policies are:

| policy | sensitive layers 0/11 | middle layers 5/8 |
| --- | --- | --- |
| aggressive | `0.50` | `0.40` |
| balanced | `0.60` | `0.45` |
| safer | `0.70` | `0.55` |

The derived budget is rounded, clamped to configured minimum and maximum
budgets, and never exceeds available visible blocks.

These ratios are offline GPT-2 observations, not production defaults. A
runtime policy must generalize layer sensitivity rather than assume layers
0, 5, 8, and 11 exist for every model.

## Conceptual Hook Points

No hooks are implemented in Phase 12.0. Candidate observation points are:

### Scheduler And KV Lifecycle Observer

The existing Kivo observer can retain:

- request lifecycle;
- allocation/free events;
- request IDs;
- metadata-only physical block IDs;
- context-token and block-count estimates.

It must not change scheduler decisions.

### Block Table Observation

After the request block table is known, a passive observer can capture:

- logical-to-physical block mapping;
- visible block count;
- block-table identity/version;
- original logical positions.

The observer must copy only bounded metadata and must not mutate the table.

### Model Runner Decode Observation

Query-dependent scoring requires a later hook where:

- the current layer and decode step are known;
- Q is available;
- key sketches or block summaries are available;
- the current block table and causal visibility are known.

The shadow collector should compute a decision alongside normal attention,
export it, and discard it. The normal backend receives its unchanged metadata
and full visible KV path.

### Before Attention Invocation

This is the last conceptual point where selected IDs can be compared with
normal attention metadata. Phase 12 can validate IDs and simulate gather bytes
there, but must not modify backend inputs.

## Metrics To Export

- event count by request, layer, and step;
- selected-block ratio by layer;
- theoretical active-block reduction;
- candidate budget and visible blocks;
- selector compute time when measured later;
- simulated gather/materialization bytes;
- full-KV versus selected-KV bytes;
- ordering validity;
- causal validity;
- duplicate, missing, or out-of-range IDs;
- preview-only versus complete-ID exports;
- warnings and fallback reasons.

## Failure Modes

- selected ratio is too low for quality;
- no runtime oracle upper bound exists;
- query-key scoring is too expensive;
- only preview IDs are exported;
- score-order IDs are used as gather order;
- duplicate or out-of-range block IDs;
- future-block or future-token leakage;
- physical block reuse invalidates stale identity;
- block-table layout differs across attention backends;
- simulated gather overhead exceeds theoretical savings;
- sparse selected blocks cannot be consumed efficiently by the active kernel;
- batching creates incompatible per-request block shapes;
- prefix-cache or sliding-window visibility is violated.

Every invariant failure must keep the normal attention path unchanged.

## Phase 12 Subphases

### Phase 12.0

Design, event contract, example fixture, and standalone validator.

### Phase 12.1

Formalize the shadow event schema and add standalone trace aggregation,
ordering validation, causal validation, and compatibility/version checks.

### Phase 12.2

Extend the existing passive vLLM observer to export bounded block-table and
request metadata. No query tensors and no attention changes.

### Phase 12.3

Compute selected-block decisions in shadow mode from runtime metadata and
real sketches where feasible. Decisions remain ignored.

### Phase 12.4

Simulate selected-KV gather/materialization and account for selector,
materialization, metadata, and synchronization overhead.

### Phase 12.5

Apply a conservative readiness gate. Require complete IDs, valid ordering and
causality, bounded overhead, clean baseline-output comparison, and explicit
fallback behavior before proposing Phase 13.

## Phase 12.4 Runtime Touchpoint Status

Phase 12.4 added a standalone runtime-facing helper:

```text
scripts/kivo_vd/phase12_vllm_runtime_touchpoint.py
```

The chosen strategy is deliberately conservative:

- no core vLLM runtime file is modified;
- no automatic scheduler, model-runner, block-table, or attention hook runs;
- helpers are disabled unless `KIVO_PHASE12_SHADOW_ENABLED` is explicitly set;
- disabled helpers return a no-op result and write no event file;
- enabled helpers copy caller-provided metadata, delegate to the Phase 12
  passive observer, and emit validator-compatible shadow events;
- all failures return fail-closed result dictionaries.

This preserves Phase 12 as shadow-only scaffolding. The active selected-KV
path remains Phase 13 because applying selected IDs would require changing
attention metadata, block-table/slot-mapping semantics, or backend inputs.

## Validator

Validate the example events with:

```bash
.venv/bin/python scripts/kivo_vd/validate_phase12_shadow_event.py \
  --input docs/kivo_vd/examples/phase12_shadow_event_example.jsonl \
  --output-json outputs/kivo_vd/phase12_shadow_event_validation.json \
  --output-md outputs/kivo_vd/phase12_shadow_event_validation.md
```

The validator fails events that:

- omit required fields;
- are not shadow-only;
- claim active routing or measured runtime reduction;
- contain duplicate, unsorted, mismatched, or out-of-range selected IDs;
- exceed total blocks with candidate budget or selected count;
- fail ordering or causal validity flags.

A selected-ratio mismatch is reported as a warning because it is an accounting
consistency issue, not evidence that an event mutated runtime behavior.

## Phase 12.0 Decision

The shadow architecture is ready for schema hardening and passive observation
work only. Active selected attention remains deferred to Phase 13 and requires
a separate explicit authorization after the Phase 12.5 gate.
