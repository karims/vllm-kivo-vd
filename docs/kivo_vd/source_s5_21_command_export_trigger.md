# Phase S5.21: Command Export Trigger

S5.20 proved that:

- cross-process counter export works
- the runtime pre-slot-mapping hook fires
- block-table apply succeeds

But the pod result still showed:

- `demotion_command_export_attempted = 0`
- `worker_envelopes_built = 0`

So the blocker was between filtered-row apply success and command-export entry.

## What S5.21 changes

S5.21 adds precise trigger counters around the command-export path and decouples
command-export planning from the separate live-ownership-apply gate.

That means:

- successful filtered-row apply can now enter the command-export path even when
  live ownership mutation remains disabled
- actual behavior is still fail-closed
- no ownership removal or free-to-pool is enabled

## New counters

- `demotion_command_export_path_entered`
- `demotion_command_export_skipped_no_apply_summary`
- `demotion_command_export_skipped_apply_not_successful`
- `demotion_command_export_skipped_no_request_id`
- `demotion_command_export_skipped_no_visible_before`
- `demotion_command_export_skipped_no_visible_after`
- `demotion_command_export_skipped_no_candidate_demote_ids`
- `demotion_command_export_skipped_empty_after_filter`
- `demotion_command_export_succeeded`

Additional compact debug fields in the S5.19 probe summary:

- block-table apply counts
- command-export path-entered / attempted / succeeded
- command-export blocker reasons
- last observed:
  - `visible_before_count`
  - `visible_after_count`
  - `candidate_demote_count`
  - `filtered_row_changed`
  - `keep_recent_blocks`
  - `policy`

## What this phase proves

S5.21 is only about trigger diagnostics.

It can prove:

- whether the command-export path is entered
- whether a command build is attempted
- whether a command is built successfully
- if not, the exact blocker category

It still does **not** prove:

- scheduler/core transport success
- memory reduction
- free-to-pool
- `req_to_blocks` removal
- quality preservation

## Next step

- if command build is observed:
  `S5.22_scheduler_core_transport_observation`
- if not:
  inspect candidate-demote selection and filtered-row conditions before any
  ownership-removal design
