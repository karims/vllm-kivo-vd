# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vllm.v1.core.kivo_vd_sketch import (
    KivoVDRoutingDecision,
    KivoVDBlockSketch,
    KivoVDSketchConfig,
    KivoVDSketchIndex,
)
from vllm.v1.core.kivo_vd_candidate_selector import (
    KivoVDCandidateSelector,
    KivoVDCandidateSelectorConfig,
)


def _env_flag_enabled(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


class KivoVDObserver:
    """Phase 0 Kivo-VD observer hook points (no-op implementation)."""

    def __init__(
        self,
        enabled: bool = False,
        max_events: int = 10_000,
        sketch_index: KivoVDSketchIndex | None = None,
        candidate_selector: KivoVDCandidateSelector | None = None,
        event_export_path: str | None = None,
        export_event_limit: int = 10_000,
        export_full_block_ids: bool | None = None,
    ) -> None:
        self.enabled = enabled
        self.max_events = max_events
        self.sketch_index = sketch_index
        self.candidate_selector = candidate_selector
        self.event_export_path = event_export_path
        self.export_event_limit = export_event_limit
        self.export_full_block_ids = (
            _env_flag_enabled("KIVO_EXPORT_FULL_BLOCK_IDS")
            if export_full_block_ids is None
            else export_full_block_ids
        )
        self.num_before_allocate_calls = 0
        self.num_after_allocate_calls = 0
        self.num_free_request_calls = 0
        self.num_dry_run_select_calls = 0
        self._event_counter = 0
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)

    def _record_event(self, event_type: str, **kwargs: Any) -> None:
        self._event_counter += 1
        event: dict[str, Any] = {
            "event_type": event_type,
            "event_id": self._event_counter,
            "ts_monotonic": time.monotonic(),
        }
        for key, value in kwargs.items():
            if value is not None:
                event[key] = value
        self._events.append(event)

    def on_before_allocate_slots(
        self,
        request_id: str,
        num_new_tokens: int,
        num_lookahead_tokens: int,
        source: str | None = None,
        num_computed_blocks: int | None = None,
    ) -> None:
        self.num_before_allocate_calls += 1
        self._record_event(
            "before_allocate_slots",
            request_id=request_id,
            num_new_tokens=num_new_tokens,
            num_tokens=num_new_tokens,
            num_computed_blocks=num_computed_blocks,
            source=source,
        )
        return

    def on_after_allocate_slots(
        self,
        request_id: str,
        block_ids_by_group: tuple[list[int], ...] | None,
        num_new_tokens: int | None = None,
        source: str | None = None,
        num_computed_blocks: int | None = None,
    ) -> None:
        self.num_after_allocate_calls += 1
        num_new_blocks = None
        if block_ids_by_group is not None:
            num_new_blocks = sum(len(group) for group in block_ids_by_group)
        self._record_event(
            "after_allocate_slots",
            request_id=request_id,
            num_new_tokens=num_new_tokens,
            num_tokens=num_new_tokens,
            num_computed_blocks=num_computed_blocks,
            num_new_blocks=num_new_blocks,
            source=source,
        )
        if self.sketch_index is not None and block_ids_by_group is not None:
            for kv_group_id, group_block_ids in enumerate(block_ids_by_group):
                for logical_block_idx, block_id in enumerate(group_block_ids):
                    self.sketch_index.add_or_update_block_sketch(
                        KivoVDBlockSketch(
                            request_id=request_id,
                            block_id=block_id,
                            logical_block_idx=logical_block_idx,
                            kv_group_id=kv_group_id,
                            layer_id=None,
                            sketch_dim=self.sketch_index.config.sketch_dim,
                            metadata={
                                "source": source,
                                "num_new_tokens": num_new_tokens,
                            },
                        )
                    )
        return

    def on_free_request(
        self,
        request_id: str,
        block_ids_by_group: tuple[list[int], ...],
        source: str | None = None,
        num_new_tokens: int | None = None,
        num_computed_blocks: int | None = None,
    ) -> None:
        self.num_free_request_calls += 1
        if num_computed_blocks is None:
            num_computed_blocks = sum(len(group) for group in block_ids_by_group)
        self._record_event(
            "free_request",
            request_id=request_id,
            num_new_tokens=num_new_tokens,
            num_tokens=num_new_tokens,
            num_computed_blocks=num_computed_blocks,
            source=source,
        )
        if self.sketch_index is not None:
            self.sketch_index.remove_request(request_id)
        return

    def on_build_attention_metadata(
        self,
        num_reqs: int,
        num_tokens: int,
        block_table_shape: Sequence[int],
    ) -> None:
        return

    def dry_run_select_candidates(
        self,
        request_id: str,
        query_metadata_or_sketch: Any | None = None,
        source: str | None = None,
    ) -> KivoVDRoutingDecision | None:
        self.num_dry_run_select_calls += 1
        if self.sketch_index is None or self.candidate_selector is None:
            return None
        decision = self.candidate_selector.select_candidates(
            request_id=request_id,
            query_metadata_or_sketch=query_metadata_or_sketch,
            sketch_index=self.sketch_index,
        )
        selector_config = self.candidate_selector.config
        event_metadata: dict[str, Any] = {
            "full_block_ids_exported": self.export_full_block_ids,
        }
        if self.export_full_block_ids:
            event_metadata.update({
                "selected_block_ids_full": decision.selected_block_ids,
                "recent_block_ids_full": decision.recent_block_ids,
                "skipped_block_ids_full": decision.skipped_block_ids,
            })
        self._record_event(
            "dry_run_routing_decision",
            request_id=request_id,
            selected_block_count=len(decision.selected_block_ids),
            recent_block_count=len(decision.recent_block_ids),
            skipped_block_count=len(decision.skipped_block_ids),
            candidate_budget_blocks=selector_config.candidate_budget_blocks,
            recent_window_blocks=selector_config.recent_window_blocks,
            selected_block_preview=decision.selected_block_ids[:8],
            recent_block_preview=decision.recent_block_ids[:8],
            skipped_block_preview=decision.skipped_block_ids[:8],
            source=source,
            reason=decision.reason,
            **event_metadata,
        )
        return decision

    def get_counters(self) -> dict[str, int]:
        return {
            "num_before_allocate_calls": self.num_before_allocate_calls,
            "num_after_allocate_calls": self.num_after_allocate_calls,
            "num_free_request_calls": self.num_free_request_calls,
            "num_dry_run_select_calls": self.num_dry_run_select_calls,
            "num_events": len(self._events),
        }

    def get_recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        return list(self._events)[-limit:]

    def _json_safe(self, value: Any) -> Any:
        if value is None or isinstance(value, str | int | float | bool):
            return value
        if isinstance(value, list | tuple):
            return [self._json_safe(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        # Avoid serializing tensor-like or arbitrary large objects.
        if hasattr(value, "shape") or hasattr(value, "dtype"):
            return {"type": type(value).__name__}
        return str(value)

    def export_events(self, path: str | None = None, limit: int | None = None) -> int:
        export_path = path or self.event_export_path
        if export_path is None:
            return 0

        effective_limit = self.export_event_limit if limit is None else limit
        if effective_limit <= 0:
            events: list[dict[str, Any]] = []
        else:
            events = list(self._events)[-effective_limit:]

        output_path = Path(export_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for event in events:
                row = self._json_safe(event)
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
        return len(events)

    def reset(self) -> None:
        self.num_before_allocate_calls = 0
        self.num_after_allocate_calls = 0
        self.num_free_request_calls = 0
        self.num_dry_run_select_calls = 0
        self._event_counter = 0
        self._events.clear()


def create_kivo_vd_observer(
    enable_kivo_vd: bool,
    event_export_path: str | None = None,
    export_event_limit: int = 10_000,
) -> KivoVDObserver | None:
    if not enable_kivo_vd:
        return None
    sketch_index = KivoVDSketchIndex(
        config=KivoVDSketchConfig(
            enabled=True,
        )
    )
    return KivoVDObserver(
        enabled=True,
        sketch_index=sketch_index,
        candidate_selector=KivoVDCandidateSelector(
            KivoVDCandidateSelectorConfig()
        ),
        event_export_path=event_export_path,
        export_event_limit=export_event_limit,
    )
