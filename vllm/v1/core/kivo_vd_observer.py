# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import deque
from collections.abc import Sequence
from typing import Any

from vllm.v1.core.kivo_vd_sketch import KivoVDSketchIndex


class KivoVDObserver:
    """Phase 0 Kivo-VD observer hook points (no-op implementation)."""

    def __init__(
        self,
        enabled: bool = False,
        max_events: int = 10_000,
        sketch_index: KivoVDSketchIndex | None = None,
    ) -> None:
        self.enabled = enabled
        self.max_events = max_events
        self.sketch_index = sketch_index
        self.num_before_allocate_calls = 0
        self.num_after_allocate_calls = 0
        self.num_free_request_calls = 0
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
        return

    def on_build_attention_metadata(
        self,
        num_reqs: int,
        num_tokens: int,
        block_table_shape: Sequence[int],
    ) -> None:
        return

    def get_counters(self) -> dict[str, int]:
        return {
            "num_before_allocate_calls": self.num_before_allocate_calls,
            "num_after_allocate_calls": self.num_after_allocate_calls,
            "num_free_request_calls": self.num_free_request_calls,
            "num_events": len(self._events),
        }

    def get_recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        return list(self._events)[-limit:]

    def reset(self) -> None:
        self.num_before_allocate_calls = 0
        self.num_after_allocate_calls = 0
        self.num_free_request_calls = 0
        self._event_counter = 0
        self._events.clear()


def create_kivo_vd_observer(enable_kivo_vd: bool) -> KivoVDObserver | None:
    if not enable_kivo_vd:
        return None
    return KivoVDObserver(enabled=True)
