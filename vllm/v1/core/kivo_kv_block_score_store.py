# SPDX-License-Identifier: Apache-2.0

"""Process-local bounded KV block score store for Kivo-VD."""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from threading import Lock
from typing import Any, Iterable

_DEFAULT_MAX_ENTRIES = 4096
_STORE_LOCK = Lock()
_SCORES: OrderedDict[int, "KivoKVBlockScore"] = OrderedDict()
_UPDATE_COUNT = 0
_EVICTED_COUNT = 0


@dataclass(frozen=True)
class KivoKVBlockScore:
    block_id: int
    score: float
    source: str
    step: int | None = None
    layer_id: int | None = None


def _max_entries() -> int:
    value = os.getenv("KIVO_KV_SCORE_STORE_MAX_ENTRIES")
    if value is None:
        return _DEFAULT_MAX_ENTRIES
    try:
        parsed = int(value)
    except ValueError:
        return _DEFAULT_MAX_ENTRIES
    return max(1, parsed)


def update_block_scores(scores: Iterable[KivoKVBlockScore]) -> None:
    """Update the latest scalar score known for each physical block id."""
    global _UPDATE_COUNT, _EVICTED_COUNT
    with _STORE_LOCK:
        for score in scores:
            block_id = int(score.block_id)
            _SCORES.pop(block_id, None)
            _SCORES[block_id] = KivoKVBlockScore(
                block_id=block_id,
                score=float(score.score),
                source=str(score.source),
                step=score.step,
                layer_id=score.layer_id,
            )
            _UPDATE_COUNT += 1
        max_entries = _max_entries()
        while len(_SCORES) > max_entries:
            _SCORES.popitem(last=False)
            _EVICTED_COUNT += 1


def get_block_scores(block_ids: Iterable[int]) -> dict[int, float]:
    """Fetch the latest scores for the requested block ids."""
    requested = [int(block_id) for block_id in block_ids]
    with _STORE_LOCK:
        return {
            block_id: float(_SCORES[block_id].score)
            for block_id in requested
            if block_id in _SCORES
        }


def clear_block_scores() -> None:
    """Reset the process-local score store."""
    global _UPDATE_COUNT, _EVICTED_COUNT
    with _STORE_LOCK:
        _SCORES.clear()
        _UPDATE_COUNT = 0
        _EVICTED_COUNT = 0


def get_score_store_summary() -> dict[str, Any]:
    """Return a compact summary for tests and local debug."""
    with _STORE_LOCK:
        sources: dict[str, int] = {}
        for score in _SCORES.values():
            sources[score.source] = sources.get(score.source, 0) + 1
        return {
            "entry_count": len(_SCORES),
            "max_entries": _max_entries(),
            "update_count": _UPDATE_COUNT,
            "evicted_count": _EVICTED_COUNT,
            "sources": sources,
            "block_ids_sample": list(_SCORES.keys())[:16],
        }

