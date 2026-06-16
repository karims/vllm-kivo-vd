# SPDX-License-Identifier: Apache-2.0

"""Lightweight gated counters for Kivo demotion transport observation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import asdict, dataclass, field


@dataclass
class KivoDemotionCounters:
    block_table_apply_attempted: int = 0
    block_table_apply_succeeded: int = 0
    block_table_apply_rejected: int = 0
    demotion_command_export_attempted: int = 0
    demotion_command_export_rejected: int = 0
    worker_envelopes_built: int = 0
    worker_envelopes_attached: int = 0
    scheduler_envelopes_received: int = 0
    core_transport_batches_received: int = 0
    core_commands_attempted: int = 0
    core_commands_accepted: int = 0
    core_commands_rejected: int = 0
    manager_mark_demoted_attempted: int = 0
    manager_mark_demoted_succeeded: int = 0
    manager_mark_demoted_rejected: int = 0
    demoted_blocks_marked: int = 0
    req_to_blocks_removed: int = 0
    free_to_pool_calls: int = 0
    blocker_reasons: dict[str, int] = field(default_factory=dict)


_COUNTERS = KivoDemotionCounters()


def _parse_bool_env(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip() == "1"


def kivo_demotion_counters_enabled() -> bool:
    return _parse_bool_env("KIVO_KV_DEMOTION_COUNTERS_ENABLE", default=False)


def get_kivo_demotion_counters_export_file() -> str | None:
    path = os.getenv("KIVO_KV_DEMOTION_COUNTERS_EXPORT_FILE")
    if path is None:
        return None
    path = path.strip()
    return path or None


def reset_kivo_demotion_counters() -> None:
    global _COUNTERS
    _COUNTERS = KivoDemotionCounters()


def get_kivo_demotion_counters_snapshot() -> dict[str, object]:
    return asdict(_COUNTERS)


def increment_kivo_demotion_counter(name: str, amount: int = 1) -> None:
    if not kivo_demotion_counters_enabled() or amount <= 0:
        return
    current = getattr(_COUNTERS, name, None)
    if not isinstance(current, int):
        return
    setattr(_COUNTERS, name, current + amount)


def add_kivo_demotion_blocker_reasons(blocker_reasons: dict[str, int]) -> None:
    if not kivo_demotion_counters_enabled():
        return
    for reason, count in blocker_reasons.items():
        if count <= 0:
            continue
        _COUNTERS.blocker_reasons[reason] = (
            _COUNTERS.blocker_reasons.get(reason, 0) + count
        )


def export_kivo_demotion_counters_snapshot_if_enabled(
    *,
    source: str = "unknown",
) -> None:
    if not kivo_demotion_counters_enabled():
        return
    export_file = get_kivo_demotion_counters_export_file()
    if export_file is None:
        return
    target = Path(export_file)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "source": source,
        "counters": get_kivo_demotion_counters_snapshot(),
    }
    tmp_path = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(target)
