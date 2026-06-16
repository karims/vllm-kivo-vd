# SPDX-License-Identifier: Apache-2.0

"""Core-owned Kivo demotion command helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass

_DEFAULT_ACTION = "off"


@dataclass(frozen=True)
class KivoCoreDemotionConfig:
    enabled: bool
    action: str
    require_block_table_applied: bool
    require_slot_mapping_refresh: bool


@dataclass(frozen=True)
class KivoDemotionCommand:
    request_id: str
    visible_before_block_ids: tuple[int, ...]
    visible_after_block_ids: tuple[int, ...]
    candidate_demote_block_ids: tuple[int, ...]
    protected_block_ids: tuple[int, ...]
    block_table_applied: bool
    slot_mapping_refresh_guaranteed: bool
    source: str = "worker_pre_slot_mapping"


@dataclass(frozen=True)
class KivoDemotionCommandResult:
    enabled: bool
    request_id: str | None
    accepted: bool
    marked_demoted_block_ids: tuple[int, ...]
    rejected_block_ids: tuple[int, ...]
    blocker_reasons: dict[str, int]
    removes_from_req_to_blocks: bool
    frees_to_pool: bool


def _parse_bool_env(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip() == "1"


def current_kivo_core_demotion_config() -> KivoCoreDemotionConfig:
    enabled = _parse_bool_env("KIVO_KV_CORE_DEMOTION_ENABLE", default=False)
    action = os.getenv("KIVO_KV_CORE_DEMOTION_ACTION", _DEFAULT_ACTION)
    if not enabled:
        action = _DEFAULT_ACTION
    return KivoCoreDemotionConfig(
        enabled=enabled,
        action=action,
        require_block_table_applied=_parse_bool_env(
            "KIVO_KV_CORE_DEMOTION_REQUIRE_BLOCK_TABLE_APPLIED",
            default=True,
        ),
        require_slot_mapping_refresh=_parse_bool_env(
            "KIVO_KV_CORE_DEMOTION_REQUIRE_SLOT_MAPPING_REFRESH",
            default=True,
        ),
    )
