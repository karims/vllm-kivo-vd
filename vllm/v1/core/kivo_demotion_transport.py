# SPDX-License-Identifier: Apache-2.0

"""Fail-closed Kivo demotion transport helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

from vllm.v1.core.kivo_demotion_command import (
    KivoCoreDemotionConfig,
    KivoDemotionCommand,
    KivoDemotionCommandResult,
    current_kivo_core_demotion_config,
)

_DEFAULT_ACTION = "off"


@dataclass(frozen=True)
class KivoDemotionTransportConfig:
    enabled: bool
    action: str


@dataclass(frozen=True)
class KivoDemotionTransportEnvelope:
    request_id: str
    command: KivoDemotionCommand
    source: str
    step: int | None = None


@dataclass(frozen=True)
class KivoDemotionTransportResult:
    enabled: bool
    accepted_count: int
    applied_count: int
    rejected_count: int
    blocker_reasons: dict[str, int]


def _parse_bool_env(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip() == "1"


def current_kivo_demotion_transport_config() -> KivoDemotionTransportConfig:
    enabled = _parse_bool_env("KIVO_KV_DEMOTION_TRANSPORT_ENABLE", default=False)
    action = os.getenv("KIVO_KV_DEMOTION_TRANSPORT_ACTION", _DEFAULT_ACTION)
    if not enabled:
        action = _DEFAULT_ACTION
    return KivoDemotionTransportConfig(enabled=enabled, action=action)


def apply_kivo_demotion_transport_envelopes(
    *,
    kv_cache_manager: object | None,
    envelopes: Sequence[KivoDemotionTransportEnvelope],
    config: KivoDemotionTransportConfig | None = None,
    core_config: KivoCoreDemotionConfig | None = None,
) -> KivoDemotionTransportResult:
    """Apply transported demotion commands through the core KV cache manager."""
    if config is None:
        config = current_kivo_demotion_transport_config()
    if core_config is None:
        core_config = current_kivo_core_demotion_config()

    if not config.enabled or config.action == "off":
        return KivoDemotionTransportResult(
            enabled=False,
            accepted_count=0,
            applied_count=0,
            rejected_count=0,
            blocker_reasons={"disabled": 1},
        )

    blocker_reasons: dict[str, int] = {}
    if config.action != "apply_core_mark_demoted":
        blocker_reasons["invalid_transport_action"] = 1
    if kv_cache_manager is None:
        blocker_reasons["kv_cache_manager_unavailable"] = 1
    elif not hasattr(kv_cache_manager, "apply_kivo_demotion_command"):
        blocker_reasons["kv_cache_manager_missing_command_api"] = 1

    if blocker_reasons:
        return KivoDemotionTransportResult(
            enabled=True,
            accepted_count=0,
            applied_count=0,
            rejected_count=len(tuple(envelopes)),
            blocker_reasons=blocker_reasons,
        )

    accepted = 0
    applied = 0
    rejected = 0
    for envelope in envelopes:
        accepted += 1
        result: KivoDemotionCommandResult = kv_cache_manager.apply_kivo_demotion_command(
            envelope.command,
            config=core_config,
        )
        if result.accepted:
            applied += 1
        else:
            rejected += 1
            for reason, count in result.blocker_reasons.items():
                blocker_reasons[reason] = blocker_reasons.get(reason, 0) + count

    return KivoDemotionTransportResult(
        enabled=True,
        accepted_count=accepted,
        applied_count=applied,
        rejected_count=rejected,
        blocker_reasons=blocker_reasons,
    )
