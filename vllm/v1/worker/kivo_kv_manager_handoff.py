# SPDX-License-Identifier: Apache-2.0

"""Fail-closed KV manager handoff checks for Kivo-VD."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KivoKVManagerHandoffDecision:
    manager_available: bool
    manager_type: str | None
    supports_kivo_demotion_mark: bool
    request_id_compatible: bool
    safe_to_call_mark_demoted: bool
    blocker_reasons: dict[str, int]


def build_kivo_kv_manager_handoff_decision(
    *,
    kv_cache_manager: object | None,
    request_id: str | None,
) -> KivoKVManagerHandoffDecision:
    """Validate whether a provided manager object is safe for demotion marking."""
    if kv_cache_manager is None:
        return KivoKVManagerHandoffDecision(
            manager_available=False,
            manager_type=None,
            supports_kivo_demotion_mark=False,
            request_id_compatible=False,
            safe_to_call_mark_demoted=False,
            blocker_reasons={"kv_cache_manager_unavailable": 1},
        )

    manager_type = type(kv_cache_manager).__name__
    blocker_reasons: dict[str, int] = {}

    supports_mark = hasattr(kv_cache_manager, "mark_kivo_demoted_blocks_if_safe")
    if not supports_mark:
        blocker_reasons["kv_cache_manager_missing_mark_helper"] = 1

    request_id_compatible = request_id is not None
    if request_id is None:
        blocker_reasons["missing_request_id"] = 1
    elif hasattr(kv_cache_manager, "req_to_blocks"):
        try:
            request_id_compatible = request_id in kv_cache_manager.req_to_blocks
        except Exception:
            request_id_compatible = False
        if not request_id_compatible:
            blocker_reasons["request_id_not_present_in_manager"] = 1

    safe_to_call = supports_mark and request_id_compatible and not blocker_reasons

    return KivoKVManagerHandoffDecision(
        manager_available=True,
        manager_type=manager_type,
        supports_kivo_demotion_mark=supports_mark,
        request_id_compatible=request_id_compatible,
        safe_to_call_mark_demoted=safe_to_call,
        blocker_reasons=blocker_reasons,
    )
