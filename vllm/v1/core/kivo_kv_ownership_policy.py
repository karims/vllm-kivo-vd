# SPDX-License-Identifier: Apache-2.0

"""Pure KV ownership policy decisions for Kivo-VD.

This module is intentionally small and side-effect free. It parses Kivo
ownership env flags and turns a list of candidate skipped block IDs into a
deterministic allow/protect decision without mutating any block objects.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

_DEFAULT_POLICY = "off"
_DEFAULT_KEEP_RECENT_BLOCKS = 1
_DEFAULT_RECORD = "summary"


@dataclass(frozen=True)
class KivoKVOwnershipConfig:
    enabled: bool
    policy: str
    keep_recent_blocks: int
    record: str


@dataclass(frozen=True)
class KivoKVOwnershipDecision:
    enabled: bool
    policy: str
    candidate_block_ids: tuple[int, ...]
    allowed_block_ids: tuple[int, ...]
    protected_block_ids: tuple[int, ...]
    reason_counts: dict[str, int]
    candidate_count: int
    allowed_count: int
    protected_count: int


def _parse_bool_env(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip() == "1"


def _parse_int_env(name: str, *, default: int, minimum: int = 0) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(minimum, parsed)


def current_kv_ownership_config() -> KivoKVOwnershipConfig:
    """Build the current ownership config from Kivo env vars."""
    return KivoKVOwnershipConfig(
        enabled=_parse_bool_env("KIVO_KV_OWNERSHIP_ENABLE", default=False),
        policy=os.getenv("KIVO_KV_OWNERSHIP_POLICY", _DEFAULT_POLICY),
        keep_recent_blocks=_parse_int_env(
            "KIVO_KV_OWNERSHIP_KEEP_RECENT_BLOCKS",
            default=_DEFAULT_KEEP_RECENT_BLOCKS,
            minimum=0,
        ),
        record=os.getenv("KIVO_KV_OWNERSHIP_RECORD", _DEFAULT_RECORD),
    )


def decide_kv_ownership(
    candidate_block_ids: Sequence[int],
    config: KivoKVOwnershipConfig | None = None,
) -> KivoKVOwnershipDecision:
    """Partition skipped/removable candidate block IDs into allow/protect sets."""
    if config is None:
        config = current_kv_ownership_config()

    candidates = tuple(int(block_id) for block_id in candidate_block_ids)
    candidate_count = len(candidates)

    if not config.enabled:
        return KivoKVOwnershipDecision(
            enabled=False,
            policy="off",
            candidate_block_ids=candidates,
            allowed_block_ids=candidates,
            protected_block_ids=(),
            reason_counts={"disabled": 1},
            candidate_count=candidate_count,
            allowed_count=candidate_count,
            protected_count=0,
        )

    policy = config.policy
    if policy == "off":
        return KivoKVOwnershipDecision(
            enabled=True,
            policy=policy,
            candidate_block_ids=candidates,
            allowed_block_ids=candidates,
            protected_block_ids=(),
            reason_counts={"policy_off": 1},
            candidate_count=candidate_count,
            allowed_count=candidate_count,
            protected_count=0,
        )
    if policy == "allow_all_skipped":
        return KivoKVOwnershipDecision(
            enabled=True,
            policy=policy,
            candidate_block_ids=candidates,
            allowed_block_ids=candidates,
            protected_block_ids=(),
            reason_counts={"allow_all_skipped": 1},
            candidate_count=candidate_count,
            allowed_count=candidate_count,
            protected_count=0,
        )
    if policy == "protect_all_skipped":
        return KivoKVOwnershipDecision(
            enabled=True,
            policy=policy,
            candidate_block_ids=candidates,
            allowed_block_ids=(),
            protected_block_ids=candidates,
            reason_counts={"protect_all_skipped": 1},
            candidate_count=candidate_count,
            allowed_count=0,
            protected_count=candidate_count,
        )
    if policy == "protect_recent_skipped":
        keep_recent = max(0, int(config.keep_recent_blocks))
        if keep_recent <= 0:
            allowed = candidates
            protected = ()
        elif candidate_count <= keep_recent:
            allowed = ()
            protected = candidates
        else:
            allowed = candidates[:-keep_recent]
            protected = candidates[-keep_recent:]
        return KivoKVOwnershipDecision(
            enabled=True,
            policy=policy,
            candidate_block_ids=candidates,
            allowed_block_ids=allowed,
            protected_block_ids=protected,
            reason_counts={"protect_recent_skipped": 1},
            candidate_count=candidate_count,
            allowed_count=len(allowed),
            protected_count=len(protected),
        )

    return KivoKVOwnershipDecision(
        enabled=True,
        policy=policy,
        candidate_block_ids=candidates,
        allowed_block_ids=(),
        protected_block_ids=candidates,
        reason_counts={"invalid_policy": 1},
        candidate_count=candidate_count,
        allowed_count=0,
        protected_count=candidate_count,
    )
