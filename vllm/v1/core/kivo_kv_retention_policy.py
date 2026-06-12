# SPDX-License-Identifier: Apache-2.0

"""Online KV retention planning for Kivo-VD at the ownership boundary."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

_DEFAULT_POLICY = "recent_only"
_DEFAULT_ACTION = "plan_only"
_DEFAULT_KEEP_RECENT_BLOCKS = 4
_DEFAULT_MAX_FULL_BLOCKS = 64
_DEFAULT_MIN_BLOCKS_BEFORE_ACTION = 8


@dataclass(frozen=True)
class KivoKVRetentionConfig:
    enabled: bool
    policy: str
    keep_recent_blocks: int
    max_full_blocks: int
    min_blocks_before_action: int
    action: str


@dataclass(frozen=True)
class KivoKVRetentionDecision:
    enabled: bool
    policy: str
    action: str
    request_id: str | None
    all_block_ids: tuple[int, ...]
    keep_block_ids: tuple[int, ...]
    candidate_drop_block_ids: tuple[int, ...]
    protected_block_ids: tuple[int, ...]
    reason_counts: dict[str, int]
    score_available_count: int
    score_missing_count: int
    would_reduce_full_blocks_by: int


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


def current_kv_retention_config() -> KivoKVRetentionConfig:
    return KivoKVRetentionConfig(
        enabled=_parse_bool_env("KIVO_KV_RETENTION_ENABLE", default=False),
        policy=os.getenv("KIVO_KV_RETENTION_POLICY", _DEFAULT_POLICY),
        keep_recent_blocks=_parse_int_env(
            "KIVO_KV_RETENTION_KEEP_RECENT_BLOCKS",
            default=_DEFAULT_KEEP_RECENT_BLOCKS,
            minimum=0,
        ),
        max_full_blocks=_parse_int_env(
            "KIVO_KV_RETENTION_MAX_FULL_BLOCKS",
            default=_DEFAULT_MAX_FULL_BLOCKS,
            minimum=1,
        ),
        min_blocks_before_action=_parse_int_env(
            "KIVO_KV_RETENTION_MIN_BLOCKS_BEFORE_ACTION",
            default=_DEFAULT_MIN_BLOCKS_BEFORE_ACTION,
            minimum=0,
        ),
        action=os.getenv("KIVO_KV_RETENTION_ACTION", _DEFAULT_ACTION),
    )


def decide_kv_retention(
    all_block_ids: Sequence[int],
    score_map: Mapping[int, float] | None = None,
    *,
    request_id: str | None = None,
    config: KivoKVRetentionConfig | None = None,
) -> KivoKVRetentionDecision:
    """Compute a retention plan without mutating runtime ownership."""
    if config is None:
        config = current_kv_retention_config()
    if score_map is None:
        score_map = {}

    block_ids = tuple(int(block_id) for block_id in all_block_ids)
    total_blocks = len(block_ids)

    if not config.enabled:
        return KivoKVRetentionDecision(
            enabled=False,
            policy="off",
            action="plan_only",
            request_id=request_id,
            all_block_ids=block_ids,
            keep_block_ids=block_ids,
            candidate_drop_block_ids=(),
            protected_block_ids=(),
            reason_counts={"disabled": 1},
            score_available_count=0,
            score_missing_count=0,
            would_reduce_full_blocks_by=0,
        )

    if config.action != "plan_only":
        return KivoKVRetentionDecision(
            enabled=True,
            policy=config.policy,
            action=config.action,
            request_id=request_id,
            all_block_ids=block_ids,
            keep_block_ids=block_ids,
            candidate_drop_block_ids=(),
            protected_block_ids=block_ids,
            reason_counts={"unsupported_action_fail_closed": 1},
            score_available_count=0,
            score_missing_count=0,
            would_reduce_full_blocks_by=0,
        )

    if total_blocks < config.min_blocks_before_action:
        return KivoKVRetentionDecision(
            enabled=True,
            policy=config.policy,
            action=config.action,
            request_id=request_id,
            all_block_ids=block_ids,
            keep_block_ids=block_ids,
            candidate_drop_block_ids=(),
            protected_block_ids=block_ids,
            reason_counts={"below_min_blocks_before_action": 1},
            score_available_count=0,
            score_missing_count=0,
            would_reduce_full_blocks_by=0,
        )

    keep_recent = min(max(0, config.keep_recent_blocks), total_blocks)
    protected_recent = tuple(block_ids[-keep_recent:]) if keep_recent > 0 else ()
    older = tuple(block_ids[:-keep_recent]) if keep_recent > 0 else block_ids
    effective_budget = max(config.max_full_blocks, len(protected_recent))
    remaining_budget = max(0, effective_budget - len(protected_recent))

    if config.policy == "recent_only":
        older_keep = older[-remaining_budget:] if remaining_budget > 0 else ()
        keep_set = set(older_keep) | set(protected_recent)
        keep_ids = tuple(block_id for block_id in block_ids if block_id in keep_set)
        candidate_drop = tuple(
            block_id for block_id in block_ids if block_id not in keep_set
        )
        return KivoKVRetentionDecision(
            enabled=True,
            policy=config.policy,
            action=config.action,
            request_id=request_id,
            all_block_ids=block_ids,
            keep_block_ids=keep_ids,
            candidate_drop_block_ids=candidate_drop,
            protected_block_ids=protected_recent,
            reason_counts={"recent_only": 1},
            score_available_count=0,
            score_missing_count=0,
            would_reduce_full_blocks_by=len(candidate_drop),
        )

    if config.policy == "countsketch_online":
        scored_older = [
            (block_id, float(score_map[block_id]))
            for block_id in older
            if block_id in score_map
        ]
        missing_older = tuple(block_id for block_id in older if block_id not in score_map)
        older_index = {block_id: idx for idx, block_id in enumerate(older)}
        scored_older.sort(
            key=lambda item: (-item[1], -older_index[item[0]]),
        )
        selected_scored_count = max(0, remaining_budget - len(missing_older))
        selected_scored = {
            block_id for block_id, _ in scored_older[:selected_scored_count]
        }
        protected_set = set(protected_recent) | set(missing_older)
        keep_set = protected_set | selected_scored
        keep_ids = tuple(block_id for block_id in block_ids if block_id in keep_set)
        protected_ids = tuple(
            block_id for block_id in block_ids if block_id in protected_set
        )
        candidate_drop = tuple(
            block_id for block_id in block_ids if block_id not in keep_set
        )
        return KivoKVRetentionDecision(
            enabled=True,
            policy=config.policy,
            action=config.action,
            request_id=request_id,
            all_block_ids=block_ids,
            keep_block_ids=keep_ids,
            candidate_drop_block_ids=candidate_drop,
            protected_block_ids=protected_ids,
            reason_counts={"countsketch_online": 1},
            score_available_count=len(scored_older),
            score_missing_count=len(missing_older),
            would_reduce_full_blocks_by=len(candidate_drop),
        )

    return KivoKVRetentionDecision(
        enabled=True,
        policy=config.policy,
        action=config.action,
        request_id=request_id,
        all_block_ids=block_ids,
        keep_block_ids=block_ids,
        candidate_drop_block_ids=(),
        protected_block_ids=block_ids,
        reason_counts={"invalid_policy_fail_closed": 1},
        score_available_count=0,
        score_missing_count=0,
        would_reduce_full_blocks_by=0,
    )


def decision_summary(decision: KivoKVRetentionDecision) -> dict[str, Any]:
    return {
        "enabled": decision.enabled,
        "policy": decision.policy,
        "action": decision.action,
        "request_id": decision.request_id,
        "total_block_count": len(decision.all_block_ids),
        "keep_block_count": len(decision.keep_block_ids),
        "candidate_drop_count": len(decision.candidate_drop_block_ids),
        "protected_block_count": len(decision.protected_block_ids),
        "score_available_count": decision.score_available_count,
        "score_missing_count": decision.score_missing_count,
        "would_reduce_full_blocks_by": decision.would_reduce_full_blocks_by,
        "reason_counts": dict(decision.reason_counts),
    }

