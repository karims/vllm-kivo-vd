from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.core.kivo_ownership_bridge import KivoOwnershipBridgeConfig
from vllm.v1.worker.kivo_runtime_demotion_mark import (
    KivoRuntimeDemotionMarkConfig,
    maybe_mark_demoted_blocks_after_block_table_apply,
)


def _config(
    *,
    enabled: bool = True,
    action: str = "mark_demoted_after_block_table_apply",
    require_block_table_applied: bool = True,
    require_slot_mapping_refresh: bool = True,
) -> KivoRuntimeDemotionMarkConfig:
    return KivoRuntimeDemotionMarkConfig(
        enabled=enabled,
        action=action,
        require_block_table_applied=require_block_table_applied,
        require_slot_mapping_refresh=require_slot_mapping_refresh,
    )


def test_disabled_demotion_mark_returns_noop_summary():
    summary = maybe_mark_demoted_blocks_after_block_table_apply(
        request_id="req0",
        demote_block_ids=(10, 11),
        visible_after_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        kv_cache_manager=None,
        config=_config(enabled=False),
    )
    assert summary.enabled is False
    assert summary.attempted_request_count == 0


def test_enabled_but_missing_manager_fails_closed():
    summary = maybe_mark_demoted_blocks_after_block_table_apply(
        request_id="req0",
        demote_block_ids=(10, 11),
        visible_after_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        kv_cache_manager=None,
        config=_config(),
    )
    assert summary.blocked_request_count == 1
    assert summary.blocker_reasons["kv_cache_manager_unavailable"] == 1


def test_manager_without_mark_helper_fails_closed():
    summary = maybe_mark_demoted_blocks_after_block_table_apply(
        request_id="req0",
        demote_block_ids=(10, 11),
        visible_after_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        kv_cache_manager=object(),
        config=_config(),
    )
    assert summary.blocked_request_count == 1
    assert summary.blocker_reasons["kv_cache_manager_missing_mark_helper"] == 1


def test_block_table_applied_false_fails_closed():
    summary = maybe_mark_demoted_blocks_after_block_table_apply(
        request_id="req0",
        demote_block_ids=(10,),
        visible_after_block_ids=(11, 12, 13),
        block_table_applied=False,
        slot_mapping_refresh_guaranteed=True,
        kv_cache_manager=object(),
        config=_config(),
    )
    assert summary.blocker_reasons["block_table_apply_required"] == 1


def test_slot_mapping_refresh_false_fails_closed():
    summary = maybe_mark_demoted_blocks_after_block_table_apply(
        request_id="req0",
        demote_block_ids=(10,),
        visible_after_block_ids=(11, 12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=False,
        kv_cache_manager=object(),
        config=_config(),
    )
    assert summary.blocker_reasons["slot_mapping_refresh_not_guaranteed"] == 1


def test_demote_id_still_visible_after_filtering_fails_closed():
    summary = maybe_mark_demoted_blocks_after_block_table_apply(
        request_id="req0",
        demote_block_ids=(10, 11),
        visible_after_block_ids=(11, 12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        kv_cache_manager=object(),
        config=_config(),
    )
    assert summary.blocker_reasons["demote_ids_still_visible_after"] == 1


@dataclass
class FakeDecision:
    safe_to_mark_demoted: bool
    demote_block_ids: tuple[int, ...]
    blocker_reasons: dict[str, int]
    safe_to_free: bool = False


class FakeManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.req_to_blocks = {"req0": [10, 11, 12, 13]}
        self.free_called = False

    def mark_kivo_demoted_blocks_if_safe(
        self,
        request_id,
        block_ids,
        *,
        visible_after_block_ids,
        protected_block_ids=(),
        block_table_applied=False,
        slot_mapping_refresh_guaranteed=False,
        config=None,
    ):
        self.calls.append(
            {
                "request_id": request_id,
                "block_ids": tuple(block_ids),
                "visible_after_block_ids": tuple(visible_after_block_ids),
                "protected_block_ids": tuple(protected_block_ids),
                "block_table_applied": block_table_applied,
                "slot_mapping_refresh_guaranteed": slot_mapping_refresh_guaranteed,
                "config": config,
            }
        )
        assert isinstance(config, KivoOwnershipBridgeConfig)
        return FakeDecision(
            safe_to_mark_demoted=True,
            demote_block_ids=tuple(block_ids),
            blocker_reasons={},
        )


def test_valid_fake_manager_marks_demoted_ids():
    manager = FakeManager()
    summary = maybe_mark_demoted_blocks_after_block_table_apply(
        request_id="req0",
        demote_block_ids=(10, 11),
        visible_after_block_ids=(12, 13),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        kv_cache_manager=manager,
        config=_config(),
    )
    assert summary.marked_request_count == 1
    assert summary.marked_block_count == 2
    assert manager.calls[0]["block_ids"] == (10, 11)


def test_valid_call_does_not_remove_from_req_to_blocks():
    manager = FakeManager()
    maybe_mark_demoted_blocks_after_block_table_apply(
        request_id="req0",
        demote_block_ids=(10,),
        visible_after_block_ids=(11, 12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        kv_cache_manager=manager,
        config=_config(),
    )
    assert manager.req_to_blocks["req0"] == [10, 11, 12, 13]


def test_valid_call_does_not_call_free_to_pool():
    manager = FakeManager()
    maybe_mark_demoted_blocks_after_block_table_apply(
        request_id="req0",
        demote_block_ids=(10,),
        visible_after_block_ids=(11, 12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        kv_cache_manager=manager,
        config=_config(),
    )
    assert manager.free_called is False


def test_summary_reports_attempted_marked_blocked_counts():
    manager = FakeManager()
    ok = maybe_mark_demoted_blocks_after_block_table_apply(
        request_id="req0",
        demote_block_ids=(10,),
        visible_after_block_ids=(11, 12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        kv_cache_manager=manager,
        config=_config(),
    )
    blocked = maybe_mark_demoted_blocks_after_block_table_apply(
        request_id="req0",
        demote_block_ids=(10,),
        visible_after_block_ids=(10, 11, 12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        kv_cache_manager=manager,
        config=_config(),
    )
    assert ok.attempted_request_count == 1
    assert ok.marked_request_count == 1
    assert blocked.blocked_request_count == 1
