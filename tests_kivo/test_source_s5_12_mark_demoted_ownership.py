from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from vllm.v1.core.kivo_ownership_bridge import KivoOwnershipBridgeConfig
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager


@dataclass
class FakeBlock:
    block_id: int
    ref_cnt: int = 1
    is_null: bool = False


class FakeBlockPool:
    def __init__(self) -> None:
        self.null_block = object()
        self.freed_calls: list[list[int]] = []

    def free_blocks(self, blocks) -> None:
        self.freed_calls.append(
            [block.block_id for block in list(blocks) if hasattr(block, "block_id")]
        )


class DummyKVManager(SingleTypeKVCacheManager):
    def __init__(self) -> None:
        self.block_pool = FakeBlockPool()
        self.req_to_blocks = defaultdict(list)
        self.num_cached_block = {}
        self._null_block = self.block_pool.null_block
        self.kivo_req_to_demoted_block_ids = {}
        self._last_kivo_ownership_bridge_decision = None

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        del running_request_id
        return 0

    @classmethod
    def find_longest_cache_hit(cls, *args, **kwargs):
        del args, kwargs
        return ()


def _config(
    *,
    enabled: bool = True,
    action: str = "mark_demoted_if_safe",
    require_block_table_applied: bool = True,
    require_slot_mapping_refresh: bool = True,
) -> KivoOwnershipBridgeConfig:
    return KivoOwnershipBridgeConfig(
        enabled=enabled,
        action=action,
        require_block_table_applied=require_block_table_applied,
        require_slot_mapping_refresh=require_slot_mapping_refresh,
    )


def _manager() -> DummyKVManager:
    manager = DummyKVManager()
    manager.req_to_blocks["req0"] = [
        FakeBlock(10),
        FakeBlock(11),
        FakeBlock(12),
        FakeBlock(13),
    ]
    return manager


def test_disabled_bridge_does_not_mark_demoted_blocks():
    manager = _manager()
    decision = manager.mark_kivo_demoted_blocks_if_safe(
        "req0",
        (10, 11),
        visible_after_block_ids=(12, 13),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        config=_config(enabled=False),
    )
    assert decision.enabled is False
    assert manager.get_kivo_demoted_block_ids("req0") == ()


def test_plan_only_does_not_mark_demoted_blocks():
    manager = _manager()
    decision = manager.mark_kivo_demoted_blocks_if_safe(
        "req0",
        (10, 11),
        visible_after_block_ids=(12, 13),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        config=_config(action="plan_only"),
    )
    assert decision.safe_to_mark_demoted is False
    assert manager.get_kivo_demoted_block_ids("req0") == ()


def test_valid_mark_demoted_if_safe_marks_only_eligible_ids():
    manager = _manager()
    decision = manager.mark_kivo_demoted_blocks_if_safe(
        "req0",
        (10, 11),
        visible_after_block_ids=(12, 13),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        config=_config(),
    )
    assert decision.safe_to_mark_demoted is True
    assert manager.get_kivo_demoted_block_ids("req0") == (10, 11)
    assert manager.get_request_block_ids_for_kivo("req0") == (10, 11, 12, 13)


def test_marked_ids_are_queryable_by_request_id():
    manager = _manager()
    manager.mark_kivo_demoted_blocks_if_safe(
        "req0",
        (10,),
        visible_after_block_ids=(11, 12, 13),
        protected_block_ids=(11, 12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        config=_config(),
    )
    assert manager.get_kivo_demoted_block_ids("req0") == (10,)


def test_demote_id_still_visible_after_block_table_apply_fails_closed():
    manager = _manager()
    decision = manager.mark_kivo_demoted_blocks_if_safe(
        "req0",
        (10, 11),
        visible_after_block_ids=(11, 12, 13),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        config=_config(),
    )
    assert decision.safe_to_mark_demoted is False
    assert decision.blocker_reasons["demote_ids_still_visible_after"] == 1
    assert manager.get_kivo_demoted_block_ids("req0") == ()


def test_missing_slot_mapping_refresh_guarantee_fails_closed():
    manager = _manager()
    decision = manager.mark_kivo_demoted_blocks_if_safe(
        "req0",
        (10,),
        visible_after_block_ids=(11, 12, 13),
        protected_block_ids=(11, 12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=False,
        config=_config(),
    )
    assert decision.safe_to_mark_demoted is False
    assert decision.blocker_reasons["slot_mapping_refresh_not_guaranteed"] == 1


def test_missing_block_table_applied_proof_fails_closed():
    manager = _manager()
    decision = manager.mark_kivo_demoted_blocks_if_safe(
        "req0",
        (10,),
        visible_after_block_ids=(11, 12, 13),
        protected_block_ids=(11, 12, 13),
        block_table_applied=False,
        slot_mapping_refresh_guaranteed=True,
        config=_config(),
    )
    assert decision.safe_to_mark_demoted is False
    assert decision.blocker_reasons["block_table_apply_required"] == 1


def test_demote_id_not_in_ownership_before_fails_closed():
    manager = _manager()
    decision = manager.mark_kivo_demoted_blocks_if_safe(
        "req0",
        (99,),
        visible_after_block_ids=(10, 11, 12, 13),
        protected_block_ids=(10, 11, 12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        config=_config(),
    )
    assert decision.safe_to_mark_demoted is False
    assert decision.blocker_reasons["demote_ids_missing_from_ownership_before"] == 1


def test_protected_recent_id_cannot_be_marked_demoted():
    manager = _manager()
    decision = manager.mark_kivo_demoted_blocks_if_safe(
        "req0",
        (10, 13),
        visible_after_block_ids=(11, 12, 13),
        protected_block_ids=(11, 12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        config=_config(),
    )
    assert decision.safe_to_mark_demoted is False
    assert decision.blocker_reasons["demote_overlaps_protected"] == 1


def test_duplicate_ids_fail_closed():
    manager = _manager()
    decision = manager.mark_kivo_demoted_blocks_if_safe(
        "req0",
        (10, 10),
        visible_after_block_ids=(11, 12, 13),
        protected_block_ids=(11, 12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        config=_config(),
    )
    assert decision.safe_to_mark_demoted is False
    assert decision.blocker_reasons["duplicate_demote_ids"] == 1


def test_clearing_request_removes_demoted_bookkeeping():
    manager = _manager()
    manager.mark_kivo_demoted_blocks_if_safe(
        "req0",
        (10,),
        visible_after_block_ids=(11, 12, 13),
        protected_block_ids=(11, 12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        config=_config(),
    )
    manager.clear_kivo_demoted_blocks("req0")
    assert manager.get_kivo_demoted_block_ids("req0") == ()


def test_free_clears_demoted_bookkeeping_and_safe_to_free_remains_false():
    manager = _manager()
    decision = manager.mark_kivo_demoted_blocks_if_safe(
        "req0",
        (10,),
        visible_after_block_ids=(11, 12, 13),
        protected_block_ids=(11, 12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        config=_config(),
    )
    assert decision.safe_to_free is False
    manager.free("req0")
    assert manager.get_kivo_demoted_block_ids("req0") == ()


def test_no_new_free_call_is_introduced_by_mark_demoted():
    manager = _manager()
    manager.mark_kivo_demoted_blocks_if_safe(
        "req0",
        (10, 11),
        visible_after_block_ids=(12, 13),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        config=_config(),
    )
    assert manager.block_pool.freed_calls == []
