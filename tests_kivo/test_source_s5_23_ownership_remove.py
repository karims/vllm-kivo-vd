from __future__ import annotations

from collections import defaultdict

from vllm.v1.core.kivo_demotion_command import (
    KivoCoreDemotionConfig,
    KivoDemotionCommand,
    KivoOwnershipRemoveConfig,
)
from vllm.v1.core.kivo_demotion_counters import (
    get_kivo_demotion_counters_snapshot,
    reset_kivo_demotion_counters,
)
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager


class FakeBlock:
    def __init__(self, block_id: int) -> None:
        self.block_id = block_id
        self.ref_cnt = 1
        self.is_null = False


class FakeBlockPool:
    def __init__(self) -> None:
        self.null_block = object()
        self.freed_calls: list[list[int]] = []

    def free_blocks(self, blocks) -> None:
        self.freed_calls.append(
            [block.block_id for block in list(blocks) if hasattr(block, "block_id")]
        )


class DummySingleTypeManager(SingleTypeKVCacheManager):
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


def _manager() -> DummySingleTypeManager:
    manager = DummySingleTypeManager()
    manager.req_to_blocks["req0"] = [
        FakeBlock(10),
        FakeBlock(11),
        FakeBlock(12),
        FakeBlock(13),
    ]
    return manager


def _command() -> KivoDemotionCommand:
    return KivoDemotionCommand(
        request_id="req0",
        visible_before_block_ids=(10, 11, 12, 13),
        visible_after_block_ids=(12, 13),
        candidate_demote_block_ids=(10, 11),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
    )


def _core_config() -> KivoCoreDemotionConfig:
    return KivoCoreDemotionConfig(
        enabled=True,
        action="mark_demoted_only",
        require_block_table_applied=True,
        require_slot_mapping_refresh=True,
    )


def _remove_config(
    *, enabled: bool = True, action: str = "remove_marked_demoted_only"
) -> KivoOwnershipRemoveConfig:
    return KivoOwnershipRemoveConfig(enabled=enabled, action=action)


def test_remove_marked_demoted_blocks_if_safe_removes_only_marked_blocks():
    manager = _manager()
    manager.kivo_req_to_demoted_block_ids["req0"] = {10, 11}

    result = manager.remove_kivo_marked_demoted_blocks_if_safe(
        "req0",
        config=_remove_config(),
    )

    assert result.accepted is True
    assert result.removed_block_ids == (10, 11)
    assert result.remaining_block_ids == (12, 13)
    assert result.removes_from_req_to_blocks is True
    assert manager.get_request_block_ids_for_kivo("req0") == (12, 13)
    assert manager.get_kivo_demoted_block_ids("req0") == ()
    assert manager.block_pool.freed_calls == []


def test_remove_marked_demoted_blocks_if_safe_rejects_missing_request():
    manager = _manager()
    result = manager.remove_kivo_marked_demoted_blocks_if_safe(
        "missing",
        config=_remove_config(),
    )
    assert result.accepted is False
    assert result.blocker_reasons["ownership_mapping_unavailable"] == 1


def test_remove_marked_demoted_blocks_if_safe_rejects_if_would_remove_all_owned():
    manager = _manager()
    manager.kivo_req_to_demoted_block_ids["req0"] = {10, 11, 12, 13}

    result = manager.remove_kivo_marked_demoted_blocks_if_safe(
        "req0",
        config=_remove_config(),
    )

    assert result.accepted is False
    assert result.blocker_reasons["would_remove_all_owned_blocks"] == 1
    assert manager.get_request_block_ids_for_kivo("req0") == (10, 11, 12, 13)


def test_remove_marked_demoted_blocks_if_safe_rejects_stale_marked_ids():
    manager = _manager()
    manager.kivo_req_to_demoted_block_ids["req0"] = {10, 99}

    result = manager.remove_kivo_marked_demoted_blocks_if_safe(
        "req0",
        config=_remove_config(),
    )

    assert result.accepted is False
    assert result.blocker_reasons["demoted_ids_missing_from_req_to_blocks"] == 1


def test_apply_kivo_demotion_command_can_mark_then_remove_when_enabled(monkeypatch):
    manager = _manager()
    reset_kivo_demotion_counters()
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_OWNERSHIP_REMOVE_ENABLE", "1")
    monkeypatch.setenv(
        "KIVO_KV_OWNERSHIP_REMOVE_ACTION", "remove_marked_demoted_only"
    )

    result = manager.apply_kivo_demotion_command(_command(), config=_core_config())
    snapshot = get_kivo_demotion_counters_snapshot()

    assert result.accepted is True
    assert result.removes_from_req_to_blocks is True
    assert manager.get_request_block_ids_for_kivo("req0") == (12, 13)
    assert manager.get_kivo_demoted_block_ids("req0") == ()
    assert snapshot["ownership_remove_attempted"] == 1
    assert snapshot["ownership_remove_succeeded"] == 1
    assert snapshot["req_to_blocks_removed"] == 2
    assert snapshot["free_to_pool_calls"] == 0
