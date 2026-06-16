from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from vllm.v1.core.kivo_demotion_command import (
    KivoCoreDemotionConfig,
    KivoDemotionCommand,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
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


class DummyCoordinator:
    def __init__(self, managers):
        self.single_type_managers = tuple(managers)

    def apply_kivo_demotion_command(self, command, *, config):
        if not config.enabled or config.action == "off":
            from vllm.v1.core.kivo_demotion_command import KivoDemotionCommandResult

            return KivoDemotionCommandResult(
                enabled=False,
                request_id=command.request_id,
                accepted=False,
                marked_demoted_block_ids=(),
                rejected_block_ids=command.candidate_demote_block_ids,
                blocker_reasons={"disabled": 1},
                removes_from_req_to_blocks=False,
                frees_to_pool=False,
            )
        if len(self.single_type_managers) != 1:
            from vllm.v1.core.kivo_demotion_command import KivoDemotionCommandResult

            return KivoDemotionCommandResult(
                enabled=True,
                request_id=command.request_id,
                accepted=False,
                marked_demoted_block_ids=(),
                rejected_block_ids=command.candidate_demote_block_ids,
                blocker_reasons={"ambiguous_single_type_manager_count": 1},
                removes_from_req_to_blocks=False,
                frees_to_pool=False,
            )
        return self.single_type_managers[0].apply_kivo_demotion_command(
            command,
            config=config,
        )


def _config(
    *,
    enabled: bool = True,
    action: str = "mark_demoted_only",
    require_block_table_applied: bool = True,
    require_slot_mapping_refresh: bool = True,
) -> KivoCoreDemotionConfig:
    return KivoCoreDemotionConfig(
        enabled=enabled,
        action=action,
        require_block_table_applied=require_block_table_applied,
        require_slot_mapping_refresh=require_slot_mapping_refresh,
    )


def _command(
    *,
    request_id: str = "req0",
    before=(10, 11, 12, 13),
    after=(12, 13),
    demote=(10, 11),
    protected=(12, 13),
    block_table_applied=True,
    slot_mapping_refresh_guaranteed=True,
) -> KivoDemotionCommand:
    return KivoDemotionCommand(
        request_id=request_id,
        visible_before_block_ids=tuple(before),
        visible_after_block_ids=tuple(after),
        candidate_demote_block_ids=tuple(demote),
        protected_block_ids=tuple(protected),
        block_table_applied=block_table_applied,
        slot_mapping_refresh_guaranteed=slot_mapping_refresh_guaranteed,
    )


def _manager() -> DummySingleTypeManager:
    manager = DummySingleTypeManager()
    manager.req_to_blocks["req0"] = [
        FakeBlock(10),
        FakeBlock(11),
        FakeBlock(12),
        FakeBlock(13),
    ]
    return manager


def test_disabled_core_demotion_rejects_noop():
    manager = _manager()
    result = manager.apply_kivo_demotion_command(
        _command(),
        config=_config(enabled=False),
    )
    assert result.enabled is False
    assert result.accepted is False


def test_valid_command_marks_demoted_blocks_in_manager():
    manager = _manager()
    result = manager.apply_kivo_demotion_command(
        _command(),
        config=_config(),
    )
    assert result.accepted is True
    assert result.marked_demoted_block_ids == (10, 11)
    assert manager.get_kivo_demoted_block_ids("req0") == (10, 11)


def test_invalid_request_id_fails_closed():
    manager = _manager()
    result = manager.apply_kivo_demotion_command(
        _command(request_id="missing"),
        config=_config(),
    )
    assert result.accepted is False
    assert result.blocker_reasons["ownership_mapping_unavailable"] == 1


def test_candidate_demote_not_owned_fails_closed():
    manager = _manager()
    result = manager.apply_kivo_demotion_command(
        _command(demote=(10, 99), after=(12, 13)),
        config=_config(),
    )
    assert result.accepted is False
    assert result.blocker_reasons["demote_ids_missing_from_ownership_before"] == 1


def test_demote_still_visible_after_fails_closed():
    manager = _manager()
    result = manager.apply_kivo_demotion_command(
        _command(after=(11, 12, 13)),
        config=_config(),
    )
    assert result.accepted is False
    assert result.blocker_reasons["demote_ids_still_visible_after"] == 1


def test_protected_id_demotion_fails_closed():
    manager = _manager()
    result = manager.apply_kivo_demotion_command(
        _command(demote=(10, 13), after=(11, 12, 13)),
        config=_config(),
    )
    assert result.accepted is False
    assert result.blocker_reasons["demote_overlaps_protected"] == 1


def test_missing_block_table_applied_proof_fails_closed():
    manager = _manager()
    result = manager.apply_kivo_demotion_command(
        _command(block_table_applied=False),
        config=_config(),
    )
    assert result.accepted is False
    assert result.blocker_reasons["block_table_apply_required"] == 1


def test_missing_slot_mapping_refresh_proof_fails_closed():
    manager = _manager()
    result = manager.apply_kivo_demotion_command(
        _command(slot_mapping_refresh_guaranteed=False),
        config=_config(),
    )
    assert result.accepted is False
    assert result.blocker_reasons["slot_mapping_refresh_not_guaranteed"] == 1


def test_command_does_not_remove_from_req_to_blocks():
    manager = _manager()
    before = manager.get_request_block_ids_for_kivo("req0")
    result = manager.apply_kivo_demotion_command(
        _command(),
        config=_config(),
    )
    assert result.accepted is True
    assert manager.get_request_block_ids_for_kivo("req0") == before


def test_command_does_not_call_free_to_pool():
    manager = _manager()
    result = manager.apply_kivo_demotion_command(
        _command(),
        config=_config(),
    )
    assert result.frees_to_pool is False
    assert manager.block_pool.freed_calls == []


def test_command_result_records_marked_and_rejected_ids():
    manager = _manager()
    accepted = manager.apply_kivo_demotion_command(
        _command(),
        config=_config(),
    )
    rejected = manager.apply_kivo_demotion_command(
        _command(after=(11, 12, 13)),
        config=_config(),
    )
    assert accepted.marked_demoted_block_ids == (10, 11)
    assert rejected.rejected_block_ids == (10, 11)


def test_kv_cache_manager_level_api_routes_cleanly():
    manager = _manager()
    kv_manager = KVCacheManager.__new__(KVCacheManager)
    kv_manager.coordinator = DummyCoordinator([manager])
    result = KVCacheManager.apply_kivo_demotion_command(
        kv_manager,
        _command(),
        config=_config(),
    )
    assert result.accepted is True
    assert manager.get_kivo_demoted_block_ids("req0") == (10, 11)


def test_kv_cache_manager_level_api_fails_closed_when_ambiguous():
    manager_a = _manager()
    manager_b = _manager()
    kv_manager = KVCacheManager.__new__(KVCacheManager)
    kv_manager.coordinator = DummyCoordinator([manager_a, manager_b])
    result = KVCacheManager.apply_kivo_demotion_command(
        kv_manager,
        _command(),
        config=_config(),
    )
    assert result.accepted is False
    assert result.blocker_reasons["ambiguous_single_type_manager_count"] == 1
