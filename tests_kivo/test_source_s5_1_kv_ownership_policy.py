from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.v1.core.kivo_kv_ownership_policy import (
    KivoKVOwnershipConfig,
    decide_kv_ownership,
)
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager


class FakeBlock:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.is_null = False

    def __eq__(self, other):
        return self is other


class FakePool:
    def __init__(self):
        self.null_block = object()
        self.freed: list[list[int]] = []

    def free_blocks(self, blocks):
        self.freed.append([block.block_id for block in blocks])


class FakeManager(SingleTypeKVCacheManager):
    @classmethod
    def find_longest_cache_hit(
        cls, *args, **kwargs
    ):  # pragma: no cover - abstract shim
        return ()

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        return 0

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        return num_computed_tokens


def _make_manager() -> tuple[FakeManager, FakePool, list[FakeBlock]]:
    pool = FakePool()
    manager = FakeManager(
        kv_cache_spec=SimpleNamespace(block_size=1),
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
    )
    blocks = [FakeBlock(i) for i in range(4)]
    manager.req_to_blocks["req"] = blocks.copy()
    return manager, pool, blocks


def test_disabled_config_allows_everything():
    decision = decide_kv_ownership(
        [1, 2, 3], KivoKVOwnershipConfig(False, "protect_recent_skipped", 1, "summary")
    )
    assert decision.allowed_block_ids == (1, 2, 3)
    assert decision.protected_block_ids == ()
    assert decision.enabled is False


def test_allow_all_skipped_allows_everything():
    decision = decide_kv_ownership(
        [1, 2, 3], KivoKVOwnershipConfig(True, "allow_all_skipped", 1, "summary")
    )
    assert decision.allowed_block_ids == (1, 2, 3)
    assert decision.protected_block_ids == ()


def test_protect_all_skipped_protects_everything():
    decision = decide_kv_ownership(
        [1, 2, 3], KivoKVOwnershipConfig(True, "protect_all_skipped", 1, "summary")
    )
    assert decision.allowed_block_ids == ()
    assert decision.protected_block_ids == (1, 2, 3)


def test_protect_recent_skipped_keep_one():
    decision = decide_kv_ownership(
        [1, 2, 3, 4],
        KivoKVOwnershipConfig(True, "protect_recent_skipped", 1, "summary"),
    )
    assert decision.allowed_block_ids == (1, 2, 3)
    assert decision.protected_block_ids == (4,)


def test_protect_recent_skipped_keep_two():
    decision = decide_kv_ownership(
        [1, 2, 3, 4],
        KivoKVOwnershipConfig(True, "protect_recent_skipped", 2, "summary"),
    )
    assert decision.allowed_block_ids == (1, 2)
    assert decision.protected_block_ids == (3, 4)


def test_empty_candidates_handled():
    decision = decide_kv_ownership(
        [], KivoKVOwnershipConfig(True, "protect_recent_skipped", 1, "summary")
    )
    assert decision.allowed_block_ids == ()
    assert decision.protected_block_ids == ()
    assert decision.candidate_count == 0


def test_invalid_policy_fails_closed_when_enabled():
    decision = decide_kv_ownership(
        [1, 2], KivoKVOwnershipConfig(True, "not_a_policy", 1, "summary")
    )
    assert decision.allowed_block_ids == ()
    assert decision.protected_block_ids == (1, 2)
    assert decision.reason_counts == {"invalid_policy": 1}


def test_decision_does_not_mutate_input():
    candidates = [7, 8, 9]
    before = candidates.copy()
    decide_kv_ownership(
        candidates, KivoKVOwnershipConfig(True, "protect_recent_skipped", 1, "summary")
    )
    assert candidates == before


def test_remove_skipped_blocks_partitions_without_gpu(monkeypatch):
    monkeypatch.setenv("KIVO_KV_OWNERSHIP_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_OWNERSHIP_POLICY", "protect_recent_skipped")
    monkeypatch.setenv("KIVO_KV_OWNERSHIP_KEEP_RECENT_BLOCKS", "1")
    manager, pool, blocks = _make_manager()

    manager.remove_skipped_blocks("req", total_computed_tokens=4)

    assert pool.freed == [[2, 1, 0]]
    req_blocks = manager.req_to_blocks["req"]
    assert req_blocks[0] == manager._null_block
    assert req_blocks[1] == manager._null_block
    assert req_blocks[2] == manager._null_block
    assert req_blocks[3].block_id == 3
