from __future__ import annotations

from types import SimpleNamespace

from vllm.v1.core.kivo_kv_block_score_store import clear_block_scores, update_block_scores
from vllm.v1.core.kivo_kv_block_score_store import KivoKVBlockScore
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


def _make_manager() -> tuple[FakeManager, FakePool]:
    pool = FakePool()
    manager = FakeManager(
        kv_cache_spec=SimpleNamespace(block_size=1),
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
    )
    manager.req_to_blocks["req"] = [FakeBlock(i) for i in range(6)]
    return manager, pool


def test_default_disabled_path_does_not_filter_candidates(monkeypatch):
    clear_block_scores()
    monkeypatch.delenv("KIVO_KV_RETENTION_ENABLE", raising=False)
    manager, pool = _make_manager()

    manager.remove_skipped_blocks("req", total_computed_tokens=3)

    assert pool.freed == [[2, 1, 0]]
    summary = manager.get_last_kivo_retention_mutation_summary()
    assert summary is not None
    assert summary["action"] == "plan_only"
    assert summary["actual_freed_candidate_count"] == 3


def test_plan_only_does_not_mutate_candidate_set(monkeypatch):
    clear_block_scores()
    monkeypatch.setenv("KIVO_KV_RETENTION_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_RETENTION_POLICY", "recent_only")
    monkeypatch.setenv("KIVO_KV_RETENTION_KEEP_RECENT_BLOCKS", "2")
    monkeypatch.setenv("KIVO_KV_RETENTION_MAX_FULL_BLOCKS", "3")
    monkeypatch.setenv("KIVO_KV_RETENTION_MIN_BLOCKS_BEFORE_ACTION", "0")
    monkeypatch.setenv("KIVO_KV_RETENTION_ACTION", "plan_only")
    manager, pool = _make_manager()

    manager.remove_skipped_blocks("req", total_computed_tokens=4)

    assert pool.freed == [[3, 2, 1, 0]]
    summary = manager.get_last_kivo_retention_mutation_summary()
    assert summary is not None
    assert summary["action"] == "plan_only"
    assert summary["actual_freed_candidate_ids"] == [3, 2, 1, 0]


def test_free_candidates_recent_only_frees_only_outside_recent_set(monkeypatch):
    clear_block_scores()
    monkeypatch.setenv("KIVO_KV_RETENTION_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_RETENTION_POLICY", "recent_only")
    monkeypatch.setenv("KIVO_KV_RETENTION_KEEP_RECENT_BLOCKS", "2")
    monkeypatch.setenv("KIVO_KV_RETENTION_MAX_FULL_BLOCKS", "3")
    monkeypatch.setenv("KIVO_KV_RETENTION_MIN_BLOCKS_BEFORE_ACTION", "0")
    monkeypatch.setenv("KIVO_KV_RETENTION_ACTION", "free_candidates")
    manager, pool = _make_manager()

    manager.remove_skipped_blocks("req", total_computed_tokens=4)

    assert pool.freed == [[0, 1, 2]]
    req_blocks = manager.req_to_blocks["req"]
    assert req_blocks[0] == manager._null_block
    assert req_blocks[1] == manager._null_block
    assert req_blocks[2] == manager._null_block
    assert req_blocks[3].block_id == 3
    assert req_blocks[4].block_id == 4
    assert req_blocks[5].block_id == 5
    summary = manager.get_last_kivo_retention_mutation_summary()
    assert summary is not None
    assert summary["allowed_free_count"] == 3
    assert summary["protected_count"] == 1


def test_free_candidates_countsketch_online_uses_scores(monkeypatch):
    clear_block_scores()
    update_block_scores(
        [
            KivoKVBlockScore(block_id=0, score=0.1, source="test"),
            KivoKVBlockScore(block_id=1, score=0.9, source="test"),
            KivoKVBlockScore(block_id=2, score=0.2, source="test"),
            KivoKVBlockScore(block_id=3, score=0.8, source="test"),
            KivoKVBlockScore(block_id=4, score=0.05, source="test"),
        ]
    )
    monkeypatch.setenv("KIVO_KV_RETENTION_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_RETENTION_POLICY", "countsketch_online")
    monkeypatch.setenv("KIVO_KV_RETENTION_KEEP_RECENT_BLOCKS", "1")
    monkeypatch.setenv("KIVO_KV_RETENTION_MAX_FULL_BLOCKS", "3")
    monkeypatch.setenv("KIVO_KV_RETENTION_MIN_BLOCKS_BEFORE_ACTION", "0")
    monkeypatch.setenv("KIVO_KV_RETENTION_ACTION", "free_candidates")
    manager, pool = _make_manager()

    manager.remove_skipped_blocks("req", total_computed_tokens=4)

    assert pool.freed == [[0, 2]]
    req_blocks = manager.req_to_blocks["req"]
    assert req_blocks[0] == manager._null_block
    assert req_blocks[1].block_id == 1
    assert req_blocks[2] == manager._null_block
    assert req_blocks[3].block_id == 3
    summary = manager.get_last_kivo_retention_mutation_summary()
    assert summary is not None
    assert summary["score_available_count"] == 5
    assert summary["actual_freed_candidate_ids"] == [0, 2]


def test_missing_scores_protect_conservatively(monkeypatch):
    clear_block_scores()
    update_block_scores([KivoKVBlockScore(block_id=0, score=0.1, source="test")])
    monkeypatch.setenv("KIVO_KV_RETENTION_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_RETENTION_POLICY", "countsketch_online")
    monkeypatch.setenv("KIVO_KV_RETENTION_KEEP_RECENT_BLOCKS", "1")
    monkeypatch.setenv("KIVO_KV_RETENTION_MAX_FULL_BLOCKS", "3")
    monkeypatch.setenv("KIVO_KV_RETENTION_MIN_BLOCKS_BEFORE_ACTION", "0")
    monkeypatch.setenv("KIVO_KV_RETENTION_ACTION", "free_candidates")
    manager, pool = _make_manager()

    manager.remove_skipped_blocks("req", total_computed_tokens=4)

    assert pool.freed == [[0]]
    summary = manager.get_last_kivo_retention_mutation_summary()
    assert summary is not None
    assert summary["score_missing_count"] == 4
    assert summary["protected_count"] == 3


def test_invalid_action_fails_closed(monkeypatch):
    clear_block_scores()
    monkeypatch.setenv("KIVO_KV_RETENTION_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_RETENTION_POLICY", "recent_only")
    monkeypatch.setenv("KIVO_KV_RETENTION_KEEP_RECENT_BLOCKS", "1")
    monkeypatch.setenv("KIVO_KV_RETENTION_MAX_FULL_BLOCKS", "2")
    monkeypatch.setenv("KIVO_KV_RETENTION_MIN_BLOCKS_BEFORE_ACTION", "0")
    monkeypatch.setenv("KIVO_KV_RETENTION_ACTION", "bad_action")
    manager, pool = _make_manager()

    manager.remove_skipped_blocks("req", total_computed_tokens=4)

    assert pool.freed == [[]]
    summary = manager.get_last_kivo_retention_mutation_summary()
    assert summary is not None
    assert summary["actual_freed_candidate_count"] == 0
    assert summary["fail_closed_reason_counts"] == {"unsupported_action_fail_closed": 1}


def test_invalid_policy_fails_closed(monkeypatch):
    clear_block_scores()
    monkeypatch.setenv("KIVO_KV_RETENTION_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_RETENTION_POLICY", "bad_policy")
    monkeypatch.setenv("KIVO_KV_RETENTION_KEEP_RECENT_BLOCKS", "1")
    monkeypatch.setenv("KIVO_KV_RETENTION_MAX_FULL_BLOCKS", "2")
    monkeypatch.setenv("KIVO_KV_RETENTION_MIN_BLOCKS_BEFORE_ACTION", "0")
    monkeypatch.setenv("KIVO_KV_RETENTION_ACTION", "free_candidates")
    manager, pool = _make_manager()

    manager.remove_skipped_blocks("req", total_computed_tokens=4)

    assert pool.freed == [[]]
    summary = manager.get_last_kivo_retention_mutation_summary()
    assert summary is not None
    assert summary["actual_freed_candidate_count"] == 0
    assert summary["fail_closed_reason_counts"] == {"invalid_policy_fail_closed": 1}

