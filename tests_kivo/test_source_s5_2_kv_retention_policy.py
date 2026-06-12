from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm.v1.core.kivo_kv_block_score_store import (
    KivoKVBlockScore,
    clear_block_scores,
    get_block_scores,
    get_score_store_summary,
    update_block_scores,
)
from vllm.v1.core.kivo_kv_retention_policy import (
    KivoKVRetentionConfig,
    decide_kv_retention,
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


def test_disabled_retention_keeps_all():
    decision = decide_kv_retention(
        [1, 2, 3],
        {},
        request_id="req",
        config=KivoKVRetentionConfig(False, "recent_only", 1, 2, 0, "plan_only"),
    )
    assert decision.keep_block_ids == (1, 2, 3)
    assert decision.candidate_drop_block_ids == ()


def test_recent_only_keeps_latest_n():
    decision = decide_kv_retention(
        [1, 2, 3, 4],
        {},
        request_id="req",
        config=KivoKVRetentionConfig(True, "recent_only", 2, 4, 0, "plan_only"),
    )
    assert decision.keep_block_ids == (1, 2, 3, 4)
    assert decision.protected_block_ids == (3, 4)


def test_recent_only_respects_max_full_blocks():
    decision = decide_kv_retention(
        [1, 2, 3, 4, 5],
        {},
        request_id="req",
        config=KivoKVRetentionConfig(True, "recent_only", 2, 3, 0, "plan_only"),
    )
    assert decision.keep_block_ids == (3, 4, 5)
    assert decision.candidate_drop_block_ids == (1, 2)


def test_countsketch_online_keeps_recent_plus_top_scored_older():
    decision = decide_kv_retention(
        [10, 11, 12, 13, 14],
        {10: 0.1, 11: 0.9, 12: 0.3},
        request_id="req",
        config=KivoKVRetentionConfig(
            True, "countsketch_online", 2, 4, 0, "plan_only"
        ),
    )
    assert decision.keep_block_ids == (11, 12, 13, 14)
    assert decision.candidate_drop_block_ids == (10,)
    assert decision.score_available_count == 3


def test_missing_scores_are_protected_conservatively():
    decision = decide_kv_retention(
        [1, 2, 3, 4],
        {1: 0.2},
        request_id="req",
        config=KivoKVRetentionConfig(
            True, "countsketch_online", 1, 2, 0, "plan_only"
        ),
    )
    assert 2 in decision.protected_block_ids
    assert 3 in decision.protected_block_ids
    assert 4 in decision.protected_block_ids
    assert decision.score_missing_count == 2


def test_invalid_policy_fails_closed():
    decision = decide_kv_retention(
        [1, 2, 3],
        {},
        request_id="req",
        config=KivoKVRetentionConfig(True, "bad_policy", 1, 2, 0, "plan_only"),
    )
    assert decision.keep_block_ids == (1, 2, 3)
    assert decision.candidate_drop_block_ids == ()
    assert decision.reason_counts == {"invalid_policy_fail_closed": 1}


def test_retention_decision_does_not_mutate_inputs():
    block_ids = [1, 2, 3]
    score_map = {1: 0.1, 2: 0.2}
    before_ids = block_ids.copy()
    before_scores = dict(score_map)
    decide_kv_retention(
        block_ids,
        score_map,
        request_id="req",
        config=KivoKVRetentionConfig(
            True, "countsketch_online", 1, 2, 0, "plan_only"
        ),
    )
    assert block_ids == before_ids
    assert score_map == before_scores


def test_score_store_update_get_and_clear():
    clear_block_scores()
    update_block_scores(
        [
            KivoKVBlockScore(block_id=7, score=0.5, source="test"),
            KivoKVBlockScore(block_id=8, score=0.9, source="test"),
        ]
    )
    assert get_block_scores([7, 8, 9]) == {7: 0.5, 8: 0.9}
    summary = get_score_store_summary()
    assert summary["entry_count"] == 2
    clear_block_scores()
    assert get_block_scores([7, 8]) == {}


def test_score_store_remains_bounded(monkeypatch):
    clear_block_scores()
    monkeypatch.setenv("KIVO_KV_SCORE_STORE_MAX_ENTRIES", "2")
    update_block_scores(
        [
            KivoKVBlockScore(block_id=1, score=0.1, source="test"),
            KivoKVBlockScore(block_id=2, score=0.2, source="test"),
            KivoKVBlockScore(block_id=3, score=0.3, source="test"),
        ]
    )
    summary = get_score_store_summary()
    assert summary["entry_count"] == 2
    assert get_block_scores([1, 2, 3]) == {2: 0.2, 3: 0.3}


def test_integration_light_default_behavior_unchanged(monkeypatch):
    clear_block_scores()
    monkeypatch.delenv("KIVO_KV_RETENTION_ENABLE", raising=False)
    manager, pool = _make_manager()
    manager.remove_skipped_blocks("req", total_computed_tokens=3)
    assert pool.freed == [[2, 1, 0]]
    decision = manager.get_last_kivo_retention_decision()
    assert decision is not None
    assert decision.enabled is False
    assert manager.get_kivo_block_score_store_summary()["entry_count"] == 0


def test_unsupported_action_fails_closed():
    decision = decide_kv_retention(
        [1, 2, 3, 4],
        {1: 0.1, 2: 0.2},
        request_id="req",
        config=KivoKVRetentionConfig(
            True, "countsketch_online", 1, 2, 0, "free_candidates"
        ),
    )
    assert decision.keep_block_ids == (1, 2, 3, 4)
    assert decision.candidate_drop_block_ids == ()
    assert decision.reason_counts == {"unsupported_action_fail_closed": 1}


def test_tensor_observer_score_bridge_updates_store(monkeypatch):
    pytest.importorskip("torch")
    from vllm.v1.worker.kivo_attention_tensor_observer import (
        _maybe_update_retention_score_store,
    )

    clear_block_scores()
    monkeypatch.setenv("KIVO_KV_RETENTION_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_RETENTION_POLICY", "countsketch_online")

    _maybe_update_retention_score_store(
        {
            "policy_name": "shadow_kv_block_sketch",
            "sequence_id": 7,
            "layer_index": 3,
            "sketch_computed": True,
            "block_sketch_sample": [
                {"block_id": 10, "score": 0.25},
                {"block_id": 11, "score": 0.75},
            ],
        }
    )

    assert get_block_scores([10, 11]) == {10: 0.25, 11: 0.75}
    summary = get_score_store_summary()
    assert summary["entry_count"] == 2
    assert summary["sources"] == {"shadow_kv_block_sketch": 2}
