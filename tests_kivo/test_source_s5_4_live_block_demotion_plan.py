from __future__ import annotations

from types import SimpleNamespace

from vllm.v1.core.kivo_kv_block_score_store import (
    KivoKVBlockScore,
    clear_block_scores,
    update_block_scores,
)
from vllm.v1.core.kivo_kv_live_block_plan import (
    KivoKVLiveBlockPlanConfig,
    build_kivo_live_block_plan,
)
from vllm.v1.core.kivo_kv_retention_policy import (
    KivoKVRetentionConfig,
    decide_kv_retention,
)
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager


class FakeBlock:
    def __init__(self, block_id: int, ref_cnt: int = 1):
        self.block_id = block_id
        self.ref_cnt = ref_cnt
        self.is_null = False

    def __eq__(self, other):
        return self is other


class FakePool:
    def __init__(self):
        self.null_block = object()

    def free_blocks(self, blocks):
        del blocks


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


def _build_plan(
    block_ids,
    scores,
    *,
    live_config,
    retention_config,
    shared_block_ids=(),
    block_table_sync_available=False,
):
    retention_decision = decide_kv_retention(
        block_ids,
        scores,
        request_id="req",
        config=retention_config,
    )
    return build_kivo_live_block_plan(
        block_ids,
        retention_decision,
        request_id="req",
        shared_block_ids=shared_block_ids,
        block_table_sync_available=block_table_sync_available,
        ownership_mutation_available=True,
        config=live_config,
    )


def _make_manager():
    manager = FakeManager(
        kv_cache_spec=SimpleNamespace(block_size=1),
        block_pool=FakePool(),
        enable_caching=True,
        kv_cache_group_id=0,
    )
    return manager


def test_default_disabled_returns_no_live_demotion_candidates():
    plan = _build_plan(
        [1, 2, 3, 4],
        {},
        live_config=KivoKVLiveBlockPlanConfig(
            False, "plan_live_demotion_only", True, 1, 0
        ),
        retention_config=KivoKVRetentionConfig(
            False, "recent_only", 1, 2, 0, "plan_only"
        ),
    )
    assert plan.enabled is False
    assert plan.candidate_demote_block_ids == ()


def test_plan_only_computes_candidates_without_mutating():
    plan = _build_plan(
        [1, 2, 3, 4, 5],
        {},
        live_config=KivoKVLiveBlockPlanConfig(
            True, "plan_live_demotion_only", False, 2, 0
        ),
        retention_config=KivoKVRetentionConfig(
            True, "recent_only", 2, 3, 0, "plan_only"
        ),
    )
    assert plan.safe_to_apply is False
    assert plan.keep_block_ids == (3, 4, 5)
    assert plan.candidate_demote_block_ids == (1, 2)


def test_recent_protected_blocks_are_never_candidates():
    plan = _build_plan(
        [10, 11, 12, 13],
        {},
        live_config=KivoKVLiveBlockPlanConfig(
            True, "plan_live_demotion_only", False, 2, 0
        ),
        retention_config=KivoKVRetentionConfig(
            True, "recent_only", 2, 2, 0, "plan_only"
        ),
    )
    assert 12 in plan.protected_block_ids
    assert 13 in plan.protected_block_ids
    assert 12 not in plan.candidate_demote_block_ids
    assert 13 not in plan.candidate_demote_block_ids


def test_countsketch_online_keeps_high_score_older_blocks():
    plan = _build_plan(
        [10, 11, 12, 13, 14],
        {10: 0.1, 11: 0.9, 12: 0.3},
        live_config=KivoKVLiveBlockPlanConfig(
            True, "plan_live_demotion_only", False, 2, 0
        ),
        retention_config=KivoKVRetentionConfig(
            True, "countsketch_online", 2, 4, 0, "plan_only"
        ),
    )
    assert plan.keep_block_ids == (11, 12, 13, 14)
    assert plan.candidate_demote_block_ids == (10,)


def test_missing_scores_are_protected_or_blocked():
    plan = _build_plan(
        [1, 2, 3, 4],
        {1: 0.2},
        live_config=KivoKVLiveBlockPlanConfig(
            True, "plan_live_demotion_only", False, 1, 0
        ),
        retention_config=KivoKVRetentionConfig(
            True, "countsketch_online", 1, 2, 0, "plan_only"
        ),
    )
    assert 2 in plan.protected_block_ids
    assert 3 in plan.protected_block_ids
    assert 4 in plan.protected_block_ids


def test_plan_blocked_if_block_table_sync_required_but_unavailable():
    plan = _build_plan(
        [1, 2, 3, 4],
        {},
        live_config=KivoKVLiveBlockPlanConfig(
            True, "apply_live_demotion_if_safe", True, 1, 0
        ),
        retention_config=KivoKVRetentionConfig(
            True, "recent_only", 1, 2, 0, "plan_only"
        ),
        block_table_sync_available=False,
    )
    assert plan.safe_to_apply is False
    assert plan.blocker_reasons == {"block_table_sync_unavailable": 1}


def test_invalid_action_fails_closed():
    plan = _build_plan(
        [1, 2, 3],
        {},
        live_config=KivoKVLiveBlockPlanConfig(True, "bad_action", True, 1, 0),
        retention_config=KivoKVRetentionConfig(
            True, "recent_only", 1, 2, 0, "plan_only"
        ),
    )
    assert plan.safe_to_apply is False
    assert plan.blocker_reasons == {"invalid_action_fail_closed": 1}


def test_natural_order_of_keep_blocks_is_preserved():
    plan = _build_plan(
        [9, 8, 7, 6],
        {},
        live_config=KivoKVLiveBlockPlanConfig(
            True, "plan_live_demotion_only", False, 2, 0
        ),
        retention_config=KivoKVRetentionConfig(
            True, "recent_only", 2, 2, 0, "plan_only"
        ),
    )
    assert plan.keep_block_ids == (7, 6)


def test_plan_reports_blocker_reasons():
    plan = _build_plan(
        [1, 2, 3, 4],
        {},
        live_config=KivoKVLiveBlockPlanConfig(
            True, "apply_live_demotion_if_safe", True, 1, 8
        ),
        retention_config=KivoKVRetentionConfig(
            True, "recent_only", 1, 2, 0, "plan_only"
        ),
        shared_block_ids=(2,),
        block_table_sync_available=False,
    )
    assert plan.blocker_reasons["below_min_blocks_before_action"] == 1
    assert plan.blocker_reasons["block_table_sync_unavailable"] == 1
    assert plan.blocker_reasons["shared_block_refcnt"] == 1


def test_manager_can_build_plan_without_changing_default_behavior(monkeypatch):
    clear_block_scores()
    update_block_scores(
        [
            KivoKVBlockScore(block_id=0, score=0.1, source="test"),
            KivoKVBlockScore(block_id=1, score=0.9, source="test"),
        ]
    )
    monkeypatch.setenv("KIVO_KV_RETENTION_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_RETENTION_POLICY", "countsketch_online")
    monkeypatch.setenv("KIVO_KV_RETENTION_MAX_FULL_BLOCKS", "3")
    monkeypatch.setenv("KIVO_KV_LIVE_DEMOTION_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_LIVE_DEMOTION_ACTION", "plan_live_demotion_only")
    monkeypatch.setenv("KIVO_KV_LIVE_DEMOTION_PROTECT_RECENT_BLOCKS", "1")
    monkeypatch.setenv("KIVO_KV_LIVE_DEMOTION_MIN_BLOCKS_BEFORE_ACTION", "0")
    manager = _make_manager()
    manager.req_to_blocks["req"] = [
        FakeBlock(0),
        FakeBlock(1),
        FakeBlock(2, ref_cnt=2),
        FakeBlock(3),
    ]

    plan = manager.build_kivo_live_block_plan("req")

    assert manager.get_request_block_ids("req") == (0, 1, 2, 3)
    assert plan.enabled is True
    assert plan.safe_to_apply is False
    assert plan.blocker_reasons["block_table_sync_unavailable"] == 1
