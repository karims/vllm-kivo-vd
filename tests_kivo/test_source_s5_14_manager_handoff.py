from __future__ import annotations

from vllm.v1.worker.kivo_kv_manager_handoff import (
    build_kivo_kv_manager_handoff_decision,
)


class FakeManagerWithHelper:
    def __init__(self) -> None:
        self.req_to_blocks = {"req0": [10, 11, 12, 13]}
        self.free_called = False

    def mark_kivo_demoted_blocks_if_safe(self, *args, **kwargs):
        del args, kwargs
        return None


class FakeManagerWithoutHelper:
    def __init__(self) -> None:
        self.req_to_blocks = {"req0": [10, 11, 12, 13]}
        self.free_called = False


def test_missing_manager_fails_closed():
    decision = build_kivo_kv_manager_handoff_decision(
        kv_cache_manager=None,
        request_id="req0",
    )
    assert decision.manager_available is False
    assert decision.safe_to_call_mark_demoted is False
    assert decision.blocker_reasons["kv_cache_manager_unavailable"] == 1


def test_wrong_manager_type_fails_closed():
    decision = build_kivo_kv_manager_handoff_decision(
        kv_cache_manager=object(),
        request_id="req0",
    )
    assert decision.manager_available is True
    assert decision.supports_kivo_demotion_mark is False
    assert decision.safe_to_call_mark_demoted is False
    assert decision.blocker_reasons["kv_cache_manager_missing_mark_helper"] == 1


def test_fake_manager_with_helper_is_accepted():
    manager = FakeManagerWithHelper()
    decision = build_kivo_kv_manager_handoff_decision(
        kv_cache_manager=manager,
        request_id="req0",
    )
    assert decision.manager_available is True
    assert decision.supports_kivo_demotion_mark is True
    assert decision.request_id_compatible is True
    assert decision.safe_to_call_mark_demoted is True


def test_fake_manager_without_helper_is_rejected():
    manager = FakeManagerWithoutHelper()
    decision = build_kivo_kv_manager_handoff_decision(
        kv_cache_manager=manager,
        request_id="req0",
    )
    assert decision.supports_kivo_demotion_mark is False
    assert decision.safe_to_call_mark_demoted is False
    assert decision.blocker_reasons["kv_cache_manager_missing_mark_helper"] == 1


def test_request_id_mismatch_fails_closed_if_detectable():
    manager = FakeManagerWithHelper()
    decision = build_kivo_kv_manager_handoff_decision(
        kv_cache_manager=manager,
        request_id="missing",
    )
    assert decision.request_id_compatible is False
    assert decision.safe_to_call_mark_demoted is False
    assert decision.blocker_reasons["request_id_not_present_in_manager"] == 1


def test_adapter_never_frees_blocks():
    manager = FakeManagerWithHelper()
    build_kivo_kv_manager_handoff_decision(
        kv_cache_manager=manager,
        request_id="req0",
    )
    assert manager.free_called is False


def test_adapter_never_removes_from_req_to_blocks():
    manager = FakeManagerWithHelper()
    before = dict(manager.req_to_blocks)
    build_kivo_kv_manager_handoff_decision(
        kv_cache_manager=manager,
        request_id="req0",
    )
    assert manager.req_to_blocks == before
