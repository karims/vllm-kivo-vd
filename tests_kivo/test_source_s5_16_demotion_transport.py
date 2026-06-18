from __future__ import annotations

from dataclasses import dataclass

import pytest

from vllm.v1.core.kivo_demotion_command import (
    KivoCoreDemotionConfig,
    KivoDemotionCommand,
    KivoDemotionCommandResult,
)
from vllm.v1.core.kivo_demotion_counters import (
    get_kivo_demotion_counters_snapshot,
    reset_kivo_demotion_counters,
)
from vllm.v1.core.kivo_demotion_transport import (
    KivoDemotionTransportConfig,
    KivoDemotionTransportEnvelope,
    apply_kivo_demotion_transport_envelopes,
)
from vllm.v1.worker.kivo_runtime_block_table_apply import (
    build_kivo_demotion_command_for_runtime_row,
    maybe_build_kivo_demotion_command_after_runtime_apply,
    reset_kivo_demotion_command_dedupe_state_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_worker_dedupe_state():
    reset_kivo_demotion_command_dedupe_state_for_tests()
    yield
    reset_kivo_demotion_command_dedupe_state_for_tests()


def _transport_config(
    *,
    enabled: bool = True,
    action: str = "apply_core_mark_demoted",
) -> KivoDemotionTransportConfig:
    return KivoDemotionTransportConfig(enabled=enabled, action=action)


def _core_config() -> KivoCoreDemotionConfig:
    return KivoCoreDemotionConfig(
        enabled=True,
        action="mark_demoted_only",
        require_block_table_applied=True,
        require_slot_mapping_refresh=True,
    )


def test_disabled_transport_rejects_noop():
    result = apply_kivo_demotion_transport_envelopes(
        kv_cache_manager=None,
        envelopes=(),
        config=_transport_config(enabled=False),
        core_config=_core_config(),
    )
    assert result.enabled is False
    assert result.applied_count == 0


def test_worker_summary_with_no_applied_row_produces_no_command():
    export = build_kivo_demotion_command_for_runtime_row(
        request_id="req0",
        visible_before_block_ids=(10, 11, 12, 13),
        visible_after_block_ids=(12, 13),
        candidate_demote_block_ids=(10, 11),
        protected_block_ids=(12, 13),
        block_table_applied=False,
        slot_mapping_refresh_guaranteed=True,
    )
    assert export.command is None
    assert export.blocker_reasons["block_table_not_applied"] == 1


def test_worker_summary_with_empty_demote_ids_produces_no_command():
    export = build_kivo_demotion_command_for_runtime_row(
        request_id="req0",
        visible_before_block_ids=(10, 11, 12, 13),
        visible_after_block_ids=(12, 13),
        candidate_demote_block_ids=(),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
    )
    assert export.command is None
    assert export.blocker_reasons["empty_candidate_demote_ids"] == 1


def test_valid_worker_summary_produces_kivo_demotion_command():
    export = build_kivo_demotion_command_for_runtime_row(
        request_id="req0",
        visible_before_block_ids=(10, 11, 12, 13),
        visible_after_block_ids=(12, 13),
        candidate_demote_block_ids=(10, 11),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
    )
    assert isinstance(export.command, KivoDemotionCommand)
    assert export.command.candidate_demote_block_ids == (10, 11)


def test_worker_dedupe_drops_repeated_block_ids(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    reset_kivo_demotion_counters()
    first = build_kivo_demotion_command_for_runtime_row(
        request_id="req0",
        visible_before_block_ids=(10, 11, 12, 13),
        visible_after_block_ids=(12, 13),
        candidate_demote_block_ids=(10, 11),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
    )
    second = build_kivo_demotion_command_for_runtime_row(
        request_id="req0",
        visible_before_block_ids=(10, 11, 12, 13),
        visible_after_block_ids=(12, 13),
        candidate_demote_block_ids=(10, 11),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
    )
    snapshot = get_kivo_demotion_counters_snapshot()
    assert first.command is not None
    assert second.command is None
    assert second.blocker_reasons["dedupe_empty_after_drop"] == 1
    assert snapshot["demotion_command_dedupe_input_blocks"] == 4
    assert snapshot["demotion_command_dedupe_dropped_blocks"] == 2
    assert snapshot["demotion_command_dedupe_output_blocks"] == 2
    assert snapshot["demotion_command_dedupe_empty_after_drop"] == 1
    assert snapshot["demotion_export_wall_time_seconds"] > 0
    assert snapshot["demotion_export_max_call_wall_time_seconds"] > 0


def test_decode_only_request_is_recorded_without_guessing_phase(monkeypatch):
    monkeypatch.setenv("KIVO_KV_DEMOTION_COUNTERS_ENABLE", "1")
    monkeypatch.setenv("KIVO_KV_DEMOTION_DECODE_ONLY", "1")
    reset_kivo_demotion_counters()
    export = maybe_build_kivo_demotion_command_after_runtime_apply(
        request_id="req0",
        visible_before_block_ids=(10, 11, 12, 13),
        visible_after_block_ids=(12, 13),
        candidate_demote_block_ids=(10, 11),
        protected_block_ids=(12, 13),
        apply_summary_present=True,
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
        filtered_row_changed=True,
        keep_recent_blocks=2,
        policy="recent_only",
    )
    snapshot = get_kivo_demotion_counters_snapshot()
    assert export.command is not None
    assert snapshot["decode_only_requested"] == 1
    assert snapshot["decode_only_supported"] == 0
    assert snapshot["demotion_skipped_not_decode_phase"] == 0


@dataclass
class FakeKVCacheManager:
    req_to_blocks: dict[str, list[int]]
    free_called: bool = False

    def __init__(self) -> None:
        self.req_to_blocks = {"req0": [10, 11, 12, 13]}
        self.free_called = False

    def apply_kivo_demotion_command(
        self,
        command: KivoDemotionCommand,
        *,
        config: KivoCoreDemotionConfig | None = None,
    ) -> KivoDemotionCommandResult:
        del config
        if command.request_id == "reject":
            return KivoDemotionCommandResult(
                enabled=True,
                request_id=command.request_id,
                accepted=False,
                marked_demoted_block_ids=(),
                rejected_block_ids=command.candidate_demote_block_ids,
                blocker_reasons={"rejected_for_test": 1},
                removes_from_req_to_blocks=False,
                frees_to_pool=False,
            )
        return KivoDemotionCommandResult(
            enabled=True,
            request_id=command.request_id,
            accepted=True,
            marked_demoted_block_ids=command.candidate_demote_block_ids,
            rejected_block_ids=(),
            blocker_reasons={},
            removes_from_req_to_blocks=False,
            frees_to_pool=False,
        )


def test_command_transport_applies_through_fake_kv_cache_manager():
    export = build_kivo_demotion_command_for_runtime_row(
        request_id="req0",
        visible_before_block_ids=(10, 11, 12, 13),
        visible_after_block_ids=(12, 13),
        candidate_demote_block_ids=(10, 11),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
    )
    envelope = KivoDemotionTransportEnvelope(
        request_id="req0",
        command=export.command,
        source="worker_pre_slot_mapping",
    )
    manager = FakeKVCacheManager()
    result = apply_kivo_demotion_transport_envelopes(
        kv_cache_manager=manager,
        envelopes=(envelope,),
        config=_transport_config(),
        core_config=_core_config(),
    )
    assert result.applied_count == 1
    assert result.rejected_count == 0


def test_rejected_command_result_is_counted():
    command = KivoDemotionCommand(
        request_id="reject",
        visible_before_block_ids=(10, 11, 12, 13),
        visible_after_block_ids=(12, 13),
        candidate_demote_block_ids=(10, 11),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
    )
    envelope = KivoDemotionTransportEnvelope(
        request_id="reject",
        command=command,
        source="worker_pre_slot_mapping",
    )
    manager = FakeKVCacheManager()
    result = apply_kivo_demotion_transport_envelopes(
        kv_cache_manager=manager,
        envelopes=(envelope,),
        config=_transport_config(),
        core_config=_core_config(),
    )
    assert result.rejected_count == 1
    assert result.blocker_reasons["rejected_for_test"] == 1


def test_no_req_to_blocks_removal_occurs():
    export = build_kivo_demotion_command_for_runtime_row(
        request_id="req0",
        visible_before_block_ids=(10, 11, 12, 13),
        visible_after_block_ids=(12, 13),
        candidate_demote_block_ids=(10, 11),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
    )
    manager = FakeKVCacheManager()
    before = dict(manager.req_to_blocks)
    envelope = KivoDemotionTransportEnvelope(
        request_id="req0",
        command=export.command,
        source="worker_pre_slot_mapping",
    )
    apply_kivo_demotion_transport_envelopes(
        kv_cache_manager=manager,
        envelopes=(envelope,),
        config=_transport_config(),
        core_config=_core_config(),
    )
    assert manager.req_to_blocks == before


def test_no_free_to_pool_occurs():
    export = build_kivo_demotion_command_for_runtime_row(
        request_id="req0",
        visible_before_block_ids=(10, 11, 12, 13),
        visible_after_block_ids=(12, 13),
        candidate_demote_block_ids=(10, 11),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
    )
    manager = FakeKVCacheManager()
    envelope = KivoDemotionTransportEnvelope(
        request_id="req0",
        command=export.command,
        source="worker_pre_slot_mapping",
    )
    apply_kivo_demotion_transport_envelopes(
        kv_cache_manager=manager,
        envelopes=(envelope,),
        config=_transport_config(),
        core_config=_core_config(),
    )
    assert manager.free_called is False


def test_missing_runtime_transport_path_is_documented_as_blocker():
    result = apply_kivo_demotion_transport_envelopes(
        kv_cache_manager=None,
        envelopes=(
            KivoDemotionTransportEnvelope(
                request_id="req0",
                command=KivoDemotionCommand(
                    request_id="req0",
                    visible_before_block_ids=(10, 11, 12, 13),
                    visible_after_block_ids=(12, 13),
                    candidate_demote_block_ids=(10, 11),
                    protected_block_ids=(12, 13),
                    block_table_applied=True,
                    slot_mapping_refresh_guaranteed=True,
                ),
                source="worker_pre_slot_mapping",
            ),
        ),
        config=_transport_config(),
        core_config=_core_config(),
    )
    assert result.blocker_reasons["kv_cache_manager_unavailable"] == 1
