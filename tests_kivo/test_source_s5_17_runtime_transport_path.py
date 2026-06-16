from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.core.kivo_demotion_command import (
    KivoCoreDemotionConfig,
    KivoDemotionCommand,
    KivoDemotionCommandResult,
)
from vllm.v1.core.kivo_demotion_transport import (
    KivoDemotionTransportConfig,
    KivoDemotionTransportEnvelope,
    apply_kivo_demotion_transport_from_model_runner_output,
)
from vllm.v1.outputs import ModelRunnerOutput


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


def _make_command(request_id: str = "req0") -> KivoDemotionCommand:
    return KivoDemotionCommand(
        request_id=request_id,
        visible_before_block_ids=(10, 11, 12, 13),
        visible_after_block_ids=(12, 13),
        candidate_demote_block_ids=(10, 11),
        protected_block_ids=(12, 13),
        block_table_applied=True,
        slot_mapping_refresh_guaranteed=True,
    )


def test_model_runner_output_transport_field_is_backward_compatible():
    output = ModelRunnerOutput(req_ids=[], req_id_to_index={})
    assert output.kivo_demotion_transport_envelopes == ()


def test_disabled_transport_adds_no_output_payload():
    output = ModelRunnerOutput(req_ids=["req0"], req_id_to_index={"req0": 0})
    result = apply_kivo_demotion_transport_from_model_runner_output(
        kv_cache_manager=FakeKVCacheManager(),
        model_runner_output=output,
        config=_transport_config(enabled=False),
        core_config=_core_config(),
    )
    assert result.enabled is False
    assert result.applied_count == 0


def test_core_side_intake_applies_valid_envelope_from_model_runner_output():
    envelope = KivoDemotionTransportEnvelope(
        request_id="req0",
        command=_make_command(),
        source="worker_pre_slot_mapping",
    )
    output = ModelRunnerOutput(
        req_ids=["req0"],
        req_id_to_index={"req0": 0},
        kivo_demotion_transport_envelopes=(envelope,),
    )
    manager = FakeKVCacheManager()
    result = apply_kivo_demotion_transport_from_model_runner_output(
        kv_cache_manager=manager,
        model_runner_output=output,
        config=_transport_config(),
        core_config=_core_config(),
    )
    assert result.applied_count == 1
    assert result.rejected_count == 0
    assert manager.req_to_blocks == {"req0": [10, 11, 12, 13]}
    assert manager.free_called is False


def test_invalid_envelope_is_rejected_through_core_side_intake():
    envelope = KivoDemotionTransportEnvelope(
        request_id="reject",
        command=_make_command("reject"),
        source="worker_pre_slot_mapping",
    )
    output = ModelRunnerOutput(
        req_ids=["reject"],
        req_id_to_index={"reject": 0},
        kivo_demotion_transport_envelopes=(envelope,),
    )
    result = apply_kivo_demotion_transport_from_model_runner_output(
        kv_cache_manager=FakeKVCacheManager(),
        model_runner_output=output,
        config=_transport_config(),
        core_config=_core_config(),
    )
    assert result.rejected_count == 1
    assert result.blocker_reasons["rejected_for_test"] == 1


def test_missing_runtime_transport_path_is_still_fail_closed():
    envelope = KivoDemotionTransportEnvelope(
        request_id="req0",
        command=_make_command(),
        source="worker_pre_slot_mapping",
    )
    output = ModelRunnerOutput(
        req_ids=["req0"],
        req_id_to_index={"req0": 0},
        kivo_demotion_transport_envelopes=(envelope,),
    )
    result = apply_kivo_demotion_transport_from_model_runner_output(
        kv_cache_manager=None,
        model_runner_output=output,
        config=_transport_config(),
        core_config=_core_config(),
    )
    assert result.rejected_count == 1
    assert result.blocker_reasons["kv_cache_manager_unavailable"] == 1
