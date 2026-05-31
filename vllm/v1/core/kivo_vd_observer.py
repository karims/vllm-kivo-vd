# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence


class KivoVDObserver:
    """Phase 0 Kivo-VD observer hook points (no-op implementation)."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self.num_before_allocate_calls = 0
        self.num_after_allocate_calls = 0
        self.num_free_request_calls = 0

    def on_before_allocate_slots(
        self,
        request_id: str,
        num_new_tokens: int,
        num_lookahead_tokens: int,
    ) -> None:
        self.num_before_allocate_calls += 1
        return

    def on_after_allocate_slots(
        self,
        request_id: str,
        block_ids_by_group: tuple[list[int], ...] | None,
    ) -> None:
        self.num_after_allocate_calls += 1
        return

    def on_free_request(
        self,
        request_id: str,
        block_ids_by_group: tuple[list[int], ...],
    ) -> None:
        self.num_free_request_calls += 1
        return

    def on_build_attention_metadata(
        self,
        num_reqs: int,
        num_tokens: int,
        block_table_shape: Sequence[int],
    ) -> None:
        return


def create_kivo_vd_observer(enable_kivo_vd: bool) -> KivoVDObserver | None:
    if not enable_kivo_vd:
        return None
    return KivoVDObserver(enabled=True)
