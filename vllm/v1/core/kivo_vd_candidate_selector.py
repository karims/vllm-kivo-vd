# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any

from vllm.v1.core.kivo_vd_sketch import (
    KivoVDRoutingDecision,
    KivoVDSketchIndex,
    KivoVDSketchType,
)


@dataclass(slots=True)
class KivoVDCandidateSelectorConfig:
    recent_window_blocks: int = 8
    candidate_budget_blocks: int = 16
    min_candidate_blocks: int = 4
    include_recent_blocks: bool = True
    fallback_to_all_on_empty: bool = True
    sketch_type: KivoVDSketchType = KivoVDSketchType.COUNT_SKETCH
    sketch_dim: int = 64


class KivoVDCandidateSelector:
    """Dry-run candidate policy using metadata-only sketch index placeholders."""

    def __init__(self, config: KivoVDCandidateSelectorConfig | None = None) -> None:
        self.config = config or KivoVDCandidateSelectorConfig()

    def _infer_recent_block_ids(
        self,
        request_id: str,
        sketch_index: KivoVDSketchIndex,
    ) -> list[int]:
        sketches = sketch_index.get_request_block_sketches(request_id)
        if not sketches:
            return []
        # Recent means highest logical block index per request.
        by_order = sorted(
            sketches,
            key=lambda s: (s.logical_block_idx, s.kv_group_id, s.block_id),
        )
        limit = max(0, self.config.recent_window_blocks)
        if limit == 0:
            return []
        tail = by_order[-limit:]
        recent: list[int] = []
        for s in tail:
            if s.block_id not in recent:
                recent.append(s.block_id)
        return recent

    def select_candidates(
        self,
        request_id: str,
        query_metadata_or_sketch: Any,
        sketch_index: KivoVDSketchIndex,
    ) -> KivoVDRoutingDecision:
        del query_metadata_or_sketch  # phase 2.1 dry-run only

        all_sketches = sketch_index.get_request_block_sketches(request_id)
        all_block_ids = [s.block_id for s in all_sketches]
        if not all_block_ids:
            return KivoVDRoutingDecision(
                request_id=request_id,
                selected_block_ids=[],
                recent_block_ids=[],
                skipped_block_ids=[],
                reason="empty_request_blocks",
            )

        budget = max(self.config.min_candidate_blocks, self.config.candidate_budget_blocks)
        budget = min(budget, len(all_block_ids))

        selected: list[int] = []
        recent_block_ids: list[int] = []
        if self.config.include_recent_blocks:
            recent_block_ids = self._infer_recent_block_ids(request_id, sketch_index)
            for block_id in recent_block_ids:
                if block_id in all_block_ids and block_id not in selected:
                    selected.append(block_id)
                    if len(selected) >= budget:
                        break

        if len(selected) < budget:
            for score in sketch_index.score_blocks_placeholder(
                request_id,
                source=f"selector_{self.config.sketch_type.value}_{self.config.sketch_dim}",
            ):
                if score.block_id in selected:
                    continue
                selected.append(score.block_id)
                if len(selected) >= budget:
                    break

        if not selected and self.config.fallback_to_all_on_empty:
            selected = list(all_block_ids)

        skipped = [block_id for block_id in all_block_ids if block_id not in selected]
        return KivoVDRoutingDecision(
            request_id=request_id,
            selected_block_ids=selected,
            recent_block_ids=[b for b in recent_block_ids if b in selected],
            skipped_block_ids=skipped,
            reason="dry_run_metadata_selector",
        )
