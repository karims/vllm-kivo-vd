# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from enum import Enum


class KivoVDSketchType(str, Enum):
    NONE = "none"
    RANDOM_PROJECTION = "random_projection"
    COUNT_SKETCH = "count_sketch"
    SRHT = "srht"
    BIDIAGONAL_SIGN = "bidiagonal_sign"
    BIDIAGONAL = "bidiagonal"
    VARIATION_DIMINISHING = "variation_diminishing"


@dataclass(slots=True)
class KivoVDSketchConfig:
    enabled: bool = False
    sketch_dim: int = 64
    sketch_type: KivoVDSketchType = KivoVDSketchType.NONE
    max_blocks_per_query: int | None = None
    recent_window_blocks: int = 0


@dataclass(slots=True)
class KivoVDBlockSketch:
    request_id: str
    block_id: int
    logical_block_idx: int
    kv_group_id: int
    layer_id: int | None
    sketch_dim: int
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class KivoVDBlockScore:
    block_id: int
    score: float
    source: str


@dataclass(slots=True)
class KivoVDRoutingDecision:
    request_id: str
    selected_block_ids: list[int]
    recent_block_ids: list[int]
    skipped_block_ids: list[int]
    reason: str


@dataclass(slots=True)
class KivoVDSketchIndex:
    """Metadata-only sketch index.

    Phase 1 placeholder: stores only small block metadata, no key/value tensors.
    """

    config: KivoVDSketchConfig = field(default_factory=KivoVDSketchConfig)
    _by_request: dict[str, dict[tuple[int, int], KivoVDBlockSketch]] = field(
        default_factory=dict
    )

    def add_or_update_block_sketch(self, sketch: KivoVDBlockSketch) -> None:
        request_entries = self._by_request.setdefault(sketch.request_id, {})
        request_entries[(sketch.kv_group_id, sketch.block_id)] = sketch

    def remove_request(self, request_id: str) -> None:
        self._by_request.pop(request_id, None)

    def get_request_block_sketches(self, request_id: str) -> list[KivoVDBlockSketch]:
        sketches = self._by_request.get(request_id, {})
        return sorted(
            sketches.values(),
            key=lambda s: (s.logical_block_idx, s.kv_group_id, s.block_id),
        )

    def score_blocks_placeholder(
        self,
        request_id: str,
        candidate_block_ids: list[int] | None = None,
        source: str = "placeholder",
    ) -> list[KivoVDBlockScore]:
        """Return deterministic dummy scores based on block metadata only."""
        sketches = self.get_request_block_sketches(request_id)
        if candidate_block_ids is not None:
            candidates = set(candidate_block_ids)
            sketches = [s for s in sketches if s.block_id in candidates]

        scores = []
        for sketch in sketches:
            # Deterministic metadata-only score (no tensor access).
            score = (
                (sketch.block_id % 10007) / 10007.0
                + (sketch.logical_block_idx % 997) / 1_000_000.0
                + (sketch.kv_group_id % 97) / 10_000_000.0
            )
            scores.append(
                KivoVDBlockScore(
                    block_id=sketch.block_id,
                    score=score,
                    source=source,
                )
            )

        scores.sort(key=lambda x: (-x.score, x.block_id))
        return scores

    def route_blocks_placeholder(
        self,
        request_id: str,
        recent_block_ids: list[int] | None = None,
        max_blocks_per_query: int | None = None,
    ) -> KivoVDRoutingDecision:
        """Select recent blocks and then top dummy-scored blocks.

        Phase 1 placeholder: this does not affect scheduler behavior.
        """
        all_sketches = self.get_request_block_sketches(request_id)
        all_block_ids = [s.block_id for s in all_sketches]

        effective_limit = (
            max_blocks_per_query
            if max_blocks_per_query is not None
            else self.config.max_blocks_per_query
        )
        if effective_limit is None:
            effective_limit = len(all_block_ids)

        selected: list[int] = []
        recent_selected: list[int] = []
        if recent_block_ids:
            max_recent = self.config.recent_window_blocks
            for block_id in recent_block_ids:
                if block_id not in all_block_ids:
                    continue
                if block_id in selected:
                    continue
                if max_recent > 0 and len(recent_selected) >= max_recent:
                    break
                selected.append(block_id)
                recent_selected.append(block_id)
                if len(selected) >= effective_limit:
                    break

        if len(selected) < effective_limit:
            for score in self.score_blocks_placeholder(request_id):
                if score.block_id in selected:
                    continue
                selected.append(score.block_id)
                if len(selected) >= effective_limit:
                    break

        skipped = [block_id for block_id in all_block_ids if block_id not in selected]
        return KivoVDRoutingDecision(
            request_id=request_id,
            selected_block_ids=selected,
            recent_block_ids=recent_selected,
            skipped_block_ids=skipped,
            reason="placeholder_metadata_only",
        )

    def reset(self) -> None:
        self._by_request.clear()
