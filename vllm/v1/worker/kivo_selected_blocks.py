# SPDX-License-Identifier: Apache-2.0

"""Source-level Kivo S1 selected-block observation and mutation helpers."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

try:
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID as _PAD_SLOT_ID
except Exception:  # pragma: no cover - defensive fallback
    _PAD_SLOT_ID = -1


def _env_flag(name: str) -> bool:
    return os.getenv(name) == "1"


def _safe_summary(value: Any, depth: int = 0) -> dict[str, Any]:
    try:
        value_type = type(value)
        type_name = f"{value_type.__module__}.{value_type.__qualname__}"
        if value is None or isinstance(value, (str, int, float, bool)):
            return {"type": type_name, "value": value}
        if depth >= 2:
            return {"type": type_name}
        if isinstance(value, dict):
            keys = []
            for index, key in enumerate(value):
                if index >= 24:
                    break
                keys.append(str(key))
            return {"type": type_name, "length": len(value), "keys": keys}
        if isinstance(value, (list, tuple)):
            return {
                "type": type_name,
                "length": len(value),
                "items": [
                    _safe_summary(item, depth + 1)
                    for item in value[:8]
                ],
            }
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        device = getattr(value, "device", None)
        if shape is not None or dtype is not None or device is not None:
            summary = {"type": type_name}
            if shape is not None:
                try:
                    summary["shape"] = [int(item) for item in shape]
                except Exception:
                    summary["shape"] = str(shape)[:120]
            if dtype is not None:
                summary["dtype"] = str(dtype)
            if device is not None:
                summary["device"] = str(device)
            return summary
        return {"type": type_name}
    except Exception as exc:  # pragma: no cover - defensive path
        return {"summary_error": f"{type(exc).__name__}: {exc}"}


def _tensor_info(value: Any) -> dict[str, Any]:
    summary = _safe_summary(value)
    return {
        "present": value is not None,
        "type": summary.get("type"),
        "shape": summary.get("shape"),
        "dtype": summary.get("dtype"),
        "device": summary.get("device"),
    }


def _tensor_numel(value: Any) -> int | None:
    if value is None:
        return None
    numel = getattr(value, "numel", None)
    if callable(numel):
        try:
            return int(numel())
        except Exception:
            return None
    shape = getattr(value, "shape", None)
    if shape is None:
        try:
            return len(value)
        except Exception:
            return None
    try:
        total = 1
        for dim in shape:
            total *= int(dim)
        return int(total)
    except Exception:
        return None


def _is_int_dtype(dtype: Any) -> bool:
    dtype_text = str(dtype).lower()
    return "int" in dtype_text and "bool" not in dtype_text


def _is_tensor_like(value: Any) -> bool:
    return any(
        getattr(value, attr, None) is not None
        for attr in ("shape", "dtype", "device")
    )


def _pad_slot_id() -> int:
    try:
        return int(_PAD_SLOT_ID)
    except Exception:  # pragma: no cover - defensive fallback
        return -1


def _append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    output_path = Path(path)
    parent = output_path.parent
    if parent:
        parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(output_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, encoded)
    finally:
        os.close(descriptor)


def _slot_mapping_view(instance: Any) -> Any:
    slot_mapping = getattr(instance, "slot_mapping", None)
    return getattr(slot_mapping, "gpu", slot_mapping)


def _block_table_view(instance: Any) -> Any:
    block_table = getattr(instance, "block_table", None)
    return getattr(block_table, "gpu", block_table)


def _slot_mapping_item(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _valid_slot_entries(slot_mapping: Any, pad_slot_id: int) -> list[tuple[int, Any]]:
    flat = slot_mapping.reshape(-1) if hasattr(slot_mapping, "reshape") else slot_mapping
    numel = _tensor_numel(flat)
    if numel is None:
        return []
    entries: list[tuple[int, Any]] = []
    for index in range(numel):
        try:
            value = _slot_mapping_item(flat[index])
        except Exception:
            continue
        try:
            if int(value) != pad_slot_id:
                entries.append((index, value))
        except Exception:
            continue
    return entries


def _find_differing_entry(
    valid_entries: list[tuple[int, Any]],
    *,
    start_index: int,
    reverse: bool = False,
) -> tuple[int, Any] | None:
    target_value = valid_entries[start_index][1]
    indices = range(start_index - 1, -1, -1) if reverse else range(start_index + 1, len(valid_entries))
    for index in indices:
        candidate_index, candidate_value = valid_entries[index]
        if candidate_value != target_value:
            return candidate_index, candidate_value
    return None


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _visible_block_ids(
    slot_mapping: Any,
    *,
    block_size: int,
    pad_slot_id: int,
) -> tuple[list[int], list[int]]:
    """Return valid slots and visible blocks in first-seen order."""

    if block_size <= 0:
        return [], []
    valid_slots: list[int] = []
    visible_blocks: list[int] = []
    seen_blocks: set[int] = set()
    for _, value in _valid_slot_entries(slot_mapping, pad_slot_id):
        try:
            slot_id = int(value)
        except (TypeError, ValueError):
            continue
        if slot_id < 0:
            continue
        valid_slots.append(slot_id)
        block_id = slot_id // block_size
        if block_id not in seen_blocks:
            seen_blocks.add(block_id)
            visible_blocks.append(block_id)
    return valid_slots, visible_blocks


def _placeholder_block_score(block_id: int) -> int:
    """Return a deterministic integer score for S2 shadow selection."""

    value = int(block_id) & 0xFFFFFFFF
    value ^= value >> 16
    value = (value * 0x7FEB352D) & 0xFFFFFFFF
    value ^= value >> 15
    value = (value * 0x846CA68B) & 0xFFFFFFFF
    return value ^ (value >> 16)


def _select_shadow_blocks(
    visible_block_ids: list[int],
    *,
    budget_ratio: float,
    keep_recent_blocks: int,
) -> list[int]:
    """Select a bounded shadow set without changing runtime state."""

    if not visible_block_ids:
        return []
    ratio = min(max(float(budget_ratio), 0.0), 1.0)
    recent_count = min(
        max(int(keep_recent_blocks), 1),
        len(visible_block_ids),
    )
    target_count = min(
        len(visible_block_ids),
        max(recent_count, int(len(visible_block_ids) * ratio + 0.999999)),
    )
    recent = visible_block_ids[-recent_count:]
    recent_set = set(recent)
    older = [
        block_id
        for block_id in visible_block_ids
        if block_id not in recent_set
    ]
    older.sort(
        key=lambda block_id: (_placeholder_block_score(block_id), block_id),
        reverse=True,
    )
    selected = older[: target_count - recent_count] + recent
    selected_set = set(selected)
    return [
        block_id
        for block_id in visible_block_ids
        if block_id in selected_set
    ]


def _choose_remap_block(
    original_block_id: int,
    selected_block_ids: list[int],
) -> int | None:
    if not selected_block_ids:
        return None
    lower_or_equal = [
        block_id
        for block_id in selected_block_ids
        if block_id <= original_block_id
    ]
    if lower_or_equal:
        return max(lower_or_equal)
    return min(
        selected_block_ids,
        key=lambda block_id: (abs(block_id - original_block_id), block_id),
    )


def _apply_active_block_mask(
    slot_mapping: Any,
    *,
    block_size: int,
    selected_block_ids: list[int],
    visible_block_ids: list[int],
    keep_recent_blocks: int,
) -> tuple[bool, dict[str, Any]]:
    """Remap older unselected slots to selected blocks when safe."""

    record: dict[str, Any] = {
        "remapped_slot_count": 0,
        "remapped_slot_sample": [],
        "mutation_blocker_reason": None,
        "keep_recent_blocks": keep_recent_blocks,
        "budget_ratio": _env_float("KIVO_SOURCE_BUDGET_RATIO", 0.5),
    }
    if slot_mapping is None:
        record["mutation_blocker_reason"] = (
            "no safe Python-level slot mapping result found"
        )
        return False, record
    if not _is_tensor_like(slot_mapping):
        record["mutation_blocker_reason"] = (
            "slot mapping is not tensor-like; active remap not attempted"
        )
        return False, record
    if not _is_int_dtype(getattr(slot_mapping, "dtype", None)):
        record["mutation_blocker_reason"] = (
            "tensor-like slot mapping requires integer dtype for remap"
        )
        return False, record
    if not selected_block_ids:
        record["mutation_blocker_reason"] = "no selected blocks available"
        return False, record
    valid_entries = _valid_slot_entries(slot_mapping, _pad_slot_id())
    observed_max_slot = (
        max(
            int(_slot_mapping_item(value))
            for _, value in valid_entries
            if int(_slot_mapping_item(value)) >= 0
        )
        if valid_entries
        else -1
    )
    slot_limit = max(_tensor_numel(slot_mapping) or 0, observed_max_slot + 1)
    if slot_limit <= 0:
        record["mutation_blocker_reason"] = "slot mapping has no capacity"
        return False, record
    recent_count = min(max(int(keep_recent_blocks), 1), len(visible_block_ids))
    recent_block_ids = visible_block_ids[-recent_count:]
    recent_block_set = set(recent_block_ids)
    selected_block_set = set(selected_block_ids)
    unselected_block_ids = [
        block_id
        for block_id in visible_block_ids
        if block_id not in selected_block_set
    ]
    if not unselected_block_ids:
        record["mutation_blocker_reason"] = "no unselected older blocks found"
        return False, record

    remapped: list[dict[str, Any]] = []
    for slot_index, slot_value in valid_entries:
        try:
            slot_id = int(slot_value)
        except (TypeError, ValueError):
            continue
        if slot_id < 0:
            continue
        block_id = slot_id // block_size
        if block_id in recent_block_set or block_id in selected_block_set:
            continue
        target_block_id = _choose_remap_block(block_id, selected_block_ids)
        if target_block_id is None:
            continue
        offset = slot_id % block_size
        new_slot = target_block_id * block_size + offset
        if new_slot < 0 or new_slot >= slot_limit:
            continue
        if new_slot == slot_id:
            continue
        old_value = _slot_mapping_item(slot_mapping[slot_index])
        try:
            slot_mapping[slot_index] = int(new_slot)
        except Exception as exc:
            record["mutation_blocker_reason"] = (
                f"{type(exc).__name__}: {exc}"
            )
            return False, record
        remapped.append(
            {
                "slot_index": int(slot_index),
                "old_value": old_value,
                "new_value": int(new_slot),
                "old_block_id": int(block_id),
                "new_block_id": int(target_block_id),
            }
        )

    if not remapped:
        record["mutation_blocker_reason"] = (
            "no remappable unselected slots found within observed range"
        )
        return False, record

    record.update(
        {
            "remapped_slot_count": len(remapped),
            "remapped_slot_sample": remapped[:32],
            "mutation_blocker_reason": None,
        }
    )
    return True, record


def _build_block_visibility_shadow_record(
    instance: Any,
    *,
    function_name: str,
) -> dict[str, Any]:
    slot_mapping = _slot_mapping_view(instance)
    block_table = _block_table_view(instance)
    slot_info = _tensor_info(slot_mapping)
    block_info = _tensor_info(block_table)
    block_size = int(getattr(instance, "block_size", 0) or 0)
    pad_slot_id = _pad_slot_id()
    budget_ratio = min(
        max(_env_float("KIVO_SOURCE_BUDGET_RATIO", 0.5), 0.0),
        1.0,
    )
    keep_recent_blocks = max(
        _env_int("KIVO_SOURCE_KEEP_RECENT_BLOCKS", 1),
        1,
    )
    valid_slots, visible_blocks = _visible_block_ids(
        slot_mapping,
        block_size=block_size,
        pad_slot_id=pad_slot_id,
    )
    selected_blocks = _select_shadow_blocks(
        visible_blocks,
        budget_ratio=budget_ratio,
        keep_recent_blocks=keep_recent_blocks,
    )
    dropped_block_count = len(visible_blocks) - len(selected_blocks)
    reduction_ratio = (
        dropped_block_count / len(visible_blocks)
        if visible_blocks
        else 0.0
    )
    return {
        "schema_version": "kivo_source_s2_0_block_visibility_shadow_v1",
        "timestamp": time.time(),
        "pid": os.getpid(),
        "hook_name": "BlockTable.compute_slot_mapping",
        "function_name": function_name,
        "block_size": block_size,
        "slot_mapping_present": slot_mapping is not None,
        "slot_mapping_shape": slot_info["shape"],
        "slot_mapping_dtype": slot_info["dtype"],
        "slot_mapping_device": slot_info["device"],
        "block_table_present": block_table is not None,
        "block_table_shape": block_info["shape"],
        "block_table_dtype": block_info["dtype"],
        "block_table_device": block_info["device"],
        "valid_slot_count": len(valid_slots),
        "valid_slot_min": min(valid_slots) if valid_slots else None,
        "valid_slot_max": max(valid_slots) if valid_slots else None,
        "visible_block_count": len(visible_blocks),
        "visible_block_ids_sample": visible_blocks[:32],
        "total_block_table_entries": _tensor_numel(block_table),
        "policy_name": "sketch_shadow_blocks",
        "budget_ratio": budget_ratio,
        "keep_recent_blocks": keep_recent_blocks,
        "selected_block_count": len(selected_blocks),
        "selected_block_ids_sample": selected_blocks[:32],
        "dropped_block_count": dropped_block_count,
        "selection_ratio_actual": (
            len(selected_blocks) / len(visible_blocks)
            if visible_blocks
            else 0.0
        ),
        "theoretical_visible_block_reduction": dropped_block_count,
        "theoretical_visible_block_reduction_ratio": reduction_ratio,
        "mutation_attempted": False,
        "mutation_applied": False,
        "runtime_behavior_changed": False,
        "active_routing": False,
        "measured_runtime_reduction": False,
        "selected_attention_claim_allowed": False,
        "performance_claim_allowed": False,
        "caveats": [
            "shadow selection only; selected blocks are not applied",
            "placeholder block scores do not inspect KV tensors",
            "theoretical block reduction is not measured memory reduction",
            "no scheduler, attention, block table, or KV cache mutation",
        ],
    }


def _build_active_block_mask_record(
    instance: Any,
    *,
    function_name: str,
) -> tuple[bool, dict[str, Any]]:
    slot_mapping = _slot_mapping_view(instance)
    block_table = _block_table_view(instance)
    slot_info = _tensor_info(slot_mapping)
    block_info = _tensor_info(block_table)
    block_size = int(getattr(instance, "block_size", 0) or 0)
    pad_slot_id = _pad_slot_id()
    budget_ratio = min(
        max(_env_float("KIVO_SOURCE_BUDGET_RATIO", 0.5), 0.0),
        1.0,
    )
    keep_recent_blocks = max(
        _env_int("KIVO_SOURCE_KEEP_RECENT_BLOCKS", 1),
        1,
    )
    valid_slots, visible_blocks = _visible_block_ids(
        slot_mapping,
        block_size=block_size,
        pad_slot_id=pad_slot_id,
    )
    selected_blocks = _select_shadow_blocks(
        visible_blocks,
        budget_ratio=budget_ratio,
        keep_recent_blocks=keep_recent_blocks,
    )
    unselected_blocks = [
        block_id
        for block_id in visible_blocks
        if block_id not in set(selected_blocks)
    ]
    attempted = (
        len(visible_blocks) >= 2
        and len(selected_blocks) < len(visible_blocks)
    )
    mutation_applied, mutation_record = (False, {"remapped_slot_count": 0, "remapped_slot_sample": [], "mutation_blocker_reason": None, "keep_recent_blocks": keep_recent_blocks, "budget_ratio": budget_ratio})
    if attempted:
        mutation_applied, mutation_record = _apply_active_block_mask(
            slot_mapping,
            block_size=block_size,
            selected_block_ids=selected_blocks,
            visible_block_ids=visible_blocks,
            keep_recent_blocks=keep_recent_blocks,
        )
    remapped_count = int(mutation_record.get("remapped_slot_count", 0) or 0)
    return (
        mutation_applied,
        {
            "schema_version": "kivo_source_s2_1_active_block_mask_v1",
            "timestamp": time.time(),
            "pid": os.getpid(),
            "hook_name": "BlockTable.compute_slot_mapping",
            "function_name": function_name,
            "block_size": block_size,
            "slot_mapping_present": slot_mapping is not None,
            "slot_mapping_shape": slot_info["shape"],
            "slot_mapping_dtype": slot_info["dtype"],
            "slot_mapping_device": slot_info["device"],
            "block_table_present": block_table is not None,
            "block_table_shape": block_info["shape"],
            "block_table_dtype": block_info["dtype"],
            "block_table_device": block_info["device"],
            "valid_slot_count": len(valid_slots),
            "valid_slot_min": min(valid_slots) if valid_slots else None,
            "valid_slot_max": max(valid_slots) if valid_slots else None,
            "visible_block_count": len(visible_blocks),
            "visible_block_ids_sample": visible_blocks[:32],
            "selected_block_count": len(selected_blocks),
            "selected_block_ids_sample": selected_blocks[:32],
            "unselected_block_count": len(unselected_blocks),
            "unselected_block_ids_sample": unselected_blocks[:32],
            "policy_name": "active_mask_unselected_blocks",
            "keep_recent_blocks": keep_recent_blocks,
            "budget_ratio": budget_ratio,
            "remapped_slot_count": remapped_count,
            "remapped_slot_sample": mutation_record.get(
                "remapped_slot_sample", []
            ),
            "mutation_attempted": attempted,
            "mutation_applied": mutation_applied,
            "active_routing": mutation_applied,
            "runtime_behavior_changed": mutation_applied,
            "measured_runtime_reduction": False,
            "selected_attention_claim_allowed": False,
            "performance_claim_allowed": False,
            "mutation_blocker_reason": mutation_record.get(
                "mutation_blocker_reason"
            ),
            "caveats": [
                "active remap only; selected blocks are approximated",
                "no KV cache, scheduler, or attention kernel mutation",
                "theoretical block control is not measured memory reduction",
            ],
        },
    )


def _maybe_mutate_slot_mapping(
    slot_mapping: Any,
    *,
    policy: str,
) -> tuple[bool, dict[str, Any]]:
    record: dict[str, Any] = {
        "mutation_policy": None,
        "mutation_blocker_reason": None,
        "mutation_target_position": None,
        "old_value": None,
        "new_value": None,
        "mutation_index": None,
        "valid_slot_count": None,
        "pad_slot_id": _pad_slot_id(),
        "valid_mutation_index": None,
        "previous_valid_index": None,
        "next_valid_index": None,
        "old_new_differ": False,
    }
    if slot_mapping is None:
        record["mutation_blocker_reason"] = (
            "no safe Python-level slot mapping result found"
        )
        return False, record
    if not _is_tensor_like(slot_mapping):
        record["mutation_blocker_reason"] = (
            "slot mapping is not tensor-like; selected-slot state "
            "mutation not attempted"
        )
        return False, record
    slot_info = _tensor_info(slot_mapping)
    record["slot_mapping_shape"] = slot_info["shape"]
    record["slot_mapping_dtype"] = slot_info["dtype"]
    record["slot_mapping_device"] = slot_info["device"]
    if not _is_int_dtype(getattr(slot_mapping, "dtype", None)):
        record["mutation_blocker_reason"] = (
            "tensor-like slot mapping requires integer dtype for safe mutation"
        )
        return False, record
    numel = _tensor_numel(slot_mapping)
    if numel is None or numel < 2:
        record["mutation_blocker_reason"] = (
            "tensor-like slot mapping needs at least two elements"
        )
        return False, record
    valid_entries = _valid_slot_entries(slot_mapping, record["pad_slot_id"])
    record["valid_slot_count"] = len(valid_entries)
    if policy == "mask_last_slot":
        try:
            old_tensor_value = slot_mapping[-1]
            new_tensor_value = slot_mapping[-2]
            old_value = _slot_mapping_item(old_tensor_value)
            new_value = _slot_mapping_item(new_tensor_value)
            if old_value == new_value:
                record["mutation_blocker_reason"] = (
                    "no differing valid slot pair found"
                )
                return False, record
            slot_mapping[-1] = new_tensor_value
            record.update(
                {
                    "mutation_policy": policy,
                    "mutation_target_position": "last",
                    "mutation_blocker_reason": None,
                    "old_value": old_value,
                    "new_value": new_value,
                    "mutation_index": numel - 1,
                    "valid_mutation_index": numel - 1,
                    "previous_valid_index": numel - 2,
                    "next_valid_index": None,
                    "old_new_differ": old_value != new_value,
                }
            )
            return True, record
        except Exception as exc:
            record["mutation_blocker_reason"] = (
                f"{type(exc).__name__}: {exc}"
            )
            return False, record

    if not valid_entries:
        record["mutation_blocker_reason"] = "fewer than two valid slot entries"
        return False, record
    if policy == "mask_last_valid_slot":
        if len(valid_entries) < 2:
            record["mutation_blocker_reason"] = (
                "fewer than two valid slot entries"
            )
            return False, record
        target_pos = len(valid_entries) - 1
        candidate = _find_differing_entry(
            valid_entries, start_index=target_pos, reverse=True
        )
        if candidate is None:
            record["mutation_blocker_reason"] = (
                "no differing valid slot pair found"
            )
            return False, record
        valid_mutation_index, new_value = candidate
        previous_valid_index = valid_entries[target_pos - 1][0]
        old_value = _slot_mapping_item(slot_mapping[valid_entries[target_pos][0]])
        try:
            slot_mapping[valid_entries[target_pos][0]] = slot_mapping[
                valid_mutation_index
            ]
        except Exception as exc:
            record["mutation_blocker_reason"] = (
                f"{type(exc).__name__}: {exc}"
            )
            return False, record
        record.update(
            {
                "mutation_policy": policy,
                "mutation_target_position": "last",
                "mutation_blocker_reason": None,
                "old_value": old_value,
                "new_value": new_value,
                "mutation_index": valid_entries[target_pos][0],
                "valid_mutation_index": valid_entries[target_pos][0],
                "previous_valid_index": previous_valid_index,
                "next_valid_index": None,
                "old_new_differ": old_value != new_value,
            }
        )
        return True, record

    if policy == "mask_oldest_valid_slot":
        if len(valid_entries) < 2:
            record["mutation_blocker_reason"] = (
                "fewer than two valid slot entries"
            )
            return False, record
        target_pos = 0
        next_valid_index, new_value = valid_entries[1]
        if valid_entries[1][1] == valid_entries[target_pos][1]:
            record["mutation_blocker_reason"] = (
                "no differing valid slot pair found"
            )
            return False, record
        old_value = _slot_mapping_item(slot_mapping[valid_entries[target_pos][0]])
        try:
            slot_mapping[valid_entries[target_pos][0]] = slot_mapping[
                next_valid_index
            ]
        except Exception as exc:
            record["mutation_blocker_reason"] = (
                f"{type(exc).__name__}: {exc}"
            )
            return False, record
        record.update(
            {
                "mutation_policy": policy,
                "mutation_target_position": "oldest",
                "mutation_blocker_reason": None,
                "old_value": old_value,
                "new_value": new_value,
                "mutation_index": valid_entries[target_pos][0],
                "valid_mutation_index": valid_entries[target_pos][0],
                "previous_valid_index": None,
                "next_valid_index": next_valid_index,
                "old_new_differ": old_value != new_value,
            }
        )
        return True, record

    if policy == "mask_middle_valid_slot":
        if len(valid_entries) < 3:
            record["mutation_blocker_reason"] = (
                "fewer than three valid slot entries"
            )
            return False, record
        target_pos = len(valid_entries) // 2
        prev_entry = valid_entries[target_pos - 1]
        next_entry = valid_entries[target_pos + 1]
        target_index, target_value = valid_entries[target_pos]
        candidate = None
        if prev_entry[1] != target_value:
            candidate = prev_entry
        elif next_entry[1] != target_value:
            candidate = next_entry
        if candidate is None:
            record["mutation_blocker_reason"] = (
                "no differing valid slot pair found"
            )
            return False, record
        chosen_index, new_value = candidate
        try:
            slot_mapping[target_index] = slot_mapping[chosen_index]
        except Exception as exc:
            record["mutation_blocker_reason"] = (
                f"{type(exc).__name__}: {exc}"
            )
            return False, record
        record.update(
            {
                "mutation_policy": policy,
                "mutation_target_position": "middle",
                "mutation_blocker_reason": None,
                "old_value": target_value,
                "new_value": new_value,
                "mutation_index": target_index,
                "valid_mutation_index": target_index,
                "previous_valid_index": prev_entry[0],
                "next_valid_index": next_entry[0],
                "old_new_differ": target_value != new_value,
            }
        )
        return True, record

    if policy == "noop_valid_slot_shadow":
        target_pos = len(valid_entries) - 1
        candidate = _find_differing_entry(
            valid_entries, start_index=target_pos, reverse=True
        )
        if candidate is None:
            record["mutation_blocker_reason"] = (
                "no differing valid slot pair found"
            )
            return False, record
        candidate_index, new_value = candidate
        old_value = _slot_mapping_item(slot_mapping[valid_entries[target_pos][0]])
        record.update(
            {
                "mutation_policy": policy,
                "mutation_target_position": "shadow",
                "mutation_blocker_reason": "shadow policy; no mutation applied",
                "old_value": old_value,
                "new_value": new_value,
                "mutation_index": valid_entries[target_pos][0],
                "valid_mutation_index": valid_entries[target_pos][0],
                "previous_valid_index": valid_entries[target_pos - 1][0]
                if target_pos > 0
                else None,
                "next_valid_index": None,
                "old_new_differ": old_value != new_value,
            }
        )
        return False, record

    record["mutation_blocker_reason"] = (
        f"unsupported mutation policy: {policy or '<unset>'}"
    )
    return False, record


def maybe_observe_compute_slot_mapping(
    instance: Any,
    *,
    module_file: str,
    function_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> None:
    if not _env_flag("KIVO_SOURCE_ENABLE"):
        return

    output_path = os.getenv("KIVO_SOURCE_OBS_PATH")
    if not output_path:
        return

    record: dict[str, Any] | None = None
    try:
        active_enabled = _env_flag("KIVO_SOURCE_ACTIVE")
        policy = os.getenv("KIVO_SOURCE_POLICY", "")
        if policy == "sketch_shadow_blocks":
            _append_jsonl(
                output_path,
                _build_block_visibility_shadow_record(
                    instance,
                    function_name=function_name,
                ),
            )
            return
        if policy == "active_mask_unselected_blocks":
            applied, record = _build_active_block_mask_record(
                instance,
                function_name=function_name,
            )
            record["mutation_attempted"] = bool(
                record.get("visible_block_count", 0) >= 2
                and record.get("selected_block_count", 0)
                < record.get("visible_block_count", 0)
            )
            record["mutation_applied"] = applied
            record["active_routing"] = applied
            record["runtime_behavior_changed"] = applied
            _append_jsonl(output_path, record)
            return
        slot_mapping = _slot_mapping_view(instance)
        block_table = _block_table_view(instance)
        mutation_attempted = active_enabled and policy in {
            "mask_last_slot",
            "mask_last_valid_slot",
            "mask_oldest_valid_slot",
            "mask_middle_valid_slot",
            "noop_valid_slot_shadow",
        }
        mutation_applied = False
        mutation_record = {
            "mutation_policy": None,
            "mutation_blocker_reason": None,
            "mutation_target_position": None,
            "old_value": None,
            "new_value": None,
            "mutation_index": None,
            "valid_slot_count": None,
            "pad_slot_id": _pad_slot_id(),
            "valid_mutation_index": None,
            "previous_valid_index": None,
            "next_valid_index": None,
            "old_new_differ": False,
        }
        if active_enabled and not mutation_attempted:
            mutation_record["mutation_blocker_reason"] = (
                f"unsupported mutation policy: {policy or '<unset>'}"
            )
        if mutation_attempted:
            mutation_applied, mutation_record = _maybe_mutate_slot_mapping(
                slot_mapping,
                policy=policy,
            )
        result_summary = _safe_summary(result)
        slot_info = _tensor_info(slot_mapping)
        block_info = _tensor_info(block_table)
        num_blocks_per_row = getattr(instance, "num_blocks_per_row", None)
        try:
            num_blocks_preview = (
                [
                    int(item)
                    for item in list(
                        num_blocks_per_row[:32]  # type: ignore[index]
                    )
                ]
                if num_blocks_per_row is not None
                else None
            )
        except Exception:
            num_blocks_preview = None
        record = {
            "schema_version": "kivo_source_s1_block_table_v1",
            "timestamp": time.time(),
            "pid": os.getpid(),
            "hook_name": "BlockTable.compute_slot_mapping",
            "class_name": type(instance).__qualname__,
            "function_name": function_name,
            "args_summary": [_safe_summary(item) for item in args[:8]],
            "slot_mapping_present": slot_mapping is not None,
            "slot_mapping_type": slot_info["type"],
            "slot_mapping_shape": slot_info["shape"],
            "slot_mapping_dtype": slot_info["dtype"],
            "slot_mapping_device": slot_info["device"],
            "block_table_present": block_table is not None,
            "block_table_type": block_info["type"],
            "block_table_shape": block_info["shape"],
            "block_table_dtype": block_info["dtype"],
            "block_table_device": block_info["device"],
            "result_type": result_summary.get("type"),
            "result_summary": result_summary,
            "block_size": getattr(instance, "block_size", None),
            "num_blocks_per_row": num_blocks_preview,
            "max_num_blocks_per_req": getattr(
                instance, "max_num_blocks_per_req", None
            ),
            "max_num_reqs": getattr(instance, "max_num_reqs", None),
            "active_enabled": active_enabled,
            "mutation_attempted": mutation_attempted,
            "mutation_applied": mutation_applied,
            "mutation_policy": mutation_record["mutation_policy"],
            "mutation_blocker_reason": mutation_record[
                "mutation_blocker_reason"
            ],
            "mutation_target_position": mutation_record[
                "mutation_target_position"
            ],
            "old_value": mutation_record["old_value"],
            "new_value": mutation_record["new_value"],
            "mutation_index": mutation_record["mutation_index"],
            "valid_slot_count": mutation_record["valid_slot_count"],
            "pad_slot_id": mutation_record["pad_slot_id"],
            "valid_mutation_index": mutation_record["valid_mutation_index"],
            "previous_valid_index": mutation_record["previous_valid_index"],
            "next_valid_index": mutation_record["next_valid_index"],
            "old_new_differ": mutation_record["old_new_differ"],
            "runtime_behavior_changed": mutation_applied,
            "active_routing": mutation_applied,
            "measured_runtime_reduction": False,
            "caveats": [
                "source-level experiment hook",
                "fail closed on any hook error",
                "no scheduler, attention, or KV cache mutation",
            ],
        }
        _append_jsonl(output_path, record)
    except Exception as exc:
        if _env_flag("KIVO_SOURCE_FAIL_CLOSED"):
            fallback_record = record or {
                "schema_version": "kivo_source_s1_block_table_v1",
                "timestamp": time.time(),
                "pid": os.getpid(),
                "hook_name": "BlockTable.compute_slot_mapping",
                "class_name": type(instance).__qualname__,
                "function_name": function_name,
                "args_summary": [_safe_summary(item) for item in args[:8]],
                "mutation_policy": None,
                "mutation_blocker_reason": None,
                "mutation_target_position": None,
                "old_value": None,
                "new_value": None,
                "mutation_index": None,
                "valid_slot_count": None,
                "pad_slot_id": _pad_slot_id(),
                "valid_mutation_index": None,
                "previous_valid_index": None,
                "next_valid_index": None,
                "old_new_differ": False,
                "mutation_attempted": False,
                "mutation_applied": False,
                "runtime_behavior_changed": False,
                "active_routing": False,
                "measured_runtime_reduction": False,
                "caveats": [
                    "source-level experiment hook",
                    "fail closed on any hook error",
                ],
            }
            fallback_record["helper_exception_type"] = type(exc).__name__
            fallback_record["helper_exception_message"] = str(exc)
            try:
                _append_jsonl(output_path, fallback_record)
            except Exception:
                pass
            return
        raise
