# Phase S5.1: KV Ownership Intervention

Phase S5.1 adds the smallest real ownership boundary hook for Kivo-VD.

## What changed

- A pure policy module now decides which skipped/removable blocks are allowed
  to be freed and which are protected.
- `SingleTypeKVCacheManager.remove_skipped_blocks(...)` now consults that
  policy before passing blocks to the free path.

## What this does not do yet

- It does not reduce memory by default.
- It does not evict arbitrary live old blocks.
- It does not add selected attention.
- It does not change CUDA kernels.
- It does not change block-table semantics.

## Why this boundary matters

This is the first place where blocks already considered removable by vLLM can
be held back from the free queue. That makes it the right place to prepare an
online retention policy without rewriting the attention backend yet.

## Supported policies

- `off`
- `allow_all_skipped`
- `protect_all_skipped`
- `protect_recent_skipped`

The default behavior remains unchanged unless Kivo ownership env flags are
enabled.

## Next phase

S5.2 should attach an online CountSketch-based retention policy at this
ownership boundary. The policy can start conservative by protecting the most
recent skipped blocks while allowing older skipped blocks to free normally.
That is the first practical step toward real KV memory reduction.
