# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pack trained `BcLatentTexture` values into BC6H mode-10 block bytes.

Shared by offline export and render-time texture upload. The small bit-layout
helpers live here so production model code does not depend on export tooling.
"""

from __future__ import annotations

import numpy as np

# Mirrors model/bc6h.slang exactly -- see that file's own header note on why
# duplicated rather than imported.
_PRECISION = 10
_SIGNED_MIN, _SIGNED_MAX = -512.0, 511.0
_UNSIGNED_MAX = 1023.0
_INDEX_LEVELS = 15.0
_ANCHOR_LEVELS = 7.0

_MODE10_BITS = 0b00011  # block's first 5 bits, mode 10


def quantize_endpoint(value_units: np.ndarray, is_signed: bool) -> np.ndarray:
    """Round-and-clamp to mode 10's 10-bit endpoint lattice. Mirrors
    `bc6h_quantize_endpoint`."""
    lo = _SIGNED_MIN if is_signed else 0.0
    hi = _SIGNED_MAX if is_signed else _UNSIGNED_MAX
    return np.clip(np.round(value_units), lo, hi).astype(np.int64)


def quantize_index(position: np.ndarray, is_anchor: bool) -> np.ndarray:
    """Round to 7 index levels at the block anchor and 15 elsewhere."""
    levels = _ANCHOR_LEVELS if is_anchor else _INDEX_LEVELS
    return np.clip(np.round(position * levels), 0, levels).astype(np.int64)


def pack_mode10_block(endpoint0: tuple[int, int, int], endpoint1: tuple[int, int, int],
                      indices: list[int]) -> bytes:
    """Assemble one 16-byte mode-10 block: 5-bit mode, six 10-bit endpoint
    fields (both endpoints stored absolutely, no delta transform), then 16
    indices -- the first (the block's anchor) at 3 bits, the rest at 4.

    Fields are written least-significant-bit first. Mode 10 needs no
    interleaved high bits or delta transform.
    """
    if len(indices) != 16:
        raise ValueError(f'Need 16 indices, got {len(indices)}.')
    if indices[0] >= 8:
        raise ValueError(
            f'Anchor index must be < 8 (its top bit is implicit), got {indices[0]}.')

    value = 0
    pos = 0

    def write(v: int, bits: int) -> None:
        nonlocal value, pos
        value |= (v & ((1 << bits) - 1)) << pos
        pos += bits

    write(_MODE10_BITS, 5)
    for v in (*endpoint0, *endpoint1):
        write(v, _PRECISION)
    write(indices[0], 3)
    for idx in indices[1:]:
        write(idx, 4)
    assert pos == 128, f'mode-10 block must be 128 bits, built {pos}'
    return value.to_bytes(16, 'little')


def build_blocks(endpoint_raw: np.ndarray, position_raw: np.ndarray,
                 endpoint_scale: float, is_signed: bool) -> tuple[bytes, int, int]:
    """Pack one grid's trained values into real mode-10 BC6H block bytes,
    row-major over the block grid.

    `endpoint_raw`: (block_res, block_res, 6) normalised, channels 0-2 are
    endpoint 0, 3-5 endpoint 1 -- `BcLatentTexture.endpoint_textures[i]
    .to_numpy()`. `position_raw`: (res, res, 1) normalised, res == block_res
    * 4 -- `BcLatentTexture.position_textures[i].to_numpy()`. Returns
    `(payload, width, height)`, `width == height == res`.
    """
    res = position_raw.shape[0]
    block_res = endpoint_raw.shape[0]
    if res != block_res * 4:
        raise ValueError(f'position resolution {res} is not 4x the endpoint '
                         f'block resolution {block_res}.')

    ep0_units = endpoint_scale * endpoint_raw[..., 0:3].astype(np.float32)
    ep1_units = endpoint_scale * endpoint_raw[..., 3:6].astype(np.float32)
    ep0_int = quantize_endpoint(ep0_units, is_signed)  # (block_res, block_res, 3)
    ep1_int = quantize_endpoint(ep1_units, is_signed)

    position = position_raw[..., 0].astype(np.float32)
    yy, xx = np.meshgrid(np.arange(res), np.arange(res), indexing='ij')
    is_anchor_mask = ((yy % 4) == 0) & ((xx % 4) == 0)
    index_grid = np.where(
        is_anchor_mask,
        quantize_index(position, True),
        quantize_index(position, False),
    )  # (res, res)

    rows = []
    for by in range(block_res):
        row_blocks = []
        for bx in range(block_res):
            e0 = tuple(int(v) for v in ep0_int[by, bx])
            e1 = tuple(int(v) for v in ep1_int[by, bx])
            # Raster order within the block: texel ty*4+tx. Texel 0 is the
            # anchor.
            indices = [int(v) for v in
                      index_grid[by * 4:by * 4 + 4, bx * 4:bx * 4 + 4].reshape(16)]
            row_blocks.append(pack_mode10_block(e0, e1, indices))
        rows.append(b''.join(row_blocks))
    return b''.join(rows), res, res
