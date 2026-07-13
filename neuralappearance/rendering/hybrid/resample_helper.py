# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight resample (tensor -> texture blit) for the visualizer."""

from __future__ import annotations

import slangpy as spy
from slangpy import uint2


class ResampleHelper:
    """Tensor-to-texture blit with optional scaling and tone mapping.

    Used by the visualizer only.
    """

    def __init__(self, device: spy.Device):
        self._device = device
        self._module = spy.Module.load_from_file(
            device,
            'rendering/hybrid/resample_helper.slang',
        )

    def resample(
        self,
        src: spy.Tensor,
        output: spy.Texture,
        output_pos: uint2,
        output_size: uint2,
        scale: float = 1,
        tone_map: bool = False,
    ):
        size = spy.int2(int(output_size.x), int(output_size.y))
        offset = spy.int2(int(output_pos.x), int(output_pos.y))
        self._module.blit_tensor_to_texture(
            spy.grid((size.y, size.x)),
            size,
            offset,
            float(scale),
            bool(tone_map),
            src,
            output,
        )
