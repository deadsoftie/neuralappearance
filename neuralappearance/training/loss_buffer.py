# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import slangpy as spy


class LossBuffer:
    """Device or readback storage for per-sample training losses.

    ``train.slang`` stores the current loss in ``x`` and its running sum in
    ``y``. A readback buffer mirrors the device-local tensor without changing
    the layout consumed by training statistics.
    """

    def __init__(
        self, device: spy.Device, shape: spy.Shape | Sequence[int], *, readback: bool = False
    ):
        if readback:
            usage = spy.BufferUsage.copy_destination
            memory_type = spy.MemoryType.read_back
        else:
            usage = spy.BufferUsage.unordered_access | spy.BufferUsage.shader_resource
            memory_type = spy.MemoryType.device_local

        self.tensor = spy.Tensor.empty(
            device,
            shape=shape,
            dtype='float2',
            usage=usage,
            memory_type=memory_type,
        )

        if not readback:
            self.tensor.clear()

    def copy_from(self, other: LossBuffer, cmd: spy.CommandEncoder) -> None:
        cmd.copy_buffer(
            self.tensor.storage,
            0,
            other.tensor.storage,
            0,
            self.tensor.storage.size,
        )
