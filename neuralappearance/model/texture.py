# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import slangpy as spy


class Texture:
    def __init__(
        self,
        device: spy.Device,
        num_channels: int,
        resolution: int,
        wrap: bool,
        optimizable: bool = True,
    ):
        self.num_channels = num_channels
        self.resolution = resolution
        self.wrap = wrap
        self.optimizable = optimizable

        # ``texture.slang`` indexes a flat, channel-interleaved texel buffer.
        shape = (self.resolution * self.resolution * self.num_channels,)
        self.buffer = spy.Tensor.empty(device, shape, 'half')
        self.buffer.storage.copy_from_numpy(np.zeros(shape, dtype=np.float16))
        if self.optimizable:
            self.buffer = self.buffer.with_grads(zero=True)

    def make_optimizable(self) -> None:
        if self.optimizable:
            return
        self.optimizable = True
        self.buffer = self.buffer.with_grads(zero=True)

    def to_numpy(self) -> np.ndarray:
        data = self.buffer.to_numpy()
        return data.reshape(self.resolution, self.resolution, self.num_channels)

    def from_numpy(self, data: np.ndarray) -> None:
        # Validate the shape and copy the data.
        data = np.ravel(data)
        if data.shape != self.buffer.shape:
            raise ValueError(
                f'Texture shape mismatch: expected {self.buffer.shape}, got {data.shape}'
            )
        self.buffer.storage.copy_from_numpy(data)

    def get_this(self) -> dict[str, Any]:
        buffer = self.buffer.storage
        buffer_grads = buffer  # SlangPy binding cannot be ``None``.
        buffer_grads_offset = 0

        if self.optimizable:
            assert self.buffer.grad_out is not None
            buffer_grads = self.buffer.grad_out.storage
            buffer_grads_offset = self.buffer.grad_out.offset

        return {
            '_type': f'Texture<{self.num_channels}>',
            'resolution': self.resolution,
            'wrap': self.wrap,
            'optimizable': self.optimizable,
            'buffer': buffer.descriptor_handle_rw,
            'buffer_grads': buffer_grads.descriptor_handle_rw,
            'buffer_grads_offset': buffer_grads_offset,
        }

    def save_checkpoint_images(
        self,
        desc: list[str],
        folder: Path,
    ) -> list[tuple[list[str], str]]:
        """Export texture to disk as a single multi-channel EXR image."""
        data = self.to_numpy()
        path = self.checkpoint_image_path(desc, folder)

        # Latent textures are linear, not sRGB.
        bitmap = spy.Bitmap(
            data,
            channel_names=[
                f'channel{i:0{len(str(self.num_channels - 1))}d}' for i in range(self.num_channels)
            ],
            srgb_gamma=False,
        )

        # Image encode and disk I/O run concurrently with the next texture's
        # compute. No downstream caller reads the file immediately.
        bitmap.write_async(path)

        return [(desc, path.name)]

    def checkpoint_image_path(
        self,
        desc: list[str],
        folder: Path,
    ) -> Path:
        """Return the file path for this texture's checkpoint image."""
        out_name = '.'.join(desc) + '.exr'
        return folder / out_name

    def _bitmap_to_numpy(self, bitmap: spy.Bitmap) -> np.ndarray:
        data = np.asarray(bitmap)
        if data.ndim == 2:
            data = data[:, :, None]
        return data

    def load_from_bitmap(
        self,
        bitmap: spy.Bitmap,
    ):
        """Load texture data from a pre-loaded bitmap."""
        data = self._bitmap_to_numpy(bitmap)
        self.from_numpy(data)

    def load_checkpoint_images(
        self,
        desc: list[str],
        folder: Path,
    ):
        path = self.checkpoint_image_path(desc, folder)
        self.load_from_bitmap(spy.Bitmap(path))
