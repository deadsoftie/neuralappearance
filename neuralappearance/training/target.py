# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import numpy as np
import slangpy as spy


class TrainingTarget:
    """Named slice within a packed prediction or reference value."""

    def __init__(self, name: str, size: int, offset: int, num_channels: int) -> None:
        self.name = name
        self.size = size
        self.offset = offset
        self.num_channels = num_channels

    def view(self, buffer: np.ndarray | spy.Tensor) -> np.ndarray:
        """Return this target's final-dimension slice from predicted values."""
        if isinstance(buffer, np.ndarray):
            buffer_np = buffer
        elif isinstance(buffer, spy.Tensor):
            buffer_np = buffer.to_numpy()
        else:
            raise TypeError(
                f'TrainingTarget.view: Unsupported argument type "{type(buffer).__name__}"'
            )

        if buffer_np.shape[-1] != self.num_channels:
            raise ValueError(
                f'TrainingTarget.view: Unexpected array shape {buffer.shape}, expected innermost size of {self.num_channels}.'
            )

        # Copy the slice so it is contiguous and can become a SlangPy tensor.
        return buffer_np[..., self.offset : self.offset + self.size].copy()


class TrainingTargets:
    """Channel layout and loss weighting for one configured training target.

    Auxiliary targets may combine several semantic values in one network
    output. Iteration exposes those components as ``TrainingTarget`` slices for
    plotting and rendering while training uses the packed channel layout.
    """

    def __init__(self, name: str = 'bsdf') -> None:
        self.name = name.lower()

        self.loss_weights: list[float] | None = None

        if self.name == 'bsdf':
            # Default target when training the main neural BSDF.
            self.component_names = ['bsdf']
            self.component_sizes = [3]
        elif self.name == 'ref_diffuse_specular_roughness':
            # Diffuse albedo, specular albedo, and roughness.
            self.component_names = ['diffuse', 'specular', 'roughness']
            self.component_sizes = [3, 3, 1]
            # Give equal weight to every albedo channel.
            self.loss_weights = [0.4 * 1.0 / 3.0] * 6 + [0.2]
        elif self.name == 'ref_diffuse_specular_roughness_normal':
            # Diffuse albedo, specular albedo, roughness, and normal.
            self.component_names = ['diffuse', 'specular', 'roughness', 'normal']
            self.component_sizes = [3, 3, 1, 3]
        elif self.name == 'mc_albedo':
            # Monte Carlo estimate of hemispherical-directional albedo.
            # Extracted from a pre-trained neural BSDF.
            self.component_names = ['mc_albedo']
            self.component_sizes = [3]
        else:
            raise ValueError(f'Unsupported training target "{self.name}".')

        assert len(self.component_sizes) == len(self.component_names)
        self.target_count = len(self.component_names)
        self.num_channels = sum(self.component_sizes)

        if self.loss_weights is None:
            self.loss_weights = [1.0 / self.num_channels] * self.num_channels

    def __len__(self) -> int:
        return self.target_count

    def __iter__(self) -> Iterator[TrainingTarget]:
        """Iterate over the target's ``TrainingTarget`` components."""
        for i in range(len(self.component_sizes)):
            yield TrainingTarget(
                name=self.component_names[i],
                size=self.component_sizes[i],
                offset=sum(self.component_sizes[:i]),
                num_channels=self.num_channels,
            )

    def __getitem__(self, key: int | str) -> TrainingTarget:
        """Return the ``TrainingTarget`` for the given component."""
        if isinstance(key, int):
            idx = key
        elif isinstance(key, str):
            try:
                idx = self.component_names.index(key)
            except ValueError as e:
                raise KeyError(key) from e
        else:
            raise TypeError('TrainingTargets index must be int or str.')

        return TrainingTarget(
            name=self.component_names[idx],
            size=self.component_sizes[idx],
            offset=sum(self.component_sizes[:idx]),
            num_channels=self.num_channels,
        )
