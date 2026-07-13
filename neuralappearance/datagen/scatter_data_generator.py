# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base class for pluggable scatter data generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import slangpy as spy


class ScatterDataGenerator(ABC):
    """Evaluate material scattering at supplied shading points.

    All directions are in local shading space.
    """

    @property
    def mip_level(self) -> int:
        """Current texture/LOD mip level for material evaluation."""
        return 0

    @mip_level.setter
    def mip_level(self, value: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def eval_scatter(
        self,
        queries: spy.Tensor,
        n_eval: int,
        compact_indices: spy.Tensor,
        iteration: int,
        depth: int,
    ) -> spy.Tensor:
        """Evaluate and importance-sample ``n_eval`` shading points.

        ``queries`` may contain extra entries. ``compact_indices``
        maps valid queries back to pixels for random-number seeding.

        Return a GPU tensor of ``ScatterEventResult`` values.
        """
        ...
