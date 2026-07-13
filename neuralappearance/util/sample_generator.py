# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Self

import numpy as np
import slangpy as spy


class UniformSampleGenerator:
    """Create falcor2 sample generators from one persistent seed stream."""

    def __init__(
        self,
        module: spy.Module,
        shape: tuple[int, ...],
        seed: int = 0,
    ):
        self.module = module
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sg = None
        self.reshape(shape)

    def reshape(
        self,
        shape: tuple[int, ...],
    ) -> Self:
        shape = tuple(shape)
        if self.sg is not None and tuple(self.sg.shape) == shape:
            return self

        seeds = self.rng.integers(0, 2**32, size=shape, dtype=np.uint32)
        self.sg = self.module.UniformSampleGenerator(seeds)
        return self

    def get_this(self) -> Any:
        return self.sg
