# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ..basetypes import Real
from .optimizer import Optimizer


class FullPrecisionOptimizer(Optimizer):
    """Run a nested optimizer against float copies of half-precision parameters.

    This avoids accumulated half-precision rounding in optimizer updates. Half
    parameters use a float shadow copy; float parameters pass directly to the
    nested optimizer. ``gradient_scale`` is removed when gradients become float.
    """

    def __init__(self, nested_optimizer: Optimizer, gradient_scale: float = 1.0):
        super().__init__()

        self.nested_optim = nested_optimizer
        self.gradient_scale = gradient_scale

    def update_state(self):
        self.nested_optim.update_state()

    def get_type_name(self, dtype: Real) -> str:
        return f'FullPrecisionOptimizer<{dtype}, {self.nested_optim.get_type_name(Real.float)}>'

    def get_this(self):
        return {'gradient_scale': self.gradient_scale, 'nested_optim': self.nested_optim.get_this()}
