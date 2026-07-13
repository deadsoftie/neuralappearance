# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ..basetypes import Real
from .optimizer import Optimizer


class RegularizedAdamOptimizer(Optimizer):
    """Adam with L2 regularization and optional finite-gradient checks.

    Moments live in Slang. Python tracks the iteration and supplies
    bias-correction factors with each dispatch.
    """

    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        l2_regularization: float = 0.0,
        ignore_invalid_gradients: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.l2_regularization = l2_regularization
        self.ignore_invalid_gradients = ignore_invalid_gradients
        self.iteration = 1

    def update_state(self):
        self.iteration += 1

    def get_type_name(self, dtype: Real) -> str:
        return f'RegularizedAdamOptimizer<{dtype}>'

    def get_this(self):
        return {
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon,
            'l2_regularization': self.l2_regularization,
            'ignore_invalid_gradients': self.ignore_invalid_gradients,
            'mean_correction_factor': 1.0 / (1.0 - self.beta1**self.iteration),
            'variance_correction_factor': 1.0 / (1.0 - self.beta2**self.iteration),
        }
