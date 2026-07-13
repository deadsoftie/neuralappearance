# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from training import TrainingTargets

losses = [
    'L1',
    'L1WithPowerLog',
    'L1WithPowerRoot',
    'L1WithSafeLog',
    'L2',
    'RelativeL1',
    'RelativeL2',
    'SMAPE',
]


class LossFunction(dict):
    """Slang-bindable loss configuration for packed training targets."""

    def __init__(self, config: dict, targets: TrainingTargets):
        type = config['type']
        if type not in losses:
            raise ValueError(f'Unknown loss function type: {type}')

        self.type_name = f'Losses::{type}<{targets.num_channels}>'

        result = {
            '_type': self.type_name,
            'weights': targets.loss_weights,
        }
        if type == 'L1WithPowerLog' or type == 'L1WithPowerRoot':
            result['power'] = config.get('power', 3.0)
        elif type == 'RelativeL1' or type == 'RelativeL2' or type == 'SMAPE':
            result['epsilon'] = config.get('epsilon', 1e-4)

        super().__init__(result)
