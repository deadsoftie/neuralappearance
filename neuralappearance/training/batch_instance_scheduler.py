# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


class BatchInstanceScheduler:
    """Step through instance-count and batch-size schedules during a phase.

    Each list divides the phase into equal-length periods. This lets training
    prune model instances over time while increasing the samples assigned to
    the survivors.
    """

    def __init__(
        self,
        num_instances: list[int],
        batch_sizes: list[int | str],
    ):
        self.num_instances = num_instances
        if not self.num_instances:
            raise ValueError('Instance-count schedule must not be empty.')

        self.batch_sizes = []
        for batch_size in batch_sizes:
            if isinstance(batch_size, int):
                self.batch_sizes.append(batch_size)
                continue

            if not isinstance(batch_size, str):
                raise TypeError(f'Unsupported batch size type: {type(batch_size).__name__}')

            value = batch_size.strip()
            if value.endswith(('k', 'K')):
                batch_size = int(float(value[:-1]) * 1024)
            elif value.endswith(('m', 'M')):
                batch_size = int(float(value[:-1]) * 1024 * 1024)
            else:
                batch_size = int(float(value))
            self.batch_sizes.append(batch_size)
        if not self.batch_sizes:
            raise ValueError('Batch-size schedule must not be empty.')

    def num_instances_at(self, iteration: int, num_iterations: int) -> int:
        return self._value_at(self.num_instances, iteration, num_iterations)

    def batch_size_at(self, iteration: int, num_iterations: int) -> int:
        return self._value_at(self.batch_sizes, iteration, num_iterations)

    def _value_at(self, values: list[int], iteration: int, num_iterations: int) -> int:
        if num_iterations <= 0:
            return values[-1]

        period = num_iterations / len(values)
        index = min(int(iteration / period), len(values) - 1)
        return values[index]
