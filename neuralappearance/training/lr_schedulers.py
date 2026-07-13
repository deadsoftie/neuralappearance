# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools

import util


class LRScheduler:
    """Base class for learning rate schedulers."""

    def __init__(self, num_iterations: int):
        self.num_iterations = num_iterations

    def get_scale(self, iteration: int) -> float:
        """Calculate the current scale based on the scheduler."""
        raise NotImplementedError('Subclasses should implement this method.')


class CosineAnnealingLRScheduler(LRScheduler):
    def __init__(self, start_scale: float, end_scale: float, num_iterations: int):
        super().__init__(num_iterations)
        self.start_scale = start_scale
        self.end_scale = end_scale

    def get_scale(self, iteration: int) -> float:
        """Calculate the scale from the cosine-annealing schedule."""
        iteration = min(iteration, self.num_iterations)
        scale = util.cosine_falloff(
            iteration, self.num_iterations, self.start_scale, self.end_scale
        )
        return scale


class LRSchedulerChain(LRScheduler):
    """Apply multiple learning-rate schedules in consecutive phase segments."""

    def __init__(self, schedulers: list[LRScheduler]):
        self.schedulers = schedulers
        num_iterations = [scheduler.num_iterations for scheduler in schedulers]
        self.num_iterations = sum(num_iterations)
        self.num_iterations_cumul = list(itertools.accumulate(num_iterations))

    def get_scale(self, iteration: int) -> float:
        # Find the scheduler covering the current iteration.
        for idx, end_iter in enumerate(self.num_iterations_cumul):
            if iteration <= end_iter:
                # Express the iteration relative to this scheduler's start.
                prev_iterations = self.num_iterations_cumul[idx - 1] if idx > 0 else 0
                local_iteration = iteration - prev_iterations
                # Evaluate the selected scheduler.
                return self.schedulers[idx].get_scale(local_iteration)

        # Past the chain, retain the final scheduler's last value.
        return self.schedulers[-1].get_scale(self.schedulers[-1].num_iterations)
