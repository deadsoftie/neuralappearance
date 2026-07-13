# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import slangpy as spy


def cosine_falloff(
    iteration: int,
    max_iterations: int,
    start: float = 1.0,
    end: float = 0.0,
) -> float:
    """Interpolate from ``start`` to ``end`` over a cosine half-wave."""

    if iteration >= max_iterations:
        return end
    alpha = alpha = 0.5 * (1 + math.cos(iteration * math.pi / max_iterations))
    return end + alpha * (start - end)


def udim_offset(udim_id: int) -> spy.float2:
    """Convert a UDIM number such as 1001 into its integer UV-tile offset."""

    return spy.float2(
        (udim_id - 1001) % 10,
        (udim_id - 1001) // 10,
    )
