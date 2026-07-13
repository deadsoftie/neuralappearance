// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off

// Per-pixel masked accumulator for ``HybridPathTracer`` auxiliary rendering.
//
// Add ``L`` to ``result`` wherever ``hit_mask`` is nonzero.

void accumulate_aux_masked<int NumChannels>(
    float[NumChannels] L,
    uint hit_mask,
    inout float[NumChannels] result,
)
{
    if (hit_mask == 0)
        return;

    [unroll]
    for (int i = 0; i < NumChannels; i++)
    {
        result[i] += L[i];
    }
}
