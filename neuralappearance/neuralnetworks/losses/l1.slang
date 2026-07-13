// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing neural_networks;

namespace Losses
{

public struct L1<int N> : ILossFunction<N>
{
    public float weights[N];

    public __init(float weights[N]) { this.weights = weights; }

    public __init()
    {
        for (int i = 0; i < N; ++i)
            this.weights[i] = 1.f / float(N);
    }

    [NoDiffThis, BackwardDifferentiable]
    public float eval(float pred[N], float ref[N])
    {
        float loss = 0.f;
        [ForceUnroll]
        for (int i = 0; i < N; ++i)
        {
            float diff = abs(ref[i] - pred[i]);
            loss += weights[i] * diff;
        }
        return loss;
    }
}
};
