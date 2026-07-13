// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing neural_networks;

namespace Losses
{

public struct L1WithPowerRoot<int N> : ILossFunction<N>
{
    public float weights[N];
    public float power;

    public __init(float weights[N], float power)
    {
        this.weights = weights;
        this.power = power;
    }

    public __init(float power)
    {
        for (int i = 0; i < N; ++i)
            this.weights[i] = 1.f / float(N);
        this.power = power;
    }

    [NoDiffThis, BackwardDifferentiable]
    float nth_root(float x) { return pow(x, 1.f / this.power); }

    [NoDiffThis, BackwardDifferentiable]
    public float eval(float pred[N], float ref[N])
    {
        float loss = 0.f;
        [ForceUnroll]
        for (int i = 0; i < N; ++i)
        {
            float r = nth_root(ref[i]);
            float p = nth_root(pred[i]);
            float diff = abs(r - p);
            loss += weights[i] * diff;
        }
        return loss;
    }
}

}
