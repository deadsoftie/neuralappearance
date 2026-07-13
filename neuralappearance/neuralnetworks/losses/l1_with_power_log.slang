// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing neural_networks;

namespace Losses
{

public struct L1WithPowerLog<int N> : ILossFunction<N>
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

    // Approximates ln(x) for power -> inf.
    [NoDiffThis, BackwardDifferentiable]
    float power_log(float x) { return this.power * (pow(x, 1.f / this.power) - 1.f); }

    [NoDiffThis, BackwardDifferentiable]
    public float eval(float pred[N], float ref[N])
    {
        float loss = 0.f;
        [ForceUnroll]
        for (int i = 0; i < N; ++i)
        {
            float r = power_log(ref[i]);
            float p = power_log(pred[i]);
            float diff = abs(r - p);
            loss += weights[i] * diff;
        }
        return loss;
    }
}

}
