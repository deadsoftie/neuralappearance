// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing neural_networks;

namespace Losses
{

public struct SMAPE<int N> : ILossFunction<N>
{
    public float weights[N];
    public float epsilon;

    public __init(float weights[N], float epsilon)
    {
        this.weights = weights;
        this.epsilon = epsilon;
    }

    public __init(float epsilon)
    {
        for (int i = 0; i < N; ++i)
            this.weights[i] = 1.f / float(N);
        this.epsilon = epsilon;
    }

    [NoDiffThis, BackwardDifferentiable]
    public float eval(float pred[N], float ref[N])
    {
        float loss = 0.f;

        [ForceUnroll]
        for (int i = 0; i < N; ++i)
        {
            float diff = abs(pred[i] - ref[i]);
            // ``detach()`` excludes symmetric normalization from the gradient,
            // leaving only the absolute residual.
            float denominator = (abs(pred[i]) + abs(ref[i])) / 2.f + this.epsilon;
            loss += this.weights[i] * diff / detach(denominator);
        }

        return loss;
    }
}

}
