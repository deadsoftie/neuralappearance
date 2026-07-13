// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing neural_networks;

namespace Activation
{

public struct LeakyReLU<T : IReal, int Width>
{
    public int model_id;
    public T negative_slope;

    [NoDiffThis, BackwardDifferentiable, PreferRecompute]
    T activate(T x) { return max(x, T(0.0f)) + min(x, T(0.0f)) * negative_slope; }
}

public extension<T : IReal, int Width> LeakyReLU<T, Width>
    : IRecordableModel<T[Width], T[Width]>
{
    [BackwardDifferentiable]
    public T[Width] model_forward(T x[Width])
    {
        T y[Width];
        [ForceUnroll]
        for (int i = 0; i < Width; ++i)
            y[i] = activate(x[i]);
        return y;
    }
}

public extension<T : IReal, int Width> LeakyReLU<T, Width>
    : IRecordableModel<vector<T, Width>, vector<T, Width>>
{
    [BackwardDifferentiable]
    public vector<T, Width> model_forward(vector<T, Width> x)
    {
        vector<T, Width> y;
        [ForceUnroll]
        for (int i = 0; i < Width; ++i)
            y[i] = activate(x[i]);
        return y;
    }
};

public extension<T : IReal, int Width> LeakyReLU<T, Width>
    : ICoopRecordableModel<T, Width, Width>
{
    [NoDiffThis, BackwardDifferentiable]
    public CoopVec<T, Width> model_forward(CoopVec<T, Width> x)
    {
        let zero = no_diff CoopVec<T, Width>(T(0.0f));
        let slope = no_diff CoopVec<T, Width>(negative_slope);
        let positive = max(x, zero);
        let negative = min(x, zero) * slope;

        return positive + negative;
    }
}

}
