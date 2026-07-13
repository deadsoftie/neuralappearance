// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing neural_networks;

namespace Activation
{

public struct ELU<T : IReal, int Width>
{
    public int model_id;
    public T a;

    [NoDiffThis, BackwardDifferentiable, PreferRecompute]
    T activate(T x) { return a * (exp(-min(x, T(0.0f))) - T(1.0f)) + max(x, T(0.0f)); }
}

public extension<T : IReal, int Width> ELU<T, Width>
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

public extension<T : IReal, int Width> ELU<T, Width>
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

public extension<T : IReal, int Width> ELU<T, Width>
    : ICoopRecordableModel<T, Width, Width>
{
    [NoDiffThis, BackwardDifferentiable]
    public CoopVec<T, Width> model_forward(CoopVec<T, Width> x)
    {
        typealias CV = CoopVec<T, Width>;

        let zero = no_diff CV(T(0.0f));
        let one = no_diff CV(T(1.0f));
        let scale = no_diff CV(a);
        let exp_term = exp(-min(x, zero));
        let curved = scale * (exp_term - one);
        let linear = max(x, zero);

        return curved + linear;
    }
}

}
