// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing neural_networks;

namespace Activation
{

public struct Identity<T : IReal, int Width>
{
    public int model_id;

    [BackwardDifferentiable, PreferRecompute]
    T activate(T x) { return x; }
}

public extension<T : IReal, int Width> Identity<T, Width>
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

public extension<T : IReal, int Width> Identity<T, Width>
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
}

public extension<T : IReal, int Width> Identity<T, Width>
    : ICoopRecordableModel<T, Width, Width>
{
    [BackwardDifferentiable]
    public CoopVec<T, Width> model_forward(CoopVec<T, Width> x)
    {
        return x;
    }
}

}
