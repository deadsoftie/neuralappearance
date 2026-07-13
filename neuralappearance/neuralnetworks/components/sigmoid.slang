// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing neural_networks;

namespace Activation
{

[BackwardDerivative(sigmoid_bwd)]
public T sigmoid<T : IReal>(T x)
{
    T neg_e = exp(-abs(x));
    T denom = neg_e + T(1);
    T num = x > T(0) ? T(1) : neg_e;
    return num / denom;
}

public void sigmoid_bwd<T : IReal>(inout DifferentialPair<T> x, T.Differential grad)
{
    T neg_e = exp(-abs(x.p));
    T d_x = neg_e / ((neg_e + T(1)) * (neg_e + T(1)));

    x = diffPair(
        x.p,
        __slang_noop_cast<T.Differential>(d_x * __slang_noop_cast<T>(grad))
    );
}

[BackwardDerivative(sigmoid_bwd_coop)]
public CoopVec<T, N> sigmoid<T : IReal, int N>(CoopVec<T, N> x)
{
    let one = CoopVec<T, N>(T(1));
    let zero = CoopVec<T, N>(T(0));
    let neg_e = exp(min(x, -x));
    let mask = step(zero, x);
    let denom = neg_e + one;
    let num = neg_e + (one - neg_e) * mask;

    return num / denom;
}

public void sigmoid_bwd_coop<T : IReal, int N>(
    inout DifferentialPair<CoopVec<T, N>> x,
    CoopVec<T, N> grad
)
{
    let one = CoopVec<T, N>(T(1));
    let neg_e = exp(min(x.p, -x.p));
    let d_x = neg_e / ((neg_e + one) * (neg_e + one));

    x = diffPair(x.p, d_x * grad);
}

public struct Sigmoid<T : IReal, int Width>
{
    public int model_id;

    [BackwardDifferentiable, PreferRecompute]
    T activate(T x) { return sigmoid(x); }
}

public extension<T : IReal, int Width> Sigmoid<T, Width>
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

public extension<T : IReal, int Width> Sigmoid<T, Width>
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

public extension<T : IReal, int Width> Sigmoid<T, Width>
    : ICoopRecordableModel<T, Width, Width>
{
    [BackwardDifferentiable, PreferRecompute]
    public CoopVec<T, Width> model_forward(CoopVec<T, Width> x)
    {
        return sigmoid(x);
    }
}

}
