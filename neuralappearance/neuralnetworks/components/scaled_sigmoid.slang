// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing neural_networks;

namespace Activation
{

[BackwardDerivative(scaled_sigmoid_bwd)]
public T scaled_sigmoid<T : IReal>(T x, no_diff T scale)
{
    T neg_e = exp(-abs(x));
    T pos = scale / (scale * neg_e + T(1));
    T neg = scale * neg_e / (neg_e + scale);
    return x > T(0) ? pos : neg;
}

public void scaled_sigmoid_bwd<T : IReal>(inout DifferentialPair<T> x, T scale, T.Differential grad)
{
    T neg_e = exp(-abs(x.p));
    T denom = x.p > T(0) ? T(1) + scale * neg_e : scale + neg_e;
    T d_x = scale * scale * neg_e / (denom * denom);

    x = diffPair(
        x.p,
        __slang_noop_cast<T.Differential>(d_x * __slang_noop_cast<T>(grad))
    );
}

[BackwardDerivative(scaled_sigmoid_bwd_coop)]
public CoopVec<T, N> scaled_sigmoid<T : IReal, int N>(CoopVec<T, N> x, no_diff T scale)
{
    typealias CV = CoopVec<T, N>;

    let one = CV(T(1));
    let zero = CV(T(0));
    let scale_vec = CV(scale);
    let neg_e = exp(min(x, -x));
    let mask = step(zero, x);
    let pos = scale_vec / (scale * neg_e + one);
    let neg = scale * neg_e / (neg_e + scale_vec);
    let y = pos * mask + (one - mask) * neg;

    return y;
}

public void scaled_sigmoid_bwd_coop<T : IReal, int N>(
    inout DifferentialPair<CoopVec<T, N>> x,
    T scale,
    CoopVec<T, N> grad
)
{
    typealias CV = CoopVec<T, N>;

    let one = CV(T(1));
    let zero = CV(T(0));
    let scale_vec = CV(scale);
    let neg_e = exp(min(x.p, -x.p));
    let mask = step(zero, x.p);
    let pos_denom = one + scale * neg_e;
    let neg_denom = scale_vec + neg_e;
    let denom = pos_denom * mask + (one - mask) * neg_denom;
    let d_x = scale * scale * neg_e / (denom * denom);

    x = diffPair(x.p, d_x * grad);
}

public struct ScaledSigmoid<T : IReal, int Width>
{
    public int model_id;
    public T scale;
}

public extension<T : IReal, int Width> ScaledSigmoid<T, Width>
    : IRecordableModel<T[Width], T[Width]>
{
    [BackwardDifferentiable, PreferRecompute, NoDiffThis]
    public T[Width] model_forward(T x[Width])
    {
        T y[Width];
        [ForceUnroll]
        for (int i = 0; i < Width; ++i)
            y[i] = scaled_sigmoid(x[i], scale);
        return y;
    }
}

public extension<T : IReal, int Width> ScaledSigmoid<T, Width>
    : IRecordableModel<vector<T, Width>, vector<T, Width>>
{
    [BackwardDifferentiable, PreferRecompute, NoDiffThis]
    public vector<T, Width> model_forward(vector<T, Width> x)
    {
        vector<T, Width> y;
        [ForceUnroll]
        for (int i = 0; i < Width; ++i)
            y[i] = scaled_sigmoid(x[i], scale);
        return y;
    }
};

public extension<T : IReal, int Width> ScaledSigmoid<T, Width>
    : ICoopRecordableModel<T, Width, Width>
{
    [BackwardDifferentiable, PreferRecompute, NoDiffThis]
    public CoopVec<T, Width> model_forward(CoopVec<T, Width> x)
    {
        return scaled_sigmoid(x, scale);
    }
}

}
