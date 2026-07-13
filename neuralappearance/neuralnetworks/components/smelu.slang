// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing neural_networks;

namespace Activation
{

public struct SmeLU<T : IReal, int Width>
{
    /* Linear ramp (slope beta) for x < -alpha.
       Linear ramp (slope 1) for x > alpha.
       Smooth quadratic interpolant for -alpha < x < alpha.
       a, b, c are the coefficients of the quadratic a*x^2 + b*x + c. */
    static const T alpha = T(0.5f);
    static const T beta = T(0.01f);
    static const T a = (T(1.f) - beta) / (T(4.f) * alpha);
    static const T b = (T(1.f) + beta) / T(2.f);
    static const T c = alpha * (T(1.f) - beta) / T(4.f);

    public int model_id;

    [BackwardDifferentiable, PreferRecompute]
    public T activate(T x)
    {
        let left = max(beta * x, x + alpha * (T(1.f) - beta));
        let middle = a * x * x + b * x + c;
        let right = max(x, alpha);

        return min(left, min(middle, right));
    }
}

public extension<T : IReal, int Width> SmeLU<T, Width>
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

public extension<T : IReal, int Width> SmeLU<T, Width>
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

public extension<T : IReal, int Width> SmeLU<T, Width>
    : ICoopRecordableModel<T, Width, Width>
{
    [BackwardDifferentiable]
    public CoopVec<T, Width> model_forward(CoopVec<T, Width> x)
    {
        typealias CV = CoopVec<T, Width>;

        let beta_vec = no_diff CV(beta);
        let alpha_vec = no_diff CV(alpha);
        let a_vec = no_diff CV(a);
        let b_vec = no_diff CV(b);
        let c_vec = no_diff CV(c);
        let shoulder = no_diff CV(T(1.f) - beta);

        let left_linear = beta_vec * x;
        let left_blend = x + alpha_vec * shoulder;
        let left = max(left_linear, left_blend);
        let middle = a_vec * x * x + b_vec * x + c_vec;
        let right = max(x, alpha_vec);
        let y = min(left, min(middle, right));

        return y;
    }
}

}
