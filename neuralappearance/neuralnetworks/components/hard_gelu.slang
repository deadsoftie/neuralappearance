// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing neural_networks;

namespace Activation
{

// Cheap GELU approximation used by NVIDIA's shipped RTXNTC decoder MLP
// (RTXNTC-Library/src/RegressionKernels.h, ``tin::ActHGELUClamp``):
//   f(x) = 0                   for x <= -1.5
//   f(x) = x/3 * (x + 1.5)     for -1.5 < x < 1.5
//   f(x) = x                   for x >= 1.5
// Like real GELU (and unlike ReLU/LeakyReLU/SmeLU), this dips slightly
// below zero just left of the origin before rejoining the identity line,
// so it is not expressible as a min/max clamp of the two linear pieces the
// way the other activations in this folder are. Written branchlessly
// instead: the quadratic clamped to [-1.5, 1.5], plus whatever excess `x`
// has beyond +1.5 (there is never excess below -1.5, since the quadratic
// already equals 0 there, matching the left piece exactly).
public struct HardGELU<T : IReal, int Width>
{
    static const T lower = T(-1.5f);
    static const T upper = T(1.5f);
    static const T third = T(1.0f) / T(3.0f);

    public int model_id;

    [BackwardDifferentiable, PreferRecompute]
    public T activate(T x)
    {
        let clamped = max(lower, min(x, upper));
        let quadratic = clamped * third * (clamped + T(1.5f));
        return quadratic + max(x - upper, T(0.0f));
    }
}

public extension<T : IReal, int Width> HardGELU<T, Width>
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

public extension<T : IReal, int Width> HardGELU<T, Width>
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

public extension<T : IReal, int Width> HardGELU<T, Width>
    : ICoopRecordableModel<T, Width, Width>
{
    [BackwardDifferentiable]
    public CoopVec<T, Width> model_forward(CoopVec<T, Width> x)
    {
        typealias CV = CoopVec<T, Width>;

        let lower_vec = no_diff CV(lower);
        let upper_vec = no_diff CV(upper);
        let third_vec = no_diff CV(third);
        let half_const = no_diff CV(T(1.5f));
        let zero = no_diff CV(T(0.0f));

        let clamped = max(lower_vec, min(x, upper_vec));
        let quadratic = clamped * third_vec * (clamped + half_const);
        let excess = max(x - upper_vec, zero);

        return quadratic + excess;
    }
}

}
