// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing neural_networks;

namespace Activation
{

// RTXNTC's exact decoder activation, ``HGELUClamp`` (RTXNTC-Library
// include/libntc/shaders/Inference.hlsli, ``NtcHGELUClamp_ForwardFloat``):
//   f(x) = min(x, 3) * saturate(x / 3 + 0.5)
// Same as ``HardGELU`` up to x = 1.5, but the linear piece is capped at 3
// instead of continuing as the identity. ``HardGELU`` stays unclamped for the
// jointly trained decoder; this variant exists so weights exported by ntc-cli
// decode bit-for-bit the same function they were trained with.
public struct HardGELUClamp<T : IReal, int Width>
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
        return quadratic + min(max(x - upper, T(0.0f)), upper);
    }
}

public extension<T : IReal, int Width> HardGELUClamp<T, Width>
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

public extension<T : IReal, int Width> HardGELUClamp<T, Width>
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

public extension<T : IReal, int Width> HardGELUClamp<T, Width>
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
        let excess = min(max(x - upper_vec, zero), upper_vec);

        return quadratic + excess;
    }
}

}
