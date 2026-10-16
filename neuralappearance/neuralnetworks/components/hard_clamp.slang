// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing neural_networks;

namespace Activation
{

// rtcnam: hard safety-net clamp, appended after BsdfDecoder's existing
// output_activation (ScaledSigmoid). See rtcnam/harness/bsdf_blowup_probe.py
// and rtcnam/harness/aliasing_probe.py's investigation: at grid_size_scale
// values other than the one validated preset, a fraction of single BSDF
// evaluations saturate ScaledSigmoid's hard ceiling (`scale`, e.g. 1e5),
// independent of (wi, wo) coverage or latent smoothness -- both of those
// axes were tried as training-time fixes and neither eliminated the render-
// level damage (latent_smoothness reduced outlier severity but not
// frequency). This clamps the value actually delivered to the renderer,
// which cannot help what the network represents internally but directly
// bounds the "firefly" magnitude a path trace can multiply across bounces.
// Has no learnable parameters -- safe to add to an already-trained
// checkpoint's decoder chain without retraining.
public struct HardClamp<T : IReal, int Width>
{
    public int model_id;
    public T cap;

    [BackwardDifferentiable, PreferRecompute]
    T activate(T x, no_diff T cap_val) { return clamp(x, T(0.0f), cap_val); }
}

public extension<T : IReal, int Width> HardClamp<T, Width>
    : IRecordableModel<T[Width], T[Width]>
{
    [BackwardDifferentiable]
    public T[Width] model_forward(T x[Width])
    {
        T y[Width];
        [ForceUnroll]
        for (int i = 0; i < Width; ++i)
            y[i] = activate(x[i], cap);
        return y;
    }
}

public extension<T : IReal, int Width> HardClamp<T, Width>
    : IRecordableModel<vector<T, Width>, vector<T, Width>>
{
    [BackwardDifferentiable]
    public vector<T, Width> model_forward(vector<T, Width> x)
    {
        vector<T, Width> y;
        [ForceUnroll]
        for (int i = 0; i < Width; ++i)
            y[i] = activate(x[i], cap);
        return y;
    }
};

public extension<T : IReal, int Width> HardClamp<T, Width>
    : ICoopRecordableModel<T, Width, Width>
{
    [BackwardDifferentiable]
    public CoopVec<T, Width> model_forward(CoopVec<T, Width> x)
    {
        let lo = no_diff CoopVec<T, Width>(T(0.0f));
        let hi = no_diff CoopVec<T, Width>(cap);
        let y = clamp(x, lo, hi);
        return y;
    }
}

}
