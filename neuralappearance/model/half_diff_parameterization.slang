// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
import neural_networks;
import base_types;
import util.math_helpers;

interface IHalfDiffParameterization<
    int NumLatents,
    int NumFrames
> : IModel<BsdfDecoderInputWithFrames<NumLatents, NumFrames>, BsdfDecoderInputWithFrames<NumLatents, NumFrames>>
{
    [Differentiable]
    static void transform(float3 wi, float3 wo, out float3 wh, out float3 wd);
}

// All implementations have a shared ``forward`` function.
extension<
    int NumLatents,
    int NumFrames,
    HalfDiffParameterization : IHalfDiffParameterization<NumLatents, NumFrames>
> HalfDiffParameterization
{
    [Differentiable]
    IOType forward(IOType x)
    {
        // Mutating only wh/wd preserves metadata, latents, and the original
        // directions for flatteners that consume multiple representations.
        [ForceUnroll]
        for (int i = 0; i < NumFrames; ++i)
            HalfDiffParameterization.transform(x.wi[i], x.wo[i], x.wh[i], x.wd[i]);
        return x;
    }
}

// Parameterization as described in "A New Change of Variables for Efficient
// BRDF Representation" (Rusinkiewicz 1998).
// Uses the spherical angles of the half-vector to rotate ``wi`` into ``wd``.
struct RusinkiewiczParameterization<int NumLatents, int NumFrames>
    : IHalfDiffParameterization<NumLatents, NumFrames>
{
    [Differentiable]
    static void transform(float3 wi, float3 wo, out float3 wh, out float3 wd)
    {
        // Compute half-vector.
        wh = normalize(wi + wo);

        // Compute the different vector from ``wi`` by rotating around Z and Y
        // based on the elevation and azimuth of ``wh``.
        float2 wh_sph = cartesian_to_spherical_rad(wh);
        float3 temp = rotate_z(wi, -wh_sph.y);
        wd = rotate_y(temp, -wh_sph.x);
    }
}

// This alternate parameterization rotates ``wi`` into ``wd`` along the
// shortest arc.
struct StableRusinkiewiczParameterization<int NumLatents, int NumFrames>
    : IHalfDiffParameterization<NumLatents, NumFrames>
{
    [Differentiable]
    static void transform(float3 wi, float3 wo, out float3 wh, out float3 wd)
    {
        // Compute half-vector.
        wh = normalize(wi + wo);

        // Compute the difference vector from wi by the smallest rotation that
        // brings wh to Z. Numerically stable version of rotating wi by
        // acos(wi.z) around cross(wh, Z).
        // Expanded Rodrigues formula avoids spherical angles and their unstable
        // derivatives near the pole. The denominator is singular only at -Z,
        // which is outside the valid reflective upper-hemisphere configuration.
        float3 v = float3(wh.y, -wh.x, 0.f);
        wd = wi * wh.z + cross(v, wi) + dot(v, wi) * v / (1.f + wh.z);
    }
}
