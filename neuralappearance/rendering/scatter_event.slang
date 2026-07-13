// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off

// These types carry external, non-falcor2 scatter evaluation requests.

// Per-sample input for material evaluation.
public struct ScatterEventQuery
{
    public float2 uv; // The surface UV coordinate.
    public float3 wi; // The incident direction.
    public float3 wo; // The NEE direction.
}

// Canonical scatter result type. 44 bytes.
public struct ScatterEventResult
{
    public float3 scatter_dir;    // 12 B. Importance-sampled outgoing direction.
    public float3 nee_weight;     // 12 B. ``f * cos`` in the NEE direction.
    public float nee_pdf;         //  4 B. BSDF mixture PDF for NEE.
    public float3 scatter_weight; // 12 B. ``f * cos`` in the sampled direction.
    public float scatter_pdf;     //  4 B. BSDF mixture PDF for scattering.

    // Default zero-init constructor (preserves ``= {}`` syntax).
    public __init()
    {
        scatter_dir = float3(0);
        nee_weight = float3(0);
        nee_pdf = 0;
        scatter_weight = float3(0);
        scatter_pdf = 0;
    }

    // Construct and sanitize (replaces NaN/Inf with zero).
    public __init(float3 scatter_dir, float3 nee_weight, float nee_pdf, float3 scatter_weight, float scatter_pdf)
    {
        this.scatter_dir = scatter_dir;
        this.nee_weight = nee_weight;
        this.nee_pdf = nee_pdf;
        this.scatter_weight = scatter_weight;
        this.scatter_pdf = scatter_pdf;
        sanitize();
    }

    // Replace any NaN/Inf values with zero.
    [mutating]
    public void sanitize()
    {
        if (any(isinf(nee_weight)) || any(isnan(nee_weight)))
            nee_weight = float3(0);
        if (any(isinf(scatter_weight)) || any(isnan(scatter_weight)))
            scatter_weight = float3(0);
        if (isinf(nee_pdf) || isnan(nee_pdf))
            nee_pdf = 0;
        if (isinf(scatter_pdf) || isnan(scatter_pdf))
            scatter_pdf = 0;
    }
}

// Half-precision version of ``ScatterEventResult``. This occupies 24 bytes.
// May be used by backends whose performance is sensitive to data size due to
// PCIe or IPC transfer.
public struct ScatterEventResultHalf
{
    public half3 scatter_dir;    // 6 B -- importance-sampled outgoing direction
    public half3 nee_weight;     // 6 B -- f*cos at NEE direction
    public half nee_pdf;         // 2 B -- BSDF mixture PDF at NEE direction
    public half3 scatter_weight; // 6 B -- f*cos at sampled direction
    public half scatter_pdf;     // 2 B -- BSDF mixture PDF at sampled direction
    public half pad;             // 2 B -- pad out to multiple of 4 bytes

    // Sanitize float inputs and clamp them to the largest finite half value.
    public __init(float3 scatter_dir, float3 nee_weight, float nee_pdf, float3 scatter_weight, float scatter_pdf)
    {
        static const float half_max = 65504.0;
        if (any(isnan(nee_weight)))
            nee_weight = float3(0);
        if (any(isnan(scatter_weight)))
            scatter_weight = float3(0);
        if (isnan(nee_pdf))
            nee_pdf = 0;
        if (isnan(scatter_pdf))
            scatter_pdf = 0;
        this.scatter_dir = half3(scatter_dir);
        this.nee_weight = half3(min(nee_weight, float3(half_max)));
        this.nee_pdf = half(min(nee_pdf, half_max));
        this.scatter_weight = half3(min(scatter_weight, float3(half_max)));
        this.scatter_pdf = half(min(scatter_pdf, half_max));
        pad = half(0);
    }

    // Convert to full-precision ``ScatterEventResult``.
    public ScatterEventResult to_float()
    {
        ScatterEventResult d;
        d.scatter_dir = float3(scatter_dir);
        d.nee_weight = float3(nee_weight);
        d.nee_pdf = float(nee_pdf);
        d.scatter_weight = float3(scatter_weight);
        d.scatter_pdf = float(scatter_pdf);
        return d;
    }
}
