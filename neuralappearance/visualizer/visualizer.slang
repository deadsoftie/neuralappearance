// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
__exported import util.math_helpers;
__exported import util.texture_helpers;

// BSDF plotting is a three-stage image pipeline: encode an outgoing hemisphere
// direction in color.xyz, replace it with a BSDF evaluation, then apply display
// mapping and draw the selected incident direction marker.
void prepare_bsdf_plot(int2 call_id, int2 resolution, out float4 color)
{
    float2 xy = (call_id + 0.5f) / resolution;
    xy = 2.f * xy - 1.f;
    float z = sqrt(max(0.f, 1.f - xy.x * xy.x - xy.y * xy.y));
    color = float4(xy, z, 1.f);
}

void colormap_bsdf_plot(float exposure, int channel_idx, int colormap_idx, inout float4 color)
{
    float3 bsdf = color.xyz;

    float scale = pow(2.f, exposure);

    // Simple case: show the actual RGB color.
    if (channel_idx == 0)
    {
        color.xyz = scale * bsdf;
        return;
    }

    // Otherwise, reduce down to a scalar value...
    float s = 0.f;
    if (channel_idx == 1) // R.
        s = bsdf.x;
    else if (channel_idx == 2) // G.
        s = bsdf.y;
    else if (channel_idx == 3) // B.
        s = bsdf.z;
    else if (channel_idx == 4) // Luminance.
        s = luminance(bsdf);
    s *= scale;

    // ... and apply a colormap.
    if (colormap_idx == 0)
        color.xyz = colormap_viridis(s);
    else if (colormap_idx == 1)
        color.xyz = colormap_plasma(s);
    else if (colormap_idx == 2)
        color.xyz = colormap_magma(s);
    else if (colormap_idx == 3)
        color.xyz = colormap_inferno(s);
    else if (colormap_idx == 4)
        color.xyz = colormap_gray(s);
}

void finalize_bsdf_plot(int2 call_id, int2 resolution, float3 selected_wi, inout float4 color)
{
    float2 xy = (call_id + 0.5f) / resolution;
    xy = 2.f * xy - 1.f;

    if (length(xy) > 1.f)
        color = float4(0.f, 0.f, 0.f, 1);

    if (selected_wi.z < 0.f)
        return;

    const float radius = 2.f * 6.f / resolution.x;
    float2 diff = xy - selected_wi.xy;
    if (dot(diff, diff) <= radius * radius) {
        color = float4(1.f, 1.f, 1.f, 1.f);
    }
    if (dot(diff, diff) <= 0.66f * radius * radius) {
        color = float4(1.f, 0.f, 0.f, 1.f);
    }
}

void visualize_aux_signal<int NumChannels>(
    float[NumChannels] values,
    uint hit_mask,
    int offset,
    int size,
    bool is_normal,
    out float4 color
)
{
    color = float4(0.f, 0.f, 0.f, 1.f);
    if (hit_mask == 0)
        return;

    float3 value = float3(values[offset]);
    if (size > 1)
        value.y = values[offset + 1];
    if (size > 2)
        value.z = values[offset + 2];

    if (is_normal)
        value = 0.5f * value + 0.5f;
    color.xyz = saturate(value);
}
