// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off

// Scale from UV coordinates to fractional texture indices.
// For resolution N, the range [0, 1]^2 will map to [-0.5, N-0.5]^2,
// where integer coordinates correspond to the texel centers.
// See also ``fractional_texel_to_uv`` for the inverse.
[BackwardDifferentiable]
float2 uv_to_fractional_texel(float2 uv, uint2 resolution, bool wrap)
{
    float2 texel_index;
    if (wrap)
    {
        // Wrap to stay inside [0, 1] domain. This assumes there are no UDIMs.
        uv = uv - no_diff floor(uv); // Map uv to canonical [0, 1]^2 range.
        texel_index = uv * resolution - 0.5f;
    }
    else
    {
        // No wrapping, used by default for the case with UDIMs.
        uv = max(0.f, uv);           // Discard negative UVs.
        uv = uv - no_diff floor(uv); // Map uv to canonical [0, 1]^2 range.
        texel_index = uv * resolution - 0.5f;
        texel_index = max(0.f, texel_index);
    }
    return texel_index;
}

// Scale from fractional texture indices to UV coordinates.
// For resolution N, the range [-0.5, N-0.5]^2 will map to [0, 1]^2.
// See also ``uv_to_fractional_texel`` for the inverse.
[BackwardDifferentiable]
float2 fractional_texel_to_uv(float2 texel, uint2 resolution)
{
    float2 uv = (texel + 0.5f) / resolution;
    return uv;
}

// For a given texture coordinate ``uv``, compute the bilinear interpolation
// weights of the four surrounding texels and return its fractional pixel
// index.
[BackwardDifferentiable]
float2 get_bilinear_interpolation_weights(float2 uv, uint2 resolution, bool wrap, out float weights[4])
{
    float2 texel = uv_to_fractional_texel(uv, resolution, wrap);

    // Compute bilinear weights based on fractional offset.
    float2 uv_offset = texel - no_diff floor(texel);
    weights[0] = (1.f - uv_offset.x) * (1.f - uv_offset.y);
    weights[1] = uv_offset.x * (1.f - uv_offset.y);
    weights[2] = (1.f - uv_offset.x) * uv_offset.y;
    weights[3] = uv_offset.x * uv_offset.y;

    return texel;
}

[BackwardDifferentiable]
void get_bilinear_interpolation_texels_and_weights(
    float2 uv,
    uint2 resolution,
    bool wrap,
    out uint2 texels[4],
    out float weights[4]
)
{
    float2 uv_scaled = get_bilinear_interpolation_weights(uv, resolution, wrap, weights);
    int2 tmp[4];
    tmp[0] = int2(floor(uv_scaled)); // top-left corner
    tmp[1] = tmp[0] + int2(1, 0);    // top-right corner
    tmp[2] = tmp[0] + int2(0, 1);    // bottom-left corner
    tmp[3] = tmp[0] + int2(1, 1);    // bottom-right corner
    if (wrap)
    {
        [ForceUnroll]
        for (int i = 0; i < 4; ++i)
            texels[i] = tmp[i] % resolution;
    }
    else // clamp
    {
        [ForceUnroll]
        for (int i = 0; i < 4; ++i)
            texels[i] = max(0, min(tmp[i], resolution - 1));
    }
}

// Stochastic bilinear texture filtering.
// For a given texture coordinate ``uv``, sample one of the four surrounding
// texels based on the bilinear interpolation weights.
[BackwardDifferentiable]
float2 sample_bilinear_texel_center(float2 uv, uint2 resolution, bool wrap, inout float sample)
{
    // Remove UDIM tile offset and work in a canonical [0, 1]^2 range.
    float2 udim_offset = no_diff floor(uv);

    // Compute the four interpolation weights and the upper left texel location.
    float weights[4];
    float2 uv_scaled = get_bilinear_interpolation_weights(uv, resolution, wrap, weights);
    int2 texel = int2(floor(uv_scaled));

    // Pick one of the four texels based on the random sample, while rescaling
    // it back into a uniform sample in [0, 1).
    if (sample < weights[0])
    {
        sample /= weights[0];
    }
    else if (sample < (weights[0] + weights[1]))
    {
        texel.x += 1;
        sample = (sample - weights[0]) / weights[1];
    }
    else if (sample < (weights[0] + weights[1] + weights[2]))
    {
        texel.y += 1;
        sample = (sample - weights[0] - weights[1]) / weights[2];
    }
    else
    {
        texel.xy += 1;
        sample = (sample - weights[0] - weights[1] - weights[2]) / weights[3];
    }

    if (wrap)
    {
        texel = texel % resolution;
    }
    else
    {
        texel = max(0, min(texel, resolution - 1));
    }

    // Compute corresponding UV location.
    uv = fractional_texel_to_uv(float2(texel), resolution);

    // Add back UDIM tile offset.
    return udim_offset + uv;
}

// Find the ``uv`` coordinate of the nearest texel center.
float2 snap_to_nearest_texel_center(float2 uv, uint2 resolution, bool wrap)
{
    // Remove UDIM tile offset and work in a canonical [0, 1]^2 range.
    float2 udim_offset = floor(uv);

    // Round to nearest texel.
    float2 texel = uv_to_fractional_texel(uv, resolution, wrap);
    uv = fractional_texel_to_uv(round(texel), resolution);

    // Add back UDIM tile offset.
    return udim_offset + uv;
}

float scalar_mip_level_from_footprint(float2 duv_dx, float2 duv_dy, uint2 resolution)
{
    float2 texel_dx = duv_dx * resolution;
    float2 texel_dy = duv_dy * resolution;

    float sqr_len_x = dot(texel_dx, texel_dx);
    float sqr_len_y = dot(texel_dy, texel_dy);

    float max_sqr_len = max(sqr_len_x, sqr_len_y);
    return 0.5f * log2(max(max_sqr_len, 1.f)); // values < 1 mean magnification.
}

int sample_integer_mip_level(float mip_level, float sample)
{
    return max(0, int(floor(mip_level + sample)));
}
