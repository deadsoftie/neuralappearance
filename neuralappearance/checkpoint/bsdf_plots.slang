// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
import util.texture_helpers;
import util.math_helpers;

// Generate a grid of fixed spatial locations and angular slices used to compare
// reference and neural lobes consistently across checkpoints.
void get_plots_coords(
    float2 min_uv,
    float2 max_uv,
    int num_uvs,
    int num_samples,
    uint2 resolution,
    int mip_level,
    int3 call_id,
    out float2 uv,
    out float3 wi,
    out float3 wo
)
{
    // ``spy.call_id`` reverses the index order relative to Python.
    int col = call_id.z;
    int row = call_id.y;
    int sample_idx = call_id.x;

    // ``uv`` is a regular ``num_uvs x num_uvs`` grid inside
    // ``(uv_min, uv_max)``.
    float2 duv = float2(max_uv - min_uv) / num_uvs;
    uv = min_uv + duv * (float2(col, row) + 0.5f);
    // Be careful about what UV coordinates are used exactly. Training happens
    // at discrete texel centers at the original material texture resolution.
    // The produced latents should never be interpolated directly. (Renderings
    // use stochastic filtering instead to still have bilinearly interpolated
    // looks.)
    uv = snap_to_nearest_texel_center(uv, resolution >> mip_level, false);

    // ``wi`` contains ``num_uvs`` regular samples inside (90°, 0°).
    // In each col, offset the sampled theta by one, with wrap-around.
    float dtheta = float.PI_OVER_TWO / num_uvs;
    int theta_idx = (num_uvs + row - col) % num_uvs;
    float theta_i = float.PI_OVER_TWO - dtheta * (theta_idx + 0.5f);
    wi = float3(sin(theta_i), 0.f, cos(theta_i));

    // ``wo`` contains ``num_samples`` regular samples across [-180°, +180°].
    dtheta = float.TWO_PI / (num_samples - 1);
    float theta_o = -float.PI + dtheta * sample_idx;
    wo = float3(sin(theta_o), 0.f, cos(theta_o));
}
