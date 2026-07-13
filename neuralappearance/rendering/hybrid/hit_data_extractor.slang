// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
import rendering.hybrid.hybrid_path_state;
import util.texture_helpers;

// This auxiliary kernel traces primary rays and writes hit data and stochastic
// filtering samples to per-pixel buffers.

extern static const uint TARGET_MATERIAL_ID;

// SlangPy dispatches primary-ray hit extraction over the image dimensions.
public void extract_hit_data(
    uniform Camera camera,
    out float2 uv,
    out float3 wi,
    out float3 wo,
    out uint hit_mask,
    out uint mip_level,
    out float texel_sample,
    uniform uint iteration,
    uniform int fixed_mip_level,
    uniform int texture_resolution,
    uniform int num_mip_levels,
    uint2 tid
)
{
    // Generate jittered primary ray.
    HybridPathState path_state = generate_path_state(
        camera,
        iteration,
        int8_t(fixed_mip_level),
        false,
        tid
    );

    // Intersect the scene with inline ray tracing.
    RayQuerySceneIntersector intersector;
    HitInfo hit = intersector.intersect(Ray(path_state.ray_origin, path_state.ray_dir));
    if (!hit.valid)
    {
        uv = float2(0);
        wi = float3(0);
        wo = float3(0);
        hit_mask = 0;
        mip_level = 0u;
        texel_sample = 0.f;
        return;
    }

    let gi_desc = g_scene.get_geometry_instance_desc(hit.geometry_instance_id);
    if (uint(gi_desc.header.material_id) != TARGET_MATERIAL_ID)
    {
        uv = float2(0);
        wi = float3(0);
        wo = float3(0);
        hit_mask = 0;
        mip_level = 0u;
        texel_sample = 0.f;
        return;
    }

    SurfaceInteraction si = g_scene.make_surface_interaction(hit, -path_state.ray_dir);

    float mip_sample = next_float(path_state.sg);
    texel_sample = next_float(path_state.sg);
    int selected_mip_level;
    if (fixed_mip_level >= 0)
    {
        selected_mip_level = fixed_mip_level;
    }
    else
    {
        float2 duv_dx;
        float2 duv_dy;
        compute_primary_uv_differentials(
            path_state.ray_dir,
            path_state.ray_origin,
            hit,
            si,
            path_state.pixel_spread,
            duv_dx,
            duv_dy
        );
        float lod = scalar_mip_level_from_footprint(duv_dx, duv_dy, texture_resolution);
        selected_mip_level = sample_integer_mip_level(lod, mip_sample);
    }
    mip_level = uint(clamp(selected_mip_level, 0, num_mip_levels - 1));

    // Match the grazing-angle normal adjustment in
    // ``hybrid_path_tracer.slang::intersect()``.
    float3 adj_normal = adjust_normal(path_state.ray_dir, si.normal_ws, si.shading_frame_ws.normal);
    let frame = Frame(adj_normal, si.shading_frame_ws.tangent);

    uv = si.uv;
    wi = frame.to_local(si.wi_ws);
    // Auxiliary signals do not depend on the outgoing direction.
    wo = float3(0, 0, 1);
    hit_mask = 1;
}
