// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
__exported import falcor2.render;
import falcor2.utils;
import slangpy;

// Minimal hit information returned to Python when the user picks a point in the
// rendered scene. wi is reported in the hit's local shading frame.
struct TraceRayResult
{
    bool valid;
    int material_id;
    float2 uv;
    float3 wi;

    __init()
    {
        valid = false;
        material_id = -1;
        uv = float2(0.f);
        wi = float3(0.f);
    }
};

TraceRayResult trace_ray(float3 ray_p, float3 ray_d)
{
    TraceRayResult result = {};

    let ray = Ray(ray_p, ray_d);
    RayQuerySceneIntersector intersector;
    HitInfo hit = intersector.intersect(ray);

    if (hit.valid)
    {
        SurfaceInteraction si = g_scene.make_surface_interaction(hit, -ray.dir);
        let gi_desc = g_scene.get_geometry_instance_desc(hit.geometry_instance_id);

        result.valid = true;
        result.material_id = int(gi_desc.header.material_id);
        result.uv = si.uv;
        result.wi = si.shading_frame_ws.to_local(si.wi_ws);
    }

    return result;
}
