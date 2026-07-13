// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
__exported import falcor2.render;
__exported import falcor2.utils;
__exported import slangpy;
__exported import rendering.scatter_event;

// Path-state types and helpers shared by ``hybrid_path_tracer.slang`` and its
// wavefront entry points.

// This sampler exposes its fields and adds per-pixel subpixel jittering, which
// the stock ``RaySampler`` does not provide.

public struct HybridRaySampler
{
    public Camera camera;
    public uint2 id;
    public float2 uv;
    public float2 distribution;

    public Ray sample_ray(inout UniformSampleGenerator sg)
    {
        float2 jittered_uv = this.uv + next_float2(sg) * this.distribution;
        return camera.get_ray(jittered_uv);
    }
}

// Bent-normal adjustment at grazing angles.
public float3 adjust_normal(float3 wi, float3 geom_normal, float3 shading_normal)
{
    const float k_cos_theta_threshold = 0.01f;
    float cos_theta = dot(-wi, shading_normal);
    if (cos_theta <= k_cos_theta_threshold)
    {
        float t = saturate(cos_theta * (1.f / k_cos_theta_threshold));
        return normalize(lerp(geom_normal, shading_normal, t));
    }
    return shading_normal;
}

// Compute primary-ray UV differentials for stochastic MIP selection.
public void compute_primary_uv_differentials(
    float3 ray_dir,
    float3 ray_origin,
    HitInfo hit,
    SurfaceInteraction si,
    float pixel_spread,
    out float2 duv_dx,
    out float2 duv_dy
)
{
    duv_dx = float2(0.f, 0.f);
    duv_dy = float2(0.f, 0.f);

    TriangleVertex tri[3] = g_scene.get_triangle_vertices(hit.geometry_instance_id, hit.primitive_index);
    float4x4 transform = g_scene.get_world_from_object(hit.geometry_instance_id);

    float3 local_dp1 = tri[1].position - tri[0].position;
    float3 local_dp2 = tri[2].position - tri[0].position;
    float3 dp1 = mul((float3x3)transform, local_dp1);
    float3 dp2 = mul((float3x3)transform, local_dp2);

    // ``TriangleVertex.uv`` is a ``float2[UV_COUNT]`` array; the
    // primary UV is at index 0.
    float2 duv1 = tri[1].uv[0] - tri[0].uv[0];
    float2 duv2 = tri[2].uv[0] - tri[0].uv[0];

    float det = duv1.x * duv2.y - duv1.y * duv2.x;
    if (abs(det) < 1e-10f)
    {
        return;
    }

    float inv_det = 1.f / det;
    float3 dpdu = (duv2.y * dp1 - duv1.y * dp2) * inv_det;
    float3 dpdv = (-duv2.x * dp1 + duv1.x * dp2) * inv_det;

    float3 up_guide = abs(ray_dir.y) < 0.999f ? float3(0.f, 1.f, 0.f) : float3(1.f, 0.f, 0.f);
    float3 cam_right = normalize(cross(ray_dir, up_guide));
    float3 cam_up = cross(cam_right, ray_dir);

    // ``HitInfo`` only carries barycentrics + geometry instance ID.
    // Reconstruct distance from position delta.
    float t = length(si.position_ws - ray_origin);
    float spread_radius = t * pixel_spread;
    float3 d_px_unproj = cam_right * spread_radius;
    float3 d_py_unproj = cam_up * spread_radius;

    float3 n = si.shading_frame_ws.normal;
    float n_dot_d = dot(n, ray_dir);
    float n_dot_d_clamped = (n_dot_d < 0.f ? -1.f : 1.f) * max(abs(n_dot_d), 1e-6f);

    float3 d_px = d_px_unproj - ray_dir * (dot(n, d_px_unproj) / n_dot_d_clamped);
    float3 d_py = d_py_unproj - ray_dir * (dot(n, d_py_unproj) / n_dot_d_clamped);

    float e = dot(dpdu, dpdu);
    float f = dot(dpdu, dpdv);
    float g = dot(dpdv, dpdv);
    float denom = e * g - f * f;
    if (denom < 1e-10f)
    {
        return;
    }
    float inv_denom = 1.f / denom;

    float d_px_du = dot(d_px, dpdu);
    float d_px_dv = dot(d_px, dpdv);
    duv_dx.x = (g * d_px_du - f * d_px_dv) * inv_denom;
    duv_dx.y = (e * d_px_dv - f * d_px_du) * inv_denom;

    float d_py_du = dot(d_py, dpdu);
    float d_py_dv = dot(d_py, dpdv);
    duv_dy.x = (g * d_py_du - f * d_py_dv) * inv_denom;
    duv_dy.y = (e * d_py_dv - f * d_py_du) * inv_denom;
}

// Generate a shadow ray from a surface interaction toward a sampled light.
public Ray create_visibility_ray(
    const SurfaceInteraction si,
    const DirectionSample sample,
    float3 geom_normal
)
{
    Ray ray;
    // Orient the offset normal so the origin is on the side facing the light.
    float orientation = (dot(si.normal_ws, sample.dir_ws) >= 0) ? 1 : -1;
    ray.origin = offset_ray_origin(si.position_ws, si.normal_ws * orientation);
    ray.dir = sample.dir_ws;
    ray.t_min = 0.0;
    ray.t_max = float.MAX;
    if (sample.emitter_type == EmitterType::emissive_triangle)
    {
        // Orient the target offset toward the shadow ray origin.
        float target_orientation = (dot(sample.normal_ws, sample.pos_ws - ray.origin) >= 0) ? 1 : -1;
        float3 target = offset_ray_origin(sample.pos_ws, sample.normal_ws * target_orientation);
        float3 to_light = target - ray.origin;
        ray.t_max = length(to_light);
        ray.dir = to_light / ray.t_max;
    }
    return ray;
}

// Per-path state shared across wavefront dispatches.

public enum HybridPathStatus : uint16_t
{
    terminated       = 0,  // The path is complete.
    intersect        = 1,  // The path awaits ``intersect()``.
    evaluate         = 2,  // The path awaits material evaluation.
    propagate        = 3,  // The path awaits ``propagate()``.
};


public struct HybridPathState
{
    // These values track random sampling and path progress.
    public UniformSampleGenerator sg;
    public HybridPathStatus status;
    // Depth starts at one for the primary ray and advances each bounce.
    public uint16_t depth;
    // This flag records whether the current NEE sample is valid.
    public uint16_t nee_valid;

    // Current ray origin and direction.
    // After ``intersect()``, ``ray_origin`` is the offset next-ray origin. For
    // intersecting paths, ``ray_dir`` is the scatter direction. For evaluation
    // and propagation, it temporarily holds the incoming direction.
    public float3 ray_origin;
    public float3 ray_dir;

    public float3 throughput;

    // ``intersect()`` sets this data and ``propagate()`` consumes
    // it. The values persist for next-bounce environment-map MIS.
    public float3 scatter_normal_ws;
    public float scatter_pdf;

    // ``propagate()`` uses this tangent to return ``wo`` to world space.
    public float3 frame_tangent_ws;

    // ``intersect()`` stores the current NEE sample for ``propagate()``.
    public float3 nee_light_weight;
    public float nee_light_pdf;

    // Pixel spread controls automatic MIP selection.
    public float pixel_spread;
    // A negative value enables dynamic MIP selection.
    public int8_t fixed_mip_level;
    // This flag selects the neural or analytic material path.
    public bool render_neural_material;
};

// Initialize path state and generate primary camera ray.
public HybridPathState generate_path_state(
    uniform Camera camera,
    uniform uint iteration,
    uniform int8_t fixed_mip_level,
    uniform bool render_neural_material,
    uint2 tid
)
{
    HybridRaySampler ray_sampler;
    ray_sampler.camera = camera;
    ray_sampler.id = tid;
    float2 pix_size = 1.0 / camera.dims;
    float2 uv_coord = float2(tid) / camera.dims;
    uv_coord.y = 1 - uv_coord.y;
    ray_sampler.uv = uv_coord;
    ray_sampler.distribution = pix_size;

    float w_len = max(length(camera.image_w), 1e-6f);
    float tan_dx = 2.f * pix_size.x * length(camera.image_u) / w_len;
    float tan_dy = 2.f * pix_size.y * length(camera.image_v) / w_len;
    float pixel_spread = max(tan_dx, tan_dy);

    HybridPathState path_state;
    path_state.sg = UniformSampleGenerator(tid, iteration);
    path_state.status = HybridPathStatus::intersect;
    path_state.nee_valid = 0;
    path_state.depth = 1;
    Ray camera_ray = ray_sampler.sample_ray(path_state.sg);
    path_state.ray_origin = camera_ray.origin;
    path_state.ray_dir = camera_ray.dir;
    path_state.throughput = float3(1, 1, 1);
    path_state.scatter_normal_ws = float3(0);
    path_state.scatter_pdf = 0;
    path_state.frame_tangent_ws = float3(0);
    path_state.nee_light_weight = float3(0);
    path_state.nee_light_pdf = 0;
    path_state.pixel_spread = pixel_spread;
    path_state.fixed_mip_level = fixed_mip_level;
    path_state.render_neural_material = render_neural_material;
    return path_state;
}
