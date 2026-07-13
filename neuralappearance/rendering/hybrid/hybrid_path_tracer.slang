// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
__exported import rendering.hybrid.hybrid_path_state;
import util.math_helpers;
import util.texture_helpers;
import datagen.data_generator;
import rendering.neural_material_globals;
import slangpy;

// Hybrid megakernel and wavefront path tracer.

// Bounce-policy configuration.
extern static const bool ENABLE_NEE = true;
extern static const bool ENABLE_MIS = true;
extern static const bool ENABLE_EMISSIVE_TRIANGLES = true;
extern static const bool ENABLE_ENV_MAP = true;
extern static const uint MAX_DEPTH = 10;

// Analytic lights remain enabled in the light sampler. If none exist,
// ``compute_type_probabilities()`` gives them zero selection probability.
static LightSampler<true, ENABLE_ENV_MAP, ENABLE_EMISSIVE_TRIANGLES> scene_light_sampler;

// The renderer supports two dispatch modes: megakernel and wavefront.
//
// In megakernel mode, falcor2 evaluates all materials inline in world space, so
// no shading-frame conversion is needed.
//
// In wavefront mode, an external scatter generator evaluates the target
// material in local space. ``intersect()`` converts directions to local space,
// while ``propagate()`` returns them to world space. Falcor2 evaluates others
// inline.

extern static const bool WAVEFRONT_MODE = false;
extern static const uint TARGET_MATERIAL_ID;
extern static const uint TARGET_MATERIAL_RESOLUTION;

public ScatterEventQuery intersect(
    inout HybridPathState path_state,
    inout float4 radiance,
    out ScatterEventResult scatter
)
{
    ScatterEventQuery eval_request = {};
    scatter = {};
    path_state.nee_valid = 0;

    if (path_state.status == HybridPathStatus::terminated)
        return eval_request;

    RayQuerySceneIntersector intersector;
    HitInfo hit = intersector.intersect(Ray(path_state.ray_origin, path_state.ray_dir));
    bool is_primary_hit = path_state.depth == 1;

    if (!hit.valid)
    {
        bool handle_env_map = ENABLE_ENV_MAP && g_scene.has_environment_lights()
            && (is_primary_hit || !ENABLE_NEE || ENABLE_MIS);

        if (handle_env_map)
        {
            // Build the surface interaction at the previous-bounce hit point so
            // the environment light eval (a directional light) can evaluate
            // without caring about position.
            SurfaceInteraction scatter_si = {};
            scatter_si.position_ws = path_state.ray_origin;
            scatter_si.normal_ws = path_state.scatter_normal_ws;
            DirectionSample ds = {};
            ds.dir_ws = path_state.ray_dir;
            ds.emitter_type = EmitterType::environment;

            float mis_weight = 1.0;
            if (!is_primary_hit && ENABLE_NEE && ENABLE_MIS)
            {
                float light_pdf = scene_light_sampler.eval_direction_pdf(scatter_si, ds);
                mis_weight = mis_heuristic(1, path_state.scatter_pdf, 1, light_pdf);
            }

            let eval = g_scene.light_system.eval_environment_direction(scatter_si, ds);
            radiance.xyz += path_state.throughput * eval * mis_weight;
        }
        path_state.status = HybridPathStatus::terminated;
        return eval_request;
    }

    SurfaceInteraction si = g_scene.make_surface_interaction(hit, -path_state.ray_dir);

    // Begin filtering-specific modifications.
    {
        float2 duv_dx = float2(0.f);
        float2 duv_dy = float2(0.f);
        float mip_sample = next_float(path_state.sg);
        float texel_sample = next_float(path_state.sg);
        if (path_state.fixed_mip_level == -1 && is_primary_hit)
        {
            // Recover hit distance from the ray origin because ``HitInfo`` does
            // not expose ``ray_t`` directly.
            compute_primary_uv_differentials(
                path_state.ray_dir,
                path_state.ray_origin,
                hit,
                si,
                path_state.pixel_spread,
                duv_dx,
                duv_dy
            );
        }

        if (path_state.render_neural_material)
        {
            // Neural materials do all filtering internally (both UVs and MIPs).
            // We just need to sneak the relevant information into the material
            // instance somehow.
            g_neural_material_globals.duv_dx = duv_dx;
            g_neural_material_globals.duv_dy = duv_dy;
            g_neural_material_globals.fixed_mip_level = path_state.fixed_mip_level;
            g_neural_material_globals.mip_sample = mip_sample;
            g_neural_material_globals.texel_sample = texel_sample;
        }
        else
        {
            // For non-neural reference materials, be careful to match the
            // behavior of the neural material here. This also matches what is
            // done on the training side.
            int texture_resolution = TARGET_MATERIAL_RESOLUTION;
            int mip_level;
            if (path_state.fixed_mip_level >= 0)
            {
                // Optionally override automatic MIP selection.
                mip_level = path_state.fixed_mip_level;
            }
            else
            {
                // Automatic MIP selection based on UV space differentials.
                // We stochastically snap to an integer MIP level.
                float lod = scalar_mip_level_from_footprint(duv_dx, duv_dy, texture_resolution);
                mip_level = sample_integer_mip_level(lod, mip_sample);
            }

            // Apply stochastic bilinear texture filtering at the appropriate
            // MIP level. This means we apply filtering after shading.
            float2 uv = si.uv;
            bool snap_to_texel_center = true;
            if (snap_to_texel_center)
            {
                uv = sample_bilinear_texel_center(uv, texture_resolution >> mip_level, false, texel_sample);
            }

            // Based on the MIP level, jitter the UV coordinate inside the
            // footprint. This will give a 1-sample Monte Carlo estimate of a
            // "ground truth" prefiltered BRDF evaluation.
            float sigma = DataGenerator::filter_sigma(mip_level, texture_resolution);
            float2 uv_sample = next_float2(path_state.sg);
            float2 uv_filter = sample_2d_gaussian(uv_sample, uv, sigma);
            if (snap_to_texel_center)
            {
                uv_filter = snap_to_nearest_texel_center(uv_filter, texture_resolution, false);
            }
            si.uv = uv_filter;
        }
    }

    // Apply bent-normal adjustment at grazing angles.
    float3 adj_normal = adjust_normal(path_state.ray_dir, si.normal_ws, si.shading_frame_ws.normal);

    path_state.ray_origin = offset_ray_origin(si.position_ws, si.normal_ws);
    path_state.scatter_normal_ws = adj_normal;
    path_state.scatter_pdf = 0;

    // Sample next-event estimation.
    float3 nee_wo_world = float3(0);
    if (ENABLE_NEE && path_state.depth < MAX_DEPTH)
    {
        // ``SurfaceInteraction`` satisfies the light sampler's ``IInteraction``
        // requirement.
        // Override the shading normal with the bent-adjusted one so light
        // sampling uses the same frame the BSDF will see.
        SurfaceInteraction nee_si = si;
        nee_si.normal_ws = adj_normal;

        DirectionSample light_sample = {};
        float3 light_weight = float3(0);

        if (scene_light_sampler.sample_direction(nee_si, path_state.sg, light_sample, light_weight))
        {
            bool occluded = intersector.is_occluded(create_visibility_ray(nee_si, light_sample, si.normal_ws));

            if (!occluded)
            {
                nee_wo_world = light_sample.dir_ws;
                path_state.nee_valid = 1;
                path_state.nee_light_weight = light_weight;
                path_state.nee_light_pdf = light_sample.pdf;
            }
        }
    }

    // In wavefront mode, delegate target-material evaluation to the external
    // scatter data generator.
    if (WAVEFRONT_MODE)
    {
        let gi_desc = g_scene.get_geometry_instance_desc(hit.geometry_instance_id);
        if (uint(gi_desc.header.material_id) == TARGET_MATERIAL_ID)
        {
            let frame = Frame(adj_normal, si.shading_frame_ws.tangent);

            eval_request.uv = si.uv;
            eval_request.wi = frame.to_local(si.wi_ws);

            if (path_state.nee_valid != 0)
                eval_request.wo = frame.to_local(nee_wo_world);

            path_state.frame_tangent_ws = frame.tangent;
            path_state.status = HybridPathStatus::evaluate;
            return eval_request;
        }
    }

    // Evaluate using falcor2's material system.
    let lod_sampler = ExplicitLodSampler(0);
    let material_instance = g_scene.get_material_instance(hit, si, lod_sampler);

    // Importance-sample new direction via the material.
    MaterialSample mat_sample = {};
    if (material_instance.sample(si, path_state.sg, mat_sample))
    {
        path_state.ray_dir = mat_sample.wo_ws;
        scatter.scatter_pdf = mat_sample.pdf;

        // ``weight`` is ``f * cos / pdf``; recover ``f * cos``.
        scatter.scatter_weight = mat_sample.weight * mat_sample.pdf;

        float orientation = (dot(si.normal_ws, mat_sample.wo_ws) >= 0) ? 1 : -1;
        path_state.ray_origin = offset_ray_origin(si.position_ws, si.normal_ws * orientation);

        // Wavefront propagation reconstructs the direction from local space.
        if (WAVEFRONT_MODE)
        {
            let frame = Frame(adj_normal, si.shading_frame_ws.tangent);
            scatter.scatter_dir = frame.to_local(mat_sample.wo_ws);
            path_state.frame_tangent_ws = frame.tangent;
        }
    }

    // NEE eval in light direction; skip if event was shadowed.
    if (path_state.nee_valid != 0)
    {
        scatter.nee_pdf = material_instance.eval_pdf(si, nee_wo_world);
        scatter.nee_weight = material_instance.eval(si, nee_wo_world, path_state.sg);
    }

    // Remove non-finite scatter results.
    scatter.sanitize();

    path_state.status = HybridPathStatus::propagate;

    return eval_request;
}

public void propagate(
    inout HybridPathState path_state,
    inout float4 radiance,
    ScatterEventResult scatter
)
{
    if (path_state.status != HybridPathStatus::evaluate &&
        path_state.status != HybridPathStatus::propagate)
        return;

    if (path_state.nee_valid != 0)
    {
        float mis_weight = 1.0;
        if (ENABLE_MIS)
        {
            mis_weight = mis_heuristic(
                1, path_state.nee_light_pdf,
                1, scatter.nee_pdf
            );
        }

        radiance.xyz += path_state.throughput * scatter.nee_weight * path_state.nee_light_weight * mis_weight;
    }

    // Update throughput from ``f(wi, wo) * cos(theta)`` and the mixture PDF.
    if (scatter.scatter_pdf > 0)
    {
        path_state.throughput *= scatter.scatter_weight / scatter.scatter_pdf;

        // Terminate if throughput is zero or invalid.
        if (all(path_state.throughput <= 0) ||
            any(isnan(path_state.throughput)) ||
            any(isinf(path_state.throughput)))
        {
            path_state.status = HybridPathStatus::terminated;
            return;
        }
    }
    else
    {
        path_state.status = HybridPathStatus::terminated;
        return;
    }

    // Wavefront mode reconstructs the world-space scatter direction from local
    // ``wo``. Megakernel mode sets ``ray_dir`` directly in ``intersect()``.
    if (WAVEFRONT_MODE)
    {
        float3 t = path_state.frame_tangent_ws;
        float3 b = normalize(cross(path_state.scatter_normal_ws, path_state.frame_tangent_ws));
        float3 n = path_state.scatter_normal_ws;

        path_state.ray_dir = t * scatter.scatter_dir.x
                           + b * scatter.scatter_dir.y
                           + n * scatter.scatter_dir.z;
    }

    // Store BSDF mixture PDF for next-bounce env-map MIS.
    path_state.scatter_pdf = scatter.scatter_pdf;

    // ``ray_origin`` already contains the offset point for the next ray.
    path_state.depth++;

    // Terminate at maximum depth; otherwise request the next intersection.
    if (path_state.depth > MAX_DEPTH)
    {
        path_state.status = HybridPathStatus::terminated;
    }
    else
    {
        path_state.status = HybridPathStatus::intersect;
    }
}

// In this material-system megakernel, ``WAVEFRONT_MODE`` must be false so
// ``intersect()`` evaluates all geometry through ``IMaterialInstance``.
public void render_megakernel_scene(
    uniform Camera camera,
    out float4 output,
    uniform uint iteration,
    uniform int8_t fixed_mip_level,
    uniform bool render_neural_material,
    uint2 tid
)
{
    HybridPathState path_state = generate_path_state(camera, iteration, fixed_mip_level, render_neural_material, tid);
    float4 radiance = float4(0, 0, 0, 1);

    [MaxIters(MAX_DEPTH)]
    for (uint d = 0; d < MAX_DEPTH; d++)
    {
        ScatterEventResult scatter;
        intersect(path_state, radiance, scatter);

        if (path_state.status == HybridPathStatus::terminated)
            break;

        propagate(path_state, radiance, scatter);
    }

    output = radiance;
}

// Clear radiance with ``w=1`` so the output is immediately valid.
public void clear_radiance(out float4 radiance)
{
    radiance = float4(0, 0, 0, 1);
}

// The wavefront intersection entry point calls ``intersect()``, compacts paths
// awaiting evaluation, and writes their request and pixel-index buffers.
public void wavefront_intersect_compact_gather(
    inout HybridPathState path_state,
    inout float4 radiance,
    out ScatterEventResult eval_samples,
    RWTensor<ScatterEventQuery, 1> compact_requests,
    RWTensor<uint, 1> compact_indices,
    RWTensor<uint, 1> reverse_compact,
    RWStructuredBuffer<uint> counters,
    int2 call_id,
    uniform uint image_width
)
{
    uint pixel_idx = uint(call_id.y) * image_width + uint(call_id.x);

    ScatterEventQuery eval_request = intersect(path_state, radiance, eval_samples);

    if (path_state.status == HybridPathStatus::terminated)
        return;

    // Count alive paths.
    uint dummy;
    InterlockedAdd(counters[0], 1, dummy);

    if (path_state.status == HybridPathStatus::evaluate)
    {
        // Reserve a slot in the compact buffer.
        uint compact_tid;
        InterlockedAdd(counters[1], 1, compact_tid);
        compact_indices[compact_tid] = pixel_idx;
        compact_requests[compact_tid] = eval_request;
        reverse_compact[pixel_idx] = compact_tid;
    }
}

// The wavefront propagation entry point reads external results for evaluation
// pixels and inline results for other pixels, then calls ``propagate()``.
public void wavefront_scatter_propagate(
    inout HybridPathState path_state,
    inout float4 radiance,
    ScatterEventResult eval_sample,
    Tensor<ScatterEventResult, 1> compact_results,
    Tensor<uint, 1> reverse_compact,
    int2 call_id,
    uniform uint image_width
)
{
    uint pixel_idx = uint(call_id.y) * image_width + uint(call_id.x);

    ScatterEventResult scatter;
    if (path_state.status == HybridPathStatus::evaluate)
        scatter = compact_results[reverse_compact[pixel_idx]];
    else
        scatter = eval_sample;

    propagate(path_state, radiance, scatter);
}
