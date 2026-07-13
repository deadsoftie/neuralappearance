// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
__exported import datagen.falcor_reference_materials;
import datagen.data_generator;
import util.math_helpers;
import util.texture_helpers;

void plot_bsdf_reference(
    int2 call_id,
    int material_backend_id,
    int mip_level,
    float2 selected_uv,
    float3 selected_wi,
    inout float4 color,
    int num_prefilter_samples,
    bool snap_to_texel_center,
    int texture_resolution
)
{
    let material = g_scene.get_material(MaterialID(material_backend_id));
    float3 wo = color.xyz;

    // Stochastic prefiltering inside a Gaussian kernel.
    float sigma = DataGenerator::filter_sigma(mip_level, texture_resolution);
    var sg = UniformSampleGenerator(call_id, 0);
    float3 fs = { 0.f, 0.f, 0.f };
    for (uint i = 0; i < num_prefilter_samples; ++i)
    {
        float2 uv_filter = sample_2d_gaussian(next_float2(sg), selected_uv, sigma);
        if (snap_to_texel_center)
        {
            uv_filter = snap_to_nearest_texel_center(uv_filter, texture_resolution, false);
        }
        let sd = ReferenceMaterials::Falcor::setup_shading_data(uv_filter, selected_wi);
        let mi = material.setup_material_instance(sd, ExplicitLodSampler(0));
        fs += mi.eval(sd, wo, sg);
    }
    fs /= num_prefilter_samples;
    color.xyz = fs;
}

void plot_uvs_reference(
    int2 call_id,
    int material_backend_id,
    float2 udim_offset,
    float2 selected_uv,
    int2 resolution,
    out float4 color
)
{
    float2 uv = (call_id + 0.5f) / resolution;
    uv += udim_offset;

    let material = g_scene.get_material(MaterialID(material_backend_id));
    let shading_data = ReferenceMaterials::Falcor::setup_shading_data(uv, float3(0.f, 0.f, 1.f));
    let lod_sampler = ExplicitLodSampler(0.f);
    let material_instance = material.setup_material_instance(shading_data, lod_sampler);
    let properties = material_instance.collect_properties(shading_data);
    color = float4(0.f, 0.f, 0.f, 1.f);
    color.xyz = properties.diffuse_reflection_albedo;

    const float radius = 6.f / resolution.x;
    float2 diff = uv - selected_uv;
    if (dot(diff, diff) <= radius * radius) {
        color = float4(1, 1, 1, 1);
    }
    if (dot(diff, diff) <= 0.66f * radius * radius) {
        color = float4(1, 0, 0, 1);
    }
}
