// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
__exported import falcor_data_generator;
import util.math_helpers;
import util.texture_helpers;

namespace DataGenerator
{
namespace Bsdf
{
namespace Falcor
{

void fill_target_bsdf<
    int NumMaterialParams
>(
    Config config,
    int texture_resolution, // to match ``fill_target_bsdf_lod`` signature.
    inout UniformSampleGenerator sg,
    inout TrainingSample<3, NumMaterialParams> sample
)
{
    sample.target = eval_target_bsdf(sample.material_backend_id, sample.mip_level, sample.uv, sample.wi, sample.wo, sg);
}

float[3] eval_target_bsdf(
    uint32_t material_backend_id,
    uint32_t mip_level,
    float2 uv,
    float3 wi,
    float3 wo,
    inout UniformSampleGenerator sg
)
{
    let material = g_scene.get_material(MaterialID(material_backend_id));
    let sd = ReferenceMaterials::Falcor::setup_shading_data(uv, wi);
    let mi = material.setup_material_instance(sd, ExplicitLodSampler(0.f));

    float3 fs = mi.eval(sd, wo, sg);
    return { fs.x, fs.y, fs.z };
}

void fill_target_bsdf_lod<
    int NumMaterialParams
>(
    Config config,
    int texture_resolution,
    inout UniformSampleGenerator sg,
    inout TrainingSample<3, NumMaterialParams> sample
)
{
    sample.target = eval_target_bsdf_lod(
        sample.material_backend_id,
        sample.mip_level,
        sample.uv,
        sample.wi,
        sample.wo,
        sg,
        config.num_prefilter_samples,
        config.snap_to_texel_center,
        texture_resolution
    );
}

float[3] eval_target_bsdf_lod(
    uint32_t material_backend_id,
    uint32_t mip_level,
    float2 uv,
    float3 wi,
    float3 wo,
    inout UniformSampleGenerator sg,
    uint32_t num_prefilter_samples,
    bool snap_to_texel_center,
    uint32_t texture_resolution
)
{
    let material = g_scene.get_material(MaterialID(material_backend_id));

    // Stochastic prefiltering inside a Gaussian kernel.
    float sigma = filter_sigma(mip_level, texture_resolution);
    float3 fs = { 0.f, 0.f, 0.f };
    for (uint i = 0; i < num_prefilter_samples; ++i)
    {
        float2 uv_filter = sample_2d_gaussian(next_float2(sg), uv, sigma);
        if (snap_to_texel_center)
        {
            uv_filter = snap_to_nearest_texel_center(uv_filter, texture_resolution, false);
        }
        let sd = ReferenceMaterials::Falcor::setup_shading_data(uv_filter, wi);
        let mi = material.setup_material_instance(sd, ExplicitLodSampler(0.f));
        fs += mi.eval(sd, wo, sg);
    }
    fs /= num_prefilter_samples;
    return { fs.x, fs.y, fs.z };
}

void fill_encoder_inputs<
    uint NumExportedBsdfs,
    int NumEncoderInputs
>(
    Config config,
    inout UniformSampleGenerator sg,
    inout TrainingSample<3, NumEncoderInputs> sample,
)
{
    uint3 indices = color_augmentation_indices(config.color_augmentation_ratio, next_float(sg));
    sample.encoder_inputs = eval_encoder_inputs_permuted<NumExportedBsdfs, NumEncoderInputs>(
        sample.material_backend_id,
        sample.mip_level,
        sample.uv,
        indices
    );
    float[3] target_p = { sample.target[indices.x], sample.target[indices.y], sample.target[indices.z] };
    sample.target = target_p;
}

half[NumEncoderInputs] eval_encoder_inputs_permuted<
    int NumExportedBsdfs,
    int NumEncoderInputs
>(
    uint32_t material_backend_id,
    uint32_t mip_level,
    float2 uv,
    uint3 indices,
)
{
    let dir = float3(0, 0, 1);
    let material = g_scene.get_material(MaterialID(material_backend_id));
    let sd = ReferenceMaterials::Falcor::setup_shading_data(uv, dir);
    let mi = material.setup_material_instance(sd, ExplicitLodSampler(0.f));

    let result = mi.collect_extra_bsdf_properties(sd, dir);
    if (!result.hasValue)
    {
        return {};
    }

    ExtraBsdfProperties extra_props = result.value;

    half3 normal[NumExportedBsdfs] = {};
    half3 tangent[NumExportedBsdfs] = {};
    half3 albedo[NumExportedBsdfs] = {};
    half2 roughness[NumExportedBsdfs] = {};
    half3 weight[NumExportedBsdfs] = {};

    // Write the first ``NumExportedBsdfs`` BSDFs.
    uint i = 0;
    for (; i < min(extra_props.bsdf_count, NumExportedBsdfs); ++i)
    {
        normal[i] = half3(extra_props.bsdf_N[i]);
        tangent[i] = half3(extra_props.bsdf_T[i]);
        albedo[i] = half3(extra_props.bsdf_albedo[i]);
        roughness[i] = half2(extra_props.bsdf_roughness[i]);
        weight[i] = half3(extra_props.bsdf_weight[i]);
    }

    // Aggregate excess BSDF parameters into the final exported layer rather
    // than dropping their contribution entirely.
    uint count = 1;
    for (; i < extra_props.bsdf_count; ++i)
    {
        normal[NumExportedBsdfs - 1] += half3(extra_props.bsdf_N[i]);
        tangent[NumExportedBsdfs - 1] += half3(extra_props.bsdf_T[i]);
        albedo[NumExportedBsdfs - 1] += half3(extra_props.bsdf_albedo[i]);
        roughness[NumExportedBsdfs - 1] += half2(extra_props.bsdf_roughness[i]);
        weight[NumExportedBsdfs - 1] += half3(extra_props.bsdf_weight[i]);
        count++;
    }
    let norm = half(1.f / count);
    normal[NumExportedBsdfs - 1] *= norm;
    tangent[NumExportedBsdfs - 1] *= norm;
    albedo[NumExportedBsdfs - 1] *= norm;
    roughness[NumExportedBsdfs - 1] *= norm;
    weight[NumExportedBsdfs - 1] *= norm;

    for (uint i = 0; i < NumExportedBsdfs; ++i)
    {
        let albedo_p = half3(albedo[i][indices.x], albedo[i][indices.y], albedo[i][indices.z]);
        albedo[i] = albedo_p;
        let weight_p = half3(weight[i][indices.x], weight[i][indices.y], weight[i][indices.z]);
        weight[i] = weight_p;
    }

    uint idx = 0;
    half[NumEncoderInputs] encoder_inputs;
    [ForceUnroll]
    for (int i = 0; i < NumExportedBsdfs; ++i)
    {
        idx = pack_encoder_inputs(encoder_inputs, idx, normal[i], tangent[i], albedo[i], roughness[i], weight[i]);
    }
    encoder_inputs[idx++] = 1.h + half(mip_level);
    return encoder_inputs;
}

half[NumEncoderInputs] eval_encoder_inputs<
    int NumExportedBsdfs,
    int NumEncoderInputs
>(
    uint32_t material_backend_id,
    uint32_t mip_level,
    float2 uv,
    float color_augmentation_ratio,
    inout UniformSampleGenerator sg,
)
{
    uint3 indices = color_augmentation_indices(color_augmentation_ratio, next_float(sg));
    return eval_encoder_inputs_permuted<NumExportedBsdfs, NumEncoderInputs>(
        material_backend_id,
        mip_level,
        uv,
        indices
    );
}

} // namespace Falcor
} // namespace Bsdf
} // namespace DataGenerator
