// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
__exported import aux_data_generator;
__exported import falcor_data_generator;

namespace DataGenerator
{
namespace Aux
{
namespace Falcor
{

AuxMaterialProperties eval_target_properties(
    uint32_t material_backend_id,
    uint32_t mip_level,
    float2 uv,
    float3 wi,
    float3 wo
)
{
    let material = g_scene.get_material(MaterialID(material_backend_id));
    let sd = ReferenceMaterials::Falcor::setup_shading_data(uv, wi);

    // Assume here that the reference BSDF properties already have a notion of
    // LOD built-in, e.g. via prefiltered albedo textures.
    let lod_sampler = ExplicitLodSampler(float(mip_level));
    let mi = material.setup_material_instance(sd, lod_sampler);

    let props = mi.collect_properties(sd);
    return AuxMaterialProperties(
        props.diffuse_reflection_albedo,
        props.specular_reflection_albedo,
        props.guide_normal,
        props.roughness
    );
}

float[7] eval_target_ref_diffuse_specular_roughness(
    uint32_t material_backend_id,
    uint32_t mip_level,
    float2 uv,
    float3 wi,
    float3 wo
)
{
    let props = eval_target_properties(material_backend_id, mip_level, uv, wi, wo);
    return props.diffuse_specular_roughness();
}

float[10] eval_target_ref_diffuse_specular_roughness_normal(
    uint32_t material_backend_id,
    uint32_t mip_level,
    float2 uv,
    float3 wi,
    float3 wo
)
{
    let props = eval_target_properties(material_backend_id, mip_level, uv, wi, wo);
    return props.diffuse_specular_roughness_normal();
}

void fill_target_ref_diffuse_specular_roughness(inout TrainingSample<7, 1> sample)
{
    sample.target = eval_target_ref_diffuse_specular_roughness(
        sample.material_backend_id,
        sample.mip_level,
        sample.uv,
        sample.wi,
        sample.wo
    );
}

void fill_target_ref_diffuse_specular_roughness_normal(inout TrainingSample<10, 1> sample)
{
    sample.target = eval_target_ref_diffuse_specular_roughness_normal(
        sample.material_backend_id,
        sample.mip_level,
        sample.uv,
        sample.wi,
        sample.wo
    );
}

} // namespace Falcor
} // namespace Aux
} // namespace DataGenerator
