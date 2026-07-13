// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
__exported import falcor2.render;
__exported import falcor2.utils;

namespace ReferenceMaterials
{
namespace Falcor
{

// Query the layer count used to size the encoder input before training.
int num_bsdf_layers(uint32_t material_backend_id)
{
    let material = g_scene.get_material(MaterialID(material_backend_id));
    let dir = float3(0, 0, 1);
    let sd = setup_shading_data(float2(0.5f), dir);
    let mi = material.setup_material_instance(sd, ExplicitLodSampler(0));
    let result = mi.collect_extra_bsdf_properties(sd, dir);
    if (result.hasValue)
        return result.value.bsdf_count;
    return 0;
}

// Build the minimal surface interaction needed to evaluate a material in its
// canonical local frame; data generation does not require scene geometry.
SurfaceInteraction setup_shading_data(float2 uv, float3 wi)
{
    SurfaceInteraction result;
    result.position_ws = float3(uv, 0);
    result.wi_ws = wi;
    result.uv = uv;

    result.shading_frame_ws = Frame(float3(0, 0, 1));
    result.normal_ws = result.shading_frame_ws.normal;
    result.front_facing = wi.z > 0;
    result.ior = 1.0;

    return result;
}

} // namespace Falcor
} // namespace ReferenceMaterials
