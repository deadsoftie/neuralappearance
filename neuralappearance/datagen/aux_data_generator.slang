// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
__exported import data_generator;

namespace DataGenerator
{
namespace Aux
{

// Canonical auxiliary properties collected from a reference material. The
// methods below pack configured subsets into the channel layouts expected by
// TrainingTargets and the auxiliary decoder.
struct AuxMaterialProperties
{
    float3 diffuse_albedo;
    float3 specular_albedo;
    float3 guide_normal;
    float roughness;

    float[7] diffuse_specular_roughness()
    {
        float[7] result;
        int idx = 0;
        result[idx++] = diffuse_albedo.x;
        result[idx++] = diffuse_albedo.y;
        result[idx++] = diffuse_albedo.z;
        result[idx++] = specular_albedo.x;
        result[idx++] = specular_albedo.y;
        result[idx++] = specular_albedo.z;
        result[idx++] = roughness;
        return result;
    }

    float[10] diffuse_specular_roughness_normal()
    {
        float[10] result;
        int idx = 0;
        result[idx++] = diffuse_albedo.x;
        result[idx++] = diffuse_albedo.y;
        result[idx++] = diffuse_albedo.z;
        result[idx++] = specular_albedo.x;
        result[idx++] = specular_albedo.y;
        result[idx++] = specular_albedo.z;
        result[idx++] = roughness;
        result[idx++] = guide_normal.x;
        result[idx++] = guide_normal.y;
        result[idx++] = guide_normal.z;
        return result;
    }
}

} // namespace Aux
} // namespace DataGenerator
