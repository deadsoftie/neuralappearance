// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
__exported import data_generator;
__exported import falcor_reference_materials;
import util.math_helpers;
import util.texture_helpers;

namespace DataGenerator
{
namespace Falcor
{

void fill_importance_sampled_wo<
    int NumTargetChannels,
    int NumMaterialParams
>(
    Config config,
    int texture_resolution,
    inout UniformSampleGenerator sg,
    inout TrainingSample<NumTargetChannels, NumMaterialParams> sample
)
{
    // Uniform and cosine strategies populate ``wo`` in
    // ``generate_training_sample()``. Zero marks strategies delegated to the
    // material.
    if (any(sample.wo != 0.f))
        return; // Already sampled. Nothing to do.

    float2 uv = sample.uv;
    if (sample.mip_level > 0)
    {
        // Resample ``uv`` in the filter region, then sample ``wo`` there.
        float sigma = filter_sigma(sample.mip_level, texture_resolution);
        uv = sample_2d_gaussian(next_float2(sg), uv, sigma);
        if (config.snap_to_texel_center)
        {
            uv = snap_to_nearest_texel_center(uv, texture_resolution, false);
        }
    }

    let material = g_scene.get_material(MaterialID(sample.material_backend_id));
    let sd = ReferenceMaterials::Falcor::setup_shading_data(uv, sample.wi);
    let mi = material.setup_material_instance(sd, ExplicitLodSampler(0));

    MaterialSample result = {};

    bool success = false;
    uint tries = 0;

    while (!success && tries < 100)
    {
        success = mi.sample(sd, sg, result);
        success &= result.has_flag(BSDFFlags::reflection);
        tries++;
    }

    sample.wo = result.wo_ws;
    sample.valid &= success;
}

} // namespace Falcor
} // namespace DataGenerator
