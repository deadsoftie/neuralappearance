// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
__exported import data_generator;
__exported import model.neural_material_model;
import util.math_helpers;

namespace DataGenerator
{
namespace Sampler
{

// Generate inputs for the sampler phase from the frozen latent texture. Unlike
// Unlike ``TrainingSample``, this contains no target or outgoing direction.
void generate_training_sample<
    int NumLatents,
    LatentTexture : ILatentTexture<NumLatents>
>(
    int[2] call_id,
    int[2] call_shape, // [batch_count, batch_size]
    Config config,
    int material_id,
    int texture_resolution,
    float2 udim_offset,
    inout UniformSampleGenerator sg,
    LatentTexture latent_texture,
    inout SamplerTrainingSample<NumLatents> sample
)
{
    // Material ID is already sampled.
    sample.material_id = material_id;

    // Threads in a warp share MIP levels to improve prefiltering performance.
    sample.mip_level = sample_mip_level_for_warp(config, sg);

    // Stratified uv sampling per batch.
    sample.uv = sample_uv(
        call_id[1],
        call_shape[1],
        sample.mip_level,
        udim_offset,
        texture_resolution,
        config.snap_to_texel_center,
        sg
    );

    // Uniform wi sampling; wo will be importance sampled during training.
    sample.wi = sample_hemisphere(next_float2(sg));

    // We can already fetch the (frozen) latents here.
    sample.latents = latent_texture.get_nearest({ sample.material_id, sample.mip_level, sample.uv });
}

} // namespace Sampler
} // namespace DataGenerator
