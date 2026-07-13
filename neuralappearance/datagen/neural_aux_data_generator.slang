// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
__exported import aux_data_generator;
__exported import model.neural_material_model;
import util.math_helpers;
import falcor2.utils;

namespace DataGenerator
{
namespace Aux
{
namespace Neural
{

void fill_target_mc_albedo<
    int NumLatents,
    int NumLobes,
    LatentTexture : ILatentTexture<NumLatents>,
    BsdfDecoder : IBsdfDecoder<NumLatents>,
    Sampler : ISampler<NumLatents, NumLobes>
>(
    LatentTexture latent_texture,
    BsdfDecoder decoder,
    Sampler sampler,
    inout UniformSampleGenerator sg,
    inout TrainingSample<3, 1> sample
)
{
    sample.target = eval_target_mc_albedo(
        latent_texture,
        decoder,
        sampler,
        sample.material_id,
        sample.mip_level,
        sample.uv,
        sample.wi,
        sample.wo,
        sg
    );
}

float[3] eval_target_mc_albedo<
    int NumLatents,
    int NumLobes,
    LatentTexture : ILatentTexture<NumLatents>,
    BsdfDecoder : IBsdfDecoder<NumLatents>,
    Sampler : ISampler<NumLatents, NumLobes>
>(
    LatentTexture latent_texture,
    BsdfDecoder decoder,
    Sampler sampler,
    uint32_t material_id,
    uint32_t mip_level,
    float2 uv,
    float3 wi,
    float3 wo,
    inout UniformSampleGenerator sg,
    uint32_t num_mc_samples = 1
)
{

    // Evaluate parameters of the neural sampler.
    float[NumLatents] latents = latent_texture.get_nearest({ material_id, mip_level, uv });
    let sampler_params = sampler.decode_params({ latents, wi });

    // Monte Carlo estimation of the hemispherical-directional albedo
    // over the upper hemisphere:
    // Estimate hemispherical-directional reflectance by integrating the BSDF
    // over outgoing directions:
    // a(\omega_i) = \int_H^2 f_r(\omega_i, \omega_o) |cos(theta_i)| d \omega_o
    float3 result = 0.f;
    for (uint i = 0; i < num_mc_samples; ++i)
    {
        float3 wo = sampler.sample(wi, next_float3(sg), sampler_params);
        float pdf = sampler.pdf(wi, wo, sampler_params);
        float pred[3] = decoder.forward({ material_id, mip_level, latents, wi, wo });
        float3 fr = { pred[0], pred[1], pred[2] };
        result += (fr * max(0.f, wo.z)) / pdf;
    }
    result /= float(num_mc_samples);

    return { result.x, result.y, result.z };
}

} // namespace Neural
} // namespace Aux
} // namespace DataGenerator
