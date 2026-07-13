// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
__exported import falcor2.utils;
import util.math_helpers;
import util.texture_helpers;

// Common sample passed to BSDF and auxiliary training kernels.
// ``material_id`` addresses the neural model. ``material_backend_id`` addresses
// the native material used to fill target and encoder data.
struct TrainingSample<int NumTargetChannels, int NumMaterialParams>
{
    bool valid;
    int material_id;
    int material_backend_id;
    int mip_level;
    float2 uv;
    float3 wi;
    float3 wo;
    float target[NumTargetChannels];
    half encoder_inputs[NumMaterialParams];

    float[NumTargetChannels] get_target() { return target; }
    half[NumMaterialParams] get_encoder_inputs() { return encoder_inputs; }
}

// Sampler training only needs the frozen latent code and incident direction;
// its loss samples outgoing directions from the learned proposal.
struct SamplerTrainingSample<int NumLatents>
{
    int material_id;
    int mip_level;
    float2 uv;
    float3 wi;
    float[NumLatents] latents;
}

namespace DataGenerator
{

void generate_training_sample<
    int NumTargetChannels,
    int NumMaterialParams
>(
    int[2] call_id,
    int[2] call_shape, // [batch_count, batch_size]
    Config config,
    int material_id,
    int material_backend_id,
    int texture_resolution,
    float2 udim_offset,
    inout UniformSampleGenerator sg,
    inout TrainingSample<NumTargetChannels, NumMaterialParams> sample,
)
{
    sample.valid = true;

    // Material ID is already sampled.
    sample.material_id = material_id;
    sample.material_backend_id = material_backend_id;

    // MIP level is shared by all threads in the warp for prefiltering perf.
    sample.mip_level = sample_mip_level_for_warp(config, sg);

    // Sample ``uv`` stratified inside UDIM.
    sample.uv = sample_uv(
        call_id[1],
        call_shape[1],
        sample.mip_level,
        udim_offset,
        texture_resolution,
        config.snap_to_texel_center,
        sg
    );

    // Sample ``wi`` (and potentially already ``wo``).
    // Some ``DirectionalSamplingStrategy`` options require ``wo`` generation
    // later by importance sampling the reference material. This requires access
    // to the falcor2 material system and is done in a separate call.
    bool sampled_wo;
    sample.valid &= sample_directions(
        config,
        sg,
        sample.wi,
        sample.wo,
        sampled_wo
    );
    if (!sampled_wo)
        sample.wo = float3(0.f); // Mark as unsampled.
}

// Sampling choices shared by the phase-specific data generators.
struct Config
{
    bool snap_to_texel_center;

    int2 mip_level_min_max;
    int num_prefilter_samples;

    int directional_sampling_strategy; // ``DirectionalSamplingStrategy`` enum.
    float balanced_directional_sampling_ratio;

    float color_augmentation_ratio;
}

// Distribution used to cover the incident/outgoing directional domain. The
// ``ImportanceWo`` variants leave ``wo`` for later material sampling.
enum DirectionalSamplingStrategy
{
    UniformRusinkiewicz,
    UniformWiUniformWo,
    CosineWiCosineWo,
    UniformWiImportanceWo,
    CosineWiImportanceWo,
    Balanced_UniformRusinkiewicz_UniformWiImportanceWo,
    Balanced_UniformWiUniformWo_UniformWiImportanceWo,
}
int get_directional_sampling_strategy<DirectionalSamplingStrategy s>()
{
    return int(s);
}

int sample_mip_level<S : ISampleGenerator>(Config config, inout S sg)
{
    // Prefer fine levels while retaining samples at every configured mip.
    let sample = next_float(sg);
    int min = config.mip_level_min_max.x;
    int max = config.mip_level_min_max.y;

    float num_levels = max - min + 1.f;
    float mip_level = min - log2(1.f - sample * (1.f - exp2(-num_levels)));
    mip_level = clamp(floor(mip_level), min, max);
    return int(mip_level);
}

int sample_mip_level_for_warp(
    Config config,
    inout UniformSampleGenerator sg
)
{
    int mip_level = 0;

    // Only the first lane consumes the MIP sample; the rest of the wave
    // receives the value through a broadcast.
    if (WaveIsFirstLane())
        mip_level = sample_mip_level(config, sg);

    return WaveReadLaneFirst(mip_level);
}

float2 sample_uv<S : ISampleGenerator>(
    int batch_id,
    int batch_size,
    int mip_level,
    float2 udim_offset,
    int texture_resolution,
    bool snap_to_texel_center,
    inout S sg
)
{
    int side_length = int(sqrt(float(batch_size)));
    int2 batch_shape = int2(side_length, side_length);
    int2 batch_id_2 = int2(batch_id % side_length, batch_id / side_length);

    float2 sample = next_float2(sg);
    float2 uv = (float2(batch_id_2) + sample) / batch_shape;
    uv += udim_offset;

    if (snap_to_texel_center)
    {
        int resolution = texture_resolution >> mip_level;
        uv = snap_to_nearest_texel_center(uv, resolution, false);
    }

    return uv;
}

bool sample_directions<S : ISampleGenerator>(
    Config config,
    inout S sg,
    out float3 wi,
    out float3 wo,
    out bool sampled_wo
)
{
    bool success = false;

    let s = DirectionalSamplingStrategy(config.directional_sampling_strategy);

    if (s == DirectionalSamplingStrategy::UniformRusinkiewicz ||
        s == DirectionalSamplingStrategy::Balanced_UniformRusinkiewicz_UniformWiImportanceWo)
    {
        int tries = 0;
        while (!success && tries < 100)
        {
            sample_uniform_rusinkiewicz(next_float4(sg), wi, wo);
            success = wi.z >= k_min_cos && wo.z >= 0.f;
            tries++;
        }
        sampled_wo = true;

        if (s == DirectionalSamplingStrategy::Balanced_UniformRusinkiewicz_UniformWiImportanceWo &&
            next_float(sg) >= config.balanced_directional_sampling_ratio)
        {
            // ``wo`` will be importance-sampled later.
            sampled_wo = false;
        }
    }
    else if (s == DirectionalSamplingStrategy::UniformWiUniformWo ||
             s == DirectionalSamplingStrategy::Balanced_UniformWiUniformWo_UniformWiImportanceWo)
    {
        wi = sample_hemisphere(next_float2(sg));
        wo = sample_hemisphere(next_float2(sg));
        success = wi.z >= k_min_cos && wo.z >= 0.f;
        sampled_wo = true;

        if (s == DirectionalSamplingStrategy::Balanced_UniformWiUniformWo_UniformWiImportanceWo &&
            next_float(sg) >= config.balanced_directional_sampling_ratio)
        {
            // ``wo`` will be importance-sampled later.
            sampled_wo = false;
        }
    }
    else if (s == DirectionalSamplingStrategy::CosineWiCosineWo)
    {
        float unused;
        wi = sample_cosine_hemisphere_concentric(next_float2(sg), unused);
        wo = sample_cosine_hemisphere_concentric(next_float2(sg), unused);
        success = wi.z >= k_min_cos && wo.z >= 0.f;
        sampled_wo = true;
    }
    else if (s == DirectionalSamplingStrategy::UniformWiImportanceWo)
    {
        wi = sample_hemisphere(next_float2(sg));
        success = wi.z >= k_min_cos;
        sampled_wo = false; // ``wo`` will be importance sampled later instead.
    }
    else if (s == DirectionalSamplingStrategy::CosineWiImportanceWo)
    {
        float unused;
        wi = sample_cosine_hemisphere_concentric(next_float2(sg), unused);
        success = wi.z >= k_min_cos;
        sampled_wo = false; // ``wo`` will be importance sampled later instead.
    }

    return success;
}

float filter_sigma(int mip_level, int texture_resolution)
{
    if (mip_level == 0)
        return 0.f;
    float radius = 0.5f * (1 << mip_level);
    return 0.65f * radius / texture_resolution;
}

uint3 color_augmentation_indices(float ratio, float sample)
{
    if (sample < ratio)
    {
        // Map sample to a 3-component [0..2] index variation with repetition.
        // 27 possibilities in total.
        sample /= ratio;
        float r = 27 * sample;
        return uint3(uint(r / 9), uint(r / 3), uint(r)) % 3;
    }
    else
    {
        // Keep original index order.
        return uint3(0, 1, 2);
    }
}

[ForceInline]
uint pack_encoder_inputs(
    inout half[] encoder_inputs,
    uint idx,
    half3 normal,
    half3 tangent,
    half3 albedo,
    half2 roughness,
    half3 weight
)
{
    encoder_inputs[idx++] = normal.x;
    encoder_inputs[idx++] = normal.y;
    encoder_inputs[idx++] = normal.z;
    encoder_inputs[idx++] = tangent.x;
    encoder_inputs[idx++] = tangent.y;
    encoder_inputs[idx++] = tangent.z;
    encoder_inputs[idx++] = albedo.x;
    encoder_inputs[idx++] = albedo.y;
    encoder_inputs[idx++] = albedo.z;
    encoder_inputs[idx++] = roughness.x;
    encoder_inputs[idx++] = roughness.y;
    encoder_inputs[idx++] = weight.x;
    encoder_inputs[idx++] = weight.y;
    encoder_inputs[idx++] = weight.z;
    return idx;
}

} // namespace DataGenerator
