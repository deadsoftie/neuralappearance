// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
__exported import base_types;
__exported import aux_decoder;
__exported import bsdf_decoder_flattener;
__exported import encoder;
__exported import half_diff_parameterization;
__exported import latent_texture;
__exported import rotation;
__exported import sampler;

// Python calls these wrappers from validation and visualization code. They
// expose uniform entry points for the concrete model assembled from config.
float[NumLatents] eval_encoder<
    int NumInputs,
    int NumLatents,
    Encoder : IEncoder<NumInputs, NumLatents>
>(
    Encoder encoder,
    int material_id,
    half[NumInputs] inputs,
)
{
    return encoder.forward({ material_id, inputs });
}

float[NumLatents] eval_latent_texture<
    int NumLatents,
    LatentTexture : ILatentTexture<NumLatents>
>(
    LatentTexture latent_texture,
    int material_id,
    int mip_level,
    float2 uv
)
{
    return latent_texture.get_nearest({ material_id, mip_level, uv });
}

float[NumLatents] eval_latent_texture_bilinear_stochastic<
    int NumLatents,
    LatentTexture : ILatentTexture<NumLatents>
>(
    LatentTexture latent_texture,
    int material_id,
    uint mip_level,
    float2 uv,
    float sample
)
{
    return latent_texture.get_bilinear_stochastic({ material_id, int(mip_level), uv }, sample);
}

float[3] eval_bsdf_decoder<
    int NumLatents,
    BsdfDecoder : IBsdfDecoder<NumLatents>
>(
    BsdfDecoder decoder,
    int material_id,
    int mip_level,
    float[NumLatents] latents,
    float3 wi,
    float3 wo
)
{
    return decoder.forward({ material_id, mip_level, latents, wi, wo });
}

float[NumLobes + 1] eval_sampler<
    int NumLatents,
    int NumLobes,
    Sampler : ISampler<NumLatents, NumLobes>
>(
    Sampler sampler,
    float[NumLatents] latents,
    float3 wi,
    float3 wo
)
{
    // Decode parameters and evaluate PDF.
    let params = sampler.decode_params({ latents, wi });

    float[NumLobes + 1] result;
    // Slot zero stores the mixture PDF; following slots expose the weighted
    // per-lobe contributions for diagnostics and sampler plots.
    result[0] = 0.f;

    float[NumLobes] pdfs = sampler.pdf_lobes(wi, wo, params);
    [ForceUnroll]
    for (int i = 0; i < NumLobes; ++i)
    {
        result[0] += pdfs[i];
        result[1 + i] = pdfs[i];
    }
    return result;
}

float[NumOutputs] eval_aux<
    int NumLatents,
    int NumOutputs,
    AuxDecoder : IAuxDecoder<NumLatents, NumOutputs>
>(
    AuxDecoder aux,
    float[NumLatents] latents,
    float3 wi
)
{
    return aux.forward({ latents, wi });
}
