// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
__exported import neural_networks;
import falcor2.utils;
import util.math_helpers;

// Shared data contracts between the neural material stages. Keeping these
// inputs separate from the implementations lets Python compose and type-check
// encoder, texture, decoder, sampler, and auxiliary models independently.

// Latent texture.
interface ILatentTexture<int NumLatents> : IModel<LatentTextureInput, float>
{
    [BackwardDifferentiable]
    float[NumLatents] get_nearest(LatentTextureInput input);

    [BackwardDifferentiable]
    float[NumLatents] get_bilinear(LatentTextureInput input);

    [BackwardDifferentiable]
    float[NumLatents] get_bilinear_stochastic(LatentTextureInput, float sample);

    int get_texture_resolution(int material_id);
}

// Latent textures are queried through filtering-specific methods. ``forward``
// only satisfies ``IModel`` and is not used for texture evaluation.
extension<int NumLatents, LatentTexture : ILatentTexture<NumLatents>> LatentTexture
{
    [BackwardDifferentiable]
    float forward(LatentTextureInput input) { return 0.f; }
}

struct LatentTextureInput : IDifferentiable
{
    // Selects one spatial lookup from the material's latent mip hierarchy.
    int material_id;
    int mip_level;
    float2 uv;
}


// Encoder.
typealias IEncoder<int NumInputs, int NumLatents> = IModel<EncoderInput<NumInputs, NumLatents>, float[NumLatents]>;

struct EncoderInput<int NumMaterialParams, int NumLatents> : IDifferentiable
{
    // ``material_id`` selects the material but is not learned; gradients
    // flow through the material parameters.
    int material_id;
    half params[NumMaterialParams];
}


// Bsdf decoder.
typealias IBsdfDecoder<int NumLatents> = IModel<BsdfDecoderInput<NumLatents>, float[3]>;

struct BsdfDecoderInputWithFrames<int NumLatents, int NumFrames> : IDifferentiable
{
    // Direction arrays contain one entry per learned shading frame. Before a
    // rotation stage is used, ``BsdfDecoderInput`` aliases this to one frame.
    int material_id;
    int mip_level;
    float latents[NumLatents];
    float3 wi[NumFrames];
    float3 wo[NumFrames];
    float3 wh[NumFrames];
    float3 wd[NumFrames];
}

typealias BsdfDecoderInput<int NumLatents> = BsdfDecoderInputWithFrames<NumLatents, 1>;

extension<int NumLatents> BsdfDecoderInput<NumLatents>
{
    [Differentiable]
    __init(int material_id, int mip_level, float[NumLatents] latents, float3 wi, float3 wo)
    {
        this.material_id = material_id;
        this.mip_level = mip_level;
        this.latents = latents;
        this.wi[0] = wi;
        this.wo[0] = wo;
        // Give every field a differentiable initializer. The half/difference
        // stage at the front of ``BsdfDecoder`` overwrites these placeholders.
        this.wh[0] = wi;
        this.wd[0] = wo;
    }
}

// Sampler.
interface ISampler<int NumLatents, int NumLobes> : IModel<SamplerInput<NumLatents>, float>
{
    associatedtype Params : IDifferentiable;

    [BackwardDifferentiable]
    Params decode_params(SamplerInput<NumLatents>);

    float3 sample(float3 wi, float3 sample, Params params);

    [BackwardDifferentiable]
    float pdf(no_diff float3 wi, no_diff float3 wo, Params params);

    [BackwardDifferentiable]
    float[NumLobes] pdf_lobes(no_diff float3 wi, no_diff float3 wo, Params params);
}

// Sampling uses ``decode_params()``, ``sample()``, and ``pdf()`` rather than
// ``forward()``.
// This implementation only fills the ``IModel`` interface requirement.
extension<int NumLatents, int NumLobes, Sampler : ISampler<NumLatents, NumLobes>> Sampler
{
    [BackwardDifferentiable]
    float forward(SamplerInput<NumLatents> input) { return 0.f; }
}

struct SamplerInput<int NumLatents> : IDifferentiable
{
    // The outgoing distribution is conditioned on both the material code and
    // the incident direction.
    float latents[NumLatents];
    float3 wi;
}


// Aux decoder.
typealias IAuxDecoder<int NumLatents, int NumOutputs> = IModel<AuxDecoderInput<NumLatents>, float[NumOutputs]>;

struct AuxDecoderInput<int NumLatents> : IDifferentiable
{
    // Auxiliary material outputs may vary with the incident direction.
    float latents[NumLatents];
    float3 wi;
}
