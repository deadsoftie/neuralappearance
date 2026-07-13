// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
__exported import model.neural_material_model;

void plot_bsdf_neural<
    int NumLatents,
    LatentTexture : ILatentTexture<NumLatents>,
    BsdfDecoder : IBsdfDecoder<NumLatents>
>(
    LatentTexture latent_texture,
    BsdfDecoder decoder,
    int material_id,
    int mip_level,
    float2 selected_uv,
    float3 selected_wi,
    inout float4 color
)
{
    let latents = eval_latent_texture(latent_texture, material_id, mip_level, selected_uv);
    float3 wo = color.xyz;
    float[3] fs = eval_bsdf_decoder(decoder, material_id, mip_level, latents, selected_wi, wo);
    color.xyz = { fs[0], fs[1], fs[2] };
}
