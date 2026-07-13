// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
import slangpy;
import model.base_types;
__exported import model.texture;
__exported import util.texture_helpers;

// Latent texture for neural materials.
// This handles multiple materials at once. Each material is allowed to have
// potentially different UDIM sets and resolutions, but all materials share the
// same number of mip levels and latent channels.
struct LatentTexture<int NumTiles, int NumMipLevels, int NumLatents>
    : ILatentTexture<NumLatents>
{
    StructuredBuffer<int>.Handle tile_indirection;
    StructuredBuffer<int>.Handle material_texture_wrap;
    Texture<NumLatents> textures[NumTiles * NumMipLevels];

    int get_texture_resolution(int material_id)
    {
        let texture_idx = get_texture_idx(material_id, 0, 0);
        return this.textures[texture_idx].resolution;
    }

    [BackwardDifferentiable]
    float[NumLatents] get_nearest(LatentTextureInput input)
    {
        let udim_idx = get_udim_index(input.material_id, input.uv);
        let texture_idx = get_texture_idx(input.material_id, udim_idx, input.mip_level);
        return this.textures[texture_idx].get_nearest(input.uv);
    }

    [BackwardDifferentiable]
    float[NumLatents] get_bilinear(LatentTextureInput input)
    {
        let udim_idx = get_udim_index(input.material_id, input.uv);
        let texture_idx = get_texture_idx(input.material_id, udim_idx, input.mip_level);
        return this.textures[texture_idx].get_bilinear(input.uv);
    }

    [BackwardDifferentiable]
    float[NumLatents] get_bilinear_stochastic(LatentTextureInput input, float sample)
    {
        let udim_idx = get_udim_index(input.material_id, input.uv);
        let texture_idx = get_texture_idx(input.material_id, udim_idx, input.mip_level);
        return this.textures[texture_idx].get_bilinear_stochastic(input.uv, sample);
    }

    int get_udim_index(int material_id, float2 uv)
    {
        // Convert UV coordinates to tile coordinates.
        if (material_texture_wrap[material_id] == 1)
        {
            uv = uv - floor(uv);
        }
        else
        {
            uv = max(0.f, uv); // Discard negative UVs.
        }
        // UDIM's decimal layout has ten columns. The clamped [0, 99] index is
        // used with the fixed per-material indirection stride built in Python.
        int2 tile_coord = int2(uv);
        return min(max(tile_coord.x + 10 * tile_coord.y, 0), 99);
    }

    int get_texture_idx(int material_id, int udim_idx, int mip_level)
    {
        let udim_stride = 100;
        let tile_idx = this.tile_indirection[material_id * udim_stride + udim_idx];
        // Python stores each tile's MIP chain contiguously in this order.
        return tile_idx * NumMipLevels + mip_level;
    }
}

// Compute UV coordinates for latent texture sub tiles.
float2 latent_texture_tile_uvs(
    int2 call_id,
    int2 shape,
    float2 uv_min,
    float2 uv_max
)
{
    // Sample texel centers. Endpoints would shift generated data by half a
    // texel relative to later texture filtering.
    float2 uv = (float2(call_id) + float2(0.5f)) / shape;
    return lerp(uv_min, uv_max, uv);
}

// Run the encoder to generate (a sub tile of) one UDIM.
void encode_latent_texture_tile<
    int NumEncoderInputs,
    int NumLatents,
    Encoder : IEncoder<NumEncoderInputs, NumLatents>
>(
    IRWTensor<half[NumLatents], 2> udim,
    Encoder encoder,
    int material_id,
    half[NumEncoderInputs] encoder_inputs,
    int2 sub_tile_offset,
    int2 call_id
)
{
    // Run the encoder to convert material attributes into the latent code.
    float[NumLatents] latents = encoder.forward({ material_id, encoder_inputs });

    // Cast to FP16.
    half[NumLatents] latents_half;
    [unroll]
    for (int i = 0; i < NumLatents; ++i)
        latents_half[i] = half(latents[i]);

    // Write the latents for the tile into the UDIM buffer at the right offset.
    udim[sub_tile_offset + call_id] = latents_half;
}
