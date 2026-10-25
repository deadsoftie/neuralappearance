// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
import slangpy;
import model.base_types;
__exported import model.texture;
__exported import util.texture_helpers;

// Renders a texture set compressed by ntc-cli (post-hoc baseline) directly from
// its grids, decoding at the shading point instead of decompressing to a dense
// latent first. Same grid storage and decode MLP as `NtcLatentTexture`; the
// differences all follow RTXNTC's own inference (RTXNTC-Library
// include/libntc/shaders/Inference.hlsli, `NtcSampleTextureSet`):
//
// - Decoding happens at one color-texture texel: grids are sampled at that
//   texel's center and the positional encoding uses the integer texel index
//   times `positionScale`. ntc-cli only ever trained on texel positions.
// - Filtering is stochastic: `get_bilinear_stochastic` picks one of the four
//   neighbouring texels by its bilinear weight, then decodes there. This is
//   the same texel choice the dense `LatentTexture` makes, so rendering this
//   texture matches rendering ntc-cli's decompressed latent in expectation.
// - All octaves are always active (no mip fade) and each decoded channel goes
//   through RTXNTC's HLG decode, then the per-channel affine that undoes the
//   [0, 1] normalization applied before compression.
struct NtcSdkLatentTexture<
    int NumTiles,
    int NumMipLevels,
    int NumFeatures,
    int NumPosEncWaves,
    int NumLatents,
    int DecoderInputWidth,
    Decoder : IModel<float[DecoderInputWidth], float[NumLatents]>
> : ILatentTexture<NumLatents>
{
    StructuredBuffer<int>.Handle tile_indirection;
    StructuredBuffer<int>.Handle material_texture_wrap;
    StructuredBuffer<int>.Handle material_texture_resolution;
    // Per material: NumLatents scales followed by NumLatents biases.
    StructuredBuffer<float>.Handle output_affine;
    Texture<NumFeatures> high_res_grids[NumTiles * NumMipLevels];
    Texture<NumFeatures> low_res_grids[NumTiles * NumMipLevels];
    Decoder decoder;

    int get_texture_resolution(int material_id)
    {
        return material_texture_resolution[material_id];
    }

    [Differentiable]
    float[NumLatents] get_nearest(LatentTextureInput input)
    {
        let color_resolution = get_color_resolution(input.material_id, input.mip_level);
        let texel = nearest_texel(input.uv, color_resolution, get_material_wrap(input.material_id));
        return decode_texel(input, texel, color_resolution);
    }

    [Differentiable]
    float[NumLatents] get_bilinear(LatentTextureInput input)
    {
        // Four decodes blended, the deterministic equivalent of
        // `get_bilinear_stochastic`. Not on the render path.
        let color_resolution = get_color_resolution(input.material_id, input.mip_level);
        let wrap = get_material_wrap(input.material_id);
        uint2 texels[4];
        float weights[4];
        no_diff get_bilinear_interpolation_texels_and_weights(
            input.uv, uint2(color_resolution), wrap, texels, weights);

        float[NumLatents] result;
        [ForceUnroll]
        for (int c = 0; c < NumLatents; ++c)
            result[c] = 0.f;
        [ForceUnroll]
        for (int i = 0; i < 4; ++i)
        {
            let decoded = decode_texel(input, int2(texels[i]), color_resolution);
            [ForceUnroll]
            for (int c = 0; c < NumLatents; ++c)
                result[c] += weights[i] * decoded[c];
        }
        return result;
    }

    [Differentiable]
    float[NumLatents] get_bilinear_stochastic(LatentTextureInput input, float sample)
    {
        let color_resolution = get_color_resolution(input.material_id, input.mip_level);
        let wrap = get_material_wrap(input.material_id);
        let texel = stochastic_texel(input.uv, color_resolution, wrap, sample);
        return decode_texel(input, texel, color_resolution);
    }

    [Differentiable]
    float[NumLatents] decode_texel(LatentTextureInput input, int2 texel, int color_resolution)
    {
        let udim_idx = get_udim_index(input.material_id, input.uv);
        let tile_idx = get_tile_idx(input.material_id, udim_idx);
        let grid_idx = get_grid_idx(tile_idx, input.mip_level);

        let uv_center = (float2(texel) + 0.5f) / float(color_resolution);
        float[NumFeatures] hi = high_res_grids[grid_idx].get_bilinear(uv_center);
        float[NumFeatures] lo = low_res_grids[grid_idx].get_bilinear(uv_center);

        // RTXNTC (TextureSet.cpp): positionScale = 0.5 * gridWidth / colorMipWidth,
        // positionLod = clamp(0.25 * log2(positionScale), -1, 1).
        let position_scale = get_position_scale(grid_idx, color_resolution);
        let lod = clamp(0.25f * log2(position_scale), -1.f, 1.f);
        float[4 * NumPosEncWaves] pos_enc = encode_texel_position<NumPosEncWaves>(float2(texel) * position_scale);

        float[DecoderInputWidth] decoder_input;
        int idx = 0;
        [ForceUnroll]
        for (int i = 0; i < NumFeatures; ++i)
            decoder_input[idx++] = hi[i];
        [ForceUnroll]
        for (int i = 0; i < NumFeatures; ++i)
            decoder_input[idx++] = lo[i];
        [ForceUnroll]
        for (int i = 0; i < 4 * NumPosEncWaves; ++i)
            decoder_input[idx++] = pos_enc[i];
        decoder_input[idx++] = lod;

        float[NumLatents] encoded = decoder.forward(decoder_input);
        float[NumLatents] result;
        [ForceUnroll]
        for (int c = 0; c < NumLatents; ++c)
        {
            let scale = get_output_scale(input.material_id, c);
            let bias = get_output_bias(input.material_id, c);
            result[c] = hlg_decode(encoded[c]) * scale + bias;
        }
        return result;
    }

    int get_grid_idx(int tile_idx, int mip_level)
    {
        return tile_idx * NumMipLevels + mip_level;
    }

    // Plain (non-[Differentiable]) helpers for every non-differentiable
    // lookup, following `ntc_latent_texture.slang` -- inline `no_diff`
    // subscripts crash the Slang compiler under `bwd_diff`.
    float get_position_scale(int grid_idx, int color_resolution)
    {
        return 0.5f * float(high_res_grids[grid_idx].resolution) / float(color_resolution);
    }

    int get_color_resolution(int material_id, int mip_level)
    {
        return max(1, material_texture_resolution[material_id] >> mip_level);
    }

    float get_output_scale(int material_id, int channel)
    {
        return output_affine[material_id * 2 * NumLatents + channel];
    }

    float get_output_bias(int material_id, int channel)
    {
        return output_affine[material_id * 2 * NumLatents + NumLatents + channel];
    }

    int2 nearest_texel(float2 uv, int color_resolution, bool wrap)
    {
        let texel = int2(round(uv_to_fractional_texel(uv, uint2(color_resolution), wrap)));
        return wrap ? (texel % color_resolution) : clamp(texel, 0, color_resolution - 1);
    }

    int2 stochastic_texel(float2 uv, int color_resolution, bool wrap, float sample)
    {
        float s = sample;
        let texel_uv = sample_bilinear_texel_center(uv, uint2(color_resolution), wrap, s);
        let local_uv = texel_uv - floor(texel_uv);
        return clamp(int2(floor(local_uv * float(color_resolution))), 0, color_resolution - 1);
    }

    bool get_material_wrap(int material_id)
    {
        return material_texture_wrap[material_id] == 1;
    }

    int get_udim_index(int material_id, float2 uv)
    {
        float2 local_uv = uv;
        if (material_texture_wrap[material_id] == 1)
        {
            local_uv = local_uv - floor(local_uv);
        }
        else
        {
            local_uv = max(0.f, local_uv);
        }
        int2 tile_coord = int2(local_uv);
        return min(max(tile_coord.x + 10 * tile_coord.y, 0), 99);
    }

    int get_tile_idx(int material_id, int udim_idx)
    {
        let udim_stride = 100;
        return this.tile_indirection[material_id * udim_stride + udim_idx];
    }
}

// RTXNTC's `NtcEncodeSamplePosition`: two offset sawtooth phases per octave on
// an already scaled texel position, frequency doubling each octave.
[Differentiable]
float[4 * NumWaves] encode_texel_position<int NumWaves>(float2 p)
{
    float[4 * NumWaves] result;
    int idx = 0;
    [ForceUnroll]
    for (int w = 0; w < NumWaves; ++w)
    {
        result[idx++] = frac(p.x) * 2.f - 1.f;
        result[idx++] = frac(p.y) * 2.f - 1.f;
        result[idx++] = frac(p.x + 0.25f) * 2.f - 1.f;
        result[idx++] = frac(p.y + 0.25f) * 2.f - 1.f;
        p *= 2.f;
    }
    return result;
}

// RTXNTC's `NtcHybridLogGammaColorSpace::Decode` (ColorSpaces.hlsli).
[Differentiable]
float hlg_decode(float h)
{
    const float a = 0.17883277f;
    const float b = 0.28466892f;
    const float c = 0.55991073f;
    let s = (h < 0.f) ? -1.f : 1.f;
    let ah = abs(h);
    return s * ((ah < 0.5f) ? 4.f * ah * ah : (exp((ah - c) / a) + b));
}
