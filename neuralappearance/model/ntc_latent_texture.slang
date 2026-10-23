// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
import slangpy;
import model.base_types;
__exported import model.texture;
__exported import util.texture_helpers;

// NTC-style compressed latent texture. A high/low-resolution feature-grid pair
// per (tile, mip) is bilinearly sampled and decoded with positional encoding and
// LOD into NumLatents channels. Grid pairs use the same flat tile-major layout
// as the dense texture's mip chain and share NumFeatures channels.
struct NtcLatentTexture<
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
    Texture<NumFeatures> high_res_grids[NumTiles * NumMipLevels];
    Texture<NumFeatures> low_res_grids[NumTiles * NumMipLevels];
    Decoder decoder;

    int get_texture_resolution(int material_id)
    {
        return material_texture_resolution[material_id];
    }

    [Differentiable]
    float[NumLatents] get_bilinear(LatentTextureInput input)
    {
        let udim_idx = get_udim_index(input.material_id, input.uv);
        let tile_idx = get_tile_idx(input.material_id, udim_idx);
        let grid_idx = get_grid_idx(tile_idx, input.mip_level);

        float[NumFeatures] hi = high_res_grids[grid_idx].get_bilinear(input.uv);
        float[NumFeatures] lo = low_res_grids[grid_idx].get_bilinear(input.uv);

        // Plain (non-[Differentiable]) helper calls, not an inline `no_diff`
        // subscript: an inline `no_diff material_texture_resolution[...]`
        // here crashes the Slang compiler (internal error) when this
        // function is later wrapped in `bwd_diff(...)` for training -- see
        // `get_udim_index`/`get_tile_idx` below for the same already-proven
        // pattern (plain helper returning a non-differentiable type, called
        // without `no_diff` at all).
        let grid_resolution = get_grid_resolution(grid_idx);
        let texture_resolution = get_texture_resolution(input.material_id);
        let wrap = get_material_wrap(input.material_id);
        // RTXNTC's `positionScale` (TextureSet.cpp:461):
        //   positionScale = 0.5 * latentGridWidth / colorMipWidth
        // and it encodes `texel * positionScale`, i.e. in UV terms
        //   posf = (uv * colorMipWidth) * 0.5 * gridWidth / colorMipWidth
        //        = uv * gridWidth * 0.5
        // so the FIRST octave's sawtooth period is two high-res grid cells and
        // the top octave (3 waves) is 2x the grid resolution -- it encodes
        // *sub-cell* position, just past the grid's own Nyquist limit.
        float[4 * NumPosEncWaves] pos_enc =
            encode_position<NumPosEncWaves>(input.uv, grid_resolution, wrap, input.mip_level);
        // RTXNTC feeds `positionLod` here, not the raw mip index
        // (TextureSet.cpp:462). Computed in a plain helper, like the others.
        float lod = get_position_lod(grid_resolution, texture_resolution, input.mip_level);

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

        // Keep decoded latents unbounded and disable latent normalization for
        // this path. Encoder targets, warm-start, and direct optimization must
        // share one unsquashed latent space with unit output gradients.
        return decoder.forward(decoder_input);
    }

    [Differentiable]
    float[NumLatents] get_nearest(LatentTextureInput input)
    {
        return get_bilinear(input);
    }

    [Differentiable]
    float[NumLatents] get_bilinear_stochastic(LatentTextureInput input, float sample)
    {
        // The grids are already continuously bilinearly sampled and run
        // through the decoder on every call; there is no cheaper discrete
        // "pick one texel" variant the way there is for a dense buffer, so
        // this ignores `sample` and decodes the same way `get_bilinear` does.
        return get_bilinear(input);
    }

    // Flat (tile, mip) index into `high_res_grids`/`low_res_grids`. Same ABI
    // as dense `LatentTexture::get_texture_idx` (`latent_texture.slang:68-74`):
    // Python stores each tile's mip chain contiguously in this order.
    int get_grid_idx(int tile_idx, int mip_level)
    {
        return tile_idx * NumMipLevels + mip_level;
    }

    // High-res feature grid resolution for a (tile, mip) grid pair. Plain
    // (non-[Differentiable]) helper for the same reason as the others below
    // -- see the comment in `get_bilinear`. Takes the already-flattened grid
    // index, not a bare tile index.
    int get_grid_resolution(int grid_idx)
    {
        return high_res_grids[grid_idx].resolution;
    }

    // RTXNTC's ``positionLod`` (TextureSet.cpp:462):
    //   clamp(0.25 * log2(positionScale), -1, 1)
    // A per-(tile, mip) constant, so the decode MLP's first-layer bias can
    // absorb it in the single-mip case.
    //
    // `texture_resolution` is the material's BASE (mip-0) color-texture
    // resolution; RTXNTC's real formula needs the color-mip width at the
    // same LOD as `grid_resolution`, so shrink it by `mip_level` here --
    // matching dense `LatentTexture.resolution()`'s own `base_resolution >>
    // mip_level` convention (`latent_texture.py:175-179`). At `mip_level ==
    // 0` this is a no-op, so the single-mip case is bit-for-bit unchanged.
    float get_position_lod(int grid_resolution, int texture_resolution, int mip_level)
    {
        let color_mip_width = max(1, texture_resolution >> mip_level);
        let position_scale = 0.5f * float(grid_resolution) / float(color_mip_width);
        return clamp(0.25f * log2(position_scale), -1.f, 1.f);
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

// RTXNTC-style positional encoding: two offset phases per octave, with
// frequency doubling each octave. Frequencies use the high-res feature-grid
// resolution, never the color texture resolution.
//
// Higher octaves fade as mip_level increases. At mip 0 every octave has unit
// weight, so filtering is disabled.
float[4 * NumWaves] encode_position<int NumWaves>(float2 uv, int grid_resolution, bool wrap, int mip_level)
{
    float2 local_uv = wrap ? (uv - floor(uv)) : max(0.f, uv);
    float2 p = local_uv * (float(grid_resolution) * 0.5f);
    float keep_octaves = float(NumWaves) - float(mip_level);

    float[4 * NumWaves] result;
    int idx = 0;
    [ForceUnroll]
    for (int w = 0; w < NumWaves; ++w)
    {
        float weight = clamp(keep_octaves - float(w), 0.f, 1.f);
        result[idx++] = weight * (frac(p.x) * 2.f - 1.f);
        result[idx++] = weight * (frac(p.y) * 2.f - 1.f);
        result[idx++] = weight * (frac(p.x + 0.25f) * 2.f - 1.f);
        result[idx++] = weight * (frac(p.y + 0.25f) * 2.f - 1.f);
        p *= 2.f;
    }
    return result;
}
