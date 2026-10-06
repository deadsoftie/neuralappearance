// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
import slangpy;
import model.base_types;
__exported import model.texture;
__exported import util.texture_helpers;

// NTC-style compressed latent texture. Instead of one dense per-texel
// buffer (``LatentTexture``), this stores a PAIR of bilinearly-sampled
// feature grids per tile -- both at a resolution lower than the output
// texture by ``grid_size_scale`` (high-res) and half that again (low-res),
// each with ``NumFeatures`` channels -- and a small fixed decode MLP that
// turns a (high-res sample, low-res sample, positional encoding, LOD)
// tuple into the ``NumLatents``-channel code every other RTNAM component
// (BsdfDecoder, Sampler, AuxDecoder, the renderer, the visualizer) already
// expects, unchanged, via the shared ``ILatentTexture<NumLatents>``
// interface. See rtcnam/PLAN.md for the exact grid-math/quantization
// recipe this mirrors (NVIDIA's shipped RTXNTC SDK, not just the 2023
// paper) and rtcnam/NTC_LATENT_TEXTURE_NOTES.md for the parts of that
// recipe this first implementation simplifies away.
//
// Both grids always use the same channel count (``NumFeatures``), matching
// RTXNTC's own ``FeatureGrid`` usage in RegressionKernels.h, where
// ``highResFeatureGrid``/``lowResFeatureGrid`` are constructed with the
// identical ``params.numFeatures``.
struct NtcLatentTexture<
    int NumTiles,
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
    Texture<NumFeatures> high_res_grids[NumTiles];
    Texture<NumFeatures> low_res_grids[NumTiles];
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

        float[NumFeatures] hi = high_res_grids[tile_idx].get_bilinear(input.uv);
        float[NumFeatures] lo = low_res_grids[tile_idx].get_bilinear(input.uv);

        // Plain (non-[Differentiable]) helper calls, not an inline `no_diff`
        // subscript: an inline `no_diff material_texture_resolution[...]`
        // here crashes the Slang compiler (internal error) when this
        // function is later wrapped in `bwd_diff(...)` for training -- see
        // `get_udim_index`/`get_tile_idx` below for the same already-proven
        // pattern (plain helper returning a non-differentiable type, called
        // without `no_diff` at all).
        let resolution = get_material_resolution(input.material_id);
        let wrap = get_material_wrap(input.material_id);
        float[4 * NumPosEncWaves] pos_enc = encode_position<NumPosEncWaves>(input.uv, resolution, wrap);
        float lod = float(input.mip_level);

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

    int get_material_resolution(int material_id)
    {
        return material_texture_resolution[material_id];
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

// Deterministic Fourier-style positional encoding, mirroring RTXNTC's own
// ``EncodeSamplePosition`` (RegressionKernels.h): per octave, two offset
// phases (0 and 0.25) of ``frac(coord) * 2 - 1``, doubling frequency each
// octave. Operates in output-texel space (``uv * resolution``), not raw
// [0,1) UV, so each octave's frequency is tied to the color-texel grid the
// same way RTXNTC's ``positionScale`` ties it to the chosen color mip --
// simplified here to the single-mip case (no mip-grouping term), since
// this repo's configs train one latent mip level.
float[4 * NumWaves] encode_position<int NumWaves>(float2 uv, int resolution, bool wrap)
{
    float2 local_uv = wrap ? (uv - floor(uv)) : max(0.f, uv);
    float2 p = local_uv * float(resolution);

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
