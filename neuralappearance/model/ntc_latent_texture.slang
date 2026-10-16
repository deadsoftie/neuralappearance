// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
import slangpy;
import model.base_types;
__exported import model.texture;
__exported import util.texture_helpers;

// NTC-style compressed latent texture. Instead of one dense per-texel
// buffer (``LatentTexture``), this stores a PAIR of bilinearly-sampled
// feature grids per (tile, mip) -- both at a resolution lower than that
// mip's output texture by ``grid_size_scale`` (high-res) and half that
// again (low-res), each with ``NumFeatures`` channels, flattened as
// ``tile_idx * NumMipLevels + mip_level`` exactly like the dense path's
// own mip chain -- and a small fixed decode MLP that
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

        // No bounding function on the output, deliberately, and
        // `latents.normalize` must be FALSE in the config for this path.
        //
        // The stock `latents.normalize: true` makes the Encoder squash its
        // own output through `normalize_latent_code` (`x/(|x|+1)`,
        // encoder.slang) so `BsdfDecoder` trains on (-1, 1). That is fine for
        // stock RTNAM, which always pairs it with `bsdf_finetuning: 0` -- and
        // train.py says why it has to:
        //
        //   assert not config[model][latents][normalize], (
        //       "Direct optimization of latents is not compatible with "
        //       "latent normalization."
        //   )
        //
        // This path DOES run direct optimization, so normalization is off and
        // everything -- encoder output, warm-start target, this grid's output,
        // the latents `BsdfDecoder` sees -- lives in one unsquashed space with
        // unit gradient throughout.
        //
        // Four output variants were measured at full scale before settling
        // here; the reasons are easy to re-introduce:
        //
        //   - Squash applied during warm-start as well, against the encoder's
        //     *post*-squash output: a double squash. To express a target of
        //     0.9 the MLP had to output 9, deep in the squash's
        //     vanishing-gradient region (derivative `1/(1+|x|)^2`), so 59% of
        //     components pinned near +-1 and never recovered. PSNR -179 dB.
        //   - Leaky clamp (bound 4 then 1, leak 0.01): self-defeating. Beyond
        //     the bound the gradient is only 0.01, nearly as dead as the
        //     squash, while 1% of a large raw value is still large, so it did
        //     not bound either. PSNR -287/-434 dB.
        //   - Squash on the output with warm-start regressing in *pre*-squash
        //     space (so warm-start itself had unit gradient -- the obvious fix
        //     to the first variant). This bounded the latents correctly:
        //     measured [-0.92, 0.94] against dense's [-0.79, 0.89]. It still
        //     rendered at -37.5 dB. The reason is that the encoder's own raw
        //     latents are large (warm-start learned mean |z| = 5.37, range
        //     [-11.8, 14.5]), so the squash ran deep in saturation and scaled
        //     direct optimization's gradients by ~0.025, effectively freezing
        //     the grid. **The latent range was never the problem** -- this
        //     variant proves it, by fixing the range and changing nothing.
        //   - No squash, `normalize: true` left on (so the warm-start target
        //     was squashed but this output was not): internally inconsistent,
        //     yet it reached +32.24 dB at 1.11 bpp because direct optimization
        //     co-adapts `BsdfDecoder`. It failed at 20 bpp (-313.8 dB), where
        //     the larger grid drifts further out of the band the decoder knows
        //     ([-3.91, 3.56] versus [-1.65, 1.76]).
        //
        // Hence: no squash anywhere, and `normalize: false`.
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

// Deterministic Fourier-style positional encoding, mirroring RTXNTC's own
// ``EncodeSamplePosition`` (RegressionKernels.h): per octave, two offset
// phases (0 and 0.25) of ``frac(coord) * 2 - 1``, doubling frequency each
// octave.
//
// ``grid_resolution`` is the HIGH-RES FEATURE GRID resolution, not the color
// texture resolution, and the 0.5 factor below is RTXNTC's: together they
// reproduce its ``posf = uv * gridWidth * 0.5`` exactly (derivation at the
// call site). The first octave therefore spans two grid cells and the last
// of three octaves reaches 2x the grid resolution.
//
// THIS SCALE IS LOAD-BEARING; DO NOT TIE IT TO THE COLOR TEXTURE RESOLUTION.
// An earlier version used ``uv * color_texture_resolution`` (4096), which at
// gridSizeScale 2 is 4x too high in every octave and at gridSizeScale 6 is
// 12x too high -- the top octave reaching 8-24x the grid resolution. Inside a
// single grid cell the feature inputs are only a bilinear (linear) ramp, so
// the decode MLP could not represent sub-cell structure from them; instead it
// fit the training residual off the aliased sawtooth, which at those
// frequencies acts as a pseudo-random hash of the UV. Measured consequence on
// the 20 bpp run: within ONE grid cell the decoded latents swung over
// [-3.07, 1.28] with 543x more total variation than net change, and held-out
// UVs hashed differently from training UVs, so latents reached [-15.2, 6.0]
// against the encoder's real [-0.79, 0.89]. Warm-start *training* MSE looked
// healthy (0.0073) while the render was -240 dB -- a pure train/test gap. See
// rtcnam/NTC_LATENT_TEXTURE_NOTES.md.
//
// Explicit band-limiting (rtcnam/MIP_SAMPLING_PLAN.md Stage 4, "explicit"
// option, stacked on top of the "implicit" one -- the two were never
// mutually exclusive): smoothly zero out the highest octaves as `mip_level`
// increases, in addition to (not instead of) `grid_resolution` already
// shrinking per mip. Standard procedural-noise mip-fade technique --
// `keep_octaves` drops by exactly one per mip level, and each octave's own
// weight is a 1-wide linear ramp from 1 (fully present) to 0 (fully faded)
// centered on that cutoff.
//
// `mip_level == 0` is an exact, provable no-op, not a tuned-to-look-right
// approximation: `keep_octaves == NumWaves` there, so for every valid
// octave index `w` (0..NumWaves-1), `keep_octaves - w >= 1`, so
// `clamp(keep_octaves - w, 0, 1) == 1` for all of them. This is true for
// every value `NumWaves` is ever configured to, so every single-mip config
// (every config in the repo before this plan, and mip 0 of every config
// after it) renders byte-identical to before this change.
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
