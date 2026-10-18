// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
import slangpy;
import model.base_types;
__exported import model.texture;
__exported import model.bc6h;
__exported import util.texture_helpers;

// Block-compressed latent texture. rtbcnam/PLAN.md Stage 2.
//
// Instead of a dense per-texel buffer (`LatentTexture`) or a feature-grid pair
// plus a decode MLP (`NtcLatentTexture`), this stores `NumSets` BC6H-shaped
// feature textures per (tile, mip) and concatenates their sampled values to
// form the latent directly. There is NO decode MLP and NO positional encoding:
//
//   latent[3 * NumSets] = concat_s( bilinear_s( bc6h_decode(texel) ) )
//
// That is BCf's own structure (docs/deep-dives/dd1-block-compressed-features.md,
// table 1's Ttheta0..Ttheta3) with RTNAM's `BsdfDecoder` in place of BCf's own
// tiny one. Multi-resolution capacity comes from the descending set
// resolutions rather than from a Fourier encoding, which is why the
// grid-lattice failure mode RTCNAM spent a day on cannot occur here.
//
// ---------------------------------------------------------------------------
// Storage: two textures per set, at different resolutions
// ---------------------------------------------------------------------------
//
// BC6H mode 10 stores, per 4x4 block, two endpoints (3 channels each) shared by
// all 16 texels, plus one index per texel. Those live at different rates, so
// they are two separate `Texture<N>` instances per set:
//
//   `endpoints[g]`  `Texture<6>` at block resolution (ceil(res / 4)):
//                   channels 0-2 are endpoint 0, channels 3-5 endpoint 1.
//   `positions[g]`  `Texture<1>` at full texel resolution: where each texel
//                   sits along its block's line segment, in [0, 1].
//
// Reusing `Texture<N>` rather than hand-rolling buffers is deliberate: it
// brings the proven differentiable fetch, the atomic gradient scatter, the
// `buffer_grads` aliasing for non-optimizable textures, and the checkpoint
// path, none of which need to be rewritten or re-tested.
//
// Endpoints are stored NORMALISED, in roughly [-1, 1], and scaled by
// `endpoint_scale` on the way into `bc6h_decode_mode10`, which works in
// endpoint-integer units. Stage 1 measured why: the position gradient runs
// three to four orders of magnitude above an endpoint gradient (the ratio is
// exactly the endpoint separation, which is physical), and a raw Adam step of
// 1e-3 in endpoint units would need a thousand steps to cross one quantization
// level. Half precision in [-1, 1] has ~11 bits of significand, so scaling by
// 512 still resolves well below one endpoint unit.
//
// ---------------------------------------------------------------------------
// Filtering: decode per texel, THEN blend
// ---------------------------------------------------------------------------
//
// `get_bilinear` decodes each of the four neighbouring texels independently and
// blends the decoded results. That ordering is not a detail: it is what the
// fixed-function texture unit does, so it is what Stage 5's hardware path will
// reproduce. Blending endpoints or indices first and decoding once would be
// cheaper and WRONG, and it would make Stage 3's quality number describe
// something no hardware will ever produce.
//
// Cost, stated plainly: 4 taps x `NumSets` sets = 16 decodes per latent fetch
// at the default 4 sets, each needing a 6-value endpoint read and a 1-value
// position read. That is heavy, and it is why Stage 5 exists -- there, one
// hardware `SampleLevel` per set replaces four software decodes. Stage 2
// deliberately accepts the cost so Stage 3's quality answer is not entangled
// with a new resource-binding path. A cheap future optimisation, not taken
// here: three of the four taps usually share a block, so the endpoint read
// could be hoisted.
struct BcLatentTexture<
    int NumSets,
    int NumTiles,
    int NumMipLevels,
    int NumLatents
> : ILatentTexture<NumLatents>
{
    StructuredBuffer<int>.Handle tile_indirection;
    StructuredBuffer<int>.Handle material_texture_wrap;
    StructuredBuffer<int>.Handle material_texture_resolution;
    Texture<6> endpoints[NumSets * NumTiles * NumMipLevels];
    Texture<1> positions[NumSets * NumTiles * NumMipLevels];
    // Endpoint-integer units per unit of stored parameter: 512 for the signed
    // format, 1023 for unsigned. See the header note.
    float endpoint_scale;
    // Non-zero selects BC6H_SF16 semantics. An int rather than a bool because
    // it crosses the Python boundary as plain buffer data.
    int is_signed;

    int get_texture_resolution(int material_id)
    {
        return material_texture_resolution[material_id];
    }

    // Decode one texel of one set, exactly as hardware would. FORWARD ONLY --
    // no differentiability attribute, deliberately, matching Texture._get_code's
    // own shape (also no attribute, only a registered backward). See the long
    // comment below on why this is load-bearing, not a style choice.
    //
    // The endpoint lookup is at BLOCK granularity -- all 16 texels in a block
    // share one line segment, which is the format's defining constraint and
    // the thing Stage 3 is really testing the latent against.
    //
    // `is_anchor`: the real format gives only 3 bits (not 4) to the texel at
    // local raster position 0 of each 4x4 block -- a hardware encoding
    // constraint, not a training choice. Found at Stage 4
    // (`rtbcnam/results/STAGE4.md`): measured on a real trained checkpoint,
    // 35.55% of all blocks needed an anchor index this file's earlier version
    // could not express, before `bc6h_quantize_index` grew an `is_anchor`
    // parameter. Do not remove this -- it is what makes Stage 4's "export is
    // lossless" claim actually true, rather than true of an unconstrained
    // simulation that the real format cannot reproduce.
    float3 decode_texel(int grid_idx, uint2 texel)
    {
        uint2 block = texel / 4;
        let is_anchor = (texel.x % 4 == 0) && (texel.y % 4 == 0);
        float[6] raw = endpoints[grid_idx]._get_code(block);
        float[1] position = positions[grid_idx]._get_code(texel);
        let scale = get_endpoint_scale();
        float3 endpoint0 = scale * float3(raw[0], raw[1], raw[2]);
        float3 endpoint1 = scale * float3(raw[3], raw[4], raw[5]);
        return bc6h_decode_mode10(endpoint0, endpoint1, position[0], get_is_signed(), is_anchor);
    }

    // Manual backward for one texel: given the incoming gradient on the
    // decoded float3, chain explicitly through bc6h_decode_mode10 and
    // scatter into the endpoint/position buffers via _get_code_bwd.
    //
    // WHY THIS CANNOT BE AUTOMATIC. bc6h_decode_mode10 has a hand-written
    // custom derivative (Stage 1) because its forward pass is an integer
    // pipeline with a true zero gradient almost everywhere. Stage 2 found,
    // the hard way -- by noticing the trained texture never moved from its
    // init value at ANY checkpoint across an entire training run -- that
    // Slang's automatic differentiation of an OUTER [Differentiable]
    // function does NOT correctly invoke a NESTED callee's registered
    // custom derivative when that callee has DIFFERENTIABLE VALUE
    // PARAMETERS (DifferentialPair-based, like bc6h_decode_mode10's
    // endpoint0/endpoint1/position). Verified with a minimal,
    // bc6h-independent repro: a trivial custom-derivative function called
    // ordinarily from inside another [Differentiable] function silently
    // produces an exact zero, with no warning and no compile error,
    // regardless of attribute choice on either function.
    //
    // What DOES survive nesting (verified with the same minimal repro, at
    // the SAME nesting depth this code actually needs): a function with NO
    // differentiable value parameters of its own, whose derivative acts
    // entirely through SIDE EFFECTS -- exactly _get_code/_get_code_bwd's own
    // shape, which NtcLatentTexture's grids have relied on all along. So
    // this function reproduces that shape one level up: decode_texel and
    // this backward both have no differentiable VALUE parameters (tex's
    // buffers carry the real state), and the backward manually invokes
    // bwd_diff(bc6h_decode_mode10) itself -- an EXPLICIT call, not a nested
    // ordinary one -- then threads the result into _get_code_bwd's own
    // atomic scatter by hand.
    //
    // Do not "simplify" this back into an ordinary nested call. Re-run
    // rtbcnam/results/STAGE2_GRADIENT_BUG.md's reproduction first.
    void decode_texel_bwd(int grid_idx, uint2 texel, float3 d_out)
    {
        if (!is_grid_optimizable(grid_idx))
            return;   // frozen: no scatter, same intent as forward's detach().

        uint2 block = texel / 4;
        let is_anchor = (texel.x % 4 == 0) && (texel.y % 4 == 0);
        float[6] raw = endpoints[grid_idx]._get_code(block);
        float[1] position = positions[grid_idx]._get_code(texel);
        let scale = get_endpoint_scale();
        float3 ep0 = scale * float3(raw[0], raw[1], raw[2]);
        float3 ep1 = scale * float3(raw[3], raw[4], raw[5]);

        var pair0 = diffPair(ep0);
        var pair1 = diffPair(ep1);
        var pair_pos = diffPair(position[0]);
        bwd_diff(bc6h_decode_mode10)(pair0, pair1, pair_pos, get_is_signed(), is_anchor, d_out);

        // Chain rule through ep = scale * float3(raw[...]):
        // d_raw[i] = d(ep[i])/d(raw[i]) * d_ep[i] = scale * d_ep[i].
        float[6] d_raw;
        d_raw[0] = scale * pair0.d.x; d_raw[1] = scale * pair0.d.y; d_raw[2] = scale * pair0.d.z;
        d_raw[3] = scale * pair1.d.x; d_raw[4] = scale * pair1.d.y; d_raw[5] = scale * pair1.d.z;
        float[1] d_position;
        d_position[0] = pair_pos.d;

        endpoints[grid_idx]._get_code_bwd(block, d_raw);
        positions[grid_idx]._get_code_bwd(texel, d_position);
    }

    // FORWARD ONLY, no attribute -- see decode_texel's comment. The
    // registered backward immediately below is what actually trains this.
    float[NumLatents] get_bilinear(no_diff LatentTextureInput input)
    {
        let udim_idx = get_udim_index(input.material_id, input.uv);
        let tile_idx = get_tile_idx(input.material_id, udim_idx);
        let wrap = get_material_wrap(input.material_id);

        float[NumLatents] result;
        [ForceUnroll]
        for (int s = 0; s < NumSets; ++s)
        {
            let grid_idx = get_grid_idx(s, tile_idx, input.mip_level);
            let resolution = get_set_resolution(grid_idx);

            uint2 texels[4];
            float weights[4];
            get_bilinear_interpolation_texels_and_weights(
                input.uv, uint2(resolution), wrap, texels, weights);

            float3 blended = float3(0.f);
            [ForceUnroll]
            for (int i = 0; i < 4; ++i)
                blended += weights[i] * decode_texel(grid_idx, texels[i]);

            result[s * 3 + 0] = blended.x;
            result[s * 3 + 1] = blended.y;
            result[s * 3 + 2] = blended.z;
        }
        return result;
    }

    [BackwardDerivativeOf(get_bilinear)]
    void get_bilinear_bwd(no_diff LatentTextureInput input, float[NumLatents] d_out)
    {
        let udim_idx = get_udim_index(input.material_id, input.uv);
        let tile_idx = get_tile_idx(input.material_id, udim_idx);
        let wrap = get_material_wrap(input.material_id);

        [ForceUnroll]
        for (int s = 0; s < NumSets; ++s)
        {
            let grid_idx = get_grid_idx(s, tile_idx, input.mip_level);
            let resolution = get_set_resolution(grid_idx);

            uint2 texels[4];
            float weights[4];
            get_bilinear_interpolation_texels_and_weights(
                input.uv, uint2(resolution), wrap, texels, weights);

            float3 d_set = float3(d_out[s * 3 + 0], d_out[s * 3 + 1], d_out[s * 3 + 2]);
            [ForceUnroll]
            for (int i = 0; i < 4; ++i)
                decode_texel_bwd(grid_idx, texels[i], weights[i] * d_set);
        }
    }

    // FORWARD ONLY, no attribute. See get_bilinear's comment above.
    float[NumLatents] get_nearest(no_diff LatentTextureInput input)
    {
        let udim_idx = get_udim_index(input.material_id, input.uv);
        let tile_idx = get_tile_idx(input.material_id, udim_idx);
        let wrap = get_material_wrap(input.material_id);

        float[NumLatents] result;
        [ForceUnroll]
        for (int s = 0; s < NumSets; ++s)
        {
            let grid_idx = get_grid_idx(s, tile_idx, input.mip_level);
            let resolution = get_set_resolution(grid_idx);
            let texel = uint2(round(
                uv_to_fractional_texel(input.uv, uint2(resolution), wrap)));

            float3 decoded = decode_texel(grid_idx, texel);
            result[s * 3 + 0] = decoded.x;
            result[s * 3 + 1] = decoded.y;
            result[s * 3 + 2] = decoded.z;
        }
        return result;
    }

    [BackwardDerivativeOf(get_nearest)]
    void get_nearest_bwd(no_diff LatentTextureInput input, float[NumLatents] d_out)
    {
        let udim_idx = get_udim_index(input.material_id, input.uv);
        let tile_idx = get_tile_idx(input.material_id, udim_idx);
        let wrap = get_material_wrap(input.material_id);

        [ForceUnroll]
        for (int s = 0; s < NumSets; ++s)
        {
            let grid_idx = get_grid_idx(s, tile_idx, input.mip_level);
            let resolution = get_set_resolution(grid_idx);
            let texel = uint2(round(
                uv_to_fractional_texel(input.uv, uint2(resolution), wrap)));

            float3 d_set = float3(d_out[s * 3 + 0], d_out[s * 3 + 1], d_out[s * 3 + 2]);
            decode_texel_bwd(grid_idx, texel, d_set);
        }
    }

    // One texel per set, chosen with probability equal to its bilinear weight.
    //
    // Implemented rather than aliased to get_bilinear (which is what
    // NtcLatentTexture does) because here it is a real 4x saving: each tap
    // costs a full BC6H decode plus two buffer reads, and each contributes
    // its own atomic gradient scatter. The same sample drives every set, so
    // the four sets pick spatially consistent taps -- correlated, but that
    // is preferable to decorrelated noise across a single shading point's
    // latent.
    //
    // FORWARD ONLY, no attribute. See get_bilinear's comment above for why.
    float[NumLatents] get_bilinear_stochastic(no_diff LatentTextureInput input, no_diff float sample)
    {
        let udim_idx = get_udim_index(input.material_id, input.uv);
        let tile_idx = get_tile_idx(input.material_id, udim_idx);
        let wrap = get_material_wrap(input.material_id);

        float[NumLatents] result;
        [ForceUnroll]
        for (int s = 0; s < NumSets; ++s)
        {
            let grid_idx = get_grid_idx(s, tile_idx, input.mip_level);
            let resolution = get_set_resolution(grid_idx);

            // sample_bilinear_texel_center consumes its sample, so each set
            // gets its own copy of the same value.
            float local_sample = sample;
            let snapped = sample_bilinear_texel_center(
                input.uv, resolution, wrap, local_sample);
            let texel = uint2(round(
                uv_to_fractional_texel(snapped, uint2(resolution), wrap)));

            float3 decoded = decode_texel(grid_idx, texel);
            result[s * 3 + 0] = decoded.x;
            result[s * 3 + 1] = decoded.y;
            result[s * 3 + 2] = decoded.z;
        }
        return result;
    }

    [BackwardDerivativeOf(get_bilinear_stochastic)]
    void get_bilinear_stochastic_bwd(no_diff LatentTextureInput input, no_diff float sample, float[NumLatents] d_out)
    {
        let udim_idx = get_udim_index(input.material_id, input.uv);
        let tile_idx = get_tile_idx(input.material_id, udim_idx);
        let wrap = get_material_wrap(input.material_id);

        [ForceUnroll]
        for (int s = 0; s < NumSets; ++s)
        {
            let grid_idx = get_grid_idx(s, tile_idx, input.mip_level);
            let resolution = get_set_resolution(grid_idx);

            float local_sample = sample;
            let snapped = sample_bilinear_texel_center(
                input.uv, resolution, wrap, local_sample);
            let texel = uint2(round(
                uv_to_fractional_texel(snapped, uint2(resolution), wrap)));

            float3 d_set = float3(d_out[s * 3 + 0], d_out[s * 3 + 1], d_out[s * 3 + 2]);
            decode_texel_bwd(grid_idx, texel, d_set);
        }
    }

    // Flat index into `endpoints`/`positions`. SET-MAJOR, then tile, then mip:
    //
    //     set_idx * (NumTiles * NumMipLevels) + tile_idx * NumMipLevels + mip
    //
    // Python's `model_init` builds the lists in exactly this nesting order, and
    // `test_bc_latent_texture.py::test_texture_count_and_resolutions_per_mip`
    // pins it. RTCNAM's equivalent bug read mip 0 for every mip and stayed
    // invisible until a multi-mip run rendered wrong, so this is one of the few
    // things worth a dedicated test.
    int get_grid_idx(int set_idx, int tile_idx, int mip_level)
    {
        return set_idx * (NumTiles * NumMipLevels) + tile_idx * NumMipLevels + mip_level;
    }

    // Plain, non-`[Differentiable]` accessors. This is the proven pattern from
    // `ntc_latent_texture.slang`: an inline `no_diff` subscript on a struct
    // field inside a function that later gets wrapped in `bwd_diff(...)`
    // crashes the Slang compiler with an internal error, while a plain helper
    // returning a non-differentiable type called without `no_diff` at all is
    // fine. Do not inline these.
    int get_set_resolution(int grid_idx)
    {
        return positions[grid_idx].resolution;
    }

    int get_block_resolution(int grid_idx)
    {
        return endpoints[grid_idx].resolution;
    }

    bool is_grid_optimizable(int grid_idx)
    {
        return endpoints[grid_idx].optimizable;
    }

    float get_endpoint_scale()
    {
        return endpoint_scale;
    }

    bool get_is_signed()
    {
        return is_signed != 0;
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
