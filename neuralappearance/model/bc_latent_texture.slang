// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
import slangpy;
import model.base_types;
__exported import model.texture;
__exported import model.bc6h;
__exported import util.texture_helpers;

// Block-compressed latent texture with no decode MLP or positional encoding:
//   latent[3 * NumSets] = concat_s(bilinear_s(bc6h_decode(texel)))
// Descending set resolutions provide the feature pyramid.
//
// Each set stores six endpoint channels at 4x4-block resolution and one
// position channel at texel resolution. Endpoints are normalized and scaled to
// endpoint-integer units before decoding. Reusing Texture<N> provides
// differentiable sampling, gradient scatter, and checkpoint persistence.
//
// Bilinear filtering decodes all four neighboring texels before blending their
// values, matching fixed-function BC6H filtering. Blending encoded parameters
// first would produce different results. Hardware sampling replaces these four
// software decodes per set with one filtered fetch.
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

    // Forward-only decode with an explicit backward below. All texels in a 4x4
    // block share one endpoint pair. The block's first texel is the anchor and
    // has a 3-bit index; all other texels have 4-bit indices.
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

    // Chain explicitly through bc6h_decode_mode10, then scatter gradients
    // into the endpoint and position buffers. Nested automatic differentiation
    // does not reliably invoke the decoder's custom derivative.
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

    // Set-major, then tile, then mip; Python uses the same allocation order.
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
