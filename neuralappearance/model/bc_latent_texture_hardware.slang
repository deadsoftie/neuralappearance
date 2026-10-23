// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
__exported import model.base_types;

// Render-only hardware BC6H sampling of packed software-training parameters.
// One compressed Texture2D per (set, tile, mip) replaces the software four-tap
// decode with fixed-function decoding and filtering. This type is never
// optimizable, so all backward functions are no-ops.
//
// SampleGrad supplies the ray footprint's UV derivatives for anisotropic
// filtering; Sample has no implicit derivatives outside fragment shaders, and
// SampleLevel selects only an isotropic LOD. Each grid is a single-level texture,
// so filtering is anisotropic within the selected mip, not trilinear across mips.
struct BcLatentTextureHardware<
    int NumSets,
    int NumTiles,
    int NumMipLevels,
    int NumLatents
> : ILatentTexture<NumLatents>
{
    StructuredBuffer<int>.Handle tile_indirection;
    StructuredBuffer<int>.Handle material_texture_wrap;
    StructuredBuffer<int>.Handle material_texture_resolution;
    Texture2D<float4>.Handle grids[NumSets * NumTiles * NumMipLevels];
    SamplerState.Handle samp;
    SamplerState.Handle clamp_samp;
    SamplerState.Handle nearest_samp;
    SamplerState.Handle nearest_clamp_samp;

    int get_texture_resolution(int material_id)
    {
        return material_texture_resolution[material_id];
    }

    // Set-major, then tile, then mip -- same convention as
    // `BcLatentTexture.get_grid_idx` and `model_init`'s build order, so a
    // checkpoint's texture ordering needs no translation between the two
    // types (`bc_latent_texture_hardware.py` builds this array by iterating
    // its software counterpart's own lists directly).
    int get_grid_idx(int set_idx, int tile_idx, int mip_level)
    {
        return set_idx * (NumTiles * NumMipLevels) + tile_idx * NumMipLevels + mip_level;
    }

    int get_udim_index(int material_id, float2 uv)
    {
        float2 local_uv = uv;
        if (material_texture_wrap[material_id] == 1)
            local_uv = local_uv - floor(local_uv);
        else
            local_uv = max(0.f, local_uv);
        int2 tile_coord = int2(local_uv);
        return min(max(tile_coord.x + 10 * tile_coord.y, 0), 99);
    }

    int get_tile_idx(int material_id, int udim_idx)
    {
        let udim_stride = 100;
        return this.tile_indirection[material_id * udim_stride + udim_idx];
    }

    // Forward-only; the explicit backward below is a no-op because hardware
    // textures are not optimizable.
    float[NumLatents] get_bilinear(no_diff LatentTextureInput input)
    {
        let udim_idx = get_udim_index(input.material_id, input.uv);
        let tile_idx = get_tile_idx(input.material_id, udim_idx);
        let sampler = material_texture_wrap[input.material_id] == 1 ? samp : clamp_samp;

        float[NumLatents] result;
        [ForceUnroll]
        for (int s = 0; s < NumSets; ++s)
        {
            let grid_idx = get_grid_idx(s, tile_idx, input.mip_level);
            float3 decoded = grids[grid_idx]
                .SampleGrad(sampler, input.uv, input.duv_dx, input.duv_dy).rgb;
            result[s * 3 + 0] = decoded.x;
            result[s * 3 + 1] = decoded.y;
            result[s * 3 + 2] = decoded.z;
        }
        return result;
    }

    [BackwardDerivativeOf(get_bilinear)]
    void get_bilinear_bwd(no_diff LatentTextureInput input, float[NumLatents] d_out)
    {
        // True no-op: this texture is never optimizable, so there is no
        // buffer to scatter into. See the type-level comment.
    }

    // FORWARD ONLY, no attribute. See get_bilinear's comment above.
    float[NumLatents] get_nearest(no_diff LatentTextureInput input)
    {
        let udim_idx = get_udim_index(input.material_id, input.uv);
        let tile_idx = get_tile_idx(input.material_id, udim_idx);
        let sampler = material_texture_wrap[input.material_id] == 1
            ? nearest_samp : nearest_clamp_samp;

        float[NumLatents] result;
        [ForceUnroll]
        for (int s = 0; s < NumSets; ++s)
        {
            let grid_idx = get_grid_idx(s, tile_idx, input.mip_level);
            // Point sampling has no footprint to speak of -- SampleLevel at
            // mip 0 is the right degenerate case, not SampleGrad with
            // zeroed derivatives (which would ask the sampler to treat a
            // true point query as an infinitely fine anisotropic one).
            float3 decoded = grids[grid_idx].SampleLevel(sampler, input.uv, 0.f).rgb;
            result[s * 3 + 0] = decoded.x;
            result[s * 3 + 1] = decoded.y;
            result[s * 3 + 2] = decoded.z;
        }
        return result;
    }

    [BackwardDerivativeOf(get_nearest)]
    void get_nearest_bwd(no_diff LatentTextureInput input, float[NumLatents] d_out) {}

    // FORWARD ONLY, no attribute. See get_bilinear's comment above.
    //
    // Hardware filtering is already a single fetch, so stochastic sampling uses
    // the filtered value rather than the software path's cheaper point tap.
    float[NumLatents] get_bilinear_stochastic(no_diff LatentTextureInput input, no_diff float sample)
    {
        return get_bilinear(input);
    }

    [BackwardDerivativeOf(get_bilinear_stochastic)]
    void get_bilinear_stochastic_bwd(
        no_diff LatentTextureInput input, no_diff float sample, float[NumLatents] d_out) {}
}
