// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
__exported import model.base_types;

// Render-time-only hardware BC6H sampling. rtbcnam/PLAN.md Stage 5.
//
// `BcLatentTexture` (Stage 2) stores endpoints and positions as software
// StructuredBuffer-backed `Texture<N>` and decodes BC6H blocks in a Slang
// integer pipeline (`model/bc6h.slang`) so the representation is
// DIFFERENTIABLE for training. This type is the render-only counterpart:
// the SAME trained values, packed into a REAL compressed BC6H `Texture2D`
// per (set, tile, mip) and sampled by the fixed-function texture unit --
// hardware decode AND hardware anisotropic filtering in one instruction per
// set, instead of software's 4-tap decode-then-blend.
//
// Never trained, never optimizable -- built once, after training, from an
// already-trained `BcLatentTexture` checkpoint
// (`model/bc_latent_texture_hardware.py`'s `from_bc_latent_texture`), by
// packing the exact same bytes `rtbcnam/harness/export_bc6h.py` already
// builds and verifies bit-exact against the real GPU
// (`rtbcnam/results/STAGE4.md`). Every backward here is a true no-op, not a
// placeholder: a frozen, non-trainable texture has no gradient to produce,
// the same semantics `BcLatentTexture.decode_texel_bwd`'s own
// `is_grid_optimizable` early-out already gives a frozen grid.
//
// Implements `ILatentTexture<NumLatents>` exactly, so every existing
// consumer -- `BsdfDecoder`, `Sampler`, `AuxDecoder`, the renderer, the
// visualizer -- works against it unchanged; only `NeuralModel`'s own
// construction needs to pick this type instead of `BcLatentTexture` for a
// render-only (not training) run.
//
// ---------------------------------------------------------------------------
// Why SampleGrad, not Sample or SampleLevel
// ---------------------------------------------------------------------------
//
// This is a ray tracer, not a rasterizer: there is no implicit screen-space
// derivative for `Texture2D.Sample` to use, and HLSL/Slang refuse to emit it
// outside a fragment-shader-like stage. `SampleLevel` compiles but only ever
// picks one ISOTROPIC lod -- real anisotropic filtering needs the sampler to
// see the UV footprint's actual shape, which is exactly what `SampleGrad`'s
// `ddx`/`ddy` arguments supply.
//
// `rendering/neural_material.slang` already computes that footprint
// (`g.duv_dx`/`g.duv_dy`) to drive the software path's mip Russian-roulette
// pick (`util/texture_helpers.slang`'s `scalar_mip_level_from_footprint`);
// `LatentTextureInput.duv_dx`/`duv_dy` (added for this stage, defaulted to
// zero so every pre-existing training/validation call site is unaffected)
// carries the same values down to here instead of collapsing them to one
// stochastic integer mip first.
//
// At Stage 3's own configs (`num_mip_levels == 1`), there is no mip chain to
// select between, so this is anisotropic filtering AT a single resolution,
// not yet trilinear ACROSS resolutions -- real and meaningful on its own
// (multiple taps along the footprint's major axis within one mip, which
// plain bilinear never does), but a true multi-mip hardware texture is
// follow-up work, not claimed here. Verified bindless-texture-array +
// SamplerState.Handle + SampleGrad as a mechanism, bit-exact against the
// same reference `export_bc6h.py` verifies against, before this file was
// written.
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

    // FORWARD ONLY, no attribute -- same shape as `BcLatentTexture.decode_texel`
    // and `Texture._get_code`, with an explicit no-op backward registered
    // below rather than letting automatic differentiation see a nested call
    // with no custom derivative (which is `STAGE2_GRADIENT_BUG.md`'s own
    // failure mode: silent, wrong, exact zero). There is nothing to
    // differentiate here regardless -- see the type-level comment -- but the
    // shape is kept consistent with the rest of this codebase rather than
    // relied on as a coincidence.
    float[NumLatents] get_bilinear(no_diff LatentTextureInput input)
    {
        let udim_idx = get_udim_index(input.material_id, input.uv);
        let tile_idx = get_tile_idx(input.material_id, udim_idx);

        float[NumLatents] result;
        [ForceUnroll]
        for (int s = 0; s < NumSets; ++s)
        {
            let grid_idx = get_grid_idx(s, tile_idx, input.mip_level);
            float3 decoded = grids[grid_idx]
                .SampleGrad(samp, input.uv, input.duv_dx, input.duv_dy).rgb;
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

        float[NumLatents] result;
        [ForceUnroll]
        for (int s = 0; s < NumSets; ++s)
        {
            let grid_idx = get_grid_idx(s, tile_idx, input.mip_level);
            // Point sampling has no footprint to speak of -- SampleLevel at
            // mip 0 is the right degenerate case, not SampleGrad with
            // zeroed derivatives (which would ask the sampler to treat a
            // true point query as an infinitely fine anisotropic one).
            float3 decoded = grids[grid_idx].SampleLevel(samp, input.uv, 0.f).rgb;
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
    // Aliased straight to get_bilinear rather than given its own point-pick
    // behaviour: the software path's version of this method trades a
    // 4-tap bilinear decode for a cheaper single stochastic tap, because
    // each software tap is a full BC6H integer decode. That trade does not
    // exist here -- the hardware anisotropic fetch above is already ONE
    // instruction, not four -- so there is nothing left to approximate
    // away, and the real filtered value is strictly better than a
    // single-sample stand-in for it.
    float[NumLatents] get_bilinear_stochastic(no_diff LatentTextureInput input, no_diff float sample)
    {
        return get_bilinear(input);
    }

    [BackwardDerivativeOf(get_bilinear_stochastic)]
    void get_bilinear_stochastic_bwd(
        no_diff LatentTextureInput input, no_diff float sample, float[NumLatents] d_out) {}
}
