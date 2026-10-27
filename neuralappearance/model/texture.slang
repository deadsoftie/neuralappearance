// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
import util.texture_helpers;

struct Texture<int NumChannels>
{
    int resolution;
    bool wrap;
    bool optimizable;

    RWStructuredBuffer<half>.Handle buffer;
    // Separate storage receives gradients accumulated by ``_get_code_bwd()``.
    RWByteAddressBuffer.Handle buffer_grads;
    int buffer_grads_offset;

    [BackwardDifferentiable]
    float[NumChannels] get_nearest(float2 uv)
    {
        let texel = uint2(round(uv_to_fractional_texel(uv, uint2(this.resolution), this.wrap)));
        var result = _get_code(texel);
        if (!optimizable)
            result = detach(result);
        return result;
    }

    [BackwardDifferentiable]
    float[NumChannels] get_bilinear(float2 uv)
    {
        uint2 texels[4];
        float weights[4];
        get_bilinear_interpolation_texels_and_weights(
            uv,
            uint2(this.resolution),
            this.wrap,
            texels,
            weights
        );

        float[NumChannels] result = {};
        [ForceUnroll]
        for (int i = 0; i < 4; ++i)
        {
            float[NumChannels] code = _get_code(texels[i]);
            [ForceUnroll]
            for (int l = 0; l < NumChannels; ++l)
                result[l] += weights[i] * code[l];
        }
        if (!optimizable)
            result = detach(result);
        return result;
    }

    // Learned 4-neighbor-concat interpolation (NTC's "G0" sampling mode,
    // docs/deep-dives/dd3-ntc.md Sec 4.3): unlike ``get_bilinear``, which
    // blends the four neighbors, this concatenates them so a later MLP layer
    // can combine them differently depending on exact texel position.
    // Reuses the exact same differentiable per-texel fetch as ``get_bilinear``
    // (``_get_code``/``_get_code_bwd``) -- no new backward derivative needed.
    [BackwardDifferentiable]
    float[4 * NumChannels] get_4neighbor_concat(float2 uv)
    {
        uint2 texels[4];
        float weights[4];
        get_bilinear_interpolation_texels_and_weights(
            uv,
            uint2(this.resolution),
            this.wrap,
            texels,
            weights
        );

        float[4 * NumChannels] result;
        [ForceUnroll]
        for (int i = 0; i < 4; ++i)
        {
            float[NumChannels] code = _get_code(texels[i]);
            [ForceUnroll]
            for (int l = 0; l < NumChannels; ++l)
                result[i * NumChannels + l] = code[l];
        }
        if (!optimizable)
            result = detach(result);
        return result;
    }

    // rtcnam: `get_bilinear` with simulated quantization noise, for
    // quantization-aware direct optimization of NTC grids
    // (`model.latents.ntc_grid.quant_noise`). Each corner value gets uniform
    // noise in [-noise_width / 2, noise_width / 2) before the blend, as the
    // RTXNTC SDK trains its grids under one quantization step of uniform
    // noise. The noise is an additive constant, so gradients reach the
    // corners exactly as in `get_bilinear` (straight-through). It is drawn
    // per lookup, hashed from the corner texel, the channel and the lookup's
    // own uv, rather than once per texel per optimizer step.
    [BackwardDifferentiable]
    float[NumChannels] get_bilinear_noisy(float2 uv, no_diff float noise_width, no_diff uint salt)
    {
        uint2 texels[4];
        float weights[4];
        get_bilinear_interpolation_texels_and_weights(
            uv,
            uint2(this.resolution),
            this.wrap,
            texels,
            weights
        );
        let lookup_seed = _quant_noise_seed(asuint(detach(uv.x)), asuint(detach(uv.y)), salt);

        float[NumChannels] result = {};
        [ForceUnroll]
        for (int i = 0; i < 4; ++i)
        {
            float[NumChannels] code = _get_code(texels[i]);
            [ForceUnroll]
            for (int l = 0; l < NumChannels; ++l)
                result[l] += weights[i] * (code[l] + _quant_noise(texels[i], l, lookup_seed, noise_width));
        }
        if (!optimizable)
            result = detach(result);
        return result;
    }

    // PCG hash (Jarzynski and Olano 2020) of the lookup's uv bits and a salt.
    uint _quant_noise_seed(uint u, uint v, uint salt)
    {
        return _pcg(u ^ _pcg(v ^ _pcg(salt)));
    }

    float _quant_noise(uint2 texel, int channel, uint lookup_seed, float noise_width)
    {
        let h = _pcg(lookup_seed ^ _pcg(texel.x ^ _pcg(texel.y ^ _pcg(uint(channel)))));
        // Top 24 bits -> [0, 1), exact in float32.
        let u = float(h >> 8) * (1.f / 16777216.f);
        return (u - 0.5f) * noise_width;
    }

    uint _pcg(uint v)
    {
        let state = v * 747796405u + 2891336453u;
        let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }

    [BackwardDifferentiable]
    float[NumChannels] get_bilinear_stochastic(float2 uv, float sample)
    {
        // Select one of the four neighbors with probability equal to its
        // bilinear weight. This gives unbiased values and texel gradients with
        // a single update, reducing atomic contention during training.
        uv = sample_bilinear_texel_center(
            uv,
            this.resolution,
            this.wrap,
            sample
        );
        return get_nearest(uv);
    }

    float[NumChannels] _get_code(uint2 texel)
    {
        int texel_offset = (texel.y * this.resolution + texel.x) * NumChannels;

        float[NumChannels] result;
        [ForceUnroll]
        for (int i = 0; i < NumChannels; i++)
            result[i] = this.buffer[texel_offset + i];
        return result;
    }

    [BackwardDerivativeOf(_get_code)]
    void _get_code_bwd(uint2 texel, float[NumChannels] grad_out)
    {
        int texel_offset = (texel.y * this.resolution + texel.x) * NumChannels;
        texel_offset += this.buffer_grads_offset;

        [ForceUnroll]
        for (int i = 0; i < NumChannels; i++)
        {
            int byte_offset = (texel_offset + i) * sizeof(half);
            half original_value;
            // Samples may share texels, so gradients are added atomically.
            this.buffer_grads.InterlockedAddF16Emulated(byte_offset, half(grad_out[i]), original_value);
        }
    }
}
