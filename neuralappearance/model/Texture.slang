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
