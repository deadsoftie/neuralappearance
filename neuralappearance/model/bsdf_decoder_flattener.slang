// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
import base_types;

interface IBsdfDecoderFlattener<
    int NumLatents,
    int NumFrames,
    int NumDirOutputs,
    int NumOutputs
> : IModel<BsdfDecoderInputWithFrames<NumLatents, NumFrames>, float[NumOutputs]>
{
    [Differentiable]
    float[NumDirOutputs] flatten_directions(BsdfDecoderInputWithFrames<NumLatents, NumFrames> input);
}

// All implementations have a shared ``forward`` function.
extension<
    int NumLatents,
    int NumFrames,
    int NumDirOutputs,
    int NumOutputs,
    Flattener : IBsdfDecoderFlattener<NumLatents, NumFrames, NumDirOutputs, NumOutputs>
> Flattener
{
    [Differentiable]
    float[NumOutputs] forward(BsdfDecoderInputWithFrames<NumLatents, NumFrames> input)
    {
        int idx = 0;
        float[NumOutputs] output;

        // Direction features always precede latents.
        float[NumDirOutputs] directions = flatten_directions(input);
        [ForceUnroll]
        for (int i = 0; i < NumDirOutputs; ++i)
            output[idx++] = directions[i];

        [ForceUnroll]
        for (int i = 0; i < NumLatents; ++i)
            output[idx++] = input.latents[i];

        return output;
    }

    // Flatten static arrays in frame-major order.
    [Differentiable]
    float[3 * NumFrames] flatten_direction<int NumFrames>(float3[NumFrames] dirs)
    {
        int idx = 0;
        float[3 * NumFrames] f;

        [ForceUnroll]
        for (int i = 0; i < NumFrames; ++i)
        {
            f[idx++] = dirs[i].x;
            f[idx++] = dirs[i].y;
            f[idx++] = dirs[i].z;
        }

        return f;
    }
}

struct WiWoBsdfDecoderFlattener<
    int NumLatents,
    int NumFrames,
    int NumDirOutputs = 2 * 3 * NumFrames,
    int NumOutputs = NumLatents + NumDirOutputs
> : IBsdfDecoderFlattener<NumLatents, NumFrames, NumDirOutputs, NumOutputs>
{
    [Differentiable]
    float[NumDirOutputs] flatten_directions(InputType input)
    {
        int idx = 0;
        float[NumDirOutputs] result;

        let wis = flatten_direction(input.wi);
        [ForceUnroll]
        for (int i = 0; i < 3 * NumFrames; ++i)
            result[idx++] = wis[i];

        let wos = flatten_direction(input.wo);
        [ForceUnroll]
        for (int i = 0; i < 3 * NumFrames; ++i)
            result[idx++] = wos[i];

        return result;
    }
}

struct WhWdBsdfDecoderFlattener<
    int NumLatents,
    int NumFrames,
    int NumDirOutputs = 2 * 3 * NumFrames,
    int NumOutputs = NumLatents + NumDirOutputs
> : IBsdfDecoderFlattener<NumLatents, NumFrames, NumDirOutputs, NumOutputs>
{
    [Differentiable]
    float[NumDirOutputs] flatten_directions(InputType input)
    {
        int idx = 0;
        float[NumDirOutputs] result;

        let whs = flatten_direction(input.wh);
        [ForceUnroll]
        for (int i = 0; i < 3 * NumFrames; ++i)
            result[idx++] = whs[i];

        let wds = flatten_direction(input.wd);
        [ForceUnroll]
        for (int i = 0; i < 3 * NumFrames; ++i)
            result[idx++] = wds[i];

        return result;
    }
}

struct WhWdZiZoBsdfDecoderFlattener<
    int NumLatents,
    int NumFrames,
    int NumDirOutputs = 2 * 3 * NumFrames + 2 * NumFrames,
    int NumOutputs = NumLatents + NumDirOutputs
> : IBsdfDecoderFlattener<NumLatents, NumFrames, NumDirOutputs, NumOutputs>
{
    [Differentiable]
    float[NumDirOutputs] flatten_directions(InputType input)
    {
        int idx = 0;
        float[NumDirOutputs] result;

        let whs = flatten_direction(input.wh);
        [ForceUnroll]
        for (int i = 0; i < 3 * NumFrames; ++i)
            result[idx++] = whs[i];

        let wds = flatten_direction(input.wd);
        [ForceUnroll]
        for (int i = 0; i < 3 * NumFrames; ++i)
            result[idx++] = wds[i];

        [ForceUnroll]
        for (int i = 0; i < NumFrames; ++i)
            result[idx++] = input.wi[i].z;

        [ForceUnroll]
        for (int i = 0; i < NumFrames; ++i)
            result[idx++] = input.wo[i].z;

        return result;
    }
}

struct WhWdWiWoBsdfDecoderFlattener<
    int NumLatents,
    int NumFrames,
    int NumDirOutputs = 2 * 6 * NumFrames,
    int NumOutputs = NumLatents + NumDirOutputs
> : IBsdfDecoderFlattener<NumLatents, NumFrames, NumDirOutputs, NumOutputs>
{
    [Differentiable]
    float[NumDirOutputs] flatten_directions(InputType input)
    {
        int idx = 0;
        float[NumDirOutputs] result;

        let whs = flatten_direction(input.wh);
        [ForceUnroll]
        for (int i = 0; i < 3 * NumFrames; ++i)
            result[idx++] = whs[i];

        let wds = flatten_direction(input.wd);
        [ForceUnroll]
        for (int i = 0; i < 3 * NumFrames; ++i)
            result[idx++] = wds[i];

        let wis = flatten_direction(input.wi);
        [ForceUnroll]
        for (int i = 0; i < 3 * NumFrames; ++i)
            result[idx++] = wis[i];

        let wos = flatten_direction(input.wo);
        [ForceUnroll]
        for (int i = 0; i < 3 * NumFrames; ++i)
            result[idx++] = wos[i];

        return result;
    }
}
