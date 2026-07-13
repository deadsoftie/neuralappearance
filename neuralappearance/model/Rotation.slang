// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
import neural_networks;
import base_types;


// Generic interface for the rotation module.
interface IRotation<
    int NumFrames,
    int NumLatents
> : IModel<BsdfDecoderInput<NumLatents>, BsdfDecoderInputWithFrames<NumLatents, NumFrames>>
{
    // Implementation specific parameter type.
    associatedtype Params : IDifferentiable;

    // Decode parameters from a nested model, typically an MLP.
    [Differentiable]
    Params decode_params(float latents[NumLatents]);

    // Rotate a given direction into multiple frames based on the parameters.
    [Differentiable]
    float3[NumFrames] to_local(Params params, float3 w);
}

extension<
    int NumFrames,
    int NumLatents,
    Rotation : IRotation<NumFrames, NumLatents>
> Rotation
{
    [Differentiable]
    OutputType forward(InputType input)
    {
        OutputType output;
        // Forward unmodified fields.
        output.material_id = input.material_id;
        output.mip_level = input.mip_level;
        output.latents = input.latents;

        // Decode once and reuse the frames for every direction. Decoding each
        // direction independently could produce inconsistent BSDF coordinates.
        Params params = decode_params(input.latents);

        // Rotate all available directions.
        output.wi = to_local(params, input.wi[0]);
        output.wo = to_local(params, input.wo[0]);
        output.wh = to_local(params, input.wh[0]);
        output.wd = to_local(params, input.wd[0]);
        return output;
    }
}

// A.k.a. the "ShadingFrameEncoding" from the original TOG paper.
// Predicts a number of normals & tangents and rotates directions based on a
// normalized (but not necessarily orthogonal) shading "frame".
struct ShadingFrameRotation<
    int NumFrames,
    int NumLatents,
    int NumDecoderOutputs,
    FrameDecoder : IModel<float[NumLatents], float[NumDecoderOutputs]>
> : IRotation<NumFrames, NumLatents>
{
    struct ShadingFrame : IDifferentiable
    {
        float3 t, b, n;

        [Differentiable]
        float3 to_local(const float3 v)
        {
            return float3(dot(v, t), dot(v, b), dot(v, n));
        }
    };
    typealias Params = ShadingFrame[NumFrames];

    FrameDecoder rotation_decoder;

    [Differentiable]
    Params decode_params(float latents[NumLatents])
    {
        let sh_frames = rotation_decoder.forward(latents);

        Params params;
        [ForceUnroll]
        for (uint i = 0; i < NumFrames; ++i)
        {
            // Normals occupy the first half and tangents the second half.
            uint idx_n = 3 * (i + 0);
            uint idx_t = 3 * (i + NumFrames);

            // Extract normal and tangent.
            float3 n = float3(sh_frames[idx_n], sh_frames[idx_n + 1], sh_frames[idx_n + 2]);
            float3 t = float3(sh_frames[idx_t], sh_frames[idx_t + 1], sh_frames[idx_t + 2]);

            // Treat predictions as residuals around the canonical frame. Zero
            // initialized/decoded values therefore preserve the input frame.
            n += float3(0.f, 0.f, 1.f);
            t += float3(1.f, 0.f, 0.f);

            // t is normalized but not made orthogonal to n. b supplies the
            // third projection axis from their cross product.
            n = normalize(n);
            t = normalize(t);
            float3 b = cross(n, t);

            params[i].t = t;
            params[i].b = b;
            params[i].n = n;
        }
        return params;
    }

    [Differentiable]
    float3[NumFrames] to_local(Params params, float3 w)
    {
        float3[NumFrames] w_local;
        [ForceUnroll]
        for (uint i = 0; i < NumFrames; ++i)
            w_local[i] = params[i].to_local(w);
        return w_local;
    }
}
