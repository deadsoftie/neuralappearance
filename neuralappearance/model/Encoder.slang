// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
import neural_networks;
import base_types;

// Maps range [-inf, inf] -> [-1, 1].
// Compared to a logistic sigmoid, this does not suffer as strongly from
// vanishing gradients for large magnitude inputs.
[BackwardDifferentiable]
float[NumChannels] normalize_latent_code<int NumChannels>(float[NumChannels] data)
{
    [ForceUnroll]
    for (int i = 0; i < NumChannels; ++i)
    {
        data[i] = data[i] / (abs(data[i]) + 1.0f);
    }
    return data;
}

struct Encoder<
    int NumInputs,
    int NumLatents,
    Network : IModel<half[NumInputs], float[NumLatents]>
> : IEncoder<NumInputs, NumLatents>
{
    Network network;
    bool normalize_latents;

    [BackwardDifferentiable]
    float[NumLatents] forward(EncoderInput<NumInputs, NumLatents> input)
    {
        // Run MLP to encode material attributes into latent code.
        var latents = network.forward(input.params);

        // Optionally bound latent values to (-1, 1) before baking them into the
        // latent texture. Unlike a clamp, the mapping remains smooth.
        if (normalize_latents)
            latents = normalize_latent_code<NumLatents>(latents);

        return latents;
    }
}
