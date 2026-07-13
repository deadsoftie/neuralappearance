// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
import neural_networks;
import base_types;

struct DummyAuxDecoder<
    int NumLatents,
    int NumOutputs
> : IAuxDecoder<NumLatents, NumOutputs>
{
    // Used when auxiliary targets are disabled. It has the same interface as a
    // trained decoder but always returns zero.
    [BackwardDifferentiable]
    float[NumOutputs] forward(AuxDecoderInput<NumLatents> input)
    {
        return {};
    }
}

struct AuxDecoder<
    int NumLatents,
    int NumOutputs,
    int NumNetworkInputs,
    Network : IModel<float[NumNetworkInputs], float[NumOutputs]>
> : IAuxDecoder<NumLatents, NumOutputs>
{
    Network network;

    [BackwardDifferentiable]
    float[NumOutputs] forward(AuxDecoderInput<NumLatents> input)
    {
        float[NumNetworkInputs] network_input;
        network_input[0] = input.wi.x;
        network_input[1] = input.wi.y;
        network_input[2] = input.wi.z;
        [ForceUnroll]
        for (uint i = 0; i < NumLatents; ++i)
        {
            network_input[3 + i] = input.latents[i];
        }
        return network.forward(network_input);
    }
}
