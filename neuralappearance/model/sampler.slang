// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
import neural_networks;
import base_types;
import util.math_helpers;

struct DiffuseSpecularSampler<
    int NumLatents,
    int NumLobes,
    int NumSpecLobes,
    int NumNetworkInputs,
    int NumNetworkOutputs,
    Network : IModel<float[NumNetworkInputs], float[NumNetworkOutputs]>
> : ISampler<NumLatents, NumLobes>
{
    struct DiffuseSpecularParams : IDifferentiable
    {
        // Diffuse parameters.
        float diff_weight; // lobe weight
        float2 diff_mu;    // 2d slope offset

        // Specular parameters.
        float spec_weight[NumSpecLobes]; // lobe weight
        float2 spec_alpha[NumSpecLobes]; // anisotropic roughness
        float spec_rho[NumSpecLobes];    // roughness correlation
        float2 spec_mu[NumSpecLobes];    // 2d slope offset
    }
    typealias Params = DiffuseSpecularParams;

    Network network;

    [BackwardDifferentiable]
    Params decode_params(SamplerInput<NumLatents> input)
    {
        // Run MLP to decode the latent code and incident direction in
        // diffuse+GGX parameters.
        float[NumNetworkInputs] network_input;
        network_input[0] = input.wi.x;
        network_input[1] = input.wi.y;
        network_input[2] = input.wi.z;
        [ForceUnroll]
        for (uint i = 0; i < NumLatents; ++i)
        {
            network_input[3 + i] = input.latents[i];
        }
        var network_output = network.forward(network_input);

        Params params;

        // Transform unconstrained network outputs into valid distribution
        // parameters.
        [ForceUnroll]
        for (uint i = 0; i < NumSpecLobes; ++i)
        {
            let alpha = float2(
                tanh_approx(network_output[i * 5 + 0]),
                tanh_approx(network_output[i * 5 + 1])
            );
            params.spec_alpha[i] = 1e-4f + 0.5f * (1.f + alpha);
            params.spec_rho[i] = tanh_approx(network_output[i * 5 + 2]);
            params.spec_mu[i] = float2(
                sinh_approx(network_output[i * 5 + 3]),
                sinh_approx(network_output[i * 5 + 4])
            );
        }
        params.diff_mu = float2(
            sinh_approx(network_output[NumSpecLobes * 5 + 0]),
            sinh_approx(network_output[NumSpecLobes * 5 + 1])
        );

        // Stable softmax over all specular lobes plus the diffuse lobe. The
        // non-negative shift keeps every exponent at or below zero.
        float m = 0.f;
        [ForceUnroll]
        for (uint i = 0; i < NumSpecLobes; ++i)
        {
            m = max(m, network_output[NumSpecLobes * 5 + 2 + i]);
        }
        m = max(m, network_output[NumSpecLobes * 5 + 2 + NumSpecLobes]);

        float norm = 0.f;
        [ForceUnroll]
        for (uint i = 0; i < NumSpecLobes; ++i)
        {
            params.spec_weight[i] = exp(network_output[NumSpecLobes * 5 + 2 + i] - m);
            norm += params.spec_weight[i];
        }
        params.diff_weight = exp(network_output[NumSpecLobes * 5 + 2 + NumSpecLobes] - m);
        norm += params.diff_weight;

        norm = 1.f / norm;
        [ForceUnroll]
        for (uint i = 0; i < NumSpecLobes; ++i)
        {
            params.spec_weight[i] *= norm;
        }
        params.diff_weight *= norm;

        return params;
    }

    float3 sample(float3 wi, float3 sample, Params params)
    {
        // sample.x chooses a mixture component; yz remain independent random
        // dimensions for sampling within the selected component.
        float cmf = 0.f;
        [ForceUnroll]
        for (uint i = 0; i < NumSpecLobes; ++i)
        {
            cmf += params.spec_weight[i];
            if (sample.x < cmf)
                return sample_ggx(wi, sample.yz, params.spec_alpha[i], params.spec_rho[i], params.spec_mu[i]);
        }
        return sample_diffuse(sample.yz, params.diff_mu);
    }

    [BackwardDifferentiable]
    float pdf(no_diff float3 wi, no_diff float3 wo, Params params)
    {
        float pdf = 0.f;
        [ForceUnroll]
        for (uint i = 0; i < NumSpecLobes; ++i)
        {
            pdf += params.spec_weight[i] * pdf_ggx(wi, wo, params.spec_alpha[i], params.spec_rho[i], params.spec_mu[i]);
        }
        pdf += params.diff_weight * pdf_diffuse(wo, params.diff_mu);
        // Keep the returned density nonzero even far from every learned lobe.
        return max_float(1e-4f, pdf);
    }

    [BackwardDifferentiable]
    float[NumLobes] pdf_lobes(no_diff float3 wi, no_diff float3 wo, Params params)
    {
        float[NumLobes] result;
        result[0] = params.diff_weight * pdf_diffuse(wo, params.diff_mu);
        [ForceUnroll]
        for (uint i = 0; i < NumSpecLobes; ++i)
        {
            let p = pdf_ggx(wi, wo, params.spec_alpha[i], params.spec_rho[i], params.spec_mu[i]);
            result[1 + i] = params.spec_weight[i] * p;
        }
        return result;
    }
}

struct CosineHemisphereSampler<
    int NumLatents
> : ISampler<NumLatents, 1>
{
    typealias Params = float; // Dummy.

    [BackwardDifferentiable]
    Params decode_params(SamplerInput<NumLatents> input)
    {
        return 0.f;
    }

    float3 sample(float3 wi, float3 sample, Params params)
    {
        float pdf;
        return sample_cosine_hemisphere_concentric(sample.xy, pdf);
    }

    [BackwardDifferentiable]
    float pdf(no_diff float3 wi, no_diff float3 wo, Params params)
    {
        return no_diff pdf_cosine_hemisphere_concentric(wo);
    }

    [BackwardDifferentiable]
    float[1] pdf_lobes(no_diff float3 wi, no_diff float3 wo, Params params)
    {
        return { pdf(wi, wo, params) };
    }
}
