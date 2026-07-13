// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
import slangpy;

__exported import neural_networks;
__exported import falcor2.utils;
__exported import datagen.data_generator;
__exported import model.neural_material_model;
__exported import util.math_helpers;
__exported import checkpoint.bsdf_plots;
__exported import checkpoint.validation;

// Each training entry point evaluates one phase-specific model path under
// reverse-mode autodiff. All phases write the same float2 loss layout so Python
// can share reduction, readback, and instance-pruning code.

// Main entry point from Python for training the BSDF model (encoding phase).
void train_bsdf_encoding<
    int NumEncoderInputs,
    int NumLatents,
    Encoder : IEncoder<NumEncoderInputs, NumLatents>,
    BsdfDecoder : IBsdfDecoder<NumLatents>,
    LossFunction : ILossFunction<3>
>(
    Encoder encoder,
    BsdfDecoder decoder,
    TrainingSample<3, NumEncoderInputs> sample,
    float loss_scale,
    int2 call_id,
    RWTensor<float2, 2> loss_buffer,
    LossFunction loss_function
)
{
    NeuralConfig::set_batch_index(call_id.x);

    // Call the main model evaluation in backwards autodiff mode.
    bwd_diff(eval_encoding_model_and_loss<NumEncoderInputs, NumLatents, Encoder, BsdfDecoder, LossFunction>)(
        encoder,
        decoder,
        sample,
        call_id,
        loss_buffer,
        loss_function,
        loss_scale
    );
}

[BackwardDifferentiable]
float eval_encoding_model_and_loss<
    int NumEncoderInputs,
    int NumLatents,
    Encoder : IEncoder<NumEncoderInputs, NumLatents>,
    BsdfDecoder : IBsdfDecoder<NumLatents>,
    LossFunction : ILossFunction<3>
>(
    Encoder encoder,
    BsdfDecoder decoder,
    no_diff TrainingSample<3, NumEncoderInputs> sample,
    int2 loss_id,
    RWTensor<float2, 2> loss_buffer,
    LossFunction loss_function
)
{
    float[NumLatents] latents = encoder.forward({ sample.material_id, sample.encoder_inputs });
    float prediction[3] = decoder.forward({ sample.material_id, sample.mip_level, latents, sample.wi, sample.wo });
    return eval_and_store_loss<3, NumEncoderInputs>(prediction, sample, loss_id, loss_buffer, loss_function);
}


// Python calls this entry point during direct BSDF optimization.
void train_bsdf_diropt<
    int NumEncoderInputs,
    int NumLatents,
    LatentTexture : ILatentTexture<NumLatents>,
    BsdfDecoder : IBsdfDecoder<NumLatents>,
    LossFunction : ILossFunction<3>
>(
    LatentTexture latent_texture,
    BsdfDecoder decoder,
    TrainingSample<3, NumEncoderInputs> sample,
    inout UniformSampleGenerator sg,
    float loss_scale,
    int2 call_id,
    RWTensor<float2, 2> loss_buffer,
    LossFunction loss_function
)
{
    NeuralConfig::set_batch_index(call_id.x);

    // Call the main model evaluation in backwards autodiff mode.
    bwd_diff(eval_diropt_model_and_loss<
        NumEncoderInputs,
        NumLatents,
        LatentTexture,
        BsdfDecoder,
        LossFunction
    >)(
        latent_texture,
        decoder,
        sample,
        call_id,
        loss_buffer,
        sg,
        loss_function,
        loss_scale
    );
}

[BackwardDifferentiable]
float eval_diropt_model_and_loss<
    int NumEncoderInputs,
    int NumLatents,
    LatentTexture : ILatentTexture<NumLatents>,
    BsdfDecoder : IBsdfDecoder<NumLatents>,
    LossFunction : ILossFunction<3>
>(
    LatentTexture latent_texture,
    BsdfDecoder decoder,
    no_diff TrainingSample<3, NumEncoderInputs> sample,
    int2 loss_id,
    RWTensor<float2, 2> loss_buffer,
    inout UniformSampleGenerator sg,
    LossFunction loss_function
)
{
    float[NumLatents] latents = latent_texture.get_nearest({ sample.material_id, sample.mip_level, sample.uv });
    float prediction[3] = decoder.forward({ sample.material_id, sample.mip_level, latents, sample.wi, sample.wo });
    return eval_and_store_loss<3, NumEncoderInputs>(prediction, sample, loss_id, loss_buffer, loss_function);
}


// Main entry point from Python for training the sampler.
void train_sampler<
    int NumLatents,
    int NumLobes,
    BsdfDecoder : IBsdfDecoder<NumLatents>,
    Sampler : ISampler<NumLatents, NumLobes>
>(
    BsdfDecoder decoder,
    Sampler sampler,
    SamplerTrainingSample<NumLatents> sample,
    inout UniformSampleGenerator sg,
    float loss_scale,
    int2 call_id,
    RWTensor<float2, 2> loss_buffer
)
{
    NeuralConfig::set_batch_index(call_id.x);

    bwd_diff(eval_sampler_and_loss<
        NumLatents,
        NumLobes,
        BsdfDecoder,
        Sampler
    >)(
        decoder,
        sampler,
        sample,
        call_id,
        loss_buffer,
        sg,
        loss_scale
    );
}

[BackwardDifferentiable]
float eval_sampler_and_loss<
    int NumLatents,
    int NumLobes,
    BsdfDecoder : IBsdfDecoder<NumLatents>,
    Sampler : ISampler<NumLatents, NumLobes>
>(
    BsdfDecoder decoder,
    Sampler sampler,
    no_diff SamplerTrainingSample<NumLatents> sample,
    int2 loss_id,
    RWTensor<float2, 2> loss_buffer,
    inout UniformSampleGenerator sg
)
{
    // Decode sampling parameters from latent texture.
    let params = sampler.decode_params({ sample.latents, sample.wi });

    // Sample outgoing direction and evaluate its probability density.
    float3 wo = no_diff sampler.sample(sample.wi, next_float3(sg), params);
    float pdf = sampler.pdf(sample.wi, wo, params);

    // Evaluate neural BSDF model to use as a target.
    float pred[3] = decoder.forward({ sample.material_id, sample.mip_level, sample.latents, sample.wi, wo });
    float target = luminance(float3(pred[0], pred[1], pred[2]));

    // Compute a Monte Carlo approximation of the KL-divergence loss.
    float loss = -detach(target) * log(pdf) / detach(pdf);
    if (sample.wi.z <= 0.f || wo.z <= 0.f)
        loss = 0.f;

    // Write out training loss (+ a running sum) to an external buffer.
    float loss_sum = loss_buffer[loss_id].y;
    loss_buffer[loss_id] = float2(loss, loss_sum + loss);
    return loss;
}


// Main entry point from Python for training the aux network.
void train_aux<
    int NumTargetChannels,
    int NumLatents,
    LatentTexture : ILatentTexture<NumLatents>,
    AuxDecoder : IAuxDecoder<NumLatents, NumTargetChannels>,
    LossFunction : ILossFunction<NumTargetChannels>
>(
    no_diff LatentTexture latent_texture,
    AuxDecoder aux,
    TrainingSample<NumTargetChannels, 1> sample,
    inout UniformSampleGenerator sg,
    float loss_scale,
    int2 call_id,
    RWTensor<float2, 2> loss_buffer,
    LossFunction loss_function,
)
{
    NeuralConfig::set_batch_index(call_id.x);

    // Call the main model evaluation in backwards autodiff mode.
    bwd_diff(eval_aux_model_and_loss<
        NumTargetChannels,
        NumLatents,
        LatentTexture,
        AuxDecoder,
        LossFunction
    >)(
        latent_texture,
        aux,
        sample,
        call_id,
        loss_buffer,
        sg,
        loss_function,
        loss_scale
    );
}

[BackwardDifferentiable]
float eval_aux_model_and_loss<
    int NumTargetChannels,
    int NumLatents,
    LatentTexture : ILatentTexture<NumLatents>,
    AuxDecoder : IAuxDecoder<NumLatents, NumTargetChannels>,
    LossFunction : ILossFunction<NumTargetChannels>
>(
    no_diff LatentTexture latent_texture,
    AuxDecoder aux,
    no_diff TrainingSample<NumTargetChannels, 1> sample,
    int2 loss_id,
    RWTensor<float2, 2> loss_buffer,
    inout UniformSampleGenerator sg,
    LossFunction loss_function
)
{
    float[NumLatents] latents = latent_texture.get_nearest({ sample.material_id, sample.mip_level, sample.uv });
    float prediction[NumTargetChannels] = aux.forward({ latents, sample.wi });
    return eval_and_store_loss<NumTargetChannels, 1>(prediction, sample, loss_id, loss_buffer, loss_function);
}


// Evaluate the configured target loss. x stores this evaluation and y keeps a
// running sum used for iteration-averaged reporting.
[BackwardDifferentiable]
float eval_and_store_loss<int NumTargetChannels, int NumEncoderInputs, LossFunction : ILossFunction<NumTargetChannels>>(
    float prediction[NumTargetChannels],
    no_diff TrainingSample<NumTargetChannels, NumEncoderInputs> sample,
    int2 loss_id,
    RWTensor<float2, 2> loss_buffer,
    LossFunction loss_function
)
{
    // Apply mappings and evaluate training loss.
    float loss = loss_function.eval(prediction, sample.target);
    loss *= sample.valid ? 1.f : 0.f; // Ignore invalid samples.

    // Write out training loss (+ a running sum) to an external buffer.
    float loss_sum = loss_buffer[loss_id].y;
    loss_buffer[loss_id] = float2(loss, loss_sum + loss);
    return loss;
}
