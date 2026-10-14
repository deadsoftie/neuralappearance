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
    LossFunction loss_function,
    // rtcnam: optional coverage-penalty regularizer (see
    // eval_diropt_model_and_loss below). weight=0 (every existing config,
    // including every dense-path one) reproduces this function's exact
    // prior behavior bit-for-bit -- the extra term is `0 * (...)^2 = 0`,
    // added to a float, so it is a true no-op, not just "small".
    float coverage_penalty_weight = 0.f,
    float coverage_penalty_cap = 0.f
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
        coverage_penalty_weight,
        coverage_penalty_cap,
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
    LossFunction loss_function,
    no_diff float coverage_penalty_weight,
    no_diff float coverage_penalty_cap
)
{
    float[NumLatents] latents = latent_texture.get_nearest({ sample.material_id, sample.mip_level, sample.uv });
    float prediction[3] = decoder.forward({ sample.material_id, sample.mip_level, latents, sample.wi, sample.wo });
    float loss = eval_and_store_loss<3, NumEncoderInputs>(prediction, sample, loss_id, loss_buffer, loss_function);

    // rtcnam coverage penalty. NOTE: the diagnosis below is SUPERSEDED --
    // the render blowup it describes was actually caused by latent
    // saturation from a squash applied in ntc_latent_texture.slang, since
    // replaced by a non-saturating leaky clamp, and this penalty was
    // treating a symptom. It is retained (disabled, weight 0) because its
    // tuning data is informative about the residual tail and it may still
    // be worth re-enabling. See rtcnam/NTC_LATENT_TEXTURE_NOTES.md for the
    // corrected analysis. Original rationale, kept for context:
    // BsdfDecoder, continuing to train jointly with a grid-derived latent
    // for `bsdf_finetuning` iterations, was observed to develop large
    // extrapolation spikes (luminance up to ~1e5, vs. the real material's
    // actual peak of ~5) at (wi, wo) pairs outside the directional
    // distribution `sample` is drawn from. Indirect path-tracing bounces
    // sample their outgoing direction from the trained *Sampler*, not from
    // this same training distribution, and routinely land in exactly that
    // unsafe region -- producing an importance weight (bsdf / pdf) that is
    // enormous but finite, so it passes the renderer's own isinf/isnan
    // guard and compounds multiplicatively over up to `max_depth` bounces.
    // Probed directly (not simulated): capping the decoder's output alone,
    // without this penalty, did not fix it -- the sampler had already
    // learned to concentrate exactly on the spike. This penalty instead
    // directly discourages the spike from forming during the training
    // that creates it: alongside the main sample, evaluate the SAME frozen
    // `latents` at one extra direction pair drawn from the uniform-
    // hemisphere distribution the diagnostic probe used (independent of
    // `sample.wi`/`sample.wo`), and penalize luminance above
    // `coverage_penalty_cap` with a one-sided quadratic hinge. Zero weight
    // (the default, and every config that doesn't opt in) makes this an
    // exact no-op below.
    //
    // DO NOT reintroduce a dynamic-trip-count loop here to average several
    // draws per iteration. That was tried (`[MaxIters(8)]` over a runtime
    // `coverage_penalty_samples`) and it hangs the GPU: 4 of 4 full-scale
    // runs with that loop present died in this phase, versus 4 of 4
    // completing without it, and the failure reproduced on a freshly
    // rebooted machine as a hard `VK_ERROR_DEVICE_LOST` (TDR) rather than
    // just a stall. Critically it hung even with `coverage_penalty_weight`
    // at 0, i.e. with the loop body never executing -- so the mere presence
    // of the dynamic loop makes Slang emit reverse-mode code that spins on
    // the GPU, not anything about the penalty math. If multi-sampling is
    // ever wanted, unroll it over a compile-time generic with
    // `[ForceUnroll]` instead of a runtime bound, and re-validate a full
    // run end to end before trusting it.
    if (coverage_penalty_weight > 0.f)
    {
        float3 wi2 = no_diff sample_hemisphere(next_float2(sg));
        float3 wo2 = no_diff sample_hemisphere(next_float2(sg));
        float coverage_prediction[3] = decoder.forward({
            sample.material_id, sample.mip_level, latents, wi2, wo2
        });
        float luminance = 0.2126f * coverage_prediction[0]
            + 0.7152f * coverage_prediction[1]
            + 0.0722f * coverage_prediction[2];
        float excess = max(0.f, luminance - coverage_penalty_cap);
        loss += coverage_penalty_weight * excess * excess;
    }

    return loss;
}


// Fit a freshly-initialized NtcLatentTexture (grid pair + decode MLP) to
// reproduce the frozen, already-trained encoder's output field, before
// `train_bsdf_diropt` takes over to jointly fine-tune grid, decode MLP,
// and BsdfDecoder against the real BSDF loss. New phase, additive --
// `train_bsdf_encoding` and `train_bsdf_diropt` above are untouched, and
// this one only runs when `training.py`'s `training_ntc_warmstart()` is
// invoked, which only happens for the `ntc_grid` latents path.
void train_ntc_warmstart<
    int NumEncoderInputs,
    int NumLatents,
    Encoder : IEncoder<NumEncoderInputs, NumLatents>,
    NtcTex : ILatentTexture<NumLatents>
>(
    no_diff Encoder encoder,
    NtcTex ntc_latents,
    TrainingSample<3, NumEncoderInputs> sample,
    float loss_scale,
    int2 call_id,
    RWTensor<float2, 2> loss_buffer
)
{
    NeuralConfig::set_batch_index(call_id.x);

    bwd_diff(eval_ntc_warmstart_and_loss<NumEncoderInputs, NumLatents, Encoder, NtcTex>)(
        encoder,
        ntc_latents,
        sample,
        call_id,
        loss_buffer,
        loss_scale
    );
}

[BackwardDifferentiable]
float eval_ntc_warmstart_and_loss<
    int NumEncoderInputs,
    int NumLatents,
    Encoder : IEncoder<NumEncoderInputs, NumLatents>,
    NtcTex : ILatentTexture<NumLatents>
>(
    no_diff Encoder encoder,
    NtcTex ntc_latents,
    no_diff TrainingSample<3, NumEncoderInputs> sample,
    int2 loss_id,
    RWTensor<float2, 2> loss_buffer
)
{
    // `latents.normalize` is false on this path (see ntc_latent_texture.slang
    // and `training_bsdf_diropt`'s assert), so the encoder does not squash and
    // this regression runs in the same unsquashed space the grid outputs and
    // `BsdfDecoder` consumes -- unit gradient throughout, no inverse needed.
    float[NumLatents] target = encoder.forward({ sample.material_id, sample.encoder_inputs });
    float[NumLatents] prediction = ntc_latents.get_nearest({ sample.material_id, sample.mip_level, sample.uv });

    float loss = 0.f;
    [ForceUnroll]
    for (int i = 0; i < NumLatents; ++i)
    {
        float diff = prediction[i] - detach(target[i]);
        loss += diff * diff;
    }
    loss /= float(NumLatents);

    float loss_sum = loss_buffer[loss_id].y;
    loss_buffer[loss_id] = float2(loss, loss_sum + loss);
    return loss;
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
