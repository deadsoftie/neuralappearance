// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
import datagen.data_generator;
import model.neural_material_model;

// Return a validity flag followed by the fixed set of validation metrics listed
// in validation.py. Keeping all metrics in one dispatch evaluates them on
// exactly the same samples and predictions.
float[7] eval_validation_losses<
    int NumLatents,
    LatentTexture : ILatentTexture<NumLatents>,
    BsdfDecoder : IBsdfDecoder<NumLatents>
>(
    LatentTexture latent_texture,
    BsdfDecoder decoder,
    no_diff TrainingSample<3, 1> sample
)
{
    float[NumLatents] latents = latent_texture.get_nearest({ sample.material_id, sample.mip_level, sample.uv });
    float prediction[3] = decoder.forward({ sample.material_id, sample.mip_level, latents, sample.wi, sample.wo });

    float epsilon = 1e-2f;
    float[7] losses;
    losses[0] = sample.valid ? 1.f : 0.f;
    losses[1] = Losses.L1<3>().eval(sample.target, prediction);
    losses[2] = Losses.RelativeL1<3>(epsilon).eval(sample.target, prediction);
    losses[3] = Losses.SMAPE<3>(epsilon).eval(sample.target, prediction);
    losses[4] = Losses.L1WithPowerRoot<3>(3.f).eval(sample.target, prediction);
    losses[5] = Losses.L1WithPowerLog<3>(3.f).eval(sample.target, prediction);
    losses[6] = Losses.L1WithSafeLog<3>().eval(sample.target, prediction);
    return losses;
}
