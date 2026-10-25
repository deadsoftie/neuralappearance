# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import slangpy as spy

from .ntc_latent_texture import NtcLatentTexture


class NtcSdkLatentTexture(NtcLatentTexture):
    """A texture set compressed by ntc-cli, decoded at the shading point.

    Inference-only. Grid storage, checkpoint I/O and the decode MLP are
    inherited from `NtcLatentTexture`; the Slang side
    (`ntc_sdk_latent_texture.slang`) follows RTXNTC's own inference instead
    of the jointly trained decode. Checkpoints are written by
    `rtcnam/harness/ntc_to_grid_checkpoint.py`, never by training.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per material: `num_latents` scales then `num_latents` biases, applied
        # after the HLG decode. Identity until a checkpoint provides them.
        self.output_affine: list[float] = []

    def _default_output_affine(self) -> list[float]:
        num_materials = len(self.material_texture_resolution)
        return ([1.0] * self.num_latents + [0.0] * self.num_latents) * num_materials

    def model_init(self, module: spy.Module, input_type):
        super().model_init(module, input_type)
        if not self.output_affine:
            self.output_affine = self._default_output_affine()
        self._upload_output_affine()

    def _upload_output_affine(self) -> None:
        self.output_affine_gpu = spy.Tensor.from_numpy(
            self.module.device, np.array(self.output_affine, dtype=np.float32)
        )

    @property
    def type_name(self) -> str:
        return (
            f'NtcSdkLatentTexture<{self.num_tiles}, {self.num_mip_levels}, {self.num_features}, '
            f'{self.num_pos_enc_waves}, {self.num_latents}, {self.decoder_input_width}, '
            f'{self.decoder.type_name}>'
        )

    def model_data(self):
        data = super().model_data()
        data['output_affine'] = self.output_affine_gpu.storage.descriptor_handle_ro
        return data

    def save_checkpoint_params(self, desc: list[str]) -> list[tuple[list[str], dict]]:
        params = super().save_checkpoint_params(desc)
        params.append((desc, {'output_affine': self.output_affine}))
        return params

    def load_checkpoint_params(self, params: dict) -> None:
        super().load_checkpoint_params(params)
        affine = [float(v) for v in params['output_affine']]
        expected = len(self._default_output_affine())
        if len(affine) != expected:
            raise ValueError(f'output_affine has {len(affine)} values, expected {expected}.')
        self.output_affine = affine
        self._upload_output_affine()
