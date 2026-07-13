# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import slangpy as spy
from datagen.data_generator import DataGenerator

if TYPE_CHECKING:
    from datagen.reference_materials import ReferenceMaterials
    from model import NeuralModel


class SamplerDataGenerator(DataGenerator):
    """Generate latent and incident-direction inputs for sampler training.

    The BSDF representation is frozen by this phase. Outgoing directions are
    drawn during the sampler loss evaluation rather than stored in these input
    samples.
    """

    def __init__(
        self,
        device: spy.Device,
        config: dict,
        reference_materials: ReferenceMaterials,
        neural_model: NeuralModel,
    ):
        super().__init__(device, config, reference_materials)

        self.module = spy.Module.load_from_file(
            self.device,
            'datagen/sampler_data_generator.slang',
        )
        self.init_sample_generator(self.module)

        self.datagen_config = self.init_config(self.module, config)

        self.latent_texture = neural_model.latent_texture.instances[0]

        self.training_sample_type = self.module[
            f'SamplerTrainingSample<{self.latent_texture.num_channels}>'
        ].as_struct()

        self.generate_training_sample_kernel = (
            self.module[
                f'DataGenerator::Sampler::generate_training_sample<{self.latent_texture.num_channels}, {self.latent_texture.type_name}>'
            ]
            .as_func()
            .map(
                material_id=(0,),
                udim_offset=(0,),
                texture_resolution=(0,),
                sg=(0, 1),
                sample=(0, 1),
            )
        )

    def generate_training_data(
        self,
        iteration: int,
        batch_count: int,
        batch_size: int,
        _append_to: spy.CommandEncoder | None = None,
    ) -> spy.Tensor:
        """Generate training data for sampler training."""

        cmd = self.device.create_command_encoder() if _append_to is None else _append_to

        config = self.prepare_training_batch(batch_count, batch_size)
        assert self.training_data is not None

        self.generate_training_sample_kernel(
            call_id=spy.call_id(),
            call_shape=list(self.training_data.shape),
            config=config,
            material_id=self.material_ids,
            texture_resolution=self.texture_resolutions,
            udim_offset=self.udim_offsets,
            sg=self.sample_generator,
            latent_texture=self.latent_texture,
            sample=self.training_data,
            _append_to=cmd,
        )
        cmd.global_barrier()

        if _append_to is None:
            self.device.submit_command_buffer(cmd.finish())

        return self.training_data
