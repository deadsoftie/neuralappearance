# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import slangpy as spy
from datagen.data_generator import DataGenerator
from util.sample_generator import UniformSampleGenerator

if TYPE_CHECKING:
    from datagen.reference_materials import ReferenceMaterial, ReferenceMaterials
    from model import NeuralModel
    from training import TrainingTargets


class NeuralAuxDataGenerator(DataGenerator):
    """Estimate auxiliary targets from the frozen neural BSDF and sampler.

    This generator is used for quantities such as Monte Carlo albedo that are
    derived from the learned model rather than exported by a reference material.
    """

    def __init__(
        self,
        device: spy.Device,
        config: dict,
        reference_materials: ReferenceMaterials,
        targets: TrainingTargets,
        neural_model: NeuralModel,
    ):
        super().__init__(device, config, reference_materials)
        self.targets = targets

        self.module = spy.Module.load_from_file(
            self.device,
            'datagen/neural_aux_data_generator.slang',
        )
        self.init_sample_generator(self.module)

        # Force a simple directional sampling strategy for aux training.
        # Some options, such as ``UniformWiImportanceWo``, would
        # require access to the falcor2 reference material or a trained neural
        # importance sampler. The extra implementation effort would likely not
        # help much as the aux signals would rarely benefit from such training
        # data distributions.
        self.datagen_config = self.init_config(
            self.module,
            config,
            directional_sampling_strategy='UniformWiUniformWo',
        )

        self.latent_texture = neural_model.latent_texture.instances[0]
        self.decoder = neural_model.decoder.instances[0]
        assert neural_model.sampler is not None
        self.sampler = neural_model.sampler.instances[0]

        self.training_sample_type = self.module[
            f'TrainingSample<{self.targets.num_channels}, 1>'
        ].as_struct()

        self.generate_training_sample_kernel = (
            self.module[f'DataGenerator::generate_training_sample<{self.targets.num_channels}, 1>']
            .as_func()
            .map(
                material_id=(0,),
                material_backend_id=(0,),
                udim_offset=(0,),
                texture_resolution=(0,),
                sg=(0, 1),
                sample=(0, 1),
            )
        )

        if self.targets.name == 'mc_albedo':
            targets_name = 'mc_albedo'
            generic_args = f'{self.latent_texture.num_channels}, {self.sampler.num_lobes}, {self.latent_texture.type_name}, {self.decoder.type_name}, {self.sampler.type_name}'
        else:
            raise ValueError(f'Unsupported aux target: {self.targets.name}')

        self.fill_target_kernel = self.module[
            f'DataGenerator::Aux::Neural::fill_target_{targets_name}<{generic_args}>'
        ].as_func()
        if self.targets.name == 'mc_albedo':
            self.fill_target_kernel = self.fill_target_kernel.map(sg=(0, 1), sample=(0, 1))
        self.eval_target_kernel = self.module[
            f'DataGenerator::Aux::Neural::eval_target_{targets_name}<{generic_args}>'
        ].as_func()

    def generate_training_data(
        self,
        iteration: int,
        batch_count: int,
        batch_size: int,
        _append_to: spy.CommandEncoder | None = None,
    ) -> spy.Tensor:
        """Generate training data for aux training."""

        cmd = self.device.create_command_encoder() if _append_to is None else _append_to

        config = self.prepare_training_batch(batch_count, batch_size)
        assert self.training_data is not None

        self.generate_training_sample_kernel(
            call_id=spy.call_id(),
            call_shape=list(self.training_data.shape),
            config=config,
            material_id=self.material_ids,
            material_backend_id=self.material_backend_ids,
            texture_resolution=self.texture_resolutions,
            udim_offset=self.udim_offsets,
            sg=self.sample_generator,
            sample=self.training_data,
            _append_to=cmd,
        )
        cmd.global_barrier()

        if self.targets.name == 'mc_albedo':
            self.fill_target_kernel(
                latent_texture=self.latent_texture,
                decoder=self.decoder,
                sampler=self.sampler,
                sg=self.sample_generator,
                sample=self.training_data,
                _append_to=cmd,
            )
        else:
            raise ValueError(f'Unsupported aux target: {self.targets.name}')
        cmd.global_barrier()

        if _append_to is None:
            self.device.submit_command_buffer(cmd.finish())

        return self.training_data

    def eval_reference(
        self,
        material: ReferenceMaterial,
        mip_level: int,
        uv: spy.Tensor,
        wi: spy.Tensor,
        wo: spy.Tensor,
    ) -> spy.Tensor:
        """Evaluate auxiliary reference target with fixed inputs."""

        if self.targets.name == 'mc_albedo':
            result = self.eval_target_kernel(
                latent_texture=self.latent_texture,
                decoder=self.decoder,
                sampler=self.sampler,
                material_id=material.id,
                mip_level=mip_level,
                uv=uv,
                wi=wi,
                wo=wo,
                sg=UniformSampleGenerator(self.module, tuple(uv.shape)),
                num_mc_samples=16,
            )
            return cast(spy.Tensor, result)
        else:
            raise ValueError(f'Unsupported aux target: {self.targets.name}')
