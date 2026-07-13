# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import slangpy as spy
from datagen.data_generator import DataGenerator
from util.math_helpers import cosine_falloff
from util.sample_generator import UniformSampleGenerator

if TYPE_CHECKING:
    from datagen.reference_materials import ReferenceMaterial, ReferenceMaterials
    from training import TrainingPhase


class FalcorBsdfDataGenerator(DataGenerator):
    """Generate BSDF targets and optional encoder inputs from falcor2 materials.

    BSDF encoding includes the reference material properties consumed by the
    encoder. Direct optimization omits them and trains against the same RGB
    reference evaluations using latent textures directly.
    """

    prefilter_target = False

    def __init__(
        self,
        device: spy.Device,
        config: dict,
        reference_materials: ReferenceMaterials,
        training_phase: TrainingPhase,
    ):
        super().__init__(device, config, reference_materials)

        self.module = reference_materials.load_scene_module(
            self.device,
            'datagen/falcor_bsdf_data_generator.slang',
        )
        self.init_sample_generator(self.module)

        self.datagen_config = self.init_config(self.module, config)
        self._configure_for_phase(training_phase)

    def _configure_for_phase(self, training_phase: TrainingPhase):
        """Select the sample layout and kernels required by one BSDF phase."""

        self.training_data = None

        if training_phase not in ['BsdfEncoding', 'BsdfDirectOptimization']:
            raise ValueError(f'Unsupported BSDF training phase: {training_phase}')

        self.training_phase = training_phase
        num_mip_levels = self.config['model']['latents']['num_mip_levels']
        self.datagen_config['mip_level_min_max'] = spy.int2(0, num_mip_levels - 1)
        self.prefilter_target = num_mip_levels > 1

        self.num_encoder_inputs = 1
        if training_phase == 'BsdfEncoding':
            # Use reference material parameters as encoder inputs.
            self.num_encoder_inputs = self.reference_materials.num_encoder_inputs

        self.training_sample_type = self.module[
            f'TrainingSample<3, {self.num_encoder_inputs}>'
        ].as_struct()

        self.generate_training_sample_kernel = (
            self.module[f'DataGenerator::generate_training_sample<3, {self.num_encoder_inputs}>']
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

        self.fill_importance_sampled_wo_kernel = (
            self.module[
                f'DataGenerator::Falcor::fill_importance_sampled_wo<3, {self.num_encoder_inputs}>'
            ]
            .as_func()
            .type_conformances(self.reference_materials.conformances)
            .write(self.reference_materials.bind_scene)
            .map(texture_resolution=(0,), sg=(0, 1), sample=(0, 1))
        )

        if self.prefilter_target:
            self.fill_target_kernel = self.module[
                f'DataGenerator::Bsdf::Falcor::fill_target_bsdf_lod<{self.num_encoder_inputs}>'
            ]
        else:
            self.fill_target_kernel = self.module[
                f'DataGenerator::Bsdf::Falcor::fill_target_bsdf<{self.num_encoder_inputs}>'
            ]
        self.fill_target_kernel = (
            self.fill_target_kernel.as_func()
            .type_conformances(self.reference_materials.conformances)
            .write(self.reference_materials.bind_scene)
            .map(texture_resolution=(0,), sg=(0, 1), sample=(0, 1))
        )

        self.fill_encoder_inputs_kernel = None
        self.eval_encoder_inputs_kernel = None

        if training_phase == 'BsdfEncoding':
            num_bsdf_layers = self.reference_materials.num_bsdf_layers

            self.fill_encoder_inputs_kernel = self.module[
                f'DataGenerator::Bsdf::Falcor::fill_encoder_inputs<{num_bsdf_layers}, {self.num_encoder_inputs}>'
            ]
            self.eval_encoder_inputs_kernel = self.module[
                f'DataGenerator::Bsdf::Falcor::eval_encoder_inputs<{num_bsdf_layers}, {self.num_encoder_inputs}>'
            ]
            self.fill_encoder_inputs_kernel = (
                self.fill_encoder_inputs_kernel.as_func()
                .type_conformances(self.reference_materials.conformances)
                .write(self.reference_materials.bind_scene)
                .map(sg=(0, 1), sample=(0, 1))
            )
            self.eval_encoder_inputs_kernel = (
                self.eval_encoder_inputs_kernel.as_func()
                .type_conformances(self.reference_materials.conformances)
                .write(self.reference_materials.bind_scene)
            )

    def generate_training_data(
        self,
        iteration: int,
        batch_count: int,
        batch_size: int,
        _append_to: spy.CommandEncoder | None = None,
    ) -> spy.Tensor:
        """Generate one or more batches from the falcor2 reference materials."""

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

        self.fill_importance_sampled_wo_kernel(
            config=config,
            texture_resolution=self.texture_resolutions,
            sg=self.sample_generator,
            sample=self.training_data,
            _append_to=cmd,
        )
        cmd.global_barrier()

        self.fill_target_kernel(
            config=config,
            texture_resolution=self.texture_resolutions,
            sg=self.sample_generator,
            sample=self.training_data,
            _append_to=cmd,
        )
        cmd.global_barrier()

        if self.fill_encoder_inputs_kernel is not None:
            assert self.config['model'].get('encoder') is not None

            # Color augmentation randomly permutes or duplicates RGB channels
            # for both the encoder inputs and BRDF target values. It forces the
            # model to pay more attention to color reproduction but can lead to
            # slower convergence.
            color_augmentation = self.config['data_generation']['color_augmentation']
            if color_augmentation['enabled']:
                config['color_augmentation_ratio'] = cosine_falloff(
                    iteration,
                    color_augmentation['num_iterations'],
                    color_augmentation['start_ratio'],
                    color_augmentation['end_ratio'],
                )

            self.fill_encoder_inputs_kernel(
                config=config,
                sg=self.sample_generator,
                sample=self.training_data,
                _append_to=cmd,
            )
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
        """Evaluate the reference BSDF with fixed inputs."""

        sg = UniformSampleGenerator(self.module, tuple(uv.shape))

        if mip_level == 0:
            eval_target_kernel = (
                self.module['DataGenerator::Bsdf::Falcor::eval_target_bsdf']
                .as_func()
                .type_conformances(self.reference_materials.conformances)
                .write(self.reference_materials.bind_scene)
            )
            result = eval_target_kernel(
                material_backend_id=material.backend_id,
                mip_level=mip_level,
                uv=uv,
                wi=wi,
                wo=wo,
                sg=sg,
            )
        else:
            eval_target_kernel = (
                self.module['DataGenerator::Bsdf::Falcor::eval_target_bsdf_lod']
                .as_func()
                .type_conformances(self.reference_materials.conformances)
                .write(self.reference_materials.bind_scene)
            )
            num_samples = self.config['data_generation']['lod']['num_prefilter_samples_reference']
            result = eval_target_kernel(
                material_backend_id=material.backend_id,
                mip_level=mip_level,
                uv=uv,
                wi=wi,
                wo=wo,
                sg=sg,
                num_prefilter_samples=num_samples,
                snap_to_texel_center=self.datagen_config['snap_to_texel_center'],
                texture_resolution=material.texture_resolution,
            )
        return cast(spy.Tensor, result)

    def eval_encoder_inputs(
        self,
        material: ReferenceMaterial,
        mip_level: int,
        uv: spy.Tensor,
    ) -> spy.Tensor:
        """Evaluate the reference BSDF parameters to use as encoder inputs."""

        if self.eval_encoder_inputs_kernel is None:
            raise RuntimeError('Encoder inputs evaluation kernel not initialized.')

        sg = UniformSampleGenerator(self.module, tuple(uv.shape))

        result = self.eval_encoder_inputs_kernel(
            material_backend_id=material.backend_id,
            mip_level=mip_level,
            uv=uv,
            color_augmentation_ratio=0.0,
            sg=sg,
        )
        return cast(spy.Tensor, result)
