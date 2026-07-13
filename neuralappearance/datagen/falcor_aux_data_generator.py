# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import slangpy as spy
from datagen.data_generator import DataGenerator

if TYPE_CHECKING:
    from datagen.reference_materials import ReferenceMaterial, ReferenceMaterials
    from training import TrainingTargets


class FalcorAuxDataGenerator(DataGenerator):
    """Generate auxiliary targets exposed directly by reference materials."""

    def __init__(
        self,
        device: spy.Device,
        config: dict,
        reference_materials: ReferenceMaterials,
        targets: TrainingTargets,
    ):
        super().__init__(device, config, reference_materials)
        self.targets = targets

        self.module = reference_materials.load_scene_module(
            self.device,
            'datagen/falcor_aux_data_generator.slang',
        )
        self.init_sample_generator(self.module)

        self.datagen_config = self.init_config(self.module, config)

        self.num_target_channels = targets.num_channels

        self.training_sample_type = self.module[
            f'TrainingSample<{self.num_target_channels}, 1>'
        ].as_struct()

        self.generate_training_sample_kernel = (
            self.module[f'DataGenerator::generate_training_sample<{self.num_target_channels}, 1>']
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
                f'DataGenerator::Falcor::fill_importance_sampled_wo<{self.num_target_channels}, 1>'
            ]
            .as_func()
            .type_conformances(self.reference_materials.conformances)
            .write(self.reference_materials.bind_scene)
            .map(texture_resolution=(0,), sg=(0, 1), sample=(0, 1))
        )

        self.fill_target_kernel = (
            self.module[f'DataGenerator::Aux::Falcor::fill_target_{self.targets.name}']
            .as_func()
            .type_conformances(reference_materials.conformances)
            .write(reference_materials.bind_scene)
        )
        self.eval_target_kernel = (
            self.module[f'DataGenerator::Aux::Falcor::eval_target_{targets.name}']
            .as_func()
            .type_conformances(reference_materials.conformances)
            .write(reference_materials.bind_scene)
        )

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

        self.fill_importance_sampled_wo_kernel(
            config=config,
            texture_resolution=self.texture_resolutions,
            sg=self.sample_generator,
            sample=self.training_data,
            _append_to=cmd,
        )
        cmd.global_barrier()

        self.fill_target_kernel(
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
        """Evaluate auxiliary reference target with fixed inputs."""

        result = self.eval_target_kernel(
            material_backend_id=material.backend_id,
            mip_level=mip_level,
            uv=uv,
            wi=wi,
            wo=wo,
        )
        return cast(spy.Tensor, result)
