# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import numpy as np
import slangpy as spy
from datagen.reference_materials import ReferenceMaterials
from util.math_helpers import udim_offset
from util.sample_generator import UniformSampleGenerator

if TYPE_CHECKING:
    from datagen import (
        FalcorAuxDataGenerator,
        FalcorBsdfDataGenerator,
        NeuralAuxDataGenerator,
        SamplerDataGenerator,
    )


class DataGenerator:
    """Shared batching and sampling setup for GPU training-data generators.

    A generated batch contains ``batch_count`` independently selected
    material/UDIM tiles, each with ``batch_size`` spatial and directional
    samples. Subclasses choose the sample type and dispatch the kernels that
    fill phase-specific targets.
    """

    def __init__(
        self,
        device: spy.Device,
        config: dict,
        reference_materials: ReferenceMaterials,
    ):
        self.device = device
        self.config = config
        self.reference_materials = reference_materials
        self.batch_index: int = 0
        self.training_data: spy.Tensor | None = None
        self.material_ids: spy.Tensor | None = None
        self.material_backend_ids: spy.Tensor | None = None
        self.udim_offsets: spy.Tensor | None = None
        self.texture_resolutions: spy.Tensor | None = None
        self.datagen_config: dict = {}
        self.training_sample_type: spy.Struct | None = None
        self.sample_generator_seed = config['data_generation'].get('seed', 0)

        # Keep an infinitely repeating iterator over all materials and UDIM
        # tiles from which we will generate training data.
        udim_tiles = []
        for material in reference_materials:
            for udim_id in material.udims:
                udim_tiles.append(
                    (material.id, material.backend_id, material.texture_resolution, udim_id)
                )
        self.udim_iterator = itertools.cycle(udim_tiles)

    def init_sample_generator(self, module: spy.Module) -> None:
        self.sample_generator = UniformSampleGenerator(
            module,
            (1,),
            self.sample_generator_seed,
        )

    def init_config(
        self,
        module: spy.Module,
        config: dict,
        directional_sampling_strategy: str | None = None,
    ) -> dict:
        """Assemble a config that is passed on to the Slang side."""

        strategy_name: str = (
            directional_sampling_strategy
            or config['data_generation']['directional_sampling_strategy']
        )
        strategy_id = module.require_function(
            f'DataGenerator::get_directional_sampling_strategy<DataGenerator::DirectionalSamplingStrategy.{strategy_name}>'
        )()

        return {
            '_type': 'DataGenerator::Config',
            # Directional sampling.
            'directional_sampling_strategy': strategy_id,
            'balanced_directional_sampling_ratio': config['data_generation'][
                'balanced_directional_sampling_ratio'
            ],
            # LOD.
            'mip_level_min_max': spy.int2(0, config['model']['latents']['num_mip_levels'] - 1),
            'num_prefilter_samples': config['data_generation']['lod']['num_prefilter_samples'],
            # Misc.
            'snap_to_texel_center': True,
            'color_augmentation_ratio': 0.0,  # Set on the fly, if appropriate.
        }

    def prepare_training_batch(self, batch_count: int, batch_size: int) -> dict:
        """Allocate reusable buffers and assign one material tile per batch."""

        assert math.isqrt(batch_size) ** 2 == batch_size, 'Batch size must be a square number.'

        buffer_shape = (batch_count, batch_size)

        if self.training_data is None or self.training_data.shape != buffer_shape:
            self.training_data = spy.Tensor.empty(
                self.device, shape=buffer_shape, dtype=self.training_sample_type
            )
        if self.material_ids is None or self.material_ids.shape[0] != batch_count:
            self.material_ids = spy.Tensor.empty(self.device, shape=(batch_count,), dtype='int')
        if self.material_backend_ids is None or self.material_backend_ids.shape[0] != batch_count:
            self.material_backend_ids = spy.Tensor.empty(
                self.device, shape=(batch_count,), dtype='int'
            )
        if self.udim_offsets is None or self.udim_offsets.shape[0] != batch_count:
            self.udim_offsets = spy.Tensor.empty(self.device, shape=(batch_count,), dtype='float2')
        if self.texture_resolutions is None or self.texture_resolutions.shape[0] != batch_count:
            self.texture_resolutions = spy.Tensor.empty(
                self.device, shape=(batch_count,), dtype='int'
            )

        self.sample_generator.reshape(buffer_shape)

        # Optimization takes place one material ID and UDIM tile at a time. We
        # fetch the next ``batch_count`` and shuffle to avoid correlations.
        udim_tiles = [next(self.udim_iterator) for _ in range(batch_count)]
        np.random.shuffle(udim_tiles)
        material_ids = [t[0] for t in udim_tiles]
        material_backend_ids = [t[1] for t in udim_tiles]
        texture_resolutions = [t[2] for t in udim_tiles]
        udim_offsets = [udim_offset(t[3]) for t in udim_tiles]
        # Upload to GPU.
        self.material_ids.copy_from_numpy(np.array(material_ids).astype(np.int32))
        self.material_backend_ids.copy_from_numpy(np.array(material_backend_ids).astype(np.int32))
        self.texture_resolutions.copy_from_numpy(np.array(texture_resolutions).astype(np.int32))
        self.udim_offsets.copy_from_numpy(np.array(udim_offsets).astype(np.float32))

        return self.datagen_config.copy()

    def generate_training_data(
        self,
        iteration: int,
        batch_count: int,
        batch_size: int,
        _append_to: spy.CommandEncoder | None = None,
    ) -> spy.Tensor:
        raise NotImplementedError('Subclasses should implement this method.')


class DataGenerators:
    """Generators currently available to the sequential training phases.

    BSDF generation is always present. Sampler and auxiliary generators are
    created when their respective phases begin.
    """

    bsdf: FalcorBsdfDataGenerator
    sampler: SamplerDataGenerator | None
    aux: FalcorAuxDataGenerator | NeuralAuxDataGenerator | None

    def __init__(self, bsdf: FalcorBsdfDataGenerator):
        self.bsdf = bsdf
        self.sampler = None
        self.aux = None
