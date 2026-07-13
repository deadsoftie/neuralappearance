# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data generation module for neural appearance."""

from .aux_data_generator import create_aux_data_generator
from .data_generator import DataGenerator, DataGenerators
from .falcor_aux_data_generator import FalcorAuxDataGenerator
from .falcor_bsdf_data_generator import FalcorBsdfDataGenerator
from .neural_aux_data_generator import NeuralAuxDataGenerator
from .reference_materials import (
    FALCOR_2_TYPES,
    ReferenceMaterial,
    ReferenceMaterials,
    get_num_material_params,
)
from .sampler_data_generator import SamplerDataGenerator
from .scatter_data_generator import ScatterDataGenerator

__all__ = [
    'FALCOR_2_TYPES',
    'DataGenerator',
    'DataGenerators',
    'FalcorAuxDataGenerator',
    'FalcorBsdfDataGenerator',
    'NeuralAuxDataGenerator',
    'ReferenceMaterial',
    'ReferenceMaterials',
    'SamplerDataGenerator',
    'ScatterDataGenerator',
    'create_aux_data_generator',
    'get_num_material_params',
]
