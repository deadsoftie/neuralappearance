# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import slangpy as spy
from datagen.falcor_aux_data_generator import FalcorAuxDataGenerator
from datagen.neural_aux_data_generator import NeuralAuxDataGenerator

if TYPE_CHECKING:
    from datagen.reference_materials import ReferenceMaterials
    from model import NeuralModel
    from training import TrainingTargets


def create_aux_data_generator(
    device: spy.Device,
    config: dict,
    reference_materials: ReferenceMaterials,
    targets: TrainingTargets,
    neural_model: NeuralModel | None = None,
) -> FalcorAuxDataGenerator | NeuralAuxDataGenerator:
    """Choose the reference or neural source for an auxiliary target."""

    if targets.name in [
        'ref_diffuse_specular_roughness',
        'ref_diffuse_specular_roughness_normal',
    ]:
        # These targets derive auxiliary properties from the reference material.
        return FalcorAuxDataGenerator(device, config, reference_materials, targets)
    elif targets.name in [
        'mc_albedo',
    ]:
        # These targets derive properties from the trained neural BSDF.
        assert neural_model is not None
        return NeuralAuxDataGenerator(device, config, reference_materials, targets, neural_model)
    else:
        raise ValueError(f'Unsupported aux target: {targets.name}')
