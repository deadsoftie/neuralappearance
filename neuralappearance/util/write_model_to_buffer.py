# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import neuralnetworks as nn
import slangpy as spy
from slangpy.core.native import unpack_arg


def write_models_to_buffer(module: spy.Module, models: Sequence[nn.IModel]):
    """Pack same-type models into the GPU array used for training."""

    model_type = module[models[0].type_name]
    model_buffer = spy.Tensor.empty(device=module.device, shape=(len(models),), dtype=model_type)
    cursor = model_buffer.cursor()
    for i, model in enumerate(models):
        cursor[i].write(unpack_arg(model))
    cursor.apply()
    return model_buffer
