# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import neuralnetworks as nn
import slangpy as spy
from slangpy.reflection import SlangType
from training import TrainingTargets

from .base_types import AuxDecoderInput
from .mlp import MLP


class AuxDecoder(nn.IModel):
    """Fit auxiliary material properties required by the rendering system,
    such as those used to populate denoiser guide buffers.

    Predict configured outputs from frozen latents and incident direction ``wi``.

    This network is trained in the final auxiliary phase after the BSDF
    representation has been learned. ``TrainingTargets`` determines both the
    semantic outputs and their total channel count.
    """

    def __init__(
        self,
        dtype: nn.Real,
        latents_config: dict,
        aux_config: dict,
    ):
        super().__init__()

        self.dtype = dtype
        self.num_latents = latents_config['num_channels']
        self.targets = TrainingTargets(aux_config['targets'])
        self.hidden_layers = aux_config['hidden_layers']
        self.hidden_activations = aux_config['hidden_activations']

        self.num_inputs = 3 + self.num_latents
        self.num_outputs = self.targets.num_channels

    def model_init(self, module: spy.Module, input_type: SlangType):
        if AuxDecoderInput.from_slangtype(input_type) is None:
            self.model_error(
                f'AuxDecoder expects a AuxDecoderInput; received {input_type.full_name}'
            )

        self.network = MLP(
            dtype=self.dtype,
            num_inputs=self.num_inputs,
            num_outputs=self.num_outputs,
            hidden_layers=self.hidden_layers,
            hidden_activations=self.hidden_activations,
        )
        self.network.initialize(module, f'float[{self.num_inputs}]')

    def resolve_input_type(self, module):
        return f'AuxDecoderInput<{self.num_latents}>'

    def children(self) -> list[nn.IModel]:
        return [self.network]

    def child_name(self, child: nn.IModel) -> str | None:
        if child is self.network:
            return 'network'
        return None

    @property
    def type_name(self) -> str:
        return (
            f'AuxDecoder<{self.num_latents}, {self.num_outputs}, '
            f'{self.num_inputs}, {self.network.type_name}>'
        )

    def model_data(self):
        return {
            'network': self.network.get_this(),
        }
