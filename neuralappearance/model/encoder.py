# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import neuralnetworks as nn
import slangpy as spy
from slangpy.reflection import SlangType

from .base_types import EncoderInput
from .mlp import MLP


class Encoder(nn.IModel):
    """Compress reference-material parameters into the shared latent code.

    The encoder is trained during BSDF encoding. Its output is subsequently
    baked into latent textures, so the encoder is not needed for material
    evaluation after training.
    """

    def __init__(
        self,
        dtype: nn.Real,
        num_inputs: int,
        latents_config: dict,
        encoder_config: dict,
    ):
        super().__init__()

        self.dtype = dtype
        self.num_inputs = num_inputs
        self.num_latents = latents_config['num_channels']
        self.normalize_latents = latents_config['normalize']
        self.hidden_layers = encoder_config['hidden_layers']
        self.hidden_activations = encoder_config['hidden_activations']

    def model_init(self, module: spy.Module, input_type: SlangType):
        if EncoderInput.from_slangtype(input_type) is None:
            self.model_error(f'Encoder expects a EncoderInput; received {input_type.full_name}')

        self.network = MLP(
            dtype=self.dtype,
            num_inputs=self.num_inputs,
            num_outputs=self.num_latents,
            hidden_layers=self.hidden_layers,
            hidden_activations=self.hidden_activations,
        )
        # Match ``encoder.slang``, whose material parameters use half precision.
        self.network.initialize(module, f'half[{self.num_inputs}]')

    def resolve_input_type(self, module):
        return f'EncoderInput<{self.num_inputs}, {self.num_latents}>'

    def children(self) -> list[nn.IModel]:
        return [self.network]

    def child_name(self, child: nn.IModel) -> str | None:
        if child is self.network:
            return 'network'
        return None

    @property
    def type_name(self) -> str:
        return f'Encoder<{self.num_inputs}, {self.num_latents}, {self.network.type_name}>'

    def model_data(self):
        return {
            'normalize_latents': self.normalize_latents,
            'network': self.network.get_this(),
        }
