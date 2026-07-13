# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import neuralnetworks as nn
import slangpy as spy
from slangpy.reflection import SlangType

from .base_types import SamplerInput
from .mlp import MLP


class Sampler(nn.IModel):
    """Provide importance sampling of the learned BSDF for path tracing.

    Decode a frozen latent code and incident direction into a sampling mixture.

    The sampler is trained after the BSDF representation and predicts one
    diffuse lobe plus a configurable number of specular lobes for importance
    sampling the outgoing direction.
    """

    def __init__(
        self,
        dtype: nn.Real,
        latents_config: dict,
        sampler_config: dict,
    ):
        super().__init__()
        self.dtype = dtype
        self.num_latents = latents_config['num_channels']
        self.num_specular_lobes = sampler_config['num_specular_lobes']
        self.num_lobes = 1 + self.num_specular_lobes

        self.num_inputs = self.num_latents + 3  # 3 for ``wi``.
        # Number of outputs: 3 for the diffuse lobe + 6 per specular lobe.
        # Diffuse: weight (1), x/y slopes (2) to encode orientation.
        # Specular per lobe: weight (1), x/y slopes (2), and roughness
        # (3: x/y param, and correlation).
        # ``sampler.slang`` transforms these values into normalized
        # mixture weights, valid roughness/correlation, and unbounded slopes.
        self.num_outputs = 3 + 6 * self.num_specular_lobes

        self.hidden_layers = sampler_config['hidden_layers']
        self.hidden_activations = sampler_config['hidden_activations']

    def model_init(self, module: spy.Module, input_type: SlangType):
        """Build the network that predicts diffuse and GGX parameters."""
        if SamplerInput.from_slangtype(input_type) is None:
            self.model_error(f'Sampler expects a SamplerInput; received {input_type.full_name}')

        self.network = MLP(
            dtype=self.dtype,
            num_inputs=self.num_inputs,
            num_outputs=self.num_outputs,
            hidden_layers=self.hidden_layers,
            hidden_activations=self.hidden_activations,
        )
        self.network.initialize(module, f'float[{self.num_inputs}]')

    def resolve_input_type(self, module):
        return f'SamplerInput<{self.num_latents}>'

    def children(self) -> list[nn.IModel]:
        return [self.network]

    def child_name(self, child: nn.IModel) -> str | None:
        if child is self.network:
            return 'network'
        return None

    @property
    def type_name(self) -> str:
        return (
            f'DiffuseSpecularSampler<{self.num_latents}, {self.num_lobes}, '
            f'{self.num_specular_lobes}, {self.num_inputs}, {self.num_outputs}, '
            f'{self.network.type_name}>'
        )

    def model_data(self):
        return {
            'network': self.network.get_this(),
        }
