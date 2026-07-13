# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import neuralnetworks as nn
import slangpy as spy
from slangpy.reflection import SlangType

from .base_types import BsdfDecoderInputWithFrames
from .mlp import MLP


class Rotation(nn.IModel):
    """Predict latent-dependent shading frames for decoder directions.

    Each learned frame provides another local view of the same direction pair,
    allowing the BSDF decoder to represent spatially varying orientation.
    """

    def __init__(
        self,
        dtype: nn.Real,
        num_frames: int,
        num_latents: nn.AutoSettable[int] = nn.Auto,
    ):
        super().__init__()
        self.dtype = dtype
        self.num_frames = num_frames
        self._num_latents = num_latents

        self.rotation_decoder = MLP(
            dtype=self.dtype,
            num_inputs=self._num_latents,
            num_outputs=self.num_frames * 6,
            hidden_layers=[],
            hidden_activations=[],
            use_biases=False,
        )

    def model_init(self, module: spy.Module, input_type: SlangType):
        parsed_input = BsdfDecoderInputWithFrames.from_slangtype(input_type)
        if parsed_input is None:
            self.model_error(
                f'Rotation expects a BsdfDecoderInputWithFrames; received {input_type.full_name}'
            )
        self.num_latents = nn.resolve_auto(self._num_latents, parsed_input.num_latents)
        self.rotation_decoder.initialize(module, f'float[{self.num_latents}]')

    def resolve_input_type(self, module: spy.Module):
        # Rotation expands one direction pair into ``num_frames`` pairs.
        if self._num_latents is nn.Auto:
            return None
        return f'BsdfDecoderInput<{self.num_latents}>'

    def children(self) -> list[nn.IModel]:
        return [self.rotation_decoder]

    def child_name(self, child: nn.IModel):
        if child is self.rotation_decoder:
            return 'rotation_decoder'
        return None

    @property
    def type_name(self) -> str:
        num_decoder_outputs = self.num_frames * 6
        return (
            f'ShadingFrameRotation<{self.num_frames}, {self.num_latents}, '
            f'{num_decoder_outputs}, {self.rotation_decoder.type_name}>'
        )

    def model_data(self):
        return {'rotation_decoder': self.rotation_decoder.get_this()}
