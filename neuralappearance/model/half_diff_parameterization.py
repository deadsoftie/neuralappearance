# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import ClassVar

import neuralnetworks as nn
import slangpy as spy
from slangpy.reflection import SlangType

from .base_types import BsdfDecoderInputWithFrames


class HalfDiffParameterization(nn.IModel):
    """Add half/difference directions to a BSDF decoder input.

    This stage has no trainable parameters. It exposes a more useful angular
    representation to the decoder while retaining the original directions for
    feature layouts that use both.
    """

    types: ClassVar = ['Rusinkiewicz', 'StableRusinkiewicz']

    def __init__(
        self,
        type: str,
        num_latents: nn.AutoSettable[int] = nn.Auto,
        num_frames: nn.AutoSettable[int] = nn.Auto,
    ):
        super().__init__()
        assert type in self.types, f'Unknown HalfDiffParameterization type: {type}'
        self.type = type
        self._num_latents = num_latents
        self._num_frames = num_frames

    def model_init(self, module: spy.Module, input_type: SlangType):
        parsed_input = BsdfDecoderInputWithFrames.from_slangtype(input_type)
        if parsed_input is None:
            self.model_error(
                f'HalfDiffParameterization expects a BsdfDecoderInputWithFrames; received {input_type.full_name}'
            )
        self.num_latents = nn.resolve_auto(self._num_latents, parsed_input.num_latents)
        self.num_frames = nn.resolve_auto(self._num_frames, parsed_input.num_frames)

    def resolve_input_type(self, module: spy.Module):
        if self._num_latents is nn.Auto or self._num_frames is nn.Auto:
            return None
        return f'BsdfDecoderInputWithFrames<{self._num_latents}, {self._num_frames}>'

    @property
    def type_name(self) -> str:
        return f'{self.type}Parameterization<{self.num_latents}, {self.num_frames}>'
