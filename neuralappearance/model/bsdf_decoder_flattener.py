# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import ClassVar

import neuralnetworks as nn
import slangpy as spy
from slangpy.reflection import SlangType

from .base_types import BsdfDecoderInputWithFrames


class BsdfDecoderFlattener(nn.IModel):
    """Select and flatten directional features for the BSDF decoder MLP.

    The configured layout controls which original and half/difference vectors
    the network sees.
    """

    types: ClassVar = [
        'WiWo',
        'WhWd',
        'WhWdZiZo',
        'WhWdWiWo',
    ]

    def __init__(
        self,
        type: str,
        num_latents: nn.AutoSettable[int] = nn.Auto,
        num_frames: nn.AutoSettable[int] = nn.Auto,
    ):
        super().__init__()
        assert type in self.types, f'Unknown BsdfDecoderFlattener type: {type}'
        self.type = type
        self._num_latents = num_latents
        self._num_frames = num_frames

    def model_init(self, module: spy.Module, input_type: SlangType):
        parsed_input = BsdfDecoderInputWithFrames.from_slangtype(input_type)
        if parsed_input is None:
            self.model_error(
                f'BsdfDecoderFlattener expects a BsdfDecoderInputWithFrames; received {input_type.full_name}'
            )
        self.num_latents = nn.resolve_auto(self._num_latents, parsed_input.num_latents)
        self.num_frames = nn.resolve_auto(self._num_frames, parsed_input.num_frames)

    def resolve_input_type(self, module: spy.Module):
        if self._num_latents is nn.Auto or self._num_frames is nn.Auto:
            return None
        return f'BsdfDecoderInputWithFrames<{self._num_latents}, {self._num_frames}>'

    @property
    def type_name(self) -> str:
        num_dir_outputs = 2 * 3 * self.num_frames
        if self.type == 'WhWdZiZo':
            num_dir_outputs += 2 * self.num_frames
        elif self.type == 'WhWdWiWo':
            num_dir_outputs *= 2
        num_outputs = num_dir_outputs + self.num_latents

        return f'{self.type}BsdfDecoderFlattener<{self.num_latents}, {self.num_frames}, {num_dir_outputs}, {num_outputs}>'
