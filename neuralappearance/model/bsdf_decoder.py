# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import neuralnetworks as nn

from .bsdf_decoder_flattener import BsdfDecoderFlattener
from .half_diff_parameterization import HalfDiffParameterization
from .mlp import MLP
from .rotation import Rotation


class BsdfDecoder(nn.ModelChain):
    """Represent the material's main BSDF for evaluation during rendering.

    Decode a spatial latent code and direction pair into RGB BSDF values.

    When an encoder is configured, the decoder is first trained alongside it
    during BSDF encoding. It is then retained while direct latent-texture
    optimization refines the representation. Its component chain prepares
    angular features before the MLP and applies the configured non-negative
    output mapping afterward.
    """

    def __init__(
        self,
        dtype: nn.Real,
        latents_config: dict,
        decoder_config: dict,
    ):
        num_latents = latents_config['num_channels']

        hidden_layers = decoder_config['hidden_layers']
        hidden_activations = decoder_config['hidden_activations']
        output_activation_config = decoder_config['output_activation']

        half_diff_parameterization: str = decoder_config.get(
            'half_diff_parameterization', 'StableRusinkiewicz'
        )
        direction_inputs: str = decoder_config.get('direction_inputs', 'WhWd')
        num_shading_frames: int = decoder_config.get('num_shading_frames', 0)

        modules = []

        # Populate wh/wd before an optional rotation duplicates all four
        # direction representations into learned shading frames.
        modules += [HalfDiffParameterization(half_diff_parameterization, num_latents, 1)]

        # Rotation module.
        self.rotation = None
        if num_shading_frames > 0:
            self.rotation = Rotation(dtype, num_shading_frames)
            modules += [self.rotation]

        # Flatten all MLP inputs.
        modules += [BsdfDecoderFlattener(direction_inputs)]

        # Main decoder MLP.
        self.network = MLP(
            dtype=dtype,
            num_inputs=nn.Auto,
            num_outputs=3,
            hidden_layers=hidden_layers,
            hidden_activations=hidden_activations,
        )
        modules += [self.network]

        # Map the final linear-layer output to a non-negative BSDF value.
        output_activation = None
        if output_activation_config['type'] == 'Exp':
            output_activation = nn.Exp()
        elif output_activation_config['type'] == 'ScaledSigmoid':
            scale = output_activation_config.get('scale', 1.0)
            output_activation = nn.ScaledSigmoid(scale=scale)
        if output_activation is not None:
            modules += [output_activation]

        super().__init__(*modules)

    def child_name(self, child: nn.IModel) -> str | None:
        if child is self.rotation:
            return 'rotation'
        if child is self.network:
            return 'network'
        return super().child_name(child)
