# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import neuralnetworks as nn
import numpy as np
import slangpy as spy


class MLP(nn.ModelChain):
    def __init__(
        self,
        dtype: nn.Real,
        num_inputs: nn.AutoSettable[int],
        num_outputs: int,
        hidden_layers: list[int],
        hidden_activations: list[str] | str,
        use_biases: bool = True,
    ):
        num_hidden_layers = len(hidden_layers)
        if isinstance(hidden_activations, str):
            hidden_activations = [hidden_activations] * num_hidden_layers
        assert len(hidden_activations) == num_hidden_layers

        chain = [
            nn.Convert.to_precision(dtype),
            nn.Convert.to_coopvec(),
        ]

        if num_hidden_layers == 0:
            chain += [
                nn.LinearLayer(
                    num_inputs,
                    num_outputs,
                    use_biases,
                )
            ]
        else:
            chain += [
                nn.LinearLayer(
                    num_inputs,
                    hidden_layers[0],
                    use_biases,
                ),
                getattr(nn, hidden_activations[0])(),
            ]

            for i in range(num_hidden_layers - 1):
                chain += [
                    nn.LinearLayer(
                        hidden_layers[i],
                        hidden_layers[i + 1],
                        use_biases,
                    ),
                    getattr(nn, hidden_activations[i + 1])(),
                ]

            chain += [
                nn.LinearLayer(
                    hidden_layers[-1],
                    num_outputs,
                    use_biases,
                )
            ]

        chain += [nn.Convert.to_array(), nn.Convert.to_float()]

        super().__init__(*chain)

    def save_checkpoint_images(
        self,
        desc: list[str],
        folder: Path,
    ) -> list[tuple[list[str], str]]:
        # Extract all ``LinearLayer`` modules, i.e. neural network layers.
        layers = [c for c in self.components() if isinstance(c, nn.LinearLayer)]

        # The image will represent weight matrices as NxM, and bias vectors as
        # Nx1 pixel blocks. Fetch these from the layers.
        blocks = []
        for layer in layers:
            blocks.append(layer.get_weights())
            blocks.append(layer.get_biases()[:, None])

        # Compute size of the final image based on these.
        height = max([block.shape[0] for block in blocks])
        width = sum([block.shape[1] for block in blocks]) + len(blocks) - 1

        # Assemble the actual image.
        horizontal_offset = 0
        param_image = np.ones((height, width, 3))
        for block in blocks:
            # Extent of the block in pixels.
            top = 0
            bottom = top + block.shape[0]
            left = horizontal_offset
            right = left + block.shape[1]

            # Initialize as black.
            param_image[top:bottom, left:right, :] = 0.0
            # Write negative values into the red channel and positive values
            # into the green channel.
            negative_values = np.maximum(0, -block)
            positive_values = np.maximum(0, block)
            param_image[top:bottom, left:right, 0] = negative_values
            param_image[top:bottom, left:right, 1] = positive_values

            horizontal_offset += block.shape[1] + 1

        out_desc = [*desc, 'mlp_visualization']
        out_name = '.'.join(out_desc) + '.exr'

        bitmap = spy.Bitmap(param_image.astype(np.float16))
        bitmap.write(folder / out_name)

        return [(out_desc, out_name)]

    def save_checkpoint_params(
        self,
        desc: list[str],
    ) -> list[tuple[list[str], dict]]:
        # Extract all ``LinearLayer`` modules, i.e. neural network layers.
        layers = [c for c in self.components() if isinstance(c, nn.LinearLayer)]

        layer_data = []
        for layer in layers:
            layer_data.append(
                {
                    'num_inputs': layer.num_inputs,
                    'num_outputs': layer.num_outputs,
                    'weights': list(layer.get_weights().flatten().astype('float')),
                    'biases': list(layer.get_biases().flatten().astype('float')),
                }
            )

        return [(desc, {'mlp_layers': layer_data})]

    def load_checkpoint_params(
        self,
        params: dict,
    ):
        # Extract all ``LinearLayer`` modules, i.e. neural network layers.
        layers = [c for c in self.components() if isinstance(c, nn.LinearLayer)]

        layer_data = params['mlp_layers']
        # A checkpoint must contain exactly one record per linear layer.
        for layer_config, layer in zip(layer_data, layers, strict=True):
            num_inputs = layer_config['num_inputs']
            num_outputs = layer_config['num_outputs']

            w = np.array(layer_config['weights'], dtype=np.float16)
            b = np.array(layer_config['biases'], dtype=np.float16)
            layer.set_weights(w.reshape(num_outputs, num_inputs))
            if layer.use_biases:
                layer.set_biases(b)
