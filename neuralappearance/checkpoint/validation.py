# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import commentjson as json
import numpy as np
from datagen import FalcorBsdfDataGenerator
from util.timer import Timer

if TYPE_CHECKING:
    from datagen import ReferenceMaterials
    from model import NeuralModel


def evaluate_validation_losses(
    base_config: dict,
    neural_model: NeuralModel,
    reference_materials: ReferenceMaterials,
    outfolder: Path,
):
    """Evaluate the trained BSDF under independent directional distributions.

    Validation regenerates large batches rather than reusing training samples.
    Reporting both uniform and importance-sampled outgoing directions exposes
    errors in broad regions as well as around high-energy BSDF lobes.
    """

    if not base_config['checkpoints']['validation_losses']:
        return

    timer = Timer()
    timer.start()
    print()
    print('Evaluating validation losses ...')

    module = neural_model.module

    batch_size = 1024 * 1024
    num_batches = 100
    strategies = [
        'UniformWiUniformWo',
        'UniformWiImportanceWo',
    ]
    metric_names = [
        'L1',
        'RelativeL1',
        'SMAPE',
        'L1WithPowerRoot<3>',
        'L1WithPowerLog<3>',
        'L1WithSafeLog',
    ]
    num_losses = len(metric_names)
    all_losses = {}

    for strategy in strategies:
        config = deepcopy(base_config)
        config['data_generation']['directional_sampling_strategy'] = strategy

        data_generator = FalcorBsdfDataGenerator(
            module.device,
            config,
            reference_materials,
            'BsdfDirectOptimization',
        )

        np_losses = np.empty((num_batches, num_losses), dtype=np.float32)
        for i in range(num_batches):
            batch = data_generator.generate_training_data(
                iteration=i,
                batch_count=1,
                batch_size=batch_size,
            )[0]

            losses = module.eval_validation_losses(
                neural_model.latent_texture_buffer, neural_model.decoder_buffer, batch
            ).to_numpy()

            valid = losses[:, 0] == 1.0
            losses = losses[:, 1:][valid]
            np_losses[i] = losses.mean(axis=0)

        losses = np_losses.mean(axis=0)
        all_losses[strategy] = {}
        for name, loss in zip(metric_names, losses, strict=True):
            all_losses[strategy][name] = float(loss)

    timer.stop()
    print(f'Evaluating took {(timer.elapsed()):.2f}s.')

    # Print the validation table.
    print()
    print('------------------------------------------')
    # All strategies expose the same metric names.
    max_name_len = max(len(s) for s in strategies)

    # Calculate column widths from displayed content.
    col_widths = {}
    for metric in metric_names:
        # Find the widest value across strategies.
        max_val_width = max(len(f'{all_losses[strategy][metric]:.3f}') for strategy in strategies)
        col_widths[metric] = max(max_val_width, len(metric))

    # Print the header.
    header_strategy = ' ' * max_name_len
    header_metrics = '  '.join(metric.center(col_widths[metric]) for metric in metric_names)
    print(f'{header_strategy}  {header_metrics}')
    # Print the data rows.
    for strategy in strategies:
        strategy_name = strategy.ljust(max_name_len)
        metric_values = '  '.join(
            f'{all_losses[strategy][metric]:.3f}'.rjust(col_widths[metric])
            for metric in metric_names
        )
        print(f'{strategy_name}  {metric_values}')
    print('------------------------------------------')
    print()

    validation_losses_path = Path(outfolder) / 'validation_losses.json'
    with validation_losses_path.open('w') as f:
        json.dump(all_losses, f, indent=2)
