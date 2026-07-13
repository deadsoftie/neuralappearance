# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from util.timer import Timer

if TYPE_CHECKING:
    from util import LossSeries


def save_loss_plots(config: dict, loss_series: LossSeries) -> list[Path]:
    """Plot every recorded loss series, grouped by metric and averaging mode."""

    loss_plots_config = config.get('checkpoints', {}).get('loss_plots', {})
    if not loss_plots_config.get('save', False):
        return []

    timer = Timer()
    timer.start()
    print('Creating loss plots ...')

    saved_paths = _save_loss_plots(loss_series, loss_plots_config)

    if saved_paths:
        outfolder = Path(loss_series.outfolder)
        saved_names = ', '.join(str(path.relative_to(outfolder)) for path in saved_paths)
        print(f'Loss plots saved: {saved_names}')
    else:
        print('No loss series found for loss plots.')

    timer.stop()
    print(f'Loss plots took {(timer.elapsed()):.2f}s.')
    return saved_paths


def _save_loss_plots(loss_series: LossSeries, config: dict | None = None) -> list[Path]:
    config = {} if config is None else config
    if not config.get('save', True):
        return []

    image_format = 'png'
    output_dir = Path(loss_series.outfolder) / 'loss_plots'

    loss_groups = _collect_loss_groups(loss_series)
    if not loss_groups:
        return []

    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for (loss_name, averaged), traces in loss_groups.items():
        use_log_axis = loss_name != 'Training Loss Sampler'
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        num_plotted = 0
        min_iteration = np.inf
        max_iteration = 0.0
        for label, iterations, values in traces:
            mask = np.isfinite(iterations) & np.isfinite(values)
            if use_log_axis:
                mask &= values > 0
            if not mask.any():
                continue

            ax.plot(iterations[mask], values[mask], linewidth=1.2, label=label)
            min_iteration = min(min_iteration, float(np.min(iterations[mask])))
            max_iteration = max(max_iteration, float(np.max(iterations[mask])))
            num_plotted += 1

        if num_plotted == 0:
            plt.close(fig)
            continue

        title = f'{loss_name} (iteration-averaged)' if averaged else loss_name
        ax.set_title(title)
        ax.set_xlabel('Iteration')
        if min_iteration == max_iteration:
            ax.set_xlim(min_iteration, min_iteration + 1.0)
        else:
            ax.set_xlim(min_iteration, max_iteration)
            ticks = ax.get_xticks()
            ticks = ticks[(ticks >= min_iteration) & (ticks <= max_iteration)]
            ticks = np.unique(np.concatenate(([min_iteration], ticks, [max_iteration])))
            ax.set_xticks(ticks)
        ax.set_yscale('log' if use_log_axis else 'linear')
        ax.grid(True, which='both', alpha=0.25)
        if num_plotted <= 16:
            ax.legend(fontsize='x-small')

        path = output_dir / f'{_loss_filename_stem(loss_name, averaged)}.{image_format}'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved_paths.append(path)

    return saved_paths


def _collect_loss_groups(
    loss_series: LossSeries,
) -> dict[tuple[str, bool], list[tuple[str, np.ndarray, np.ndarray]]]:
    """Group per-instance traces so candidate models share one figure."""

    loss_groups = {}
    for name, entries in loss_series.series.items():
        name_lower = name.lower()
        if 'loss' not in name_lower:
            continue

        iterations = np.asarray(entries['iteration'], dtype=np.float32)
        values = np.asarray(entries['value'], dtype=np.float32)
        if values.size <= 0:
            continue

        loss_name, has_instance_suffix, instance_id = name.rpartition(' #')
        if not has_instance_suffix:
            loss_name = name
            instance_id = None

        is_iteration_averaged = loss_name.endswith(' (iteration-averaged)')
        if is_iteration_averaged:
            loss_name = loss_name[: -len(' (iteration-averaged)')]

        if instance_id is None:
            label = loss_name
        else:
            label = f'#{instance_id}'

        loss_groups.setdefault((loss_name, is_iteration_averaged), []).append(
            (
                label,
                iterations,
                values,
            )
        )

    return loss_groups


def _loss_filename_stem(loss_name: str, averaged: bool = False) -> str:
    filenames = {
        'Training Loss Bsdf': 'training_loss.bsdf',
        'Training Loss Sampler': 'training_loss.sampler',
        'Training Loss Aux': 'training_loss.aux',
    }
    if loss_name in filenames:
        stem = filenames[loss_name]
        return f'{stem}.iteration_averaged' if averaged else stem

    stem = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in loss_name).strip('_')
    while '__' in stem:
        stem = stem.replace('__', '_')
    if not stem:
        stem = 'loss'
    return f'{stem}.iteration_averaged' if averaged else stem
