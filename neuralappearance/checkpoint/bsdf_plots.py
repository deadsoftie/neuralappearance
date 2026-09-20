# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import slangpy as spy
from training import TrainingTargets
from util.math_helpers import udim_offset
from util.timer import Timer

if TYPE_CHECKING:
    from checkpoint.checkpoint import Checkpoint
    from datagen import DataGenerators, ReferenceMaterials


def save_bsdf_plots(
    ckpt: Checkpoint,
    reference_materials: ReferenceMaterials,
    data_generators: DataGenerators,
):
    """Create polar target plots at the current optimization state."""

    plots_config = ckpt.config['checkpoints']['bsdf_plots']
    if not plots_config['save']:
        return

    timer = Timer()
    timer.start()
    print('Creating BSDF plots ...')

    for idx, material in enumerate(reference_materials):
        print(f'  Material {idx + 1}/{len(reference_materials)} ...')

        resolution = plots_config.get('resolution', 450)
        axes = plots_config.get('axes', ['log'])

        # The final images contain ``num_uvs * num_uvs`` subplots, each focused
        # on one UV location in the first UDIM tile.
        num_uvs = plots_config['num_uvs']
        # Each subplot uses ``num_samples`` points along the angular domain.
        num_samples = plots_config['num_samples']
        assert num_uvs * num_samples > 0

        # Generate (uv, wi, wo) samples where the target should be evaluated.
        grid_shape = (num_uvs, num_uvs, num_samples)
        uvs = spy.Tensor.empty(ckpt.module.device, shape=grid_shape, dtype='float2')
        wis = spy.Tensor.empty(ckpt.module.device, shape=grid_shape, dtype='float3')
        wos = spy.Tensor.empty(ckpt.module.device, shape=grid_shape, dtype='float3')

        udim_id = plots_config.get('udim_id', -1)
        if udim_id > 1000:
            plot_udim = udim_id
        else:
            plot_udim = min(material.udims)
        min_uv = udim_offset(plot_udim)
        max_uv = min_uv + spy.float2(1, 1)

        latent_texture = ckpt.model.latent_texture
        encoder = ckpt.model.encoder
        decoder = ckpt.model.decoder
        sampler = ckpt.model.sampler
        aux = ckpt.model.aux

        for mip_level in range(ckpt.model.num_mip_levels):
            ckpt.module.get_plots_coords(
                min_uv,
                max_uv,
                num_uvs,
                num_samples,
                material.texture_resolution,
                mip_level,
                spy.call_id(),
                uvs,
                wis,
                wos,
            )
            wis_np = wis.to_numpy()
            wos_np = wos.to_numpy()

            if latent_texture:
                latents = ckpt.module.eval_latent_texture(
                    latent_texture, material.id, mip_level, uvs
                )
            elif encoder:
                encoder_inputs = data_generators.bsdf.eval_encoder_inputs(material, mip_level, uvs)
                latents = ckpt.module.eval_encoder(encoder, material.id, encoder_inputs)
            else:
                raise RuntimeError('No latent texture or encoder found in the model.')

            mips = spy.Tensor.from_numpy(
                ckpt.module.device, np.full(grid_shape, mip_level, dtype=np.int32)
            )
            prediction_brdf = ckpt.module.eval_bsdf_decoder(
                decoder, material.id, mips, latents, wis, wos
            ).to_numpy()
            if sampler:
                prediction_sampler = ckpt.module.eval_sampler(sampler, latents, wis, wos).to_numpy()
            if aux is not None:
                prediction_aux = ckpt.module.eval_aux(aux, latents, wis).to_numpy()

            training_targets: list[TrainingTargets] = [TrainingTargets('bsdf')]
            # aux can be architecturally configured (ckpt.model.aux is not
            # None) without having been trained yet in this job -- e.g. a
            # checkpoint saved during BsdfDirectOptimization/decoder_repair,
            # which runs before training_aux() ever creates the aux data
            # generator. Skip the aux comparison plot in that case instead
            # of crashing; there's nothing to compare against yet.
            if mip_level == 0 and ckpt.model.aux is not None and data_generators.aux is not None:
                training_targets += [ckpt.model.aux.targets]
            for targets in training_targets:
                if targets.name == 'bsdf':
                    reference = data_generators.bsdf.eval_reference(
                        material, mip_level, uvs, wis, wos
                    ).to_numpy()
                    prediction = prediction_brdf
                else:
                    aux_data_generator = data_generators.aux
                    if aux_data_generator is None:
                        raise RuntimeError('Auxiliary data generator is not initialized.')
                    reference = aux_data_generator.eval_reference(
                        material, mip_level, uvs, wis, wos
                    ).to_numpy()
                    prediction = prediction_aux

                for target in targets:
                    # Slice the data to the current target.
                    reference_slice = target.view(reference)
                    prediction_slice = target.view(prediction)

                    for axis_type in axes:
                        figures_eval = []
                        for row in range(num_uvs):
                            for col in range(num_uvs):
                                figures_eval.append(
                                    create_fig(
                                        row,
                                        col,
                                        wis_np,
                                        wos_np,
                                        reference_slice,
                                        prediction_slice,
                                        axis_type,
                                        resolution,
                                    )
                                )

                        name = f'bsdf_plots.material{material.id}.mip{mip_level}.{axis_type}.{target.name}'
                        save_subplots(figures_eval, name, resolution, num_uvs, ckpt.folder)

            if sampler:
                for axis_type in axes:
                    figures_sampler = []
                    for row in range(num_uvs):
                        for col in range(num_uvs):
                            figures_sampler.append(
                                create_fig(
                                    row,
                                    col,
                                    wis_np,
                                    wos_np,
                                    prediction_brdf,
                                    prediction_sampler,
                                    axis_type,
                                    resolution,
                                    sampler=True,
                                )
                            )

                    name = f'bsdf_plots.material{material.id}.mip{mip_level}.{axis_type}.pdf'
                    save_subplots(figures_sampler, name, resolution, num_uvs, ckpt.folder)

    timer.stop()
    print(f'BSDF plots took {(timer.elapsed()):.2f}s.')


def create_fig(
    row,
    col,
    wis_np,
    wos_np,
    targets,
    outputs,
    axis_type,
    resolution,
    sampler=False,
):
    """Build one polar subplot for ``save_subplots()``."""

    target = targets[row][col]
    neural = outputs[row][col]
    assert target.shape[0] == neural.shape[0]
    theta_i = np.arctan2(wis_np[row, col, :, 0], wis_np[row, col, :, 2])
    theta_o = np.arctan2(wos_np[row, col, :, 0], wos_np[row, col, :, 2])

    # Entries contain radius, theta, color, name, line style, and line width.
    traces: list[tuple] = []
    line_width = 0.8 * max(1, int(resolution / 225))
    _legend = row == 0 and col == 0

    def add_plot(radius, theta, color, name, linestyle):
        traces.append(
            (
                np.asarray(radius, dtype=np.float64),
                np.asarray(theta, dtype=np.float64),
                color,
                name if _legend else None,
                linestyle,
                line_width,
            )
        )

    # Plot angular lobes.
    if sampler:
        # Luminance prediction for PDF.
        target_lum = 0.2126 * target[:, 0] + 0.7152 * target[:, 1] + 0.0722 * target[:, 2]
        neural_lum = neural[:, 0]

        # Normalize plots to the same range.
        upper_mask = wos_np[row, col, :, 2] > 0.0
        target_max = np.max(target_lum[upper_mask])
        if target_max > 0.0:
            target_lum /= target_max
        neural_max = np.max(neural_lum[upper_mask])
        if neural_max > 0.0:
            neural /= neural_max

        # Plot target and neural prediction.
        add_plot(target_lum, theta_o, 'black', 'target', ':')
        add_plot(neural_lum, theta_o, 'black', 'neural', '-')

        # And plot individual lobe PDFs.
        for lobe_idx in range(neural.shape[-1] - 1):
            neural_l = neural[:, 1 + lobe_idx]
            add_plot(neural_l, theta_o, None, f'neural lobe {lobe_idx}', '-')

        radial_max = np.max(target_lum)
    else:
        if neural.shape[-1] < 1 or neural.shape[-1] > 3:
            raise ValueError(
                'Unsupported target size {neural.shape[-1]} for {name} in save_target_plots.'
            )

        colors = ['red', 'green', 'blue'][: neural.shape[-1]]
        for ch, cl in enumerate(colors):
            add_plot(target[:, ch], theta_o, cl, f'target.{cl[0].upper()}', ':')
        for ch, cl in enumerate(colors):
            add_plot(neural[:, ch], theta_o, cl, f'neural.{cl[0].upper()}', '-')

        radial_max = np.max(target)

    if axis_type == 'log':
        radial_max = np.log10(radial_max) + 0.5 if radial_max > 0.0 else 1.0
        radial_min = min(-3, radial_max - 5)
    elif axis_type == 'linear':
        radial_max = radial_max * 1.1 if radial_max > 0.0 else 1.0
        radial_min = 0.0
    else:
        raise ValueError(f'Invalid axis type: {axis_type}')

    add_plot([0, 1], theta_i[:2], 'magenta', 'incident dir', '-')

    return (traces, axis_type, radial_min, radial_max)


def save_subplots(figures_data, name, res, num_uvs, ckpt_folder):
    if len(figures_data) == 0:
        return

    legend_size = max(120, int(0.55 * res))
    dpi = 100
    fig_w = (res * num_uvs + legend_size) / dpi
    fig_h = (res * num_uvs) / dpi

    fig, axes = plt.subplots(
        num_uvs,
        num_uvs,
        subplot_kw={'projection': 'polar'},
        figsize=(fig_w, fig_h),
        dpi=dpi,
        squeeze=False,
    )

    angular_tick_font_size = max(5, int(res / 55))
    radial_tick_font_size = max(5, int(res / 55))

    for i, (traces, axis_type, radial_min, radial_max) in enumerate(figures_data):
        row = i % num_uvs
        col = i // num_uvs
        ax: plt.Axes = axes[row][col]  # type: ignore[assignment]

        is_log = axis_type == 'log'
        r_lo = 10**radial_min if is_log else radial_min
        r_hi = 10**radial_max if is_log else radial_max

        for r, theta, color, label, ls, lw in traces:
            if is_log:
                r = np.clip(r, r_lo, None)
            ax.plot(theta, r, color=color, linestyle=ls, linewidth=lw, label=label)

        # Polar-axis orientation (0 deg at top, counter-clockwise).
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(1)

        if is_log:
            ax.set_rscale('log')
        ax.set_rlim(r_lo, r_hi)

        ax.tick_params(axis='x', labelsize=angular_tick_font_size)
        ax.tick_params(axis='y', labelsize=radial_tick_font_size)
        # Place radial labels on the right side.
        ax.set_rlabel_position(270)

        ax.set_facecolor('#E5ECF6')
        ax.grid(True, color='white', linewidth=0.8)
        ax.spines['polar'].set_visible(False)

    # Reserve the right strip for the legend.
    legend_frac = legend_size / (dpi * fig_w)
    fig.subplots_adjust(
        left=0.01,
        right=1.0 - legend_frac - 0.02,
        top=0.97,
        bottom=0.03,
        wspace=0.2,
        hspace=0.2,
    )

    # Collect legend handles from the first subplot.
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        legend_font_size = max(6, int(res / 30))
        fig.legend(
            handles,
            labels,
            loc='center left',
            bbox_to_anchor=(1.0 - legend_frac + 0.005, 0.5),
            fontsize=legend_font_size,
            frameon=False,
            borderaxespad=0,
        )

    plot_file = ckpt_folder / f'{name}.png'
    try:
        fig.savefig(str(plot_file), format='png', dpi=dpi, facecolor='white')
    except Exception as e:
        print(f'Error saving plot {plot_file}: {e}')

    plt.close(fig)
