# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING

import commentjson as json
import slangpy as spy
from util.timer import Timer

from .bsdf_plots import save_bsdf_plots

if TYPE_CHECKING:
    from datagen import (
        DataGenerators,
        ReferenceMaterials,
    )
    from model import (
        Encoder,
        LatentTexture,
        NeuralModel,
        NeuralModelCheckpoint,
    )
    from rendering import Renderer


class Checkpoint:
    """Model snapshot plus the output location for checkpoint artifacts.

    Renderings, plots, model images, and JSON metadata produced for one
    iteration share this object so they use the same selected model instances
    and directory.
    """

    def __init__(
        self,
        module: spy.Module,
        config: dict,
        model: NeuralModelCheckpoint,
        outfolder: Path,
        iteration: int,
        create_folder: bool = True,
    ):
        self.module = module
        self.config = config
        self.model = model

        self.subpath = f'checkpoints/{iteration:08d}'
        self.folder = outfolder / self.subpath
        if create_folder:
            self.folder.mkdir(exist_ok=True, parents=True)


def save_checkpoint(
    module: spy.Module,
    config: dict,
    neural_model: NeuralModel,
    renderer: Renderer,
    reference_materials: ReferenceMaterials,
    data_generators: DataGenerators,
    outfolder: Path,
    iteration: int,
    best_instance_index: int = 0,
):
    """Save renderings, diagnostic plots, and model data for one iteration.

    The best surviving instance is selected before any artifact is generated.
    If the representation still comes from an encoder, its current output is
    first baked into the checkpoint's latent texture.
    """

    timer = Timer()
    timer.start()
    print('Saving checkpoint ...')

    # Get the relevant model instances to export in the checkpoint.
    model_ckpt = neural_model.get_checkpoint(best_instance_index)

    # Make sure the latent texture is up-to-date.
    encode_latent_texture_if_needed(
        module,
        neural_model,
        data_generators,
        reference_materials,
        model_ckpt.latent_texture,
        model_ckpt.encoder,
    )

    ckpt = Checkpoint(module, config, model_ckpt, outfolder, iteration)

    # Save model data even when an uninitialized latent texture is not yet safe
    # to render. Dense textures default to renderable.
    if getattr(model_ckpt.latent_texture, 'ready_for_render', True):
        renderer.render_neural_materials(ckpt, reference_materials, data_generators)
        save_bsdf_plots(ckpt, reference_materials, data_generators)
    else:
        print('Latent texture is not renderable yet; skipping neural render and BSDF plots.')
    save_model(ckpt)

    timer.stop()
    print(f'Checkpoint took {(timer.elapsed()):.2f}s.')


def encode_latent_texture_if_needed(
    module: spy.Module,
    neural_model: NeuralModel,
    data_generators: DataGenerators,
    reference_materials: ReferenceMaterials,
    latent_texture: LatentTexture,
    encoder: Encoder | None,
):
    """Bake the active encoder into the checkpoint texture while it is training.

    Once encoder training is done, its final texture has already been generated
    and subsequent checkpoint phases can reuse it.
    """

    assert (encoder is None) == (neural_model.encoder is None)

    if encoder is None:
        return

    if neural_model.encoder is None or neural_model.encoder.status == 'Done':
        return

    timer = Timer()
    timer.start()
    print('Encoding latent texture ...')

    latent_texture.generate_from_encoder(
        module,
        data_generators,
        reference_materials,
        neural_model.num_mip_levels,
        encoder,
    )

    timer.stop()
    print(f'Encoding took {(timer.elapsed()):.2f}s.')


def merge_checkpoint_params(model_config: dict, desc: list[str], params: dict) -> None:
    """Write one ``save_checkpoint_params`` record into ``model_config`` at ``desc``.

    Records are merged, not assigned: a component's own record shares its path
    with the records its children already wrote there, and assigning would
    drop the children (e.g. a latent texture's decode MLP weights).
    """
    c = model_config
    for key in desc[:-1]:
        c = c.setdefault(key, {})
    if isinstance(c.get(desc[-1]), dict) and isinstance(params, dict):
        c[desc[-1]].update(params)
    else:
        c[desc[-1]] = params


def save_model(ckpt: Checkpoint) -> None:
    """Write model tensors and architecture metadata enabled by the config."""

    save_model_images = ckpt.config['checkpoints']['model']['save_images']
    save_model_params = ckpt.config['checkpoints']['model']['save_params']
    if not (save_model_images or save_model_params):
        return

    timer = Timer()
    timer.start()
    print('Saving model ...')

    if save_model_images:
        # Textures and other images.
        for name, component in ckpt.model.components():
            if component is not None:
                component.save_checkpoint_images([name], ckpt.folder)

    if save_model_params:
        # Other parameters and configs.
        config = copy.deepcopy(ckpt.config)
        model_config = copy.deepcopy(config['model'])

        for name, component in ckpt.model.components():
            if component is None:
                continue

            for desc, params in component.save_checkpoint_params([name, '_params']):
                merge_checkpoint_params(model_config, desc, params)

            model_config[name]['_type_name'] = component.type_name

        with open(ckpt.folder / 'model.json', 'w') as fout:
            json.dump(model_config, fout, indent=4)

        with open(ckpt.folder / 'config.json', 'w') as fout:
            json.dump(config, fout, indent=4)

    timer.stop()
    print(f'Model took {(timer.elapsed()):.2f}s.')
