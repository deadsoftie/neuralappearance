# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import slangpy as spy
from neuralnetworks import IModel, Real, SlangType

from .mlp import MLP
from .texture import Texture

if TYPE_CHECKING:
    from datagen import ReferenceMaterials


class NtcLatentTexture(IModel):
    """NTC-style compressed latent texture: a pair of bilinearly-sampled
    feature grids per tile (high-res, low-res, both at a resolution lower
    than the output texture), decoded into the shared ``num_latents``-
    channel code by a small fixed MLP, instead of one dense per-texel
    buffer. See ``model/ntc_latent_texture.slang`` and rtcnam/PLAN.md for
    the exact grid-shape/decoder recipe this mirrors (NVIDIA's shipped
    RTXNTC SDK) and the Slang file's header comment for what this first
    implementation simplifies away.

    Implements the same ``ILatentTexture`` contract as ``LatentTexture``,
    so every existing consumer (BsdfDecoder, Sampler, AuxDecoder, every
    training kernel, the renderer, the visualizer) works against it
    unchanged. A config flag in ``NeuralModel`` picks which one gets built;
    nothing downstream needs to know which.
    """

    @classmethod
    def from_reference_materials(
        cls,
        reference_materials: ReferenceMaterials,
        num_latents: int,
        grid_size_scale: int,
        num_features: int,
        num_pos_enc_waves: int,
        decoder_hidden_layers: list[int],
        decoder_activation: str,
        optimizable: bool = True,
    ) -> NtcLatentTexture:
        tex = cls(
            num_latents,
            grid_size_scale,
            num_features,
            num_pos_enc_waves,
            decoder_hidden_layers,
            decoder_activation,
            optimizable,
        )

        num_materials = max(reference_materials.ids) + 1
        udim_stride = 100
        tex.tile_indirection = [0] * udim_stride * num_materials
        tex.material_texture_resolution = [0] * num_materials
        tex.material_texture_wrap = [True] * num_materials
        for material in reference_materials:
            id = material.id
            tex.num_tiles += len(material.udims)
            tex.material_ids.append(id)
            tex.has_udims[id] = material.has_udims
            tex.udim_mapping[id] = material.udims
            tex.material_texture_resolution[id] = material.texture_resolution
            tex.material_texture_wrap[id] = material.texture_wrap
        return tex

    def __init__(
        self,
        num_latents: int,
        grid_size_scale: int,
        num_features: int,
        num_pos_enc_waves: int,
        decoder_hidden_layers: list[int],
        decoder_activation: str,
        optimizable: bool = True,
    ):
        super().__init__()

        self.num_latents = num_latents
        # Alias expected by callers written against the dense `LatentTexture`
        # (e.g. `rendering/neural_material.py`'s fallback sampler/aux type
        # strings) that only care about the decoded channel count, not this
        # texture's internal grid representation.
        self.num_channels = num_latents
        self.grid_size_scale = grid_size_scale
        self.num_features = num_features
        self.num_pos_enc_waves = num_pos_enc_waves
        self.decoder_hidden_layers = decoder_hidden_layers
        self.decoder_activation = decoder_activation
        self.optimizable = optimizable

        self.num_tiles = 0
        self.material_ids: list[int] = []
        self.has_udims: dict[int, bool] = {}
        self.udim_mapping: dict[int, list[int]] = {}
        self.tile_indirection: list[int] = []
        self.material_texture_resolution: list[int] = []
        self.material_texture_wrap: list[bool] = []

    def _grid_resolutions(self, material_id: int) -> tuple[int, int]:
        base = self.material_texture_resolution[material_id]
        high_res = max(1, base // self.grid_size_scale)
        low_res = max(1, high_res // 2)
        return high_res, low_res

    def wrap(self, material_id: int) -> bool:
        return self.material_texture_wrap[material_id]

    def model_init(self, module: spy.Module, input_type: SlangType):
        self.high_res_grids: list[Texture] = []
        self.low_res_grids: list[Texture] = []

        tile_idx = 0
        for material_id, udims in self.udim_mapping.items():
            high_res, low_res = self._grid_resolutions(material_id)
            for udim_id in udims:
                udim_idx = udim_id - 1001
                udim_stride = 100
                self.tile_indirection[material_id * udim_stride + udim_idx] = tile_idx
                tile_idx += 1

                self.high_res_grids.append(
                    Texture(module.device, self.num_features, high_res, self.wrap(material_id), self.optimizable)
                )
                self.low_res_grids.append(
                    Texture(module.device, self.num_features, low_res, self.wrap(material_id), self.optimizable)
                )

        self.tile_indirection_gpu = spy.Tensor.from_numpy(
            module.device, np.array(self.tile_indirection, dtype=np.int32)
        )
        self.material_texture_wrap_gpu = spy.Tensor.from_numpy(
            module.device, np.array(self.material_texture_wrap, dtype=np.int32)
        )
        self.material_texture_resolution_gpu = spy.Tensor.from_numpy(
            module.device, np.array(self.material_texture_resolution, dtype=np.int32)
        )

        decoder_input_width = 2 * self.num_features + 4 * self.num_pos_enc_waves + 1
        self.decoder = MLP(
            dtype=Real.half,
            num_inputs=decoder_input_width,
            num_outputs=self.num_latents,
            hidden_layers=self.decoder_hidden_layers,
            hidden_activations=self.decoder_activation,
        )
        self.decoder.set_parent(self)
        self.decoder.initialize(module, f'float[{decoder_input_width}]')

        self.module = module

    def generate_from_encoder(self, *args, **kwargs) -> None:
        """No-op: `train.py`'s `training_bsdf_encoding()` and the checkpoint
        machinery (`checkpoint/checkpoint.py`'s `encode_latent_texture_if_needed`)
        call this unconditionally on whatever `latent_texture.instances[0]` is,
        both at BsdfEncoding checkpoints and once at that phase's end, the same
        way they do for the dense `LatentTexture`. For the dense path this
        bakes the encoder's live output into the texture; for this grid path
        there is nothing meaningful to bake yet -- the grid stays at its
        zero/random init until `train.py`'s `training_ntc_warmstart()` phase
        actually trains it (a real regression, not a bake; see that
        function's docstring for why a simple copy can't work here). Any
        checkpoint saved during `BsdfEncoding` for this path will show
        whatever the untrained grid currently decodes to -- expected and
        harmless, since those checkpoints aren't the ones that matter for
        this path.
        """

    def make_optimizable(self) -> None:
        if self.optimizable:
            return
        self.optimizable = True
        for grid in self.high_res_grids + self.low_res_grids:
            grid.make_optimizable()

    def children(self) -> list[IModel]:
        return [self.decoder]

    def child_name(self, child: IModel) -> str | None:
        if child is self.decoder:
            return 'decoder'
        return None

    def resolve_input_type(self, module: spy.Module):
        return 'LatentTextureInput'

    @property
    def type_name(self) -> str:
        decoder_input_width = 2 * self.num_features + 4 * self.num_pos_enc_waves + 1
        return (
            f'NtcLatentTexture<{self.num_tiles}, {self.num_features}, '
            f'{self.num_pos_enc_waves}, {self.num_latents}, {decoder_input_width}, '
            f'{self.decoder.type_name}>'
        )

    def model_params(self):
        # Grid buffers are this model's own parameters; the decoder's
        # weights are collected separately through `children()`.
        return [grid.buffer for grid in self.high_res_grids + self.low_res_grids]

    def model_data(self):
        return {
            'tile_indirection': self.tile_indirection_gpu.storage.descriptor_handle_ro,
            'material_texture_wrap': self.material_texture_wrap_gpu.storage.descriptor_handle_ro,
            'material_texture_resolution': self.material_texture_resolution_gpu.storage.descriptor_handle_ro,
            'high_res_grids': [grid.get_this() for grid in self.high_res_grids],
            'low_res_grids': [grid.get_this() for grid in self.low_res_grids],
            'decoder': self.decoder.get_this(),
        }

    def save_checkpoint_images(
        self,
        desc: list[str],
        folder: Path,
    ) -> list[tuple[list[str], str]]:
        """Save grid textures as EXR files, plus the decoder's own image
        state (its weight-visualization EXR) via the default children
        traversal."""

        out_files: list[tuple[list[str], str]] = super().save_checkpoint_images(desc, folder)

        tile_idx = 0
        for material_id in self.material_ids:
            for udim_id in self.udim_mapping[material_id]:
                out_desc = [*desc, f'material{material_id}']
                if self.has_udims[material_id]:
                    out_desc += [f'udim{udim_id}']

                out_files += self.high_res_grids[tile_idx].save_checkpoint_images(
                    [*out_desc, 'grid_high_res'], folder
                )
                out_files += self.low_res_grids[tile_idx].save_checkpoint_images(
                    [*out_desc, 'grid_low_res'], folder
                )
                tile_idx += 1

        return out_files

    def load_checkpoint_images(
        self,
        desc: list[str],
        folder: Path,
    ):
        super().load_checkpoint_images(desc, folder)

        load_items: list[tuple[Texture, Path]] = []
        tile_idx = 0
        for material_id in self.material_ids:
            for udim_id in self.udim_mapping[material_id]:
                out_desc = [*desc, f'material{material_id}']
                if self.has_udims[material_id]:
                    out_desc += [f'udim{udim_id}']

                hi = self.high_res_grids[tile_idx]
                lo = self.low_res_grids[tile_idx]
                load_items.append((hi, hi.checkpoint_image_path([*out_desc, 'grid_high_res'], folder)))
                load_items.append((lo, lo.checkpoint_image_path([*out_desc, 'grid_low_res'], folder)))
                tile_idx += 1

        bitmaps = spy.Bitmap.read_multiple([path for _, path in load_items])
        for (tex, _), bitmap in zip(load_items, bitmaps, strict=True):
            tex.load_from_bitmap(bitmap)
