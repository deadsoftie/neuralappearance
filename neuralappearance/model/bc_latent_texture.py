# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import slangpy as spy
from neuralnetworks import IModel, SlangType

from .texture import Texture

if TYPE_CHECKING:
    from datagen import ReferenceMaterials


# BC6H mode 10 constants, mirrored from bc6h.slang because Slang constants
# are not directly accessible from Python.
_BLOCK_SIZE = 4
_CHANNELS_PER_SET = 3          # BC6H is RGB-only
_ENDPOINT_CHANNELS = 6         # two endpoints, 3 channels each
_PRECISION = 10
_SIGNED_SCALE = 512.0          # |endpoint| range for BC6H_SF16
_UNSIGNED_SCALE = 1023.0       # endpoint range for BC6H_UF16

_SUPPORTED_FORMATS = ('bc6h_sfloat', 'bc6h_ufloat')
_SUPPORTED_BLOCK_MODES = (10,)


def endpoint_from_value(value: float, is_signed: bool) -> float:
    """Approximate the normalized endpoint parameter for a decoded value.

    This inverse of half-bit reinterpretation, finish_unquantize, and unquantize
    is used only to initialize endpoints in latent-value units.
    """
    bits = int(np.float16(value).view(np.uint16))
    magnitude = bits & 0x7FFF
    negative = bool(bits & 0x8000)

    if is_signed:
        intermediate = min(0x7FFF, (magnitude << 5) // 31)
    else:
        intermediate = min(0xFFFF, (magnitude << 6) // 31)
    # `unquantize` is 64 * |x| + 32 at precision 10 (verified exhaustively in
    # Stage 1), so the inverse is (intermediate - 32) / 64.
    units = max(0.0, (intermediate - 32) / 64.0)

    scale = _SIGNED_SCALE if is_signed else _UNSIGNED_SCALE
    normalized = min(units, scale - 1.0) / scale
    if negative and is_signed:
        return -normalized
    return normalized


class BcLatentTexture(IModel):
    """BC6H-shaped feature sets sampled directly as latent channels.

    Implements the common latent-sampling interface without a decode MLP.
    """

    @classmethod
    def from_reference_materials(
        cls,
        reference_materials: ReferenceMaterials,
        num_latents: int,
        feature_resolutions: list[int],
        bc_format: str = 'bc6h_sfloat',
        block_mode: int = 10,
        num_mip_levels: int = 1,
        optimizable: bool = True,
        init_amplitude: float = 0.1,
    ) -> BcLatentTexture:
        tex = cls(
            num_latents,
            feature_resolutions,
            bc_format,
            block_mode,
            num_mip_levels,
            optimizable,
            init_amplitude,
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
        feature_resolutions: list[int],
        bc_format: str = 'bc6h_sfloat',
        block_mode: int = 10,
        num_mip_levels: int = 1,
        optimizable: bool = True,
        init_amplitude: float = 0.1,
    ):
        super().__init__()

        if bc_format not in _SUPPORTED_FORMATS:
            raise ValueError(
                f'model.latents.bc_grid.bc_format must be one of '
                f'{_SUPPORTED_FORMATS}, got {bc_format!r}.'
            )
        if block_mode not in _SUPPORTED_BLOCK_MODES:
            raise ValueError(
                f'model.latents.bc_grid.block_mode must be one of '
                f'{_SUPPORTED_BLOCK_MODES}, got {block_mode}. Mode 10 is the '
                'only mode implemented by bc6h.slang.'
            )
        if not feature_resolutions:
            raise ValueError('model.latents.bc_grid.feature_resolutions is empty.')
        for set_idx, resolution in enumerate(feature_resolutions):
            if resolution % _BLOCK_SIZE != 0:
                raise ValueError(
                    'model.latents.bc_grid.feature_resolutions must contain only '
                    f'resolutions divisible by {_BLOCK_SIZE} (the BC6H block size); '
                    f'got {resolution} at index {set_idx}.'
                )

        expected_latents = _CHANNELS_PER_SET * len(feature_resolutions)
        if num_latents != expected_latents:
            # Not a warning: a mismatch here is a silent Python/Slang layout
            # divergence, since `NumLatents` is baked into the Slang type's
            # template arguments while the texture list length comes from this
            # list. `run_sweep.build_config` derives one from the other for
            # exactly this reason.
            raise ValueError(
                f'model.latents.num_channels must be {expected_latents} for '
                f'{len(feature_resolutions)} BC6H sets (3 channels each), got '
                f'{num_latents}. BC6H is RGB-only, so latent width is a '
                'consequence of the pyramid depth rather than an independent '
                'knob.'
            )

        self.num_latents = num_latents
        # Alias expected by callers written against the dense `LatentTexture`
        # (e.g. `rendering/neural_material.py`'s fallback sampler/aux type
        # strings) that only care about the decoded channel count.
        self.num_channels = num_latents
        self.feature_resolutions = list(feature_resolutions)
        self.bc_format = bc_format
        self.block_mode = block_mode
        self.num_mip_levels = num_mip_levels
        self.optimizable = optimizable
        self.init_amplitude = init_amplitude

        self.is_signed = bc_format == 'bc6h_sfloat'
        self.endpoint_scale = _SIGNED_SCALE if self.is_signed else _UNSIGNED_SCALE

        self.num_tiles = 0
        self.material_ids: list[int] = []
        self.has_udims: dict[int, bool] = {}
        self.udim_mapping: dict[int, list[int]] = {}
        self.tile_indirection: list[int] = []
        self.material_texture_resolution: list[int] = []
        self.material_texture_wrap: list[bool] = []

        # Checkpoint rendering waits until warm-start or direct optimization
        # replaces the untrained initial textures.
        self.ready_for_render = False

    # ------------------------------------------------------------------ shapes

    @property
    def num_sets(self) -> int:
        return len(self.feature_resolutions)

    def set_resolution(self, set_idx: int, mip_level: int) -> int:
        """Return an absolute feature resolution, floored at one BC6H block."""
        return max(_BLOCK_SIZE, self.feature_resolutions[set_idx] >> mip_level)

    def block_resolution(self, set_idx: int, mip_level: int) -> int:
        resolution = self.set_resolution(set_idx, mip_level)
        return max(1, -(-resolution // _BLOCK_SIZE))  # ceil

    def wrap(self, material_id: int) -> bool:
        return self.material_texture_wrap[material_id]

    def get_texture_idx(self, set_idx: int, tile_idx: int, mip_level: int) -> int:
        """Flat index into `endpoint_textures`/`position_textures`.

        Must match `bc_latent_texture.slang`'s `get_grid_idx` exactly:
        set-major, then tile, then mip.
        """
        return (set_idx * (self.num_tiles * self.num_mip_levels)
                + tile_idx * self.num_mip_levels + mip_level)

    # ------------------------------------------------------------------ IModel

    def _tiles(self) -> list[tuple[int, int]]:
        """Yield tiles in the order used by allocation and checkpoint I/O."""
        return [(material_id, udim_id)
                for material_id in self.material_ids
                for udim_id in self.udim_mapping[material_id]]

    def model_init(self, module: spy.Module, input_type: SlangType):
        self.endpoint_textures: list[Texture] = []
        self.position_textures: list[Texture] = []

        tiles = self._tiles()

        # Built here rather than in `from_reference_materials` because it has to
        # agree with the texture ordering below.
        udim_stride = 100
        for tile_idx, (material_id, udim_id) in enumerate(tiles):
            udim_idx = udim_id - 1001
            self.tile_indirection[material_id * udim_stride + udim_idx] = tile_idx

        # SET-MAJOR, then tile, then mip. `get_texture_idx` and the Slang
        # `get_grid_idx` both assume this nesting; changing it here without
        # changing both is a silent mis-indexing.
        for set_idx in range(self.num_sets):
            for material_id, _udim_id in tiles:
                for mip_level in range(self.num_mip_levels):
                    wrap = self.wrap(material_id)
                    self.endpoint_textures.append(Texture(
                        module.device, _ENDPOINT_CHANNELS,
                        self.block_resolution(set_idx, mip_level),
                        wrap, self.optimizable,
                    ))
                    self.position_textures.append(Texture(
                        module.device, 1,
                        self.set_resolution(set_idx, mip_level),
                        wrap, self.optimizable,
                    ))

        self._initialize_values()

        self.tile_indirection_gpu = spy.Tensor.from_numpy(
            module.device, np.array(self.tile_indirection, dtype=np.int32)
        )
        self.material_texture_wrap_gpu = spy.Tensor.from_numpy(
            module.device, np.array(self.material_texture_wrap, dtype=np.int32)
        )
        self.material_texture_resolution_gpu = spy.Tensor.from_numpy(
            module.device, np.array(self.material_texture_resolution, dtype=np.int32)
        )
        self.module = module

    def _initialize_values(self) -> None:
        """Seed the endpoints to bracket +-`init_amplitude` and the positions to
        the segment midpoint.

        Not zero. With both endpoints at zero the decoded field is zero
        everywhere, and the backward pass's slope at a zero intermediate is
        `2**-24`, so the warm start would begin with gradients around 1e-8 and
        crawl. Bracketing a real latent-sized amplitude puts the initial
        intermediate in a normal half exponent bin where the slope is useful.

        Also well inside the saturation clamp, which matters: at saturation
        `d(unquantize)/d(endpoint)` is exactly zero, so an endpoint initialised
        into the clamp never recovers (Stage 1 asserts that dead zone).
        """
        amplitude = endpoint_from_value(self.init_amplitude, self.is_signed)
        if self.is_signed:
            low, high = -amplitude, amplitude
        else:
            # Unsigned cannot hold a negative endpoint, so bracket [0, 2a]
            # instead of [-a, a] and keep the same span.
            low, high = 0.0, min(2.0 * amplitude, 1.0)

        for texture in self.endpoint_textures:
            values = np.empty((texture.resolution, texture.resolution,
                               _ENDPOINT_CHANNELS), dtype=np.float16)
            values[..., 0:3] = np.float16(low)
            values[..., 3:6] = np.float16(high)
            texture.from_numpy(values)

        for texture in self.position_textures:
            texture.from_numpy(np.full(
                (texture.resolution, texture.resolution, 1), 0.5, dtype=np.float16))

    def generate_from_encoder(self, *args, **kwargs) -> None:
        """No-op, same contract as `NtcLatentTexture.generate_from_encoder`.

        `train.py`'s `training_bsdf_encoding()` and the checkpoint machinery
        call this unconditionally on whatever `latent_texture.instances[0]` is.
        For the dense path it bakes the encoder's live output into the texture;
        here there is nothing meaningful to bake, because the BC6H
        parameterisation cannot represent an arbitrary field by assignment. The
        warm-start phase performs a real regression against the encoder's field
        instead. Any checkpoint saved during `BsdfEncoding` therefore shows the
        init values, which is expected and harmless.
        """

    def make_optimizable(self) -> None:
        if self.optimizable:
            return
        self.optimizable = True
        for texture in self.endpoint_textures + self.position_textures:
            texture.make_optimizable()

    def children(self) -> list[IModel]:
        # No decode MLP: the sampled BC6H values ARE the latent. This emptiness
        # is the architecture's whole point, not an omission.
        return []

    def resolve_input_type(self, module: spy.Module):
        return 'LatentTextureInput'

    @property
    def type_name(self) -> str:
        return (
            f'BcLatentTexture<{self.num_sets}, {self.num_tiles}, '
            f'{self.num_mip_levels}, {self.num_latents}>'
        )

    def model_params(self):
        return [t.buffer for t in self.endpoint_textures + self.position_textures]

    def model_data(self):
        return {
            'tile_indirection': self.tile_indirection_gpu.storage.descriptor_handle_ro,
            'material_texture_wrap': self.material_texture_wrap_gpu.storage.descriptor_handle_ro,
            'material_texture_resolution': self.material_texture_resolution_gpu.storage.descriptor_handle_ro,
            'endpoints': [t.get_this() for t in self.endpoint_textures],
            'positions': [t.get_this() for t in self.position_textures],
            'endpoint_scale': self.endpoint_scale,
            'is_signed': 1 if self.is_signed else 0,
        }

    # -------------------------------------------------------------- checkpoints

    def _descs(self):
        """Yield `(texture_index, endpoint_desc_tail, position_desc_tail)` for
        every texture, in the flat storage order.

        One generator shared by save and load so the two cannot disagree about
        filenames -- the failure mode that silently cost `NtcLatentTexture` a
        multi-mip run (every mip wrote to one filename, last one won).
        """
        tiles = self._tiles()

        for set_idx in range(self.num_sets):
            for tile, (material_id, udim_id) in enumerate(tiles):
                for mip_level in range(self.num_mip_levels):
                    tail = [f'material{material_id}']
                    if self.has_udims[material_id]:
                        tail += [f'udim{udim_id}']
                    # Mip 0 carries no suffix, matching the sibling latent
                    # types' convention so single-mip checkpoints have stable
                    # names.
                    if mip_level != 0:
                        tail += [f'mip{mip_level}']
                    tail += [f'set{set_idx}']
                    idx = self.get_texture_idx(set_idx, tile, mip_level)
                    yield idx, [*tail, 'endpoints'], [*tail, 'positions']

    def save_checkpoint_images(
        self,
        desc: list[str],
        folder: Path,
    ) -> list[tuple[list[str], str]]:
        out_files: list[tuple[list[str], str]] = super().save_checkpoint_images(desc, folder)
        for idx, endpoint_tail, position_tail in self._descs():
            out_files += self.endpoint_textures[idx].save_checkpoint_images(
                [*desc, *endpoint_tail], folder)
            out_files += self.position_textures[idx].save_checkpoint_images(
                [*desc, *position_tail], folder)
        return out_files

    def save_checkpoint_params(self, desc: list[str]) -> list[tuple[list[str], dict]]:
        params = super().save_checkpoint_params(desc)
        params.append((desc, {'ready_for_render': self.ready_for_render}))
        return params

    def load_checkpoint_params(self, params: dict) -> None:
        super().load_checkpoint_params(params)
        # Missing metadata is ambiguous, so preserve the historical behavior
        # for trained checkpoints written before readiness was persisted.
        self.ready_for_render = bool(params.get('ready_for_render', True))

    def load_checkpoint_images(
        self,
        desc: list[str],
        folder: Path,
    ):
        super().load_checkpoint_images(desc, folder)

        load_items: list[tuple[Texture, Path]] = []
        for idx, endpoint_tail, position_tail in self._descs():
            for texture, tail in ((self.endpoint_textures[idx], endpoint_tail),
                                  (self.position_textures[idx], position_tail)):
                load_items.append((texture, texture.checkpoint_image_path(
                    [*desc, *tail], folder)))

        missing = [str(path) for _, path in load_items if not path.is_file()]
        if missing:
            # Fail with the list rather than letting `read_multiple` raise on
            # the first one, so a partial checkpoint is diagnosable in one go.
            raise FileNotFoundError(
                f'Incomplete bc_grid checkpoint in {folder}: '
                f'{len(missing)} of {len(load_items)} feature textures are '
                f'missing, starting with {missing[0]}.'
            )

        bitmaps = spy.Bitmap.read_multiple([path for _, path in load_items])
        for (texture, _), bitmap in zip(load_items, bitmaps, strict=True):
            texture.load_from_bitmap(bitmap)

        # Legacy default for checkpoints without readiness metadata. New
        # checkpoints override this in load_checkpoint_params(), which the
        # model loader calls immediately after loading images.
        self.ready_for_render = True
