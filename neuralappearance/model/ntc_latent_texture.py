# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import slangpy as spy
from neuralnetworks import IModel, Real, SlangType, WeightQuantization

from .mlp import MLP
from .texture import Texture

if TYPE_CHECKING:
    from datagen import ReferenceMaterials


# Quantized grids use self-describing binary files because checkpoint images
# load before the parameter metadata.
_QUANTIZED_GRID_MAGIC = b'NTC4'
_QUANTIZED_GRID_VERSION = 1
_QUANTIZED_GRID_HEADER = struct.Struct('<4sBBIIff')  # magic, version, bits, resolution, num_channels, lo, hi


def _pack_quantized_grid(data: np.ndarray, lo: float, hi: float, bits: int) -> bytes:
    """Pack a quantized grid as a header followed by two indices per byte.

    The header records the quantization levels and range for dequantization on
    load; consecutive flat values occupy the low and high nibbles.
    """
    if not (1 <= bits <= 4):
        # Packing below hardcodes one nibble (4 bits) per index -- a larger
        # `bits` would silently truncate instead of raising, which is worse
        # than refusing outright.
        raise ValueError(f'Packed grid format only supports 1-4 bits, got {bits}.')

    resolution = data.shape[0]
    num_channels = data.shape[-1]
    levels = (1 << bits) - 1
    flat = data.astype(np.float32).ravel()

    span = hi - lo
    if span > 0.0:
        indices = np.clip(np.round((flat - lo) / span * levels), 0, levels).astype(np.uint8)
    else:
        indices = np.zeros_like(flat, dtype=np.uint8)

    if indices.size % 2 == 1:
        indices = np.concatenate([indices, np.zeros(1, dtype=np.uint8)])
    packed = (indices[0::2] & 0x0F) | ((indices[1::2] & 0x0F) << 4)

    header = _QUANTIZED_GRID_HEADER.pack(
        _QUANTIZED_GRID_MAGIC, _QUANTIZED_GRID_VERSION, bits, resolution, num_channels, lo, hi
    )
    return header + packed.astype(np.uint8).tobytes()


def _unpack_quantized_grid(buf: bytes) -> tuple[int, int, int, float, float, np.ndarray]:
    """Inverse of ``_pack_quantized_grid()``. Returns ``(bits, resolution,
    num_channels, lo, hi, flat_values)`` where ``flat_values`` is a flat
    float32 array of length ``resolution * resolution * num_channels``, in
    the same order ``Texture.to_numpy()``/``from_numpy()`` already use."""
    header = buf[: _QUANTIZED_GRID_HEADER.size]
    magic, version, bits, resolution, num_channels, lo, hi = _QUANTIZED_GRID_HEADER.unpack(header)
    if magic != _QUANTIZED_GRID_MAGIC:
        raise ValueError(f'Not a packed quantized-grid file (bad magic {magic!r}).')
    if version != _QUANTIZED_GRID_VERSION:
        raise ValueError(f'Unsupported packed-grid version {version}.')

    packed = np.frombuffer(buf, dtype=np.uint8, offset=_QUANTIZED_GRID_HEADER.size)
    indices = np.empty(packed.size * 2, dtype=np.uint8)
    indices[0::2] = packed & 0x0F
    indices[1::2] = (packed >> 4) & 0x0F

    n = resolution * resolution * num_channels
    indices = indices[:n]
    levels = (1 << bits) - 1
    if levels > 0:
        values = lo + indices.astype(np.float32) / levels * (hi - lo)
    else:
        values = np.full(n, lo, dtype=np.float32)
    return bits, resolution, num_channels, float(lo), float(hi), values


def decoder_activation_from_config(ntc_config: dict) -> str:
    """Decode-network activation for a `model.latents.ntc_grid` block.

    `activation_clamp: 3` selects the SDK's HardGELUClamp, whose output is
    bounded to [-3/16, 3]; unset keeps `decoder_activation` (HardGELU, which
    continues as the identity). HardGELUClamp's bound is fixed in its Slang
    source, so 3 is the only value accepted.
    """
    clamp = ntc_config.get('activation_clamp')
    if clamp is None:
        return ntc_config.get('decoder_activation', 'HardGELU')
    if clamp != 3:
        raise ValueError(
            f'model.latents.ntc_grid.activation_clamp must be 3 (HardGELUClamp\'s fixed bound) '
            f'or unset, got {clamp!r}.'
        )
    return 'HardGELUClamp'


def quant_noise_from_config(ntc_config: dict) -> dict | None:
    """Settings for quantization-aware direct optimization, or None when off.

    `model.latents.ntc_grid.quant_noise: {"enabled": true}` trains the grids
    under one quantization step of simulated uniform noise for the first
    `freeze_fraction` of direct optimization, then rounds and freezes them
    there, as the RTXNTC SDK does at 95% of its steps. `refresh_period`
    iterations between updates of the noise width to the grids' current
    range. Requires `quantize.enabled`, whose `bits` sets the step.
    """
    qat = ntc_config.get('quant_noise') or {}
    if not qat.get('enabled', False):
        return None
    if not ntc_config.get('quantize', {}).get('enabled', False):
        raise ValueError('model.latents.ntc_grid.quant_noise needs quantize.enabled: it sets the bit depth.')
    settings = {
        'bits': ntc_config['quantize'].get('bits', 4),
        'freeze_fraction': qat.get('freeze_fraction', 0.95),
        'refresh_period': qat.get('refresh_period', 5000),
    }
    if not (0.0 < settings['freeze_fraction'] < 1.0):
        raise ValueError(f'quant_noise.freeze_fraction must be in (0, 1), got {settings["freeze_fraction"]}.')
    if settings['refresh_period'] < 1:
        raise ValueError(f'quant_noise.refresh_period must be >= 1, got {settings["refresh_period"]}.')
    return settings


class NtcLatentTexture(IModel):
    """NTC-style high/low-resolution feature grids decoded by a small MLP.

    Implements the same latent-sampling interface as `LatentTexture`, with two
    grids per (tile, mip) instead of a dense per-texel buffer.
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
        num_mip_levels: int = 1,
        optimizable: bool = True,
        decoder_quantization: WeightQuantization | None = None,
    ) -> NtcLatentTexture:
        tex = cls(
            num_latents,
            grid_size_scale,
            num_features,
            num_pos_enc_waves,
            decoder_hidden_layers,
            decoder_activation,
            num_mip_levels,
            optimizable,
            decoder_quantization,
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
        num_mip_levels: int = 1,
        optimizable: bool = True,
        decoder_quantization: WeightQuantization | None = None,
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
        self.num_mip_levels = num_mip_levels
        self.decoder_quantization = decoder_quantization
        self.optimizable = optimizable

        self.num_tiles = 0
        self.material_ids: list[int] = []
        self.has_udims: dict[int, bool] = {}
        self.udim_mapping: dict[int, list[int]] = {}
        self.tile_indirection: list[int] = []
        self.material_texture_resolution: list[int] = []
        self.material_texture_wrap: list[bool] = []

        # Untrained grids can produce extreme latents and unstable path-tracing
        # weights, so checkpoint rendering waits for warm-start or optimization.
        self.ready_for_render = False

        # Set by `quantize_grids()`; drives `save_checkpoint_images()`'s
        # choice of packed-binary vs plain-EXR grid format. `quantize_bits`
        # and the two range lists are the per-grid metadata the packed
        # format's header needs (`_pack_quantized_grid()`/`_unpack_quantized_grid()`).
        self.quantized = False
        self.quantize_bits: int | None = None
        self.quantize_high_res_ranges: list[tuple[float, float]] = []
        self.quantize_low_res_ranges: list[tuple[float, float]] = []

        # Set by `freeze_grids()`. Splits this model's two parameter groups
        # apart: the grids stop being trainable while the decode MLP stays
        # trainable. See that method for why the quantize tail needs this.
        self.grids_frozen = False

        # Simulated quantization noise width per grid (high-res grids, then
        # low-res), read by the Slang side on every lookup. All zero except
        # while `set_quant_noise()` is in effect during quantization-aware
        # direct optimization; checkpoints never store it.
        self.quant_noise_widths: list[float] = []

    @property
    def decoder_input_width(self) -> int:
        """Width of the decode MLP's input vector: both grids' features, the
        positional encoding's four terms per octave, and the LOD scalar.

        Single source of truth on purpose -- `model_init` builds the MLP from
        this and `type_name` bakes it into the Slang type's template
        arguments, so two independent copies of the expression would be a
        silent Python/Slang layout mismatch the moment one of them changed.
        """
        return 2 * self.num_features + 4 * self.num_pos_enc_waves + 1

    def _grid_resolutions(self, material_id: int, mip_level: int) -> tuple[int, int]:
        # Mirrors dense `LatentTexture.resolution()`'s `base_resolution >>
        # mip_level` downscale, applied before the grid-size-scale division
        # (not after) so each mip's grid stays an integer fraction of that
        # mip's own color-texture resolution, same as mip 0.
        base = max(1, self.material_texture_resolution[material_id] >> mip_level)
        high_res = max(1, base // self.grid_size_scale)
        low_res = max(1, high_res // 2)
        return high_res, low_res

    def wrap(self, material_id: int) -> bool:
        return self.material_texture_wrap[material_id]

    def model_init(self, module: spy.Module, input_type: SlangType):
        self.high_res_grids: list[Texture] = []
        self.low_res_grids: list[Texture] = []

        # Each tile owns `num_mip_levels` consecutive (high_res, low_res) grid
        # pairs, flattened as `tile_idx * num_mip_levels + mip_level` -- the
        # same ABI dense `LatentTexture.model_init()` uses for its own
        # per-(tile, mip) `textures` list (`latent_texture.py:140-148`).
        tile_idx = 0
        for material_id, udims in self.udim_mapping.items():
            for udim_id in udims:
                udim_idx = udim_id - 1001
                udim_stride = 100
                self.tile_indirection[material_id * udim_stride + udim_idx] = tile_idx
                tile_idx += 1

                for mip_level in range(self.num_mip_levels):
                    high_res, low_res = self._grid_resolutions(material_id, mip_level)
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
        self.quant_noise_widths = [0.0] * (len(self.high_res_grids) + len(self.low_res_grids))
        self.quant_noise_widths_gpu = spy.Tensor.from_numpy(
            module.device, np.array(self.quant_noise_widths, dtype=np.float32)
        )

        self.decoder = MLP(
            dtype=Real.half,
            num_inputs=self.decoder_input_width,
            num_outputs=self.num_latents,
            hidden_layers=self.decoder_hidden_layers,
            hidden_activations=self.decoder_activation,
            quantization=self.decoder_quantization,
        )
        self.decoder.set_parent(self)
        self.decoder.initialize(module, f'float[{self.decoder_input_width}]')

        self.module = module

    def generate_from_encoder(self, *args, **kwargs) -> None:
        """Skip dense encoder baking; NTC grids learn during warm-start."""

    def make_optimizable(self) -> None:
        # `grids_frozen` has to survive this call. `InstancedComponent.
        # start_training()` invokes it unconditionally for any latent-texture
        # component, so without this guard, asking the training loop to train
        # the decode MLP would silently re-arm the grids as well -- exactly
        # what the quantize tail must not do.
        if self.grids_frozen:
            return
        if self.optimizable:
            return
        self.optimizable = True
        for grid in self.high_res_grids + self.low_res_grids:
            grid.make_optimizable()

    def freeze_grids(self) -> None:
        """Freeze grids while keeping the decode MLP trainable.

        ``model_params()`` excludes grid buffers from the optimizer. Clearing
        each texture's flag also detaches fetches and skips gradient atomics.
        """
        self.grids_frozen = True
        for grid in self.high_res_grids + self.low_res_grids:
            grid.optimizable = False

    def set_quant_noise(self, bits: int) -> list[tuple[float, float]]:
        """Turn on simulated quantization noise of one ``bits``-bit step.

        Each grid's step is its current range over ``2**bits - 1``, the step
        `quantize_grids()` would round to now. The grids are unbounded, so a
        caller that trains for long should call this again to follow the
        range. The widths live in a buffer that the model data points to, so
        an update needs no `write_to_buffer()`. Returns each grid's range,
        high-res grids first.
        """
        levels = (1 << bits) - 1
        ranges = []
        for grid in self.high_res_grids + self.low_res_grids:
            data = grid.to_numpy().astype(np.float32)
            ranges.append((float(data.min()), float(data.max())))
        self.quant_noise_widths = [(hi - lo) / levels for lo, hi in ranges]
        self.quant_noise_widths_gpu.copy_from_numpy(np.array(self.quant_noise_widths, dtype=np.float32))
        return ranges

    def clear_quant_noise(self) -> None:
        """Turn simulated quantization noise off again."""
        self.quant_noise_widths = [0.0] * len(self.quant_noise_widths)
        self.quant_noise_widths_gpu.copy_from_numpy(np.array(self.quant_noise_widths, dtype=np.float32))

    def quantize_grids(self, bits: int) -> list[dict]:
        """Round every grid to ``2**bits`` levels spanning its own range.

        Records ranges for packed checkpoints and returns per-grid quantization
        statistics. Freezing and decoder retraining are controlled by the caller.
        """
        levels = (1 << bits) - 1  # e.g. 15 for 4 bits, matching RTXNTC's own step = 1/15
        stats = []
        high_res_ranges: list[tuple[float, float]] = [(0.0, 0.0)] * len(self.high_res_grids)
        low_res_ranges: list[tuple[float, float]] = [(0.0, 0.0)] * len(self.low_res_grids)
        for kind, grids, ranges_out in (
            ('high_res', self.high_res_grids, high_res_ranges),
            ('low_res', self.low_res_grids, low_res_ranges),
        ):
            for grid_idx, grid in enumerate(grids):
                # `grids` is flattened as `tile_idx * num_mip_levels +
                # mip_level` (see `model_init`) -- decode back to the pair
                # callers actually want to log.
                tile_idx, mip_level = divmod(grid_idx, self.num_mip_levels)
                data = grid.to_numpy().astype(np.float32)
                lo, hi = float(data.min()), float(data.max())
                ranges_out[grid_idx] = (lo, hi)
                span = hi - lo
                if span <= 0.0:
                    # Degenerate (constant) grid -- nothing to quantize, and
                    # dividing by span below would be a divide-by-zero.
                    stats.append(
                        {'tile_idx': tile_idx, 'mip_level': mip_level, 'kind': kind,
                         'min': lo, 'max': hi, 'step': 0.0, 'rms_error': 0.0}
                    )
                    continue
                normalized = (data - lo) / span  # -> [0, 1]
                quantized = np.round(normalized * levels) / levels
                error = quantized * span - (data - lo)
                rms_error = float(np.sqrt(np.mean(error * error)))
                # Texture.from_numpy() writes straight into a 'half' buffer
                # with no dtype conversion (see Texture.__init__'s own
                # explicit dtype=np.float16 zero-fill) -- float32 here would
                # silently misinterpret the bytes rather than convert them.
                grid.from_numpy((quantized * span + lo).astype(np.float16))
                stats.append(
                    {'tile_idx': tile_idx, 'mip_level': mip_level, 'kind': kind,
                     'min': lo, 'max': hi, 'step': span / levels, 'rms_error': rms_error}
                )

        self.quantized = True
        self.quantize_bits = bits
        self.quantize_high_res_ranges = high_res_ranges
        self.quantize_low_res_ranges = low_res_ranges
        return stats

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
        return (
            f'NtcLatentTexture<{self.num_tiles}, {self.num_mip_levels}, {self.num_features}, '
            f'{self.num_pos_enc_waves}, {self.num_latents}, {self.decoder_input_width}, '
            f'{self.decoder.type_name}>'
        )

    def model_params(self):
        # Grid buffers are this model's own parameters; the decoder's
        # weights are collected separately through `children()`. Reporting
        # none of them is how `freeze_grids()` keeps the grids out of the
        # optimizer while the decode MLP keeps training -- see that method.
        if self.grids_frozen:
            return []
        return [grid.buffer for grid in self.high_res_grids + self.low_res_grids]

    def model_data(self):
        return {
            'tile_indirection': self.tile_indirection_gpu.storage.descriptor_handle_ro,
            'material_texture_wrap': self.material_texture_wrap_gpu.storage.descriptor_handle_ro,
            'material_texture_resolution': self.material_texture_resolution_gpu.storage.descriptor_handle_ro,
            'high_res_grids': [grid.get_this() for grid in self.high_res_grids],
            'low_res_grids': [grid.get_this() for grid in self.low_res_grids],
            'quant_noise_widths': self.quant_noise_widths_gpu.storage.descriptor_handle_ro,
            'decoder': self.decoder.get_this(),
        }

    def _quantized_grid_path(self, desc: list[str], folder: Path) -> Path:
        return folder / ('.'.join(desc) + '.bin')

    def _save_grid(
        self,
        grid: Texture,
        desc: list[str],
        folder: Path,
        quantize_range: tuple[float, float] | None,
    ) -> tuple[list[str], str]:
        if self.quantized:
            assert quantize_range is not None
            lo, hi = quantize_range
            path = self._quantized_grid_path(desc, folder)
            path.write_bytes(_pack_quantized_grid(grid.to_numpy(), lo, hi, self.quantize_bits))
            return (desc, path.name)
        else:
            return grid.save_checkpoint_images(desc, folder)[0]

    def save_checkpoint_images(
        self,
        desc: list[str],
        folder: Path,
    ) -> list[tuple[list[str], str]]:
        """Save grid textures (packed 4-bit binary once ``self.quantized``,
        plain EXR otherwise), plus the decoder's own image state (its
        weight-visualization EXR) via the default children traversal."""

        out_files: list[tuple[list[str], str]] = super().save_checkpoint_images(desc, folder)

        tile_idx = 0
        for material_id in self.material_ids:
            for udim_id in self.udim_mapping[material_id]:
                out_desc = [*desc, f'material{material_id}']
                if self.has_udims[material_id]:
                    out_desc += [f'udim{udim_id}']

                for mip_level in range(self.num_mip_levels):
                    grid_idx = tile_idx * self.num_mip_levels + mip_level
                    # mip 0 keeps the exact filenames this format already
                    # used before mips existed, so single-mip checkpoints
                    # (every config in the repo today) stay byte-identical
                    # on disk; mip > 0 is new, so it gets a new suffix.
                    mip_desc = out_desc if mip_level == 0 else [*out_desc, f'mip{mip_level}']

                    out_files.append(self._save_grid(
                        self.high_res_grids[grid_idx], [*mip_desc, 'grid_high_res'], folder,
                        self.quantize_high_res_ranges[grid_idx] if self.quantized else None,
                    ))
                    out_files.append(self._save_grid(
                        self.low_res_grids[grid_idx], [*mip_desc, 'grid_low_res'], folder,
                        self.quantize_low_res_ranges[grid_idx] if self.quantized else None,
                    ))
                tile_idx += 1

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

        exr_load_items: list[tuple[Texture, Path]] = []
        high_res_ranges: list[tuple[float, float]] = [(0.0, 0.0)] * len(self.high_res_grids)
        low_res_ranges: list[tuple[float, float]] = [(0.0, 0.0)] * len(self.low_res_grids)
        packed_count = 0
        bits_seen: int | None = None

        tile_idx = 0
        for material_id in self.material_ids:
            for udim_id in self.udim_mapping[material_id]:
                out_desc = [*desc, f'material{material_id}']
                if self.has_udims[material_id]:
                    out_desc += [f'udim{udim_id}']

                for mip_level in range(self.num_mip_levels):
                    grid_idx = tile_idx * self.num_mip_levels + mip_level
                    # Matches save_checkpoint_images(): mip 0 has no suffix.
                    mip_desc = out_desc if mip_level == 0 else [*out_desc, f'mip{mip_level}']

                    for grid, kind, ranges_out in (
                        (self.high_res_grids[grid_idx], 'grid_high_res', high_res_ranges),
                        (self.low_res_grids[grid_idx], 'grid_low_res', low_res_ranges),
                    ):
                        grid_desc = [*mip_desc, kind]
                        bin_path = self._quantized_grid_path(grid_desc, folder)
                        if bin_path.is_file():
                            bits, resolution, num_channels, lo, hi, values = _unpack_quantized_grid(
                                bin_path.read_bytes()
                            )
                            assert resolution == grid.resolution and num_channels == grid.num_channels, (
                                f'Packed grid shape mismatch at {bin_path}: file has '
                                f'resolution={resolution} num_channels={num_channels}, expected '
                                f'resolution={grid.resolution} num_channels={grid.num_channels}.'
                            )
                            grid.from_numpy(values.astype(np.float16))
                            ranges_out[grid_idx] = (lo, hi)
                            packed_count += 1
                            bits_seen = bits
                        else:
                            exr_load_items.append((grid, grid.checkpoint_image_path(grid_desc, folder)))

                tile_idx += 1

        if exr_load_items:
            bitmaps = spy.Bitmap.read_multiple([path for _, path in exr_load_items])
            for (tex, _), bitmap in zip(exr_load_items, bitmaps, strict=True):
                tex.load_from_bitmap(bitmap)

        # All-or-nothing, enforced rather than assumed. A checkpoint with SOME
        # grids packed and some left as EXR (an interrupted save, a partially
        # copied directory) would otherwise set `self.quantized = True` while
        # the EXR-loaded grids kept their default range of (0.0, 0.0). The next
        # `save_checkpoint_images()` would then pack those grids with a zero
        # span, and `_pack_quantized_grid()` maps a zero span to all-zero
        # indices -- silently destroying real trained data on disk, with no
        # error and no obvious symptom until the render came back wrong.
        #
        # Failing here instead costs a clear message and loses nothing: the
        # files on disk are still intact at this point.
        total_grids = len(self.high_res_grids) + len(self.low_res_grids)
        if packed_count and packed_count != total_grids:
            raise ValueError(
                f'Mixed checkpoint in {folder}: {packed_count} of {total_grids} '
                'grids are in the packed 4-bit format and the rest are EXR. '
                'Saving from this state would pack the EXR grids with a zero '
                'range and destroy them. Restore a complete checkpoint.'
            )

        self.quantized = packed_count > 0
        self.quantize_bits = bits_seen
        self.quantize_high_res_ranges = high_res_ranges
        self.quantize_low_res_ranges = low_res_ranges

        if self.quantized:
            # A grid that was loaded from the packed format is already
            # rounded to its quantization lattice, and continuous optimizer
            # updates would walk it straight back off that lattice -- the
            # values would no longer survive a round trip through the same
            # packed format they were just read from. Freezing here makes
            # that failure impossible rather than merely unlikely; the decode
            # MLP stays trainable, which is the half that is supposed to
            # adapt to quantization anyway (see `freeze_grids`).
            self.freeze_grids()

        # Legacy default for checkpoints without readiness metadata. New
        # checkpoints override this in load_checkpoint_params(), which the
        # model loader calls immediately after loading images.
        self.ready_for_render = True
