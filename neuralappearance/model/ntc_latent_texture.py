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


# Packed-4-bit grid checkpoint format: a self-describing binary file per
# grid, written only once `NtcLatentTexture.quantized` is True (see
# `quantize_grids()`). Self-describing (carries its own bits/range) rather
# than relying on `model.json`'s `_params` because `NeuralModel.load_checkpoint()`
# loads checkpoint images before checkpoint params (`neural_material_model.py`'s
# `load_checkpoint()`), so this file can't assume range metadata is available
# yet at image-load time.
_QUANTIZED_GRID_MAGIC = b'NTC4'
_QUANTIZED_GRID_VERSION = 1
_QUANTIZED_GRID_HEADER = struct.Struct('<4sBBIIff')  # magic, version, bits, resolution, num_channels, lo, hi


def _pack_quantized_grid(data: np.ndarray, lo: float, hi: float, bits: int) -> bytes:
    """Pack an already-quantized grid (values at ``2**bits`` levels spanning
    ``[lo, hi]``, as produced by ``quantize_grids()``) into a self-describing
    binary blob: a fixed header followed by two 4-bit indices per byte.

    Deliberately diverges from RTXNTC's literal on-disk layout (4 features
    packed into one uint16 per grid cell): this packs 2 consecutive flat-
    buffer values per byte instead. Same 4-bits/value storage cost and
    resulting file size; the byte order only matters if something reads it
    directly on the GPU, which this project defers (dequantize-at-load,
    QUANTIZATION_PLAN.md Decision 3).
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


class NtcLatentTexture(IModel):
    """NTC-style compressed latent texture: a pair of bilinearly-sampled
    feature grids per (tile, mip) -- high-res and low-res, both at a
    resolution lower than the output texture -- decoded into the shared
    ``num_latents``-channel code by a small fixed MLP, instead of one dense
    per-texel buffer. See ``model/ntc_latent_texture.slang`` and rtcnam/PLAN.md for
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

        # Until `training_ntc_warmstart()` runs, the grid is still at its
        # zero init and the decode MLP is untrained, so this texture decodes
        # to values `BsdfDecoder` was never trained for. Rendering that is
        # not merely meaningless (see `generate_from_encoder` below) -- it
        # hangs: those latents drive `BsdfDecoder` (ScaledSigmoid, scale
        # 1e5) to extreme values, and the resulting importance weights are
        # huge but finite, so they slip past the path tracer's isinf/isnan
        # guard and stall its Russian-roulette termination indefinitely
        # (observed: GPU pegged at 100% with the host idle on a fence, for
        # 30+ minutes, at the BsdfEncoding end-of-phase checkpoint).
        # `checkpoint.py` skips the neural render while this is False.
        # Bounding the decoded latents hard enough to survive that render
        # was tried first and is NOT a viable alternative: it saturates the
        # representation and wrecks quality (see ntc_latent_texture.slang).
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
        """Make the grids non-trainable while leaving the decode MLP trainable.

        The quantize tail needs exactly this split. RTXNTC's recipe (and
        rtcnam/PLAN.md's own description of it) is "freeze the grid, fine-tune
        the network": the decode MLP is the thing that reads the rounded grid
        values directly, so it is the thing best placed to absorb the
        rounding error. Without this method the two could not be separated --
        both are parameters of this one model, so marking the component
        trainable trained the grids too, and leaving it untrained froze the
        decode MLP along with them.

        Two mechanisms, both needed:

        - ``model_params()`` stops reporting the grid buffers, so the
          optimizer never sees them. This is the load-bearing one: merely
          clearing ``Texture.optimizable`` would not be enough, because
          ``Texture.get_this()`` aliases ``buffer_grads`` onto the data
          buffer for non-optimizable textures (SlangPy cannot bind null), so
          an optimizer still holding those params would read grid *data* as
          if it were gradients and destroy the grid.
        - Each grid's ``optimizable`` flag is cleared, which makes
          ``texture.slang``'s ``get_bilinear`` call ``detach()`` and skip the
          atomic gradient scatter entirely. Not required for correctness
          given the above, but it removes the wasted atomics.
        """
        self.grids_frozen = True
        for grid in self.high_res_grids + self.low_res_grids:
            grid.optimizable = False

    def quantize_grids(self, bits: int) -> list[dict]:
        """4-bit (or ``bits``-bit) quantize every grid in place: round each
        feature value to the nearest of ``2**bits`` evenly spaced levels
        spanning that grid's own observed [min, max] range.

        rtcnam/QUANTIZATION_PLAN.md, Milestone A ("fast gate"): a one-off,
        training-unaware round -- no dither, no optimizer involvement, no
        change to ``ntc_latent_texture.slang``'s sampling code at all. The
        caller (`train.py`'s `training_ntc_quantize()`) is responsible for
        then excluding these grids from the optimizer (they are already
        ``status == 'Done'`` by the time this runs, post-diropt) and running
        a short decoder-only refinetune so `BsdfDecoder` adapts to the
        rounding error -- this method only rounds; it does not freeze or
        retrain anything itself.

        Unlike RTXNTC's own scheme (additive-dither QAT against a fixed
        [0, 1] range, Milestone B in the same plan), this uses each grid's
        *own* range rather than [0, 1] -- this implementation's grids are not
        squashed to any fixed interval (see ntc_latent_texture.slang's long
        comment on why no output bound is applied), so a fixed [0, 1] would
        either clip real values or waste most of the 16 levels on an interval
        the grid never uses. Levels still span the data, so the quantization
        step comparison against Milestone B remains meaningful despite the
        different range convention.

        Returns one stats dict per grid (``tile_idx``, ``mip_level``,
        ``kind`` -- 'high_res' or 'low_res' --, ``min``, ``max``, ``step``,
        ``rms_error``) for the caller to log. Also sets
        ``self.quantized``/``self.quantize_bits``/
        ``self.quantize_high_res_ranges``/``self.quantize_low_res_ranges``,
        which ``save_checkpoint_images()`` reads to decide whether (and at
        what range) to pack the grid on disk -- that's the only persistence
        this method causes; the checkpoint *file* itself is untouched here
        (see QUANTIZATION_PLAN.md's "report first, change the file format
        second").
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

    def load_checkpoint_images(
        self,
        desc: list[str],
        folder: Path,
    ):
        super().load_checkpoint_images(desc, folder)

        exr_load_items: list[tuple[Texture, Path]] = []
        high_res_ranges: list[tuple[float, float]] = [(0.0, 0.0)] * len(self.high_res_grids)
        low_res_ranges: list[tuple[float, float]] = [(0.0, 0.0)] * len(self.low_res_grids)
        quantized_any = False
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
                            quantized_any = True
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
        packed_count = sum(
            1 for lo, hi in high_res_ranges + low_res_ranges if (lo, hi) != (0.0, 0.0)
        )
        if quantized_any and packed_count != total_grids:
            raise ValueError(
                f'Mixed checkpoint in {folder}: {packed_count} of {total_grids} '
                'grids are in the packed 4-bit format and the rest are EXR. '
                'Saving from this state would pack the EXR grids with a zero '
                'range and destroy them. Restore a complete checkpoint.'
            )

        self.quantized = quantized_any
        self.quantize_bits = bits_seen
        self.quantize_high_res_ranges = high_res_ranges
        self.quantize_low_res_ranges = low_res_ranges

        if quantized_any:
            # A grid that was loaded from the packed format is already
            # rounded to its quantization lattice, and continuous optimizer
            # updates would walk it straight back off that lattice -- the
            # values would no longer survive a round trip through the same
            # packed format they were just read from. Freezing here makes
            # that failure impossible rather than merely unlikely; the decode
            # MLP stays trainable, which is the half that is supposed to
            # adapt to quantization anyway (see `freeze_grids`).
            self.freeze_grids()

        # A checkpoint on disk holds real trained grid values, so anything
        # loading one (the visualizer, render_headless, a repeat training
        # job) can render it.
        self.ready_for_render = True
