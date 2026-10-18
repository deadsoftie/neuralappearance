# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import slangpy as spy
from neuralnetworks import IModel, SlangType

from .bc6h_pack import build_blocks
from .bc_latent_texture import BcLatentTexture


class BcLatentTextureHardware(IModel):
    """Render-time-only hardware counterpart to `BcLatentTexture`.
    rtbcnam/PLAN.md Stage 5.

    Never trained: built once, from an already-trained `BcLatentTexture`
    (`from_bc_latent_texture()`), by packing its current endpoint/position
    values into real BC6H block bytes (`model/bc6h_pack.py`, the same
    packer `rtbcnam/harness/export_bc6h.py` uses and verifies bit-exact
    against the real GPU, `rtbcnam/results/STAGE4.md`) and uploading them as
    real compressed `Texture2D`s. `model/bc_latent_texture_hardware.slang`
    samples them with the fixed-function texture unit via `SampleGrad`,
    which is what drives real hardware anisotropic filtering -- see that
    file's own header comment for why `SampleGrad` rather than `Sample` or
    `SampleLevel`.

    Implements the same `ILatentTexture` contract as `BcLatentTexture`,
    `LatentTexture` and `NtcLatentTexture`, so every existing consumer --
    `BsdfDecoder`, `Sampler`, `AuxDecoder`, the renderer, the visualizer --
    works against it unchanged. There is deliberately no path from this
    class back to a trainable one and no checkpoint save/load of its own:
    it is a one-way, render-only VIEW of a `BcLatentTexture` checkpoint that
    already exists, not a parallel model.
    """

    @classmethod
    def from_bc_latent_texture(
        cls, source: BcLatentTexture, max_anisotropy: int = 16,
    ) -> BcLatentTextureHardware:
        """Snapshot `source`'s CURRENT values. `source` must already be
        initialized (`model_init` run) and trained -- this reads
        `endpoint_textures`/`position_textures` directly, the same buffers
        `rtbcnam/harness/export_bc6h.py` reads from a loaded checkpoint.

        `max_anisotropy` is exposed here (rather than hardcoded) so Stage 5's
        own open question -- whether a quality gap against the software path
        is real anisotropic filtering behaving differently on
        high-frequency content, or a bug -- can be tested directly: pass 1
        to disable anisotropic filtering and isolate the effect
        (`rtbcnam/results/STAGE5.md`).
        """
        if not hasattr(source, 'endpoint_textures'):
            raise ValueError(
                'from_bc_latent_texture() needs an initialized BcLatentTexture '
                '(model_init() must have run, e.g. via a loaded checkpoint).'
            )

        tex = cls(source.num_latents, source.feature_resolutions, source.bc_format,
                 source.num_mip_levels, max_anisotropy)
        tex.num_tiles = source.num_tiles
        tex.material_ids = list(source.material_ids)
        tex.has_udims = dict(source.has_udims)
        tex.udim_mapping = dict(source.udim_mapping)
        tex.tile_indirection = list(source.tile_indirection)
        tex.material_texture_resolution = list(source.material_texture_resolution)
        tex.material_texture_wrap = list(source.material_texture_wrap)

        tex._source_endpoint_raw = [t.to_numpy() for t in source.endpoint_textures]
        tex._source_position_raw = [t.to_numpy() for t in source.position_textures]
        return tex

    def __init__(
        self,
        num_latents: int,
        feature_resolutions: list[int],
        bc_format: str = 'bc6h_sfloat',
        num_mip_levels: int = 1,
        max_anisotropy: int = 16,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.num_channels = num_latents
        self.feature_resolutions = list(feature_resolutions)
        self.bc_format = bc_format
        self.num_mip_levels = num_mip_levels
        self.max_anisotropy = max_anisotropy
        self.is_signed = bc_format == 'bc6h_sfloat'
        self.endpoint_scale = 512.0 if self.is_signed else 1023.0

        self.num_tiles = 0
        self.material_ids: list[int] = []
        self.has_udims: dict[int, bool] = {}
        self.udim_mapping: dict[int, list[int]] = {}
        self.tile_indirection: list[int] = []
        self.material_texture_resolution: list[int] = []
        self.material_texture_wrap: list[bool] = []

        self._source_endpoint_raw: list[np.ndarray] | None = None
        self._source_position_raw: list[np.ndarray] | None = None

        # Built from an already-trained source, so there is no warm-start
        # gap to wait out -- unlike BcLatentTexture.ready_for_render, which
        # starts False.
        self.ready_for_render = True

    @property
    def num_sets(self) -> int:
        return len(self.feature_resolutions)

    def get_texture_idx(self, set_idx: int, tile_idx: int, mip_level: int) -> int:
        """Must match `BcLatentTexture.get_texture_idx` exactly -- this class
        is built by iterating that one's own texture lists in this order."""
        return (set_idx * (self.num_tiles * self.num_mip_levels)
                + tile_idx * self.num_mip_levels + mip_level)

    def wrap(self, material_id: int) -> bool:
        return self.material_texture_wrap[material_id]

    def generate_from_encoder(self, *args, **kwargs) -> None:
        """No-op -- same contract as BcLatentTexture.generate_from_encoder.
        This class is never trained, so nothing here ever runs it for real."""

    def model_init(self, module: spy.Module, input_type: SlangType):
        if self._source_endpoint_raw is None:
            raise RuntimeError(
                'BcLatentTextureHardware must be constructed via '
                'from_bc_latent_texture(), not the plain constructor.'
            )

        device = module.device
        spy_format = getattr(spy.Format, self.bc_format)

        self.grids: list[spy.Texture] = []
        for endpoint_raw, position_raw in zip(
            self._source_endpoint_raw, self._source_position_raw, strict=True
        ):
            payload, width, height = build_blocks(
                endpoint_raw, position_raw, self.endpoint_scale, self.is_signed)
            texture = device.create_texture(
                format=spy_format, width=width, height=height, mip_count=1,
                usage=spy.TextureUsage.shader_resource,
            )
            # Not `create_texture(data=...)` -- see verify_bc6h_reference.py's
            # own note on why that path mis-computes a compressed format's
            # row pitch. `copy_from_numpy` honours the real block layout.
            texture.copy_from_numpy(np.frombuffer(payload, dtype=np.uint8).copy())
            self.grids.append(texture)

        # Real hardware trilinear + anisotropic. At today's single-mip-level
        # configs (Stage 3) there is no mip chain to blend across, so this
        # buys anisotropic filtering at one resolution rather than true
        # trilinear across several -- see bc_latent_texture_hardware.slang's
        # header comment.
        self.sampler = device.create_sampler(
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            mip_filter=spy.TextureFilteringMode.linear,
            max_anisotropy=self.max_anisotropy,
        )

        self.tile_indirection_gpu = spy.Tensor.from_numpy(
            device, np.array(self.tile_indirection, dtype=np.int32))
        self.material_texture_wrap_gpu = spy.Tensor.from_numpy(
            device, np.array(self.material_texture_wrap, dtype=np.int32))
        self.material_texture_resolution_gpu = spy.Tensor.from_numpy(
            device, np.array(self.material_texture_resolution, dtype=np.int32))

        # Freed now that the real GPU textures hold the data -- keeping them
        # around would just double the host-side memory for no reason.
        self._source_endpoint_raw = None
        self._source_position_raw = None

    def make_optimizable(self) -> None:
        raise NotImplementedError(
            'BcLatentTextureHardware is render-only and can never be made '
            'optimizable -- train a BcLatentTexture and convert it with '
            'from_bc_latent_texture() instead.'
        )

    def children(self) -> list[IModel]:
        return []

    def resolve_input_type(self, module: spy.Module):
        return 'LatentTextureInput'

    @property
    def type_name(self) -> str:
        return (
            f'BcLatentTextureHardware<{self.num_sets}, {self.num_tiles}, '
            f'{self.num_mip_levels}, {self.num_latents}>'
        )

    def model_params(self):
        return []

    def model_data(self):
        return {
            'tile_indirection': self.tile_indirection_gpu.storage.descriptor_handle_ro,
            'material_texture_wrap': self.material_texture_wrap_gpu.storage.descriptor_handle_ro,
            'material_texture_resolution': self.material_texture_resolution_gpu.storage.descriptor_handle_ro,
            'grids': [t.descriptor_handle_ro for t in self.grids],
            'samp': self.sampler.descriptor_handle,
        }
