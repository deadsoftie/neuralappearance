# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import slangpy as spy
from neuralnetworks import IModel, SlangType

from .bc6h_pack import build_blocks
from .bc_latent_texture import BcLatentTexture


class BcLatentTextureHardware(IModel):
    """Render-only BC6H view of a trained `BcLatentTexture`.

    Conversion packs the current software parameters into compressed textures
    sampled by the fixed-function texture unit. This snapshot is not trainable
    and has no independent checkpoint state.
    """

    @classmethod
    def from_bc_latent_texture(
        cls, source: BcLatentTexture, max_anisotropy: int = 16,
    ) -> BcLatentTextureHardware:
        """Snapshot an initialized, trained software texture's current values.

        Set `max_anisotropy` to 1 to disable anisotropic filtering.
        """
        if not hasattr(source, 'endpoint_textures'):
            raise ValueError(
                'from_bc_latent_texture() needs an initialized BcLatentTexture '
                '(model_init() must have run, e.g. via a loaded checkpoint).'
            )

        tex = cls(source.num_latents, source.feature_resolutions, source.bc_format,
                 source.num_mip_levels, max_anisotropy)
        tex.ready_for_render = source.ready_for_render
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

        # Direct construction has no warm-start phase. Conversion from a
        # software texture overwrites this with the source's readiness.
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
            # Upload explicitly so compressed blocks use their actual row pitch.
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
        self.clamp_sampler = device.create_sampler(
            min_filter=spy.TextureFilteringMode.linear,
            mag_filter=spy.TextureFilteringMode.linear,
            mip_filter=spy.TextureFilteringMode.linear,
            max_anisotropy=self.max_anisotropy,
            address_u=spy.TextureAddressingMode.clamp_to_edge,
            address_v=spy.TextureAddressingMode.clamp_to_edge,
        )
        self.nearest_sampler = device.create_sampler(
            min_filter=spy.TextureFilteringMode.point,
            mag_filter=spy.TextureFilteringMode.point,
            mip_filter=spy.TextureFilteringMode.point,
        )
        self.nearest_clamp_sampler = device.create_sampler(
            min_filter=spy.TextureFilteringMode.point,
            mag_filter=spy.TextureFilteringMode.point,
            mip_filter=spy.TextureFilteringMode.point,
            address_u=spy.TextureAddressingMode.clamp_to_edge,
            address_v=spy.TextureAddressingMode.clamp_to_edge,
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
            'clamp_samp': self.clamp_sampler.descriptor_handle,
            'nearest_samp': self.nearest_sampler.descriptor_handle,
            'nearest_clamp_samp': self.nearest_clamp_sampler.descriptor_handle,
        }
