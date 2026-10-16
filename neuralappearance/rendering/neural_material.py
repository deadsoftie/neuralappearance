# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import falcor2 as f2
import slangpy as spy
from slangpy.core.native import unpack_arg

if TYPE_CHECKING:
    from model import NeuralModelCheckpoint


class NeuralMaterial(f2.Material):
    """Expose a trained neural appearance model as a falcor2 material.

    The model architecture becomes generated Slang type aliases, while model
    parameters and textures live in a bound configuration buffer. A source hash
    gives each distinct architecture its own material type.
    """

    def __init__(self) -> None:
        super().__init__()
        self.module: spy.SlangModule | None = None
        self.spy_module: spy.Module | None = None
        self.config_tensor: spy.Tensor | None = None
        self.module_hash = ''
        self.reference_material_id = 0

    def configure(
        self,
        device: spy.Device,
        neural_model: NeuralModelCheckpoint,
        material_id: int,
    ) -> None:
        """Compile and bind a neural material for a reference material ID."""
        self.reference_material_id = int(material_id)

        generated_source = self.code_generate(neural_model)
        source_hash = hashlib.sha256(generated_source.encode('utf-8')).hexdigest()[:16]

        module_changed = source_hash != self.module_hash
        if module_changed:
            self.module_hash = source_hash
            source_path = Path(__file__).parent / 'neural_material.slang'
            source = source_path.read_text()
            source += generated_source
            source = source.replace('<HASH>', self.module_hash)
            source = source.replace('<NUM_LATENTS>', str(neural_model.latent_texture.num_channels))

            self.module = device.load_module_from_source(
                f'neural_material_{self.module_hash}',
                source,
            )
            self.spy_module = spy.Module(self.module)
            self.slang_type_name = f'NeuralMaterial_{self.module_hash}'

        self._write_config_buffer(device, neural_model)

        dirty_flags = f2.Material.DirtyFlags.properties
        if module_changed:
            dirty_flags |= f2.Material.DirtyFlags.resources
        self.mark_dirty(dirty_flags)

    def required_module(self) -> spy.SlangModule | None:
        return self.module

    def write_to_cursor(self, cursor: spy.ShaderCursor) -> None:
        if self.config_tensor is None:
            raise RuntimeError('NeuralMaterial.configure() must be called before scene.update().')

        cursor['material_id'] = self.reference_material_id
        cursor['config_handle'] = f2.to_handle(self.config_tensor.storage).data

    def code_generate(self, neural_model: NeuralModelCheckpoint) -> str:
        """Generate concrete Slang aliases for the checkpoint's model types."""

        latents = neural_model.latent_texture
        decoder = neural_model.decoder

        if neural_model.sampler is not None:
            sampler_type = neural_model.sampler.type_name
        else:
            sampler_type = f'CosineHemisphereSampler<{latents.num_channels}>'

        if neural_model.aux is not None:
            aux_type = neural_model.aux.type_name
        else:
            aux_type = f'DummyAuxDecoder<{latents.num_channels}, 1>'

        # rtcnam: None (every pre-existing checkpoint) substitutes a huge
        # sentinel so `min(weight, sampler_weight_cap)` in
        # neural_material.slang's sample() is a true no-op, same convention
        # as every other opt-in rtcnam knob in this project.
        sampler_weight_cap = neural_model.sampler_weight_cap
        if sampler_weight_cap is None:
            sampler_weight_cap = 3.402823e38

        # rtcnam: decode the latent once per shading-point visit and reuse it
        # across eval/sample/eval_pdf, instead of each of them re-fetching.
        # Worth it only when a fetch is expensive. For `NtcLatentTexture` a
        # fetch is 8 texel taps plus a decode-MLP forward pass, and caching
        # measured ~25% faster end to end. For the dense `LatentTexture` a
        # fetch is one buffer read, and caching measured ~30% SLOWER: the
        # cache is populated eagerly in setup_material_instance(), before the
        # `wi.z <= 0` early-outs in eval/sample/eval_pdf, so backfacing hits
        # go from zero fetches to one, and the cached vector adds per-thread
        # state through the hot path for no gain.
        #
        # That regression is not just a lost optimization -- the dense path is
        # this project's own measurement baseline, so slowing it down biases
        # every dense-vs-NTC render comparison in NTC's favour. Hence a
        # compile-time switch rather than one behavior for both.
        # Imported here rather than at module scope only to keep this module's
        # import graph one-directional (`model` never imports `rendering`).
        from model import NtcLatentTexture

        cache_latents = isinstance(latents, NtcLatentTexture)

        return _NEURAL_MATERIAL_CONFIG_TEMPLATE.format(
            latents_type=latents.type_name,
            decoder_type=decoder.type_name,
            sampler_type=sampler_type,
            aux_type=aux_type,
            num_mip_levels=neural_model.num_mip_levels,
            sampler_weight_cap=sampler_weight_cap,
            cache_latents='true' if cache_latents else 'false',
        )

    def _write_config_buffer(
        self,
        device: spy.Device,
        neural_model: NeuralModelCheckpoint,
    ) -> None:
        if self.spy_module is None:
            raise RuntimeError('Neural material module has not been compiled.')

        config_type_name = f'NeuralMaterialConfig_{self.module_hash}'
        config_type = self.spy_module[config_type_name]
        self.config_tensor = spy.Tensor.empty(device, shape=(1,), dtype=config_type)

        config: dict[str, Any] = {}
        for name, component in neural_model.components():
            if component is not None:
                config[name] = unpack_arg(component)

        cursor = self.config_tensor.cursor()
        cursor[0].write(config)
        cursor.apply()


_NEURAL_MATERIAL_CONFIG_TEMPLATE = """
public typealias LatentTexture_<HASH> = {latents_type};
public typealias Decoder_<HASH> = {decoder_type};
public typealias Sampler_<HASH> = {sampler_type};
public typealias AuxDecoder_<HASH> = {aux_type};

public struct NeuralMaterialConfig_<HASH>
{{
    static const int num_mip_levels = {num_mip_levels};
    static const float sampler_weight_cap = {sampler_weight_cap};
    static const bool cache_latents = {cache_latents};
    public LatentTexture_<HASH> latents;
    public Decoder_<HASH> decoder;
    public Sampler_<HASH> sampler;
    public AuxDecoder_<HASH> aux;
}};
"""
