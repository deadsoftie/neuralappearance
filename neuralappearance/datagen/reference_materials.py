# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass, field

import falcor2
import slangpy as spy

FALCOR_2_TYPES = {'mtlx', 'mdl'}


def get_num_material_params(num_layers: int) -> int:
    params = 3 + 3 + 3 + 2 + 3  # normal, tangent, albedo, roughness, weight.
    return num_layers * params + 1  # + 1 for MIP level.


@dataclass
class ReferenceMaterial:
    """Training metadata and optional native handle for one source material."""

    id: int
    type: str
    num_bsdf_layers: int
    num_encoder_inputs: int
    texture_resolution: int
    texture_wrap: bool
    has_udims: bool
    udims: list[int]
    # Native material handle for MaterialX/MDL materials.
    falcor2_handle: falcor2.Material | None = field(default=None, repr=False)

    @property
    def backend_id(self) -> int:
        """Underlying material ID used by the native material system.

        Loaded reference materials use the falcor2-internal material_id.
        Checkpoint material records do not carry native handles, so they fall
        back to the trainer-side id.
        """
        if self.falcor2_handle is not None:
            return self.falcor2_handle.material_id
        return self.id

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != 'falcor2_handle'}
        return d


class ReferenceMaterials:
    """Reference materials and the falcor2 scene used to evaluate them.

    Instances created from a live scene retain its modules and type conformances
    for data generation. Instances reconstructed from checkpoint config contain
    metadata only and are sufficient for loading or inspecting a neural model.
    """

    def __init__(
        self,
        materials: list[ReferenceMaterial],
        *,
        f2_scene: falcor2.Scene | None = None,
        modules: list[spy.SlangModule] | None = None,
        conformances: list[spy.TypeConformance] | None = None,
    ) -> None:
        self.materials = materials
        self.num_bsdf_layers = max(mat.num_bsdf_layers for mat in materials)
        self.num_encoder_inputs = max(mat.num_encoder_inputs for mat in materials)

        # Native scene metadata is unavailable when loading materials solely
        # from a checkpoint config.
        self.f2_scene = f2_scene
        self.modules = modules if modules is not None else []
        self.conformances = conformances if conformances is not None else []

    @property
    def ids(self) -> list[int]:
        return [mat.id for mat in self.materials]

    def bind_scene(self, cursor: spy.ShaderCursor) -> None:
        """SlangPy ``.write(...)`` callback that binds the native scene into a
        shader cursor.
        """
        assert self.f2_scene is not None, 'bind_scene called before scene was built.'
        self.f2_scene.bind(cursor)

    def load_scene_module(self, device: spy.Device, module_name: str) -> spy.Module:
        """Load a Slang module linked with the native scene render module."""
        assert self.f2_scene is not None, 'load_scene_module called before scene was built.'
        return spy.Module(
            device.load_module(module_name),
            link=[
                spy.Module(device.load_module('falcor2.utils')),
                spy.Module(self.f2_scene.render_module),
            ]
            + [spy.Module(module) for module in self.f2_scene.requirements.modules],
        )

    @classmethod
    def create(
        cls,
        device: spy.Device,
        config: dict,
        assets_paths: str | None = None,
        scene: falcor2.Scene | None = None,
    ) -> ReferenceMaterials:
        reference_materials = config['reference_materials']

        unsupported_types = [
            item.get('type')
            for item in reference_materials
            if item.get('type') not in FALCOR_2_TYPES
        ]
        if unsupported_types:
            raise ValueError(f'Unsupported reference material type: {unsupported_types[0]}.')

        result = ReferenceMaterials.from_falcor2(
            device=device,
            config=config,
            assets_paths=assets_paths,
            scene=scene,
        )

        # Sneak the material information back into the config, so it can be
        # saved alongside the checkpoint.
        # This allows loading back a neural material at a later point (e.g. for
        # rendering results) without re-loading all reference materials in the
        # native material system.
        for idx, mat in enumerate(result.materials):
            mat_config = reference_materials[idx]
            mat_config.update(mat.to_dict())

        return result

    @classmethod
    def from_falcor2(
        cls,
        device: spy.Device,
        config: dict,
        assets_paths: str | None = None,
        scene: falcor2.Scene | None = None,
    ) -> ReferenceMaterials:
        """Build reference materials on a native ``falcor2.Scene``.

        If ``scene`` is ``None``, a fresh scene is created. Otherwise, materials
        are added to the supplied scene. Callers rendering a preloaded scene
        must pass it here so the renderer and materials share one
        ``falcor2.Scene``.
        """
        if scene is None:
            scene = falcor2.Scene.create(device)

        configs = config['reference_materials']
        native_materials: list[falcor2.Material] = []
        for id, material_config in enumerate(configs):
            props = falcor2.Properties()
            if material_config['type'] == 'mtlx':
                props['mtlx_basepath'] = _build_mtlx_basepath(
                    material_config['path'],
                    assets_paths,
                )
                props['mtlx_search_paths'] = _build_mtlx_search_paths(
                    material_config['path'],
                    assets_paths,
                )
                props['mtlx_path'] = material_config['path']
                props['mtlx_source'] = 'houdini'
                props['mtlx_transmissive_bsdfs'] = 'mtlxdielectric_bsdf;mtlxdielectric_bsdf2'
                props['mtlx_layering_mode'] = falcor2.MaterialXLayeringMode.bsdf_mix
                kind = 'MaterialXMaterial'

            elif material_config['type'] == 'mdl':
                mdl_path = material_config['path']
                mdl_library_path = _select_asset_path_that_contains(
                    mdl_path,
                    assets_paths,
                )
                if not mdl_library_path:
                    raise ValueError(
                        f'Could not find MDL module {mdl_path!r} below any --assets-paths root.'
                    )

                props['mdl_material_name'] = _build_mdl_material_name(
                    mdl_path,
                    material_config['material'],
                )
                props['mdl_library_path'] = str(mdl_library_path)
                props['mdl_class_compilation'] = material_config.get(
                    'mdl_class_compilation',
                    material_config.get('class_compilation', False),
                )
                props['learnable'] = material_config.get(
                    'learnable',
                    material_config.get('make_learnable', True),
                )
                kind = 'MDLMaterial'
            else:
                raise ValueError(f'Unsupported reference material type: {material_config["type"]}')

            material = scene.create_material(kind, props)
            material.name = f'training_material_{id}'
            native_materials.append(material)

        # Update the scene so ``material_id`` is assigned, textures are loaded,
        # scene.requirements settle.
        scene.update()

        modules = list(scene.requirements.modules)
        conformances = list(scene.requirements.type_conformances)

        layer_query = None
        materials: list[ReferenceMaterial] = []
        for id, (mat_config, native_material) in enumerate(
            zip(configs, native_materials, strict=True)
        ):
            num_bsdf_layers = mat_config.get('num_bsdf_layers')
            if num_bsdf_layers is None:
                if layer_query is None:
                    layer_query = _build_num_bsdf_layers_query(
                        device,
                        scene,
                        conformances,
                    )
                num_bsdf_layers = int(layer_query(int(native_material.material_id)))
            num_bsdf_layers = int(num_bsdf_layers)
            if num_bsdf_layers <= 0:
                raise ValueError(
                    f'Could not determine "num_bsdf_layers" for reference '
                    f'material {mat_config.get("path", mat_config.get("name", "?"))}. '
                    'The native material did not expose extra BSDF layer '
                    'properties; add an explicit "num_bsdf_layers" entry.'
                )

            num_encoder_inputs = get_num_material_params(num_bsdf_layers)

            # UDIM + texture-resolution auto-detection via the material's
            # registered textures. Config entries override the detected values
            # if both are present.
            detected_udims, detected_has_udims, detected_res = _detect_texture_metadata(
                native_material
            )

            has_udims = bool(mat_config.get('has_udims', detected_has_udims))
            udims = mat_config.get('udims', detected_udims)
            res = mat_config.get('texture_resolution', detected_res)
            wrap = mat_config.get('texture_wrap', not has_udims)

            if not res:
                raise ValueError(
                    f'Could not determine texture resolution for material '
                    f'{mat_config.get("path", mat_config.get("name", "?"))}. '
                    'Either the material has no textures or texture loading did not '
                    'complete. Add an explicit "texture_resolution" entry to the '
                    'material config.'
                )

            materials.append(
                ReferenceMaterial(
                    id=id,
                    type=mat_config['type'],
                    num_bsdf_layers=num_bsdf_layers,
                    num_encoder_inputs=num_encoder_inputs,
                    texture_resolution=res,
                    texture_wrap=wrap,
                    has_udims=has_udims,
                    udims=udims,
                    falcor2_handle=native_material,
                )
            )

        # Keep the native scene and its requirements alive for rendering and
        # data generation.
        return cls(
            materials,
            f2_scene=scene,
            modules=modules,
            conformances=conformances,
        )

    @classmethod
    def from_config(cls, config: dict) -> ReferenceMaterials:
        materials: list[ReferenceMaterial] = []
        for mat_config in config['reference_materials']:
            materials.append(
                ReferenceMaterial(
                    id=mat_config['id'],
                    type=mat_config['type'],
                    num_bsdf_layers=mat_config['num_bsdf_layers'],
                    num_encoder_inputs=mat_config['num_encoder_inputs'],
                    texture_resolution=mat_config['texture_resolution'],
                    texture_wrap=mat_config['texture_wrap'],
                    has_udims=mat_config['has_udims'],
                    udims=mat_config['udims'],
                )
            )
        reference_materials = cls(materials)
        return reference_materials

    def __iter__(self):
        return iter(self.materials)

    def __len__(self):
        return len(self.materials)

    def __getitem__(self, index):
        return self.materials[index]


def _detect_texture_metadata(material: falcor2.Material) -> tuple[list[int], bool, int]:
    """Detect a loaded material's UDIM tiles and maximum resolution.

    Returns ``(udims, has_udims, max_resolution)``. The material must have been
    through ``scene.update()`` so its texture resources are loaded.
    """
    if not hasattr(material, 'build_texture_list'):
        return ([1001], False, 0)

    texture_handles: list = material.build_texture_list()

    udim_set: set[int] = {1001}
    has_udims = False
    max_res = 0

    for handle in texture_handles:
        if not handle.is_valid():
            continue
        if handle.is_udim():
            has_udims = True
            for tile in handle.udim_tiles:
                udim_set.add(tile.tile_index)
                tile_handle = tile.texture_handle
                if tile_handle is not None and tile_handle.is_valid():
                    tex = tile_handle.texture
                    if tex is not None:
                        max_res = max(max_res, tex.width, tex.height)
        else:
            tex = handle.texture
            if tex is not None:
                max_res = max(max_res, tex.width, tex.height)

    return (sorted(udim_set), has_udims, max_res)


def _build_num_bsdf_layers_query(
    device: spy.Device,
    scene: falcor2.Scene,
    conformances: list[spy.TypeConformance],
):
    """Build a tiny native-scene query for material BSDF layer count."""
    module = spy.Module(
        device.load_module('datagen/falcor_reference_materials.slang'),
        link=[
            spy.Module(device.load_module('falcor2.utils')),
            spy.Module(scene.render_module),
        ]
        + [spy.Module(module) for module in scene.requirements.modules],
    )
    return (
        module['ReferenceMaterials::Falcor::num_bsdf_layers']
        .as_func()
        .type_conformances(conformances)
        .write(scene.bind)
    )


def _first_valid_asset_path(assets_paths: str | None) -> str | None:
    """Pick the first existing path from a semicolon-separated list.

    Used as a fallback for MDL library paths, which still expect one root.
    """
    if not assets_paths:
        return None
    for p in assets_paths.split(';'):
        if p and os.path.isdir(p):
            return p
    return None


def _split_search_paths(paths: str | None) -> list[str]:
    if not paths:
        return []
    return [p.strip() for p in paths.split(';') if p.strip()]


def _select_asset_path_that_contains(
    material_path: str,
    assets_paths: str | None,
) -> str | None:
    if not assets_paths:
        return None

    if os.path.isabs(material_path):
        return None

    for path in _split_search_paths(assets_paths):
        if os.path.isfile(os.path.join(path, material_path)):
            return path
    return None


def _build_mdl_material_name(module_path: str, material_name: str) -> str:
    """Build a qualified MDL material name from a module path and export name."""
    normalized_path = os.path.normpath(module_path)
    module_path_without_suffix, suffix = os.path.splitext(normalized_path)
    if suffix.lower() != '.mdl':
        raise ValueError(f'MDL module path must end in ".mdl": {module_path!r}')
    module_parts = module_path_without_suffix.split(os.sep)
    if os.path.isabs(normalized_path) or '..' in module_parts:
        raise ValueError(f'MDL module path must be relative to --assets-paths: {module_path!r}')
    unqualified_name = material_name.split('(', maxsplit=1)[0]
    if not unqualified_name or '::' in unqualified_name:
        raise ValueError(f'MDL material must be an unqualified export name: {material_name!r}')

    module_name = '::'.join(module_parts)
    return f'{module_name}::{material_name}'


def _build_mtlx_basepath(material_path: str, assets_paths: str | None) -> str:
    asset_path = _select_asset_path_that_contains(material_path, assets_paths)
    basepath = asset_path or _first_valid_asset_path(assets_paths)
    if not basepath:
        basepath = os.path.dirname(material_path)
    return basepath


def _build_mtlx_search_paths(material_path: str, assets_paths: str | None) -> str:
    search_paths = _split_search_paths(assets_paths)
    asset_path = _select_asset_path_that_contains(material_path, assets_paths)
    material_parts = os.path.normpath(material_path).split(os.sep)
    if asset_path and len(material_parts) > 1:
        search_paths.append(os.path.join(asset_path, material_parts[0]))

    return ';'.join(search_paths)
