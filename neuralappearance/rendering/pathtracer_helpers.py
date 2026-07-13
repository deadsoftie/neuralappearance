# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import falcor2 as f2
import numpy as np
import slangpy as spy

if TYPE_CHECKING:
    from training import TrainingTargets

ASSETS_DIR = Path(__file__).parent.parent.parent / 'assets'
USD_SHADERBALL_PATH = ASSETS_DIR / 'geometry/shaderball.usdc'
ENV_MAP_PATH = ASSETS_DIR / 'envmaps/brown_photostudio_02_1k.exr'
ENV_MAP_INTENSITY_SCALE = 0.66


def _focal_length_to_fov_degrees(focal_length_mm: float, sensor_size_mm: float) -> float:
    return float(spy.math.degrees(2 * spy.math.atan(sensor_size_mm / (2 * focal_length_mm))))


def apply_udim_offset_to_scene(
    scene: f2.Scene,
    offset: tuple[float, float] | spy.float2,
) -> None:
    """Add ``offset`` to every static-mesh UV coordinate in ``scene``.

    Updated arrays are written through
    ``StaticMeshGeometry.set_texcoords()``.
    """
    offset_np = np.array([offset[0], offset[1]], dtype=np.float32)
    for geometry in scene.geometries:
        if not isinstance(geometry, f2.StaticMeshGeometry):
            continue
        for sub_idx in range(geometry.sub_mesh_count):
            texcoords = geometry.texcoords(sub_idx)
            texcoords += offset_np
            geometry.set_texcoords(sub_idx, texcoords)


def create_preview_quad(scene: f2.Scene, name: str) -> f2.StaticMeshGeometry:
    """Create a single-quad ``StaticMeshGeometry`` in ``scene``.

    The quad lies in the Y=0 plane, spans [-0.5, +0.5] in X and Z, and uses a
    (0, 0) to (1, 1) UV mapping. The caller attaches it to a
    ``GeometryInstance`` and applies the required transform.
    """
    positions = np.array(
        [
            [-0.5, 0.0, -0.5],
            [+0.5, 0.0, -0.5],
            [-0.5, 0.0, +0.5],
            [+0.5, 0.0, +0.5],
        ],
        dtype=np.float32,
    )
    normals = np.tile(np.array([0, 1, 0], dtype=np.float32), (4, 1))
    tangents = np.tile(np.array([1, 0, 0], dtype=np.float32), (4, 1))
    handedness = np.ones((4,), dtype=np.float32)
    texcoords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    indices = np.array([[2, 1, 0], [1, 2, 3]], dtype=np.uint32)

    geometry = scene.create_geometry(f2.StaticMeshGeometry)
    geometry.name = name
    geometry.set_mesh_data(
        positions=positions,
        sub_mesh_indices=[indices],
        normals=normals,
        tangents=tangents,
        handedness=handedness,
        texcoords=texcoords,
        name=name,
    )
    return geometry


def _default_material(scene: f2.Scene) -> f2.Material:
    """Create the placeholder material used before scene preparation.

    A ``StandardMaterial`` keeps ``GeometryInstance`` creation valid until
    ``prepare_scene_material()`` installs the actual material.
    """
    existing = scene.materials.find('preview_quad_placeholder')
    if existing is not None:
        return existing
    material = scene.create_material('StandardMaterial')
    material.name = 'preview_quad_placeholder'
    return material


def create_testscene(device: spy.Device) -> f2.Scene:
    """Build the standard rendering test scene used by ``Renderer``.

    The scene starts from the USD shaderball and adds a quad, camera, and
    environment map. ``ReferenceMaterials`` adds the target material later;
    ``prepare_scene_material()`` installs it before each render.
    """
    scene = f2.Scene.create(device, str(USD_SHADERBALL_PATH))

    quad_geometry = create_preview_quad(scene, name='preview_quad')
    quad_entity = scene.create_entity()
    quad_entity.name = 'preview_quad'
    quad_entity.transform = f2.Transform(
        translation=spy.float3(0, 0.15, -0.3),
        rotation=spy.math.quat_from_euler_angles(spy.math.radians(spy.float3(70, 90, 0))),
        scale=spy.float3(0.25, 0.25, 0.25),
    )
    quad_instance = quad_entity.create_component(f2.GeometryInstance)
    quad_instance.geometry = quad_geometry
    quad_instance.materials = [_default_material(scene)]

    camera_entity = scene.create_entity()
    camera_entity.name = 'preview_camera'
    camera = camera_entity.create_component(f2.Camera)
    camera.fov_y = _focal_length_to_fov_degrees(focal_length_mm=33.25, sensor_size_mm=24.0)
    camera_entity.transform = f2.Transform(
        spy.math.inverse(
            spy.math.matrix_from_look_at(
                spy.float3(0.483926147, 0.328784376, -0.193543628),
                spy.float3(-0.4428505, -0.0412656367, -0.129138097),
                spy.float3(0, 1, 0),
            )
        )
    )
    camera.recompute()

    env_entity = scene.create_entity()
    env_entity.name = 'env_light'
    env_map = env_entity.create_component(f2.EnvMapLight)
    env_map.env_map_path = ENV_MAP_PATH
    env_map.exposure = math.log2(ENV_MAP_INTENSITY_SCALE)

    return scene


def split_aux_channels(
    render_buffer: np.ndarray | spy.Tensor,
    targets: TrainingTargets,
) -> list[spy.Bitmap]:
    """Split a multi-channel aux render buffer into per-target bitmaps.

    Handles channel-count conversion (1->grayscale, 2->reconstruct normal z,
    3->RGB) and saturates diffuse/specular/roughness to [0,1].
    """
    saturate_targets = {'diffuse', 'specular', 'roughness'}

    bitmaps: list[spy.Bitmap] = []
    for target in targets:
        slice = target.view(render_buffer)

        if slice.shape[-1] == 1:
            slice = np.repeat(slice, 3, axis=-1)
        elif slice.shape[-1] == 2:
            if target.name == 'normal':
                x, y = slice[..., 0], slice[..., 1]
                # Reconstruct ``z`` from the two-channel normal.
                norm2 = x**2 + y**2
                z = np.sqrt(np.clip(1.0 - norm2, 0.0, 1.0))
                slice = np.concatenate([slice, z[..., np.newaxis]], axis=-1)
            else:
                zeros = np.zeros((*slice.shape[:-1], 1), dtype=slice.dtype)
                slice = np.concatenate((slice, zeros), axis=-1)
        elif slice.shape[-1] != 3:
            raise ValueError('Cannot render targets with more than 3 channels.')

        if target.name in saturate_targets:
            slice = np.clip(slice, 0.0, 1.0)

        bitmaps.append(spy.Bitmap(slice))
    return bitmaps
