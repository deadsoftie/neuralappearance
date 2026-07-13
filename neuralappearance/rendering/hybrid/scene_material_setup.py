# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helper for assigning materials to preview geometry and rebuilding the scene.

This module provides a single helper function used by clients of
HybridPathTracer to set up the scene material before calling render().
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import falcor2 as f2


def prepare_scene_material(
    scene: f2.Scene,
    material: f2.Material | None = None,
) -> int:
    """Assign a material to preview geometry and update the scene.

    Iterates all geometry instances in scene and assigns material to
    those whose geometry name contains 'preview'. Then calls
    scene.update().

    Returns the falcor2 material system ID of the assigned material, suitable
    for passing to set_target_material() and scatter generator constructors.

    If material is None, the existing material on the first preview
    geometry instance is applied to all 'preview' geometry instances so
    that they share the same ID for detection by HybridPathTracer's wavefront
    mode.

    Args:
        scene: The falcor2 scene to modify.
        material: A falcor2 material handle to assign, or None to keep the
            current material on preview geometry.

    Returns:
        The integer falcor2 material system ID of the preview material.

    Raises:
        RuntimeError: If no geometry instance with 'preview' in its name
            is found in scene.
    """
    import falcor2 as f2

    preview_instances = [
        component
        for component in scene.components
        if isinstance(component, f2.GeometryInstance)
        and 'preview' in component.geometry.name.lower()
    ]
    if not preview_instances:
        raise RuntimeError('No preview geometry found in scene')

    if material is not None:
        target_material = material
    else:
        # Reuse the first preview instance's material so every preview
        # geometry shares the same material ID.
        target_material = next(iter(preview_instances[0].materials))

    for gi in preview_instances:
        gi.materials = [target_material] * len(list(gi.materials))

    scene.update()

    return int(target_material.material_id)
