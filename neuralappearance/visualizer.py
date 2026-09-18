# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import copy
from pathlib import Path
from typing import Any

import falcor2 as f2
import numpy as np
import slangpy as spy
from datagen import ReferenceMaterials, create_aux_data_generator
from falcor2.editor.camera_controller import CameraController
from model import NeuralModel, NeuralModelCheckpoint
from neuralnetworks import utils as nn_utils
from rendering.hybrid.hybrid_pathtracer import HybridPathTracer
from rendering.hybrid.resample_helper import ResampleHelper
from rendering.hybrid.scene_material_setup import prepare_scene_material
from rendering.neural_material import NeuralMaterial
from rendering.pathtracer_helpers import apply_udim_offset_to_scene, create_testscene
from util import get_default_asset_paths
from util.config import load_config
from util.math_helpers import udim_offset
from util.orbit_camera_controller import OrbitCameraController

_FALCOR2_ROOT = Path(__file__).resolve().parent.parent / 'external' / 'falcor2'


def _resolve_checkpoint_paths(checkpoint: str) -> tuple[Path, Path]:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir():
        checkpoint_dir = checkpoint_path
        model_path = checkpoint_dir / 'model.json'
    else:
        model_path = checkpoint_path
        checkpoint_dir = model_path.parent

    config_path = checkpoint_dir / 'config.json'
    if not model_path.is_file():
        raise FileNotFoundError(f'Checkpoint model not found: {model_path}')
    if not config_path.is_file():
        raise FileNotFoundError(f'Checkpoint config not found: {config_path}')
    return config_path, model_path


def _get_scene_camera(scene: f2.Scene) -> f2.Camera:
    for component in scene.components:
        if isinstance(component, f2.Camera):
            return component
    raise RuntimeError('No Camera component found in visualizer scene')


class GpuAccumulator:
    """Progressively average one GPU image stream without CPU readback."""

    def __init__(self, device: spy.Device, width: int, height: int, utils_module: spy.Module):
        self._utils_module = utils_module
        self.tensor = spy.Tensor.empty(device, shape=(height, width), dtype=spy.float4)
        self.reset()

    def reset(self) -> None:
        self._utils_module.accumulator_reset(self.tensor)

    def update_and_output(self, input: spy.Tensor, output: spy.Tensor) -> None:
        if input is output:
            self._utils_module.accumulator_update_and_output_inplace(input, self.tensor)
        else:
            self._utils_module.accumulator_update_and_output(input, self.tensor, output)


class AppWindow:
    """Shared window, UI, presentation, and navigation behavior."""

    def __init__(self, vis: 'NeuralMaterialVisualizer', width: int, height: int, x: int, y: int):
        self.vis = vis
        self.device = vis.device

        self.window = spy.Window(width=width, height=height, resizable=False)
        self.window.position = spy.int2(x, y)

        self.output_texture = self.device.create_texture(
            format=spy.Format.rgba32_float,
            width=self.window.width,
            height=self.window.height,
            mip_count=1,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label='output_texture',
        )

        self.surface = self.device.create_surface(self.window)
        self.surface.configure(width=self.window.width, height=self.window.height)

        self.ui = spy.ui.Context(self.device)
        self.ui_window = spy.ui.Window(parent=self.ui.screen, title='Options')

    def keyboard_event(self, event: spy.KeyboardEvent) -> bool:
        if self.ui.handle_keyboard_event(event):
            return True

        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                self.window.close()
                return True

            if event.key == spy.KeyCode.tab:
                self.vis.set_show_ui(not self.vis.show_ui)
                return True

            if event.key == spy.KeyCode.space:
                # Toggle between neural-only and reference-only views. The UI
                # control also provides the side-by-side view.
                if self.vis.view_mode == 2:
                    self.vis.set_view_mode(0)
                else:
                    self.vis.set_view_mode(1 - self.vis.view_mode)
                return True

            if event.key == spy.KeyCode.left:
                self.vis.set_material_idx(self.vis.material_idx - 1)
                return True
            if event.key == spy.KeyCode.right:
                self.vis.set_material_idx(self.vis.material_idx + 1)
                return True

            if event.key == spy.KeyCode.up:
                self.vis.set_udim_idx(self.vis.udim_idx - 1)
                return True
            if event.key == spy.KeyCode.down:
                self.vis.set_udim_idx(self.vis.udim_idx + 1)
                return True

            if event.key == spy.KeyCode.page_up:
                self.vis.set_mip_level(self.vis.mip_level + 1)
                return True
            if event.key == spy.KeyCode.page_down:
                self.vis.set_mip_level(self.vis.mip_level - 1)
                return True

        return False

    def mouse_event(self, event: spy.MouseEvent) -> bool:
        if self.ui.handle_mouse_event(event):
            return True
        return False

    def update(self, dt: float) -> None:
        self.window.process_events()

    def present_output_texture(self) -> None:
        image = self.surface.acquire_next_image()
        if image is None:
            return

        cmd = self.device.create_command_encoder()
        cmd.blit(image, self.output_texture)
        self.ui.begin_frame(image.width, image.height)
        self.ui.end_frame(image, cmd)
        cmd.set_texture_state(image, spy.ResourceState.present)
        self.device.submit_command_buffer(cmd.finish())

        del image
        self.surface.present()


class BsdfPlotAppWindow(AppWindow):
    """Interactive angular slice of the reference or neural BSDF."""

    def __init__(self, vis: 'NeuralMaterialVisualizer'):
        x = 50 + 860 + 10
        super().__init__(vis, width=860, height=860, x=x, y=50 + 1080 + 50)

        self.window.title = 'BRDF plot'
        self.window.on_keyboard_event = self.on_keyboard_event
        self.window.on_mouse_event = self.on_mouse_event
        self.ui_window.size = spy.float2(430, 230)
        self.redraw = True

        self.plot_bsdf_neural = self.vis.module_neural.plot_bsdf_neural
        self.plot_bsdf_reference = (
            self.vis.module_reference.plot_bsdf_reference.as_func()
            .type_conformances(self.vis.reference_materials.conformances)
            .write(self.vis.reference_materials.bind_scene)
        )

        # UI.
        self.theta_slider = spy.ui.SliderFloat(
            self.ui_window,
            'Theta_i',
            value=0,
            min=0,
            max=90,
            callback=lambda theta: self.vis.set_theta_i(np.radians(theta)),
        )
        self.phi_slider = spy.ui.SliderFloat(
            self.ui_window,
            'Phi_i',
            value=0,
            min=0,
            max=360,
            callback=lambda phi: self.vis.set_phi_i(np.radians(phi)),
        )
        self.exposure_slider = spy.ui.SliderFloat(
            self.ui_window,
            'Exposure',
            value=0,
            min=-10,
            max=10,
            callback=lambda exposure: self.vis.set_plot_exposure(exposure),
        )
        self.channel_idx_combobox = spy.ui.ComboBox(
            self.ui_window,
            'Channel',
            value=0,
            items=['RGB', 'R', 'G', 'B', 'Luminance'],
            callback=lambda idx: self.vis.set_plot_channel_idx(idx),
        )
        self.colormap_idx_combobox = spy.ui.ComboBox(
            self.ui_window,
            'Colormap',
            value=0,
            items=['Viridis', 'Plasma', 'Magma', 'Inferno', 'Gray'],
            callback=lambda idx: self.vis.set_plot_colormap_idx(idx),
        )
        self.colormap_idx_combobox.enabled = False

    def update(self, dt: float) -> None:
        super().update(dt)

        if self.redraw:
            material = self.vis.reference_materials[self.vis.material_idx]
            mip_level = max(0, self.vis.mip_level)

            selected_sph = spy.float2(self.vis.theta_i, self.vis.phi_i)
            selected_wi = self.vis.module.spherical_to_cartesian_rad(selected_sph)

            cmd = self.device.create_command_encoder()

            res = [self.output_texture.width, self.output_texture.height]

            self.vis.module.prepare_bsdf_plot(
                call_id=spy.call_id(), resolution=res, color=self.output_texture, _append_to=cmd
            )

            if self.vis.show_neural:
                latents = self.vis.neural_model.latent_texture
                decoder = self.vis.neural_model.decoder
                checkpoint_material = self.vis.checkpoint_reference_materials[self.vis.material_idx]

                self.plot_bsdf_neural(
                    latent_texture=latents,
                    decoder=decoder,
                    material_id=checkpoint_material.id,
                    mip_level=mip_level,
                    selected_uv=self.vis.uv,
                    selected_wi=selected_wi,
                    color=self.output_texture,
                    _append_to=cmd,
                )
            else:
                num_prefilter_samples = 1 if mip_level == 0 else self.vis.num_prefilter_samples
                self.plot_bsdf_reference(
                    call_id=spy.call_id(),
                    material_backend_id=material.backend_id,
                    mip_level=mip_level,
                    selected_uv=self.vis.uv,
                    selected_wi=selected_wi,
                    color=self.output_texture,
                    num_prefilter_samples=num_prefilter_samples,
                    snap_to_texel_center=True,
                    texture_resolution=material.texture_resolution,
                    _append_to=cmd,
                )

            self.vis.module.colormap_bsdf_plot(
                exposure=self.vis.plot_exposure,
                channel_idx=self.vis.plot_channel,
                colormap_idx=self.vis.plot_colormap_idx,
                color=self.output_texture,
                _append_to=cmd,
            )
            self.vis.module.finalize_bsdf_plot(
                call_id=spy.call_id(),
                resolution=res,
                selected_wi=selected_wi,
                color=self.output_texture,
                _append_to=cmd,
            )

            self.device.submit_command_buffer(cmd.finish())
            self.redraw = False
        self.present_output_texture()

    def on_keyboard_event(self, event: spy.KeyboardEvent) -> None:
        if super().keyboard_event(event):
            return

    def on_mouse_event(self, event: spy.MouseEvent) -> None:
        if super().mouse_event(event):
            return

        if event.type == spy.MouseEventType.button_down:
            if event.button == spy.MouseButton.left:
                uv = spy.float2(event.pos.x / self.window.width, event.pos.y / self.window.height)
                xy = 2 * uv - 1
                z = np.sqrt(np.maximum(0, 1 - xy.x**2 - xy.y**2))
                wi = spy.float3(xy, z)
                sph = self.vis.module.cartesian_to_spherical_rad(wi)
                self.vis.set_theta_i(sph.x)
                self.vis.set_phi_i(sph.y)


class UvSpaceAppWindow(AppWindow):
    """Material UV preview used to choose the BSDF plot location."""

    def __init__(self, vis: 'NeuralMaterialVisualizer'):
        super().__init__(vis, width=860, height=860, x=50, y=50 + 1080 + 50)
        self.window.title = 'UV space'
        self.window.on_keyboard_event = self.on_keyboard_event
        self.window.on_mouse_event = self.on_mouse_event
        self.ui_window.size = spy.float2(350, 120)
        self.redraw = True

        self.plot_uvs_reference = (
            self.vis.module_reference.plot_uvs_reference.as_func()
            .type_conformances(self.vis.reference_materials.conformances)
            .write(self.vis.reference_materials.bind_scene)
        )

        # UI.
        self.u_slider = spy.ui.SliderFloat(
            self.ui_window,
            'U',
            value=0,
            min=0,
            max=1,
            callback=lambda u: self.vis.set_uv(spy.float2(u, self.vis.uv.y)),
        )
        self.v_slider = spy.ui.SliderFloat(
            self.ui_window,
            'V',
            value=0,
            min=0,
            max=1,
            callback=lambda v: self.vis.set_uv(spy.float2(self.vis.uv.x, v)),
        )

    def update(self, dt: float) -> None:
        super().update(dt)

        if self.redraw:
            material = self.vis.reference_materials[self.vis.material_idx]
            res = [self.output_texture.width, self.output_texture.height]

            cmd = self.device.create_command_encoder()
            self.plot_uvs_reference(
                call_id=spy.call_id(),
                material_backend_id=material.backend_id,
                udim_offset=self.vis.udim_offset,
                selected_uv=self.vis.uv,
                resolution=res,
                color=self.output_texture,
                _append_to=cmd,
            )
            self.device.submit_command_buffer(cmd.finish())
            self.redraw = False
        self.present_output_texture()

    def on_keyboard_event(self, event: spy.KeyboardEvent) -> None:
        if super().keyboard_event(event):
            return

    def on_mouse_event(self, event: spy.MouseEvent) -> None:
        if super().mouse_event(event):
            return

        uv = spy.float2(event.pos.x / self.window.width, event.pos.y / self.window.height)
        uv += self.vis.udim_offset

        self.window.title = f'UV space: U={uv.x:.4f}, V={uv.y:.4f}'

        if event.type == spy.MouseEventType.button_down:
            if event.button == spy.MouseButton.left:
                self.vis.set_uv(uv)


class RenderAppWindow(AppWindow):
    """Progressive scene preview for reference/neural comparison."""

    def __init__(self, vis: 'NeuralMaterialVisualizer'):
        super().__init__(vis, width=1920, height=1080, x=50, y=50)
        self.window.title = 'Rendering preview'
        self.window.on_keyboard_event = self.on_keyboard_event
        self.window.on_mouse_event = self.on_mouse_event
        self.ui_window.size = spy.float2(430, 310)

        # Camera logic — shared across both scenes.
        self.camera = _get_scene_camera(self.vis.ref_scene)
        self.camera.width = self.window.width
        self.camera.height = self.window.height
        self.reset_accumulation = False
        self.frame = 0

        self.camera_controller = CameraController(self.camera)
        self.camera_controller.move_speed *= 0.1
        self.orbit_controller = OrbitCameraController(self.camera)
        self.orbit_controller.move_speed *= 0.1
        self.use_orbit = False

        # Half-resolution buffers for the 2-way side-by-side (1:1 aspect per panel).
        self.half_w = self.window.width // 2
        half_w = self.half_w
        self.sbs_tensors = [
            spy.Tensor.empty(self.device, shape=(self.window.height, half_w), dtype=spy.float4),
            spy.Tensor.empty(self.device, shape=(self.window.height, half_w), dtype=spy.float4),
        ]
        self.sbs_accumulators = [
            GpuAccumulator(self.device, half_w, self.window.height, self.vis.utils_module),
            GpuAccumulator(self.device, half_w, self.window.height, self.vis.utils_module),
        ]

        # Third-resolution buffers for the 3-way (reference/neural/compressed) side-by-side.
        self.third_w = self.window.width // 3
        third_w = self.third_w
        self.sbs3_tensors = [
            spy.Tensor.empty(self.device, shape=(self.window.height, third_w), dtype=spy.float4)
            for _ in range(3)
        ]
        self.sbs3_accumulators = [
            GpuAccumulator(self.device, third_w, self.window.height, self.vis.utils_module)
            for _ in range(3)
        ]

        # Full-resolution tensors for single-view modes.
        self.neural_tensor = spy.Tensor.empty(
            self.device,
            shape=(self.window.height, self.window.width),
            dtype=spy.float4,
        )
        self.ref_tensor = spy.Tensor.empty(
            self.device,
            shape=(self.window.height, self.window.width),
            dtype=spy.float4,
        )
        self.neural_accum = GpuAccumulator(
            self.device,
            self.window.width,
            self.window.height,
            self.vis.utils_module,
        )
        self.ref_accum = GpuAccumulator(
            self.device,
            self.window.width,
            self.window.height,
            self.vis.utils_module,
        )

        # UI.
        view_items = ['Neural only', 'Reference only', 'Side-by-side']
        if self.vis.has_compressed:
            view_items.append('Reference - Neural - Compressed')
        self.view_combobox = spy.ui.ComboBox(
            self.ui_window,
            'View',
            value=self.vis.view_mode,
            items=view_items,
            callback=lambda idx: self.vis.set_view_mode(idx),
        )
        self.signal_combobox = spy.ui.ComboBox(
            self.ui_window,
            'Signal',
            value=self.vis.signal_idx,
            items=self.vis.signal_names,
            callback=lambda idx: self.vis.set_signal_idx(idx),
        )
        self.material_combobox = spy.ui.ComboBox(
            self.ui_window,
            'Material',
            value=self.vis.material_idx,
            items=[f'material {mat.id}' for mat in self.vis.reference_materials],
            callback=lambda idx: self.vis.set_material_idx(idx),
        )
        self.udim_combobox = spy.ui.ComboBox(
            self.ui_window,
            'UDIM',
            value=0,
            items=['1001'],
            callback=lambda idx: self.vis.set_udim_idx(idx),
        )
        self.mip_combobox = spy.ui.ComboBox(
            self.ui_window,
            'MIP level',
            value=0,
            items=['dynamic'] + [f'{level}' for level in range(vis.num_mip_levels)],
            callback=lambda idx: self.vis.set_mip_level(idx - 1),
        )
        self.num_prefilter_samples_slider = spy.ui.SliderInt(
            self.ui_window,
            'Prefilter samples',
            value=16,
            min=1,
            max=512,
            callback=lambda num_samples: self.vis.set_num_prefilter_samples(num_samples),
        )
        self.num_prefilter_samples_slider.enabled = False
        self.orbit_checkbox = spy.ui.CheckBox(
            self.ui_window,
            'Orbit camera',
            value=self.use_orbit,
            callback=lambda v: self.set_orbit(v),
        )

        # Which-side-is-which labels for side-by-side view.
        self.label_y = self.window.height - 50
        self.reference_label = spy.ui.Window(
            parent=self.ui.screen,
            title='Reference',
            position=spy.float2(10, self.label_y),
            size=spy.float2(140, 40),
        )
        self.neural_label = spy.ui.Window(
            parent=self.ui.screen,
            title='Neural',
            position=spy.float2(half_w + 10, self.label_y),
            size=spy.float2(140, 40),
        )
        self.compressed_label = spy.ui.Window(
            parent=self.ui.screen,
            title='Compressed',
            position=spy.float2(2 * third_w + 10, self.label_y),
            size=spy.float2(140, 40),
        )
        self.reference_label.visible = False
        self.neural_label.visible = False
        self.compressed_label.visible = False

    def set_orbit(self, use_orbit: bool) -> None:
        self.use_orbit = use_orbit
        self.orbit_checkbox.value = use_orbit
        if use_orbit:
            self.orbit_controller.init_from_camera()
        self.reset_accumulation = True

    @property
    def active_camera_controller(self):
        return self.orbit_controller if self.use_orbit else self.camera_controller

    def on_keyboard_event(self, event: spy.KeyboardEvent) -> None:
        if super().keyboard_event(event):
            return
        if event.type == spy.KeyboardEventType.key_press and event.key == spy.KeyCode.o:
            self.set_orbit(not self.use_orbit)
            return
        self.active_camera_controller.handle_keyboard_event(event)

    def on_mouse_event(self, event: spy.MouseEvent) -> None:
        if super().mouse_event(event):
            return

        self.active_camera_controller.handle_mouse_event(event)

        if event.type == spy.MouseEventType.button_down:
            # Split pointer coordinates across the side-by-side view.
            side_by_side = self.vis.view_mode in (2, 3)
            if self.vis.view_mode == 2:
                panel_w = self.half_w
                panel_idx = 1 if event.pos.x >= panel_w else 0
            elif self.vis.view_mode == 3:
                panel_w = self.third_w
                panel_idx = min(2, int(event.pos.x // panel_w))
            else:
                panel_w = None
                panel_idx = None

            if side_by_side:
                pixel = spy.float2(event.pos.x - panel_idx * panel_w, event.pos.y)
                self.camera.width = panel_w
                self.camera.recompute()
                scene = self.vis.scene_for_panel(panel_idx, self.vis.view_mode)
            else:
                pixel = event.pos
                scene = self.vis.scene_for(self.vis.show_neural)

            reference_id = self.vis.reference_materials[self.vis.material_idx].backend_id
            neural_id = int(self.vis.neural_material.material_id)
            compressed_id = (
                int(self.vis.compressed_material.material_id) if self.vis.has_compressed else None
            )

            valid, material_id, uv, wi = self.trace_ray(pixel, scene)

            if side_by_side:
                self.camera.width = self.window.width
                self.camera.recompute()

            if valid and material_id in (reference_id, neural_id, compressed_id):
                self.vis.set_uv(uv)
                sph = self.vis.module.cartesian_to_spherical_rad(wi)
                self.vis.set_theta_i(sph.x)
                self.vis.set_phi_i(sph.y)

    def update(self, dt) -> None:
        super().update(dt)

        if self.vis.view_mode in (2, 3):
            self.ui_window.position = spy.float2(10, 10)

        if self.active_camera_controller.update(dt):
            self.active_camera_controller.camera.recompute()
            self.reset_accumulation = True

        self._update_render()
        self.present_output_texture()
        self.frame += 1

    def _update_render(self) -> None:
        two_way = self.vis.view_mode == 2
        three_way = self.vis.view_mode == 3

        if self.reset_accumulation:
            for acc in self.sbs_accumulators:
                acc.reset()
            for acc in self.sbs3_accumulators:
                acc.reset()
            self.neural_accum.reset()
            self.ref_accum.reset()
            self.frame = 0
            self.reset_accumulation = False

        material = self.vis.reference_materials[self.vis.material_idx]
        self.vis.hybrid_pt_neural.set_target_material(None, 0)
        self.vis.hybrid_pt_ref.set_target_material(None, material.texture_resolution)
        if self.vis.has_compressed:
            self.vis.hybrid_pt_compressed.set_target_material(None, 0)

        if two_way or three_way:
            if two_way:
                panel_w = self.half_w
                panels = [
                    (self.vis.hybrid_pt_ref, self.sbs_tensors[0], self.sbs_accumulators[0], 0, 'reference'),
                    (self.vis.hybrid_pt_neural, self.sbs_tensors[1], self.sbs_accumulators[1], panel_w, 'neural'),
                ]
            else:
                panel_w = self.third_w
                panels = [
                    (self.vis.hybrid_pt_ref, self.sbs3_tensors[0], self.sbs3_accumulators[0], 0, 'reference'),
                    (self.vis.hybrid_pt_neural, self.sbs3_tensors[1], self.sbs3_accumulators[1], panel_w, 'neural'),
                    (self.vis.hybrid_pt_compressed, self.sbs3_tensors[2], self.sbs3_accumulators[2], 2 * panel_w, 'compressed'),
                ]

            prev_w = self.camera.width
            self.camera.width = panel_w
            self.camera.recompute()

            for hpt, tensor, accum, x_offset, panel in panels:
                self._render_signal(hpt, tensor, panel)
                accum.update_and_output(tensor, tensor)
                self.vis.resample_helper.resample(
                    tensor,
                    self.output_texture,
                    output_pos=spy.uint2(x_offset, 0),
                    output_size=spy.uint2(panel_w, self.output_texture.height),
                )

            self.camera.width = prev_w
            self.camera.recompute()
        elif self.vis.show_neural:
            self._render_signal(self.vis.hybrid_pt_neural, self.neural_tensor, 'neural')
            self.neural_accum.update_and_output(self.neural_tensor, self.neural_tensor)
            self.vis.resample_helper.resample(
                self.neural_tensor,
                self.output_texture,
                output_pos=spy.uint2(0, 0),
                output_size=spy.uint2(self.window.width, self.window.height),
            )
        else:
            self._render_signal(self.vis.hybrid_pt_ref, self.ref_tensor, 'reference')
            self.ref_accum.update_and_output(self.ref_tensor, self.ref_tensor)
            self.vis.resample_helper.resample(
                self.ref_tensor,
                self.output_texture,
                output_pos=spy.uint2(0, 0),
                output_size=spy.uint2(self.window.width, self.window.height),
            )

    def _render_signal(
        self,
        path_tracer: HybridPathTracer,
        output: spy.Tensor,
        panel: str,
    ) -> None:
        if self.vis.signal_idx == 0:
            path_tracer.render(
                self.camera,
                output,
                self.frame,
                self.vis.mip_level,
                panel != 'reference',
            )
        else:
            self.vis.render_aux_signal(self.camera, output, self.frame, panel)

    def trace_ray(self, pixel: spy.float2, scene) -> tuple[bool, int, spy.float2, spy.float3]:
        """Trace a ray through pixel for UV/direction picking."""
        c = self.camera.calc_uniforms()
        ray_p = c.position
        xy = (pixel + 0.5) / spy.float2(c.width, c.height)
        xy.y = 1 - xy.y
        xy = 2 * xy - 1
        ray_d = spy.math.normalize(xy.x * c.image_u + xy.y * c.image_v + c.image_w)

        trace_ray = (
            self.vis.scene_module(scene)
            .trace_ray.type_conformances(scene.requirements.type_conformances)
            .write(scene.bind)
        )

        result: Any = trace_ray(ray_p=ray_p, ray_d=ray_d)
        return result['valid'], result['material_id'], result['uv'], result['wi']


class NeuralMaterialVisualizer:
    """Load a checkpoint and coordinate its render, BSDF, and UV windows.

    Reference and neural materials use separate scenes so both can accumulate
    independently and be presented side by side with matching view settings.
    """

    show_ui: bool = True

    # View modes are
    # 0: neural only 1: reference only, and 2: reference/neural split.
    view_mode: int = 0
    show_neural: bool = True
    material_idx: int = 0
    udim_idx: int = 0
    udim_offset: spy.float2 = spy.float2(0.0)
    mip_level: int = 0
    num_prefilter_samples: int = 16
    signal_idx: int = 0

    uv: spy.float2
    theta_i: float
    phi_i: float

    plot_exposure: float
    plot_channel: int
    plot_colormap_idx: int

    def __init__(self, checkpoint: str, assets_paths: str, compressed_checkpoint: str | None = None):
        super().__init__()

        self.device = spy.Device(
            type=spy.DeviceType.vulkan,
            enable_debug_layers=False,
            compiler_options={
                'include_paths': [
                    _FALCOR2_ROOT / 'slang',
                    Path(__file__).parent,
                    spy.SHADER_PATH,
                    *nn_utils.slang_include_paths(),
                ],
                'disable_warnings': [
                    '41018',  # returning without initializing out parameter 'value'
                    '41016',  # use of uninitialized variable
                    '41021',  # default initializer will not initialize field
                    '41035',  # possible use of uninitialized variable
                ],
            },
            bindless_options=spy.BindlessDesc(buffer_count=262144),
            enable_cuda_interop=False,
            enable_print=True,
            enable_hot_reload=True,
        )

        def hot_reload_cb(x):
            self.render_window.reset_accumulation = True

        self.device.register_shader_hot_reload_callback(hot_reload_cb)

        # Load the checkpoint model using the material IDs saved with the
        # checkpoint. The reference scene below may allocate different native
        # material IDs, so keep these IDs separate for latent lookup.
        config_path, model_path = _resolve_checkpoint_paths(checkpoint)
        checkpoint_config = load_config(config_path)
        model_config = copy.deepcopy(checkpoint_config)
        self.config = model_config

        self.checkpoint_reference_materials = ReferenceMaterials.from_config(model_config)
        training_module = spy.Module.load_from_file(self.device, 'training/train.slang')
        neural_model = NeuralModel(
            training_module,
            model_config,
            self.checkpoint_reference_materials,
        )
        neural_model.load_checkpoint(model_path)
        aux_trained = neural_model.aux is not None and neural_model.aux.status == 'Done'
        self.neural_model: NeuralModelCheckpoint = neural_model.get_checkpoint()

        aux = self.neural_model.aux if aux_trained else None
        self.aux_targets = aux.targets if aux is not None else None
        self.signal_names = ['BSDF']
        if self.aux_targets is not None:
            self.signal_names += [target.name.capitalize() for target in self.aux_targets]

        # Load reference materials into a native render scene for comparison.
        self.ref_scene = create_testscene(self.device)
        self.reference_materials = ReferenceMaterials.from_falcor2(
            self.device,
            copy.deepcopy(checkpoint_config),
            assets_paths,
            scene=self.ref_scene,
        )
        assert len(self.reference_materials) == len(self.checkpoint_reference_materials)

        self.aux_data_generator = None
        if self.aux_targets is not None:
            self.aux_data_generator = create_aux_data_generator(
                self.device,
                model_config,
                self.reference_materials,
                self.aux_targets,
                neural_model,
            )

        # Separate native scene for the neural material keeps side-by-side
        # material swaps straightforward.
        self.neural_scene = create_testscene(self.device)
        self.neural_material = self.neural_scene.create_material(NeuralMaterial)
        self.neural_material.name = 'neural_material'

        # Optional third instance: another full checkpoint (e.g. Project 3's
        # NTC-swapped-latent condition C) previewed as a "Compressed" panel
        # alongside reference/neural. Same loading pattern as the primary
        # checkpoint above, just against a second checkpoint directory --
        # config.json is a verbatim copy in that workflow, but this loads its
        # own rather than assuming that.
        self.has_compressed = compressed_checkpoint is not None
        self.compressed_model: NeuralModelCheckpoint | None = None
        self.compressed_scene = None
        self.compressed_material = None
        self.checkpoint_reference_materials_compressed = None
        if self.has_compressed:
            compressed_config_path, compressed_model_path = _resolve_checkpoint_paths(
                compressed_checkpoint
            )
            compressed_checkpoint_config = load_config(compressed_config_path)
            compressed_model_config = copy.deepcopy(compressed_checkpoint_config)
            self.checkpoint_reference_materials_compressed = ReferenceMaterials.from_config(
                compressed_model_config
            )
            compressed_model = NeuralModel(
                training_module,
                compressed_model_config,
                self.checkpoint_reference_materials_compressed,
            )
            compressed_model.load_checkpoint(compressed_model_path)
            self.compressed_model = compressed_model.get_checkpoint()

            self.compressed_scene = create_testscene(self.device)
            self.compressed_material = self.compressed_scene.create_material(NeuralMaterial)
            self.compressed_material.name = 'compressed_material'

        self.scenes = [self.ref_scene, self.neural_scene]
        if self.has_compressed:
            self.scenes.append(self.compressed_scene)

        self.num_mip_levels = self.neural_model.num_mip_levels

        config_max_depth = model_config['checkpoints']['rendering'].get('max_depth', 10)

        self.utils_module = spy.Module.load_from_file(self.device, 'falcor2/utils.slang')
        self.resample_helper = ResampleHelper(self.device)

        self.hybrid_pt_neural = HybridPathTracer(self.device)
        self.hybrid_pt_neural.max_depth = config_max_depth
        self.hybrid_pt_neural.scene = self.neural_scene

        self.hybrid_pt_ref = HybridPathTracer(self.device)
        self.hybrid_pt_ref.max_depth = config_max_depth
        self.hybrid_pt_ref.scene = self.ref_scene

        self.hybrid_pt_compressed = None
        if self.has_compressed:
            self.hybrid_pt_compressed = HybridPathTracer(self.device)
            self.hybrid_pt_compressed.max_depth = config_max_depth
            self.hybrid_pt_compressed.scene = self.compressed_scene

        self.module: Any = spy.Module.load_from_file(self.device, 'visualizer/visualizer.slang')
        self.module_neural: Any = spy.Module.load_from_file(
            self.device, 'visualizer/visualizer_neural.slang'
        )
        self.module_reference: Any = self.reference_materials.load_scene_module(
            self.device,
            'visualizer/visualizer_reference.slang',
        )
        self.module_model: Any = training_module
        self._hit_data_device_module = self.device.load_module(
            'rendering/hybrid/hit_data_extractor.slang'
        )
        self._hit_kernel_cache: dict[tuple[int, int, tuple[int, ...]], Any] = {}
        self._aux_hit_buffers: dict[tuple[int, int], tuple[spy.Tensor, ...]] = {}
        self._scene_device_module = self.device.load_module(
            'visualizer/visualizer_scene.slang',
        )
        self._scene_module_cache: dict[int, tuple[tuple[int, ...], spy.Module]] = {}

        self.render_window = RenderAppWindow(self)
        self.plot_window = BsdfPlotAppWindow(self)
        self.uv_window = UvSpaceAppWindow(self)

        self.uv = spy.float2(0.5, 0.5)
        self.set_view_mode(0)
        self.set_signal_idx(0)
        self.set_material_idx(0)
        self.set_udim_idx(0)
        self.set_mip_level(0)
        self.set_num_prefilter_samples(16)
        self.set_uv(spy.float2(0.5, 0.5))
        self.set_theta_i(np.radians(30.0))
        self.set_phi_i(np.radians(180.0))
        self.set_plot_exposure(0.0)
        self.set_plot_channel_idx(0)
        self.set_plot_colormap_idx(0)

        self.timer = spy.Timer()

    def scene_for(self, use_neural: bool):
        """Return the scene for the given material type."""
        return self.neural_scene if use_neural else self.ref_scene

    def scene_for_panel(self, panel_idx: int, view_mode: int):
        """Return the scene for one panel of a side-by-side view (2- or 3-way)."""
        if view_mode == 2:
            return (self.ref_scene, self.neural_scene)[panel_idx]
        return (self.ref_scene, self.neural_scene, self.compressed_scene)[panel_idx]

    def scene_module(self, scene: f2.Scene) -> spy.Module:
        """Return a VisualizerScene module linked against a native scene."""
        requirements_key = tuple(id(module) for module in scene.requirements.modules)
        cache_entry = self._scene_module_cache.get(id(scene))
        if cache_entry is not None and cache_entry[0] == requirements_key:
            return cache_entry[1]

        module = spy.Module(
            self._scene_device_module,
            link=[
                spy.Module(self.device.load_module('falcor2.utils')),
                spy.Module(scene.render_module),
            ]
            + [spy.Module(required_module) for required_module in scene.requirements.modules],
        )
        self._scene_module_cache[id(scene)] = (requirements_key, module)
        return module

    def render_aux_signal(
        self,
        camera: f2.Camera,
        output: spy.Tensor,
        iteration: int,
        panel: str,
    ) -> None:
        """Render the selected auxiliary signal at primary surface hits."""
        assert self.aux_targets is not None

        is_neural = panel != 'reference'
        is_compressed = panel == 'compressed'
        model = self.compressed_model if is_compressed else self.neural_model
        neural_material = self.compressed_material if is_compressed else self.neural_material
        checkpoint_reference_materials = (
            self.checkpoint_reference_materials_compressed
            if is_compressed
            else self.checkpoint_reference_materials
        )
        assert not is_neural or model.aux is not None

        material = self.reference_materials[self.material_idx]
        scene = self.compressed_scene if is_compressed else self.scene_for(is_neural)
        material_id = int(neural_material.material_id) if is_neural else int(material.backend_id)
        hit_kernel = self._get_hit_kernel(scene, material_id)

        shape = (output.shape[0], output.shape[1])
        buffers = self._aux_hit_buffers.get(shape)
        if buffers is None:
            buffers = (
                spy.Tensor.empty(self.device, shape, dtype='float2'),
                spy.Tensor.empty(self.device, shape, dtype='float3'),
                spy.Tensor.empty(self.device, shape, dtype='float3'),
                spy.Tensor.empty(self.device, shape, dtype='uint'),
                spy.Tensor.empty(self.device, shape, dtype='uint'),
                spy.Tensor.empty(self.device, shape, dtype='float'),
            )
            self._aux_hit_buffers[shape] = buffers
        uv, wi, wo, hit_mask, mip_level, texel_sample = buffers

        hit_kernel.call(
            camera,
            uv,
            wi,
            wo,
            hit_mask,
            mip_level,
            texel_sample,
            iteration,
            self.mip_level,
            material.texture_resolution,
            self.num_mip_levels,
            tid=spy.grid(shape),
        )
        if is_neural:
            checkpoint_material = checkpoint_reference_materials[self.material_idx]
            latents = self.module_model.eval_latent_texture_bilinear_stochastic(
                model.latent_texture,
                checkpoint_material.id,
                mip_level,
                uv,
                texel_sample,
            )
            values = self.module_model.eval_aux(model.aux, latents, wi)
        else:
            assert self.aux_data_generator is not None
            values = self.aux_data_generator.eval_reference(
                material,
                mip_level,
                uv,
                wi,
                wo,
            )

        target = self.aux_targets[self.signal_idx - 1]
        self.module.visualize_aux_signal(
            values,
            hit_mask,
            target.offset,
            target.size,
            target.name == 'normal',
            color=output,
        )

    def _get_hit_kernel(self, scene: f2.Scene, material_id: int):
        """Return a scene-linked primary-hit extraction kernel."""
        requirements_key = tuple(id(module) for module in scene.requirements.modules)
        key = (id(scene), material_id, requirements_key)
        kernel = self._hit_kernel_cache.get(key)
        if kernel is None:
            module = spy.Module(
                self._hit_data_device_module,
                link=[spy.Module(scene.render_module)]
                + [spy.Module(required) for required in scene.requirements.modules],
            )
            constants = {
                'MAX_DEPTH': self.hybrid_pt_neural.max_depth,
                'ENABLE_NEE': self.hybrid_pt_neural.enable_nee,
                'ENABLE_MIS': self.hybrid_pt_neural.enable_mis,
                'ENABLE_EMISSIVE_TRIANGLES': self.hybrid_pt_neural.enable_emissive_triangles,
                'ENABLE_ENV_MAP': self.hybrid_pt_neural.enable_env_map,
                'TARGET_MATERIAL_ID': material_id,
            }
            kernel = module.extract_hit_data.constants(constants).type_conformances(
                scene.requirements.type_conformances
            )
            self._hit_kernel_cache[key] = kernel
        return kernel.write(scene.bind)

    def update_material(self) -> None:
        """Assign correct materials to preview geometry in all scenes."""
        self.render_window.reset_accumulation = True

        ref_mat = self.reference_materials[self.material_idx]
        checkpoint_mat = self.checkpoint_reference_materials[self.material_idx]
        prepare_scene_material(self.ref_scene, ref_mat.falcor2_handle)

        self.neural_material.configure(
            self.device,
            self.neural_model,
            checkpoint_mat.id,
        )
        prepare_scene_material(self.neural_scene, self.neural_material)

        if self.has_compressed:
            checkpoint_mat_compressed = self.checkpoint_reference_materials_compressed[
                self.material_idx
            ]
            self.compressed_material.configure(
                self.device,
                self.compressed_model,
                checkpoint_mat_compressed.id,
            )
            prepare_scene_material(self.compressed_scene, self.compressed_material)

        self._scene_module_cache.clear()
        self._hit_kernel_cache.clear()

    def set_show_ui(self, show_ui: bool) -> None:
        self.show_ui = show_ui
        self.render_window.ui_window.visible = show_ui
        self.plot_window.ui_window.visible = show_ui
        self.uv_window.ui_window.visible = show_ui
        self._update_side_by_side_labels()

    def set_show_neural(self, show_neural: bool) -> None:
        self.show_neural = show_neural
        self.view_mode = 0 if show_neural else 1
        self.apply_view_mode()

    def set_view_mode(self, view_mode: int) -> None:
        max_mode = 3 if self.has_compressed else 2
        view_mode = max(0, min(max_mode, view_mode))
        self.view_mode = view_mode
        self.show_neural = view_mode != 1
        self.apply_view_mode()
        self.render_window.reset_accumulation = True

    def set_signal_idx(self, signal_idx: int) -> None:
        self.signal_idx = max(0, min(len(self.signal_names) - 1, signal_idx))
        self.render_window.signal_combobox.value = self.signal_idx
        self.render_window.window.title = f'Rendering preview: {self.render_title}'
        self.render_window.reset_accumulation = True

    def apply_view_mode(self) -> None:
        tag = self.render_title
        plot_tag = 'Neural' if self.show_neural else 'Reference'
        self.render_window.window.title = f'Rendering preview: {tag}'
        self.render_window.view_combobox.value = self.view_mode
        self.render_window.num_prefilter_samples_slider.enabled = not self.show_neural
        self.plot_window.window.title = f'BRDF plot: {plot_tag}'
        self.plot_window.redraw = True
        self._update_side_by_side_labels()

    def _update_side_by_side_labels(self) -> None:
        rw = self.render_window
        three_way = self.view_mode == 3
        visible = self.show_ui and self.view_mode in (2, 3)

        neural_x = rw.third_w if three_way else rw.half_w
        rw.reference_label.position = spy.float2(10, rw.label_y)
        rw.neural_label.position = spy.float2(neural_x + 10, rw.label_y)
        rw.compressed_label.position = spy.float2(2 * rw.third_w + 10, rw.label_y)

        rw.reference_label.visible = visible
        rw.neural_label.visible = visible
        rw.compressed_label.visible = self.show_ui and three_way

    @property
    def render_title(self) -> str:
        view_titles = ('Neural', 'Reference', 'Reference | Neural', 'Reference | Neural | Compressed')
        return f'{view_titles[self.view_mode]} - {self.signal_names[self.signal_idx]}'

    def set_material_idx(self, material_idx: int) -> None:
        material_idx = material_idx % len(self.reference_materials)
        self.material_idx = material_idx
        self.update_material()
        self.set_udim_idx(0)

        self.render_window.material_combobox.value = material_idx
        udims = self.reference_materials[material_idx].udims
        self.render_window.udim_combobox.items = [str(udim) for udim in udims]
        self.uv_window.redraw = True

    def set_udim_idx(self, udim_idx: int) -> None:
        udims = self.reference_materials[self.material_idx].udims
        udim_idx = udim_idx % len(udims)
        self.udim_idx = udim_idx

        for scene in self.scenes:
            apply_udim_offset_to_scene(scene, (-self.udim_offset[0], -self.udim_offset[1]))
        self.udim_offset = udim_offset(udims[self.udim_idx])
        for scene in self.scenes:
            apply_udim_offset_to_scene(scene, self.udim_offset)
        for scene in self.scenes:
            scene.update()

        self.render_window.reset_accumulation = True
        self.render_window.udim_combobox.value = udim_idx
        self.uv_window.u_slider.min = self.udim_offset.x
        self.uv_window.u_slider.max = self.udim_offset.x + 1.0
        self.uv_window.v_slider.min = self.udim_offset.y
        self.uv_window.v_slider.max = self.udim_offset.y + 1.0

        # Move the selected UV to the same fractional position in the new tile.
        frac_u = self.uv.x - int(self.uv.x)
        frac_v = self.uv.y - int(self.uv.y)
        self.set_uv(spy.float2(self.udim_offset.x + frac_u, self.udim_offset.y + frac_v))

    def set_mip_level(self, mip_level: int) -> None:
        mip_level = min(max(-1, mip_level), self.num_mip_levels - 1)
        self.mip_level = mip_level
        self.render_window.reset_accumulation = True

        self.render_window.mip_combobox.value = mip_level + 1
        self.plot_window.redraw = True

    def set_num_prefilter_samples(self, num_samples: int) -> None:
        num_samples = max(1, num_samples)
        self.num_prefilter_samples = num_samples

        self.render_window.num_prefilter_samples_slider.value = num_samples
        self.plot_window.redraw = True

    def set_uv(self, uv: spy.float2) -> None:
        res = self.reference_materials[self.material_idx].texture_resolution
        uv = self.module.snap_to_nearest_texel_center(uv, res, False)

        self.uv = uv
        self.uv_window.u_slider.value = uv.x
        self.uv_window.v_slider.value = uv.y
        self.uv_window.redraw = True
        self.plot_window.redraw = True

    def set_theta_i(self, theta_i: float) -> None:
        self.theta_i = theta_i
        self.plot_window.theta_slider.value = np.degrees(theta_i)
        self.plot_window.redraw = True

    def set_phi_i(self, phi_i: float) -> None:
        self.phi_i = phi_i
        self.plot_window.phi_slider.value = np.degrees(phi_i)
        self.plot_window.redraw = True

    def set_plot_exposure(self, exposure: float) -> None:
        self.plot_exposure = exposure
        self.plot_window.exposure_slider.value = exposure
        self.plot_window.redraw = True

    def set_plot_channel_idx(self, channel_idx: int) -> None:
        self.plot_channel = channel_idx
        self.plot_window.channel_idx_combobox.value = channel_idx
        self.plot_window.colormap_idx_combobox.enabled = channel_idx >= 1
        self.plot_window.redraw = True

    def set_plot_colormap_idx(self, colormap_idx: int) -> None:
        self.plot_colormap_idx = colormap_idx
        self.plot_window.colormap_idx_combobox.value = colormap_idx
        self.plot_window.redraw = True

    def main_loop(self) -> None:
        while True:
            render_should_close = self.render_window.window.should_close()
            plot_should_close = self.plot_window.window.should_close()
            uv_should_close = self.uv_window.window.should_close()
            if render_should_close or plot_should_close or uv_should_close:
                break

            dt = self.timer.elapsed_s()
            self.timer.reset()

            self.render_window.update(dt)
            self.plot_window.update(dt)
            self.uv_window.update(dt)

        self.device.wait()


def find_latest_checkpoint(path: str | Path) -> str:
    """Find the latest checkpoint from a job folder or jobs folder."""
    jobs_dir = Path(path)
    if (jobs_dir / 'checkpoints').is_dir():
        job_dirs = [jobs_dir]
    else:
        job_dirs = sorted(
            [d for d in jobs_dir.iterdir() if d.is_dir() and (d / 'checkpoints').is_dir()],
            key=lambda d: d.name,
        )
    if not job_dirs:
        raise FileNotFoundError(f'No jobs with checkpoints found in {jobs_dir}')
    latest_job = job_dirs[-1]
    ckpt_dirs = sorted(
        [d for d in (latest_job / 'checkpoints').iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    if not ckpt_dirs:
        raise FileNotFoundError(f'No checkpoints found in {latest_job / "checkpoints"}')
    latest_ckpt = ckpt_dirs[-1]
    print(f'Using checkpoint: {latest_ckpt}')
    return str(latest_ckpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural materials visualizer')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--checkpoint', type=str, help='Path to the neural material checkpoint directory'
    )
    group.add_argument(
        '--jobs',
        type=str,
        help='Path to the jobs folder; the latest job and its latest checkpoint will be used',
    )
    parser.add_argument(
        '-a',
        '--assets-paths',
        default=get_default_asset_paths(),
        help='Semicolon-separated asset roots. Defaults to the repo root. '
        'For MDL materials, this must include the parent of the vMaterials_2 folder '
        '(e.g. "C:\\Users\\rahul\\OneDrive\\Documents\\mdl"), not just the repo root.',
    )
    parser.add_argument(
        '--compressed-checkpoint',
        type=str,
        default=None,
        help='Optional path to a second checkpoint directory (e.g. one of Project 3\'s '
        'NTC-swapped-latent checkpoints, 00200000_ntc_aggressive or '
        '00200000_ntc_high_fidelity) to compare as a third "Compressed" panel in the '
        'side-by-side view. Same config.json as --checkpoint/--jobs, so it needs the '
        'same --assets-paths, not a separate one.',
    )
    args = parser.parse_args()

    checkpoint = args.checkpoint if args.checkpoint else find_latest_checkpoint(args.jobs)

    app = NeuralMaterialVisualizer(checkpoint, args.assets_paths, args.compressed_checkpoint)
    app.main_loop()
