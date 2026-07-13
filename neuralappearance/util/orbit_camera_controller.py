# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from typing import ClassVar

import slangpy as spy


class OrbitCameraController:
    """Orbit camera that rotates around a target point.

    Controls:
        Right-drag:  orbit (rotate around target)
        Middle-drag: pan (move target along camera right/up)
        Scroll:      dolly (zoom in/out by changing distance)
        WASD/QE:     move target in camera-relative directions
        Shift:       fast movement   Ctrl: slow movement
    """

    MOVE_KEYS: ClassVar = {
        spy.KeyCode.a: spy.float3(-1, 0, 0),
        spy.KeyCode.d: spy.float3(1, 0, 0),
        spy.KeyCode.e: spy.float3(0, 1, 0),
        spy.KeyCode.q: spy.float3(0, -1, 0),
        spy.KeyCode.w: spy.float3(0, 0, 1),
        spy.KeyCode.s: spy.float3(0, 0, -1),
    }

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.on_disable_cursor = None
        self.on_enable_cursor = None

        self.orbit_speed = 0.005
        self.pan_speed = 0.001
        self.zoom_speed = 0.1
        self.move_speed = 0.5

        self.target = spy.float3(0, 0, 0)
        self.distance = 1.0
        self.yaw = 0.0
        self.pitch = 0.0

        self._state = 0  # 0=idle, 1=orbit (right), 2=pan (middle)
        self._mouse_pos = spy.float2()
        self._mouse_delta = spy.float2()
        self._first_mouse_delta = True
        self._mouse_scroll_delta = spy.float2()

        self._key_state = dict.fromkeys(OrbitCameraController.MOVE_KEYS, False)
        self._shift_down = False
        self._ctrl_down = False

        self.init_from_camera()

    def _get_pose(self):
        if hasattr(self.camera, 'entity'):
            transform = self.camera.entity.transform
            return transform.translation, transform.rotation
        return self.camera.transform.pos, self.camera.transform.rot

    def _set_pose(self, pos, rot):
        if hasattr(self.camera, 'entity'):
            transform = self.camera.entity.transform
            transform.translation = pos
            transform.rotation = rot
            self.camera.entity.transform = transform
        else:
            self.camera.transform.pos = pos
            self.camera.transform.rot = rot

    def init_from_camera(self):
        """Derive orbit parameters from the camera's current transform."""
        pos, rotation = self._get_pose()
        rot = spy.math.matrix_from_quat(rotation)
        fwd = spy.float3(-rot.get_col(2).x, -rot.get_col(2).y, -rot.get_col(2).z)
        self.distance = max(0.1, spy.math.length(pos))
        self.target = pos + fwd * self.distance
        dx = self.target.x - pos.x
        dy = self.target.y - pos.y
        dz = self.target.z - pos.z
        dist_xz = math.sqrt(dx * dx + dz * dz)
        self.yaw = math.atan2(dx, dz)
        self.pitch = -math.atan2(dy, dist_xz)
        self.apply()

    def apply(self):
        """Set camera pos/rot from orbit parameters."""
        self.pitch = max(-math.pi / 2 + 0.01, min(math.pi / 2 - 0.01, self.pitch))
        self.distance = max(0.01, self.distance)
        cp = math.cos(self.pitch)
        dx = math.sin(self.yaw) * cp
        dy = -math.sin(self.pitch)
        dz = math.cos(self.yaw) * cp
        arm = spy.float3(dx, dy, dz)
        pos = self.target - arm * self.distance
        fwd = spy.math.normalize(self.target - pos)
        rotation = spy.math.quat_from_look_at(fwd, spy.float3(0, 1, 0))
        self._set_pose(pos, rotation)

    def handle_keyboard_event(self, event: spy.KeyboardEvent):
        if event.is_key_press() or event.is_key_release():
            down = event.is_key_press()
            if event.key in OrbitCameraController.MOVE_KEYS:
                self._key_state[event.key] = down
            elif event.key == spy.KeyCode.left_shift:
                self._shift_down = down
            elif event.key == spy.KeyCode.left_control:
                self._ctrl_down = down

    def handle_mouse_event(self, event: spy.MouseEvent):
        self._shift_down = event.has_modifier(spy.KeyModifier.shift)
        self._ctrl_down = event.has_modifier(spy.KeyModifier.ctrl)

        if self._state == 0:
            self._first_mouse_delta = True
            if event.is_button_down():
                if event.button == spy.MouseButton.right:
                    self._state = 1
                elif event.button == spy.MouseButton.middle:
                    self._state = 2
        elif self._state == 1:
            if event.is_button_up() and event.button == spy.MouseButton.right:
                self._state = 0
        elif self._state == 2:
            if event.is_button_up() and event.button == spy.MouseButton.middle:
                self._state = 0

        if event.is_move():
            self._mouse_delta = event.pos - self._mouse_pos
            if self._first_mouse_delta:
                self._mouse_delta = spy.float2()
                self._first_mouse_delta = False
            self._mouse_pos = event.pos

        if event.is_scroll():
            self._mouse_scroll_delta = event.scroll

    def update(self, dt: float) -> bool:
        changed = False

        if self._state == 1:
            if spy.math.length(self._mouse_delta) > 0:
                self.yaw -= self._mouse_delta.x * self.orbit_speed
                self.pitch += self._mouse_delta.y * self.orbit_speed
                self.apply()
                changed = True
        elif self._state == 2:
            if spy.math.length(self._mouse_delta) > 0:
                _, rotation = self._get_pose()
                rot = spy.math.matrix_from_quat(rotation)
                right = rot.get_col(0)
                up = rot.get_col(1)
                offset = -right * self._mouse_delta.x + up * self._mouse_delta.y
                self.target += offset * self.pan_speed * self.distance
                self.apply()
                changed = True

        if self._mouse_scroll_delta.y != 0:
            factor = 1.0 - self._mouse_scroll_delta.y * self.zoom_speed
            self.distance *= max(0.01, factor)
            self.apply()
            changed = True

        move_delta = spy.float3()
        for key, state in self._key_state.items():
            if state:
                move_delta += OrbitCameraController.MOVE_KEYS[key]
        if spy.math.length(move_delta) > 0:
            speed = self.move_speed
            if self._shift_down:
                speed *= 10.0
            if self._ctrl_down:
                speed *= 0.1
            _, rotation = self._get_pose()
            rot = spy.math.matrix_from_quat(rotation)
            right = rot.get_col(0)
            up = rot.get_col(1)
            fwd = spy.float3(-rot.get_col(2).x, -rot.get_col(2).y, -rot.get_col(2).z)
            offset = right * move_delta.x + up * move_delta.y + fwd * move_delta.z
            self.target += offset * speed * dt
            self.apply()
            changed = True

        self._mouse_delta = spy.float2()
        self._mouse_scroll_delta = spy.float2()
        return changed
