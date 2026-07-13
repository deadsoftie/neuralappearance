# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import itertools
import operator

import numpy as np
import slangpy as spy
from slangpy import CommandEncoder, Module
from slangpy.core.native import Shape

from ..basetypes import IModel, Real
from ..components import LinearLayer
from .optimizer import Optimizer


class GradReducer:
    """Collect and reduce linear-layer gradients in a reproducible order.

    Linear layers normally add atomically into one gradient tensor. Here they
    training instead writes one slice per 32 batch elements. This helper reduces
    those slices in a fixed order before the nested optimizer runs.
    """

    def __init__(
        self,
        module: Module,
        dtype: Real,
        layers: list[LinearLayer],
        grad_count: int,
        block_size: int = 8,
        use_coop: bool = False,
    ):
        self.device = module.device
        self.module = module
        self.dtype = dtype
        self.grad_count: int
        self.input_tensors: list[spy.Tensor]
        self.output_tensors: list[spy.Tensor]
        self.unreduced_grads: list[spy.Tensor]
        self.reduced_grads: list[spy.Tensor]
        self.block_size = block_size
        self.use_coop = use_coop
        self.layers = layers
        self.mapping_buffer: spy.Tensor | None = None
        self.grad_storage: spy.Tensor | None = None
        self.task_buffer: spy.Tensor | None = None

        if self.use_coop:
            entry_point_name = f'wave_cooperative_accumulate<{self.dtype}, {self.block_size}>'
            group_size_y = 32
        else:
            entry_point_name = f'single_threaded_accumulate<{self.dtype}, {self.block_size}>'
            group_size_y = 1

        self.reduce_func = self.module[entry_point_name]
        self.dispatch_size_y = group_size_y

        task_struct = self.module.find_struct('AccumulateTask')
        if task_struct is None:
            raise ValueError(
                f'Could not find type "AccumulateTask" in slang module "{module.name}". '
                'Make sure the module imports the neural_networks module.'
            )
        self.task_struct = task_struct

        self.set_grad_count(grad_count)

    def set_grad_count(self, grad_count: int):
        self.grad_count = grad_count
        self.build_buffers()

    def build_buffers(self):
        for layer in self.layers:
            layer._request_padding(self.block_size)

        params = functools.reduce(
            operator.iadd, (layer.model_params() for layer in self.layers), []
        )

        if len(params) == 0:
            self.mapping_buffer = None
            self.grad_storage = None
            self.task_buffer = None
            self.dispatch_size_x = 0
            return

        block_counts = [(p.element_count + self.block_size - 1) // self.block_size for p in params]
        block_offsets = [0, *np.cumsum(block_counts).tolist()]
        total_blocks = block_offsets[-1]
        grad_buf_shape = (self.grad_count, total_blocks * self.block_size)
        self.dispatch_size_x = total_blocks

        if self.mapping_buffer is None or self.mapping_buffer.element_count != total_blocks:
            self.mapping_buffer = None
            self.mapping_buffer = spy.Tensor.empty(self.device, shape=(total_blocks,), dtype='uint')
        if self.grad_storage is None or self.grad_storage.shape.as_tuple() != grad_buf_shape:
            self.grad_storage = None
            self.grad_storage = spy.Tensor.zeros(self.device, grad_buf_shape, self.dtype.name)
        if self.task_buffer is None or self.task_buffer.element_count != len(params):
            self.task_buffer = None
            self.task_buffer = spy.Tensor.empty(
                self.device, shape=(len(params),), dtype=self.task_struct
            )

        mapping = np.zeros((total_blocks,), dtype=np.uint32)
        for i, (a, b) in enumerate(itertools.pairwise(block_offsets)):
            mapping[a:b] = i
        self.mapping_buffer.copy_from_numpy(mapping)

        cursor = self.task_buffer.cursor()
        unreduced_grads: list[spy.Tensor] = []
        for i, param in enumerate(params):
            if param.grad_out is None:
                raise ValueError('Trainable parameter has no gradient (grad_out == None)')

            reduced_shape = param.shape.as_tuple()
            unreduced_shape = (self.grad_count, *reduced_shape)
            unreduced_grad = self.grad_storage.view(
                Shape(unreduced_shape),
                Shape((self.grad_storage.strides[0], *param.strides.as_tuple())),
                block_offsets[i] * self.block_size,
            )
            unreduced_grads.append(unreduced_grad)

            cursor[i]['src'] = unreduced_grad.storage.descriptor_handle_rw
            cursor[i]['dst'] = param.grad_out.storage.descriptor_handle_rw
            cursor[i]['group_base'] = block_offsets[i]
            cursor[i]['src_offset'] = unreduced_grad.offset
            cursor[i]['dst_offset'] = param.grad_out.offset
            cursor[i]['accum_stride'] = unreduced_grad.strides[0]
            cursor[i]['accum_count'] = unreduced_grad.shape[0]
        cursor.apply()

        grad_index = 0
        for layer in self.layers:
            weight_grads = unreduced_grads[grad_index]
            bias_grads = unreduced_grads[grad_index + 1] if layer.use_biases else None
            grad_index += 2 if layer.use_biases else 1
            layer._request_deterministic_grads(weight_grads, bias_grads)

    def prune_layers(self, layers_to_keep: set[LinearLayer]):
        self.layers = [layer for layer in self.layers if layer in layers_to_keep]
        self.build_buffers()

    def execute(self, cmd: CommandEncoder):
        if self.dispatch_size_x > 0:
            launch_grid = spy.grid((self.dispatch_size_x * self.dispatch_size_y,))
            self.reduce_func(
                self.task_buffer, self.mapping_buffer, call_id=launch_grid, _append_to=cmd
            )

    def clear_grads(self, cmd: CommandEncoder):
        if self.grad_storage is not None:
            cmd.clear_buffer(self.grad_storage.storage)


class DeterministicOptimizer(Optimizer):
    """Make linear-layer gradient accumulation deterministic.

    The wrapper redirects each ``LinearLayer`` through a ``GradReducer``, then
    delegates parameter updates to ``nested_optimizer``. The batch size
    determines the number of independent gradient slices.
    """

    def __init__(self, nested_optimizer: Optimizer, batch_size: int):
        super().__init__()
        self._initialized = False
        self.nested_optimizer = nested_optimizer
        self.batch_size = batch_size

    def initialize(self, module: Module, models: IModel | list[IModel]):
        if isinstance(models, IModel):
            models = [models]
        self._initialized = True
        self.device = module.device

        linear_layers: list[LinearLayer] = functools.reduce(
            operator.iadd,
            ([c for c in model.components() if isinstance(c, LinearLayer)] for model in models),
            [],
        )
        grad_count = self._compute_grad_size()
        self.reducers: dict[Real, GradReducer] = {}
        dtypes = {layer.dtype for layer in linear_layers}
        for dtype in dtypes:
            layers = [layer for layer in linear_layers if layer.dtype == dtype]
            self.reducers[dtype] = GradReducer(module, dtype, layers, grad_count)

        self.nested_optimizer.initialize(module, models)

    def set_batch_size(self, batch_size: int):
        self.check_initialized()

        self.batch_size = batch_size
        grad_count = self._compute_grad_size()

        for reducer in self.reducers.values():
            reducer.set_grad_count(grad_count)

    def prune_models(self, models_to_keep: set[IModel]):
        self.check_initialized()

        layers_to_keep = set(
            functools.reduce(
                operator.iadd,
                (
                    [c for c in model.components() if isinstance(c, LinearLayer)]
                    for model in models_to_keep
                ),
                [],
            )
        )

        reducers = self.reducers
        self.reducers = {}
        for dtype, reducer in reducers.items():
            reducer.prune_layers(layers_to_keep)
            if len(reducer.layers) > 0:
                self.reducers[dtype] = reducer

        self.nested_optimizer.prune_models(models_to_keep)

    def step(self, cmd: CommandEncoder | None = None):
        self.check_initialized()

        do_wait = False
        if cmd is None:
            cmd = self.device.create_command_encoder()
            do_wait = True

        for reducer in self.reducers.values():
            reducer.execute(cmd)

        cmd.global_barrier()
        for reducer in self.reducers.values():
            reducer.clear_grads(cmd)
        cmd.global_barrier()

        self.nested_optimizer.step(cmd)

        if do_wait:
            self.device.wait_for_submit(self.device.submit_command_buffer(cmd.finish()))

    def _compute_grad_size(self):
        warp_size = 32
        return (self.batch_size + warp_size - 1) // warp_size
