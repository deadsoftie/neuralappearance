# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import operator

import numpy as np
import slangpy as spy
from slangpy import CommandEncoder, Module

from ..basetypes import IModel, Real
from ..components import LinearLayer


def sort_tensors_by_dtype(tensors: list[spy.Tensor]) -> dict[Real, list[spy.Tensor]]:
    by_dtype: dict[Real, list[spy.Tensor]] = {}

    for i, param in enumerate(tensors):
        dtype = Real.from_slangtype(param.dtype)
        if dtype is None:
            raise ValueError(
                f"Unsupported element type '{param.dtype.full_name}' "
                f'of parameter {i}: Must be half, float or double'
            )

        dst = by_dtype.get(dtype, [])
        dst.append(param)
        by_dtype[dtype] = dst

    return by_dtype


class OptimizerPool:
    """Dispatch data for parameters sharing a type and optimizer.

    A pool owns per-parameter state and maps flat dispatch indices to
    ``(parameter, element)`` pairs. One SlangPy call can therefore update
    tensors of different shapes while retaining their state buffers.
    """

    def __init__(self, module: Module, params: list[spy.Tensor], optim_type_name: str):
        optim_type = module.find_struct(optim_type_name)
        if optim_type is None:
            raise ValueError(
                f"Could not find optimizer type '{optim_type_name}' in slang module '{module.name}'. "
                'This could be due to a missing import or a type error. Make sure '
                'this is a valid type in the module, e.g. by pasting in the type above '
                'and checking for compile errors'
            )

        batch_type = module.find_struct(f'{optim_type_name}::Batch')
        if batch_type is None:
            raise ValueError(
                f"Could not find optimizer batch type '{optim_type_name}::State' in slang module "
                f"'{module.name}'. Make sure the type {optim_type_name} implements IOptimizer"
            )

        state_type = module.find_struct(f'{optim_type_name}::State')
        if state_type is None:
            raise ValueError(
                f"Could not find optimizer state type '{optim_type_name}::State' in slang module "
                f"'{module.name}'. Make sure the type {optim_type_name} implements IOptimizer"
            )

        step_func = module.find_function_in_struct(optim_type, 'step')
        if step_func is None:
            raise ValueError(
                f"Could not find method '{optim_type_name}::step()' in slang module '{module.name}'. "
                f'Make sure the type {optim_type_name} implements IOptimizer'
            )

        batch_step_func = module.find_function_in_struct(optim_type, 'batch_step')
        if batch_step_func is None:
            raise ValueError(
                f"Could not find method '{optim_type_name}::batch_step()' in slang module '{module.name}'. "
                f'Make sure the type {optim_type_name} implements IOptimizer'
            )

        self.module = module
        self.optim_type = optim_type
        self.state_type = state_type
        self.batch_type = batch_type
        self.step_func = step_func
        self.batch_step_func = batch_step_func

        self.build_buffers(params, None)

    def build_buffers(self, params: list[spy.Tensor], states: list[spy.Tensor] | None):
        """Build dispatch buffers, optionally retaining existing state."""

        if states is None:
            states = [self.state_type(param) for param in params]

        numel = sum(p.element_count for p in params)
        mapping = np.ndarray((numel, 2), dtype=np.int32)
        offset = 0

        batch_buffer = spy.Tensor.empty(
            self.module.device, shape=(len(params),), dtype=self.batch_type
        )
        cursor = batch_buffer.cursor()
        for i, (param, state) in enumerate(zip(params, states, strict=True)):
            n = param.element_count
            mapping[offset : offset + n, 0] = np.full(n, i, dtype=np.int32)
            mapping[offset : offset + n, 1] = np.arange(n, dtype=np.int32)
            offset += n

            cursor[i]['params'].write(param.storage.descriptor_handle_rw)
            cursor[i]['grads'].write(param.grad.storage.descriptor_handle_rw)
            cursor[i]['states'].write(state.storage.descriptor_handle_rw)
        cursor.apply()

        self.params = params
        self.states = states
        self.batch_buffer = batch_buffer
        self.mapping_buffer = spy.Tensor.empty(self.module.device, shape=(numel,), dtype='int2')
        self.mapping_buffer.copy_from_numpy(mapping)

    def prune_parameters(self, parameters_to_keep: set[spy.Tensor]):
        """Remove parameters while retaining their associated state."""
        keep_indices = [i for i in range(len(self.params)) if self.params[i] in parameters_to_keep]
        params = [self.params[i] for i in keep_indices]
        states = [self.states[i] for i in keep_indices]

        self.build_buffers(params, states)


class Optimizer:
    """Python-side lifecycle and batched dispatch for Slang optimizers.

    Construction stores hyperparameters. ``initialize()`` discovers trainable
    parameters, groups them by precision, reflects their Slang optimizer types,
    and allocates state. ``step()`` updates every pool, clears gradients, and
    refreshes quantized forward weights.

    Concrete optimizers implement ``get_type_name()`` and ``get_this()``.
    Override ``update_state()`` only for host state that advances after a step,
    such as Adam's bias-correction iteration.
    """

    def __init__(self):
        super().__init__()
        self._initialized = False

    def initialize(self, module: Module, models: IModel | list[IModel]):
        """Reflect optimizer types and allocate state for ``models``.

        Each precision in a mixed-precision model receives a separate pool.
        """

        if isinstance(models, IModel):
            models = [models]

        self._initialized = True
        self.device = module.device
        self.models = models
        self.parameters = functools.reduce(
            operator.iadd, (model.parameters() for model in models), []
        )

        self._find_quantized_layers()

        self.pools: dict[Real, OptimizerPool] = {}
        for dtype, param_list in sort_tensors_by_dtype(self.parameters).items():
            self.pools[dtype] = OptimizerPool(module, param_list, self.get_type_name(dtype))

    def step(self, cmd: CommandEncoder | None = None):
        """Perform one optimizer step and reset the network gradients.

        If ``cmd`` is supplied, append the Slang calls to that command encoder.
        """
        self.check_initialized()

        do_wait = False
        if cmd is None:
            cmd = self.device.create_command_encoder()
            do_wait = True

        this = self.get_this()
        for pool in self.pools.values():
            pool.batch_step_func(this, pool.batch_buffer, pool.mapping_buffer, _append_to=cmd)
        cmd.global_barrier()

        self._quantize_layer_weights(cmd)

        if do_wait:
            self.device.wait_for_submit(self.device.submit_command_buffer(cmd.finish()))

        self.update_state()

    def prune_models(self, models_to_keep: set[IModel]):
        self.check_initialized()

        self.models = [model for model in self.models if model in models_to_keep]
        self.parameters = functools.reduce(
            operator.iadd, (model.parameters() for model in self.models), []
        )

        self._find_quantized_layers()

        parameters_to_keep = set(self.parameters)
        pools = self.pools
        self.pools = {}
        for dtype, pool in pools.items():
            pool.prune_parameters(parameters_to_keep)
            if len(pool.params) > 0:
                self.pools[dtype] = pool

    def update_state(self):
        """Advance optional host-side state after ``step`` completes."""

    def get_type_name(self, dtype: Real) -> str:
        """Return the Slang type implementing ``IOptimizer<dtype>``.

        Concrete optimizers must override this method.
        """
        raise NotImplementedError()

    def get_this(self):
        """Return the Python value bound as the Slang optimizer instance.

        The value, commonly a hyperparameter dictionary, must be compatible
        with the optimizer type for every parameter precision. Concrete
        optimizers must override this method.
        """
        raise NotImplementedError()

    def check_initialized(self):
        if not self._initialized:
            raise RuntimeError(
                'Optimizer is uninitialized. Make sure to '
                'call .initialize() before using the optimizer'
            )

    def _quantize_layer_weights(self, cmd: CommandEncoder):
        if len(self.quantized_layers) == 0:
            return

        for layer in self.quantized_layers:
            cmd.convert_coop_vec_matrix(
                layer.forward_weights.storage,
                layer.forward_weight_desc,
                layer.backward_weights.storage,
                layer.backward_weight_desc,
            )
        cmd.global_barrier()

    def _find_quantized_layers(self):
        # Quantized forward weights are derived rather than optimized. Refresh
        # them after each step to match the trainable half-precision weights.
        self.quantized_layers: list[LinearLayer] = []
        for model in self.models:
            for layer in model.components():
                if isinstance(layer, LinearLayer) and layer.quantization is not None:
                    self.quantized_layers.append(layer)
