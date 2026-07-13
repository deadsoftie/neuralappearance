# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import enum
import math

import numpy as np
from slangpy import BufferUsage, CoopVecMatrixLayout, DataType, Feature, Module, Tensor

from ..basetypes import (
    ArrayKind,
    Auto,
    AutoSettable,
    IModel,
    Real,
    RealArray,
    SlangType,
    resolve_auto,
)

_COOPVEC_MATRIX_ALIGNMENT = 64
_COOPVEC_VECTOR_ALIGNMENT = 16


class WeightQuantization(enum.Enum):
    """FP8 storage format used for the weights of the forward pass.

    The optimizer updates trainable half-precision weights used by
    backpropagation, then refreshes the FP8 forward copy after each step.
    """

    FloatE4M3 = 1
    FloatE5M2 = 2

    def sgl(self):
        if self is WeightQuantization.FloatE4M3:
            return DataType.float8_e4m3
        elif self is WeightQuantization.FloatE5M2:
            return DataType.float8_e5m2
        else:
            raise ValueError(f"No sgl.DataType for WeightQuantization '{self}'")

    def coopvec(self):
        if self is WeightQuantization.FloatE4M3:
            return 'CoopVecComponentType::FloatE4M3'
        elif self is WeightQuantization.FloatE5M2:
            return 'CoopVecComponentType::FloatE5M2'
        else:
            raise ValueError(f"No CoopVecComponentType for WeightQuantization '{self}'")


class LinearLayer(IModel):
    """A linear neural-network layer computing ``A * x + b``.

    The layer accepts a regular Slang array or ``CoopVec``. Cooperative
    vectors use hardware-assisted matrix operations and require half precision;
    regular arrays use float or double precision.

    Cooperative-vector weights use a device-specific matrix layout. Access
    them through ``get_weights()`` and ``set_weights()`` rather than treating
    ``model_params()`` as a row-major matrix.
    """

    def __init__(
        self,
        num_inputs: AutoSettable[int],
        num_outputs: int,
        use_biases: bool = True,
        dtype: AutoSettable[Real] = Auto,
        use_coopvec: AutoSettable[bool] = Auto,
        quantization: WeightQuantization | None = None,
    ):
        super().__init__()

        self.use_biases = use_biases
        self.quantization = quantization
        self.num_outputs = num_outputs
        self._num_inputs = num_inputs
        self._dtype = dtype
        self._use_coopvec = use_coopvec

    def set_weights(self, weights_np: np.ndarray):
        self.check_initialized()
        if weights_np.shape != (self.num_outputs, self.num_inputs):
            raise ValueError(
                f'LinearLayer weights must have shape ({self.num_outputs}, {self.num_inputs}), rather than {weights_np.shape}'
            )

        weights_np = weights_np.astype(self.dtype.numpy())

        if self.use_coopvec:
            device = self.forward_weights.device
            src_desc = device.create_coop_vec_matrix_desc(
                self.num_outputs, self.num_inputs, CoopVecMatrixLayout.row_major, self.dtype.sgl()
            )
            # The descriptor can be larger than the weights due to padding.
            # Create a larger tensor, then copy the weights into it.
            assert src_desc.size % self.dtype.size() == 0, (
                'size of CoopVecMatrix is not a multiple of the element type size'
            )
            src_weights = Tensor.zeros(
                device, (src_desc.size // self.dtype.size(),), str(self.dtype)
            )
            src_weights[: self.num_inputs * self.num_outputs].copy_from_numpy(weights_np.flatten())

            cmd = device.create_command_encoder()
            cmd.convert_coop_vec_matrix(
                self.forward_weights.storage,
                self.forward_weight_desc,
                src_weights.storage,
                src_desc,
            )
            if self.backward_weights is not self.forward_weights:
                cmd.convert_coop_vec_matrix(
                    self.backward_weights.storage,
                    self.backward_weight_desc,
                    src_weights.storage,
                    src_desc,
                )
            device.wait_for_submit(device.submit_command_buffer(cmd.finish()))
        else:
            self.forward_weights.storage.copy_from_numpy(weights_np)
            if self.backward_weights is not self.forward_weights:
                self.backward_weights.storage.copy_from_numpy(weights_np)

    def get_weights(self) -> np.ndarray:
        self.check_initialized()

        if self.use_coopvec:
            device = self.forward_weights.device
            dst_desc = device.create_coop_vec_matrix_desc(
                self.num_outputs, self.num_inputs, CoopVecMatrixLayout.row_major, DataType.float32
            )
            # The descriptor can be larger than the matrix due to padding.
            # Create a larger tensor, then copy the weights into it.
            assert dst_desc.size % Real.float.size() == 0, (
                'size of CoopVecMatrix is not a multiple of sizeof(float)'
            )
            dst_weights = Tensor.empty(device, (dst_desc.size // Real.float.size(),), 'float')
            cmd = device.create_command_encoder()
            cmd.convert_coop_vec_matrix(
                dst_weights.storage,
                dst_desc,
                self.forward_weights.storage,
                self.forward_weight_desc,
            )
            device.wait_for_submit(device.submit_command_buffer(cmd.finish()))

            flattened_weights = dst_weights[: self.num_inputs * self.num_outputs].to_numpy()

            return flattened_weights.reshape((self.num_outputs, self.num_inputs))
        else:
            return self.forward_weights.to_numpy()

    def set_biases(self, biases_np: np.ndarray):
        self.check_initialized()
        if not self.use_biases:
            raise ValueError('LinearLayer does not use biases')
        if biases_np.shape != (self.num_outputs,):
            raise ValueError(
                f'LinearLayer biases must have shape ({self.num_outputs}), rather than {biases_np.shape}'
            )

        self.biases.storage.copy_from_numpy(biases_np)

    def get_biases(self) -> np.ndarray:
        self.check_initialized()
        if not self.use_biases:
            return np.zeros((self.num_outputs,), dtype=self.dtype.numpy())

        return self.biases.to_numpy()[: self.num_outputs]

    @property
    def type_name(self) -> str:
        base_type = 'CoopVecLinearLayer' if self.use_coopvec else 'LinearLayer'
        use_biases_str = '1' if self.use_biases else '0'

        if self.use_coopvec:
            cv_arg = ', ' + (
                self.quantization.coopvec()
                if self.quantization is not None
                else self.dtype.coopvec()
            )
        else:
            cv_arg = ''

        return f'{base_type}<{self.dtype}, {self.num_inputs}, {self.num_outputs}{cv_arg}, {use_biases_str}>'

    def model_init(self, module: Module, input_type: SlangType):
        input_array = RealArray.from_slangtype(input_type)
        self.num_inputs = resolve_auto(self._num_inputs, input_array.length)
        self.dtype = resolve_auto(self._dtype, input_array.dtype)
        self.use_coopvec = resolve_auto(self._use_coopvec, input_array.kind == ArrayKind.coopvec)

        if input_array.kind not in (ArrayKind.array, ArrayKind.coopvec):
            self.model_error(
                f'LinearLayer only supports arrays or CoopVec as input type. Received {input_array}'
            )

        if self.use_coopvec:
            if self.dtype != Real.half:
                self.model_error(
                    'LinearLayer currently only supports half precision as input '
                    f'when using CoopVec. Received {input_array}'
                )

            if Feature.cooperative_vector not in module.device.features:
                self.model_error(
                    'LinearLayer was requested to use the CoopVec API, '
                    'but the device does not support it.'
                )
        else:
            if self.dtype not in (Real.float, Real.double):
                self.model_error(
                    'LinearLayer currently only supports float or double precision '
                    f'as input when not using CoopVec. Received {input_array}'
                )

        if self.quantization is not None:
            if not self.use_coopvec:
                self.model_error(
                    f'Quantization {self.quantization} is only supported for layers with CoopVec enabled'
                )
            if self.dtype != Real.half:
                self.model_error(
                    f'Quantization {self.quantization} is only supported for layers in half precision'
                )

        # Initialize weights with the Xavier uniform distribution.
        fan_in = self.num_inputs
        fan_out = self.num_outputs
        std = math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        weights_np = np.random.uniform(-a, a, (fan_out, fan_in)).astype(self.dtype.numpy())

        device = module.device
        usage = BufferUsage.shader_resource | BufferUsage.unordered_access | BufferUsage.shared

        if self.use_coopvec:
            dtype_size = self.dtype.size()
            layout = CoopVecMatrixLayout.training_optimal
            backward_weight_desc = device.create_coop_vec_matrix_desc(
                rows=fan_out, cols=fan_in, layout=layout, element_type=self.dtype.sgl()
            )
            backward_weight_count = backward_weight_desc.size // dtype_size

            params_np = np.zeros((backward_weight_count,), dtype=self.dtype.numpy())
            device.convert_coop_vec_matrix(dst=params_np, src=weights_np, dst_layout=layout)

            backward_weights = Tensor.empty(
                device, (backward_weight_count,), str(self.dtype), usage=usage
            )
            backward_weights.storage.copy_from_numpy(params_np)

            self.backward_weight_desc = backward_weight_desc

            if self.quantization is not None:
                quantized_desc = device.create_coop_vec_matrix_desc(
                    fan_out, fan_in, layout, self.quantization.sgl(), 0
                )
                quantized_weights = Tensor.empty(
                    device, (quantized_desc.size,), 'uint8_t', usage=usage
                )

                cmd = device.create_command_encoder()
                cmd.convert_coop_vec_matrix(
                    quantized_weights.storage,
                    quantized_desc,
                    backward_weights.storage,
                    backward_weight_desc,
                )
                device.wait_for_submit(device.submit_command_buffer(cmd.finish()))
                self.forward_weight_desc = quantized_desc
                self.forward_weights = quantized_weights
            else:
                self.forward_weight_desc = backward_weight_desc
                self.forward_weights = backward_weights
        else:
            backward_weights = Tensor.empty(device, weights_np.shape, str(self.dtype), usage=usage)
            backward_weights.storage.copy_from_numpy(weights_np)

            self.backward_weight_desc = None
            self.forward_weight_desc = None
            self.forward_weights = backward_weights

        self.backward_weights = backward_weights.with_grads(zero=True)
        assert self.backward_weights.grad_out is not None
        self.weight_grads = self.backward_weights.grad_out
        self.weight_grad_stride = 0

        if self.use_biases:
            num_biases = fan_out
            if self.use_coopvec:
                num_bias_bytes = fan_out * self.dtype.size()
                if num_bias_bytes % _COOPVEC_VECTOR_ALIGNMENT:
                    num_bias_bytes += _COOPVEC_VECTOR_ALIGNMENT - (
                        num_bias_bytes % _COOPVEC_VECTOR_ALIGNMENT
                    )
                num_biases = num_bias_bytes // self.dtype.size()
            self.biases = Tensor.zeros(
                device, (num_biases,), str(self.dtype), usage=usage
            ).with_grads(zero=True)

            assert self.biases.grad_out is not None
            self.bias_grads = self.biases.grad_out
            self.bias_grad_stride = 0

    def model_params(self):
        return [self.backward_weights, self.biases] if self.use_biases else [self.backward_weights]

    def resolve_input_type(self, module: Module):
        if self._num_inputs is Auto:
            return None

        return RealArray(ArrayKind.array, resolve_auto(self._dtype, Real.float), self._num_inputs)

    def _request_padding(self, multiple: int):
        """Pad parameter storage for deterministic gradient reduction."""

        weights_padded = (
            (self.backward_weights.element_count + multiple - 1) // multiple
        ) * multiple
        if weights_padded != self.backward_weights.element_count:
            padded = Tensor.zeros(
                self.backward_weights.device,
                (weights_padded,),
                str(self.dtype),
                usage=self.backward_weights.usage,
            ).with_grads(zero=True)
            padded = padded.view(self.backward_weights.shape)
            padded.copy_from_numpy(self.backward_weights.to_numpy())
            self.backward_weights = padded

        if self.use_biases:
            biases_padded = ((self.biases.element_count + multiple - 1) // multiple) * multiple
            if biases_padded != self.biases.element_count:
                padded = Tensor.zeros(
                    self.biases.device, (biases_padded,), str(self.dtype), usage=self.biases.usage
                ).with_grads(zero=True)
                padded = padded.view(self.biases.shape)
                padded.copy_from_numpy(self.biases.to_numpy())
                self.biases = padded

    def _request_deterministic_grads(self, weight_grads: Tensor, bias_grads: Tensor | None):
        """Redirect gradient writes to the deterministic reducer's buffers."""

        self.weight_grads = weight_grads
        self.weight_grad_stride = weight_grads.strides[0]

        if self.use_biases:
            if bias_grads is None:
                raise ValueError('Bias gradients are required when layer uses biases')
            self.bias_grads = bias_grads
            self.bias_grad_stride = bias_grads.strides[0]
        elif bias_grads is not None:
            raise ValueError('bias_grads are not None, but layer does not use biases')

    def model_data(self):
        data = {
            'forward_weights': self.forward_weights.storage.descriptor_handle_ro,
            'backward_weights': self.backward_weights.storage.descriptor_handle_ro,
            'forward_weight_offset': self.forward_weights.offset,
            'backward_weight_offset': self.backward_weights.offset,
            'weight_grads': self.weight_grads.storage.descriptor_handle_rw,
            'weight_grad_offset': self.weight_grads.offset,
            'weight_grad_stride': self.weight_grad_stride,
        }

        if self.use_biases:
            assert self.biases.is_contiguous()
            data['biases'] = self.biases.storage.descriptor_handle_ro
            data['bias_offset'] = self.biases.offset
            data['bias_grads'] = self.bias_grads.storage.descriptor_handle_rw
            data['bias_grad_offset'] = self.bias_grads.offset
            data['bias_grad_stride'] = self.bias_grad_stride
        else:
            data['bias_offset'] = 0
            data['bias_grad_offset'] = 0
            data['bias_grad_stride'] = 0

        return data
