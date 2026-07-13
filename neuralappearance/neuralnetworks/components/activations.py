# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from slangpy import Module

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


class Activation(IModel):
    """Python wrapper for element-wise activation models implemented in Slang.

    Width and precision may be inferred from the reflected input. Subclasses
    select a struct in the Slang ``Activation`` namespace and expose runtime
    coefficients through ``model_data()`` when needed.
    """

    def __init__(self, act_name: str, width: AutoSettable[int], dtype: AutoSettable[Real]):
        super().__init__()

        self.act_name = act_name
        self._width = width
        self._dtype = dtype

    def model_init(self, module: Module, input_type: SlangType):
        input_array = RealArray.from_slangtype(input_type)
        self.width = resolve_auto(self._width, input_array.length)
        self.dtype = resolve_auto(self._dtype, input_array.dtype)

    def resolve_input_type(self, module: Module):
        # Width must be known because it has no independent default.
        if self._width is Auto:
            return None

        # Unless specified, default to a float array for input/output.
        return RealArray(ArrayKind.array, resolve_auto(self._dtype, Real.float), self._width)

    @property
    def type_name(self) -> str:
        return f'Activation::{self.act_name}<{self.dtype}, {self.width}>'


class Identity(Activation):
    def __init__(self, width: AutoSettable[int] = Auto, dtype: AutoSettable[Real] = Auto):
        super().__init__('Identity', width, dtype)


class ReLU(Activation):
    def __init__(self, width: AutoSettable[int] = Auto, dtype: AutoSettable[Real] = Auto):
        super().__init__('ReLU', width, dtype)


class LeakyReLU(Activation):
    def __init__(
        self,
        negative_slope: float = 0.01,
        width: AutoSettable[int] = Auto,
        dtype: AutoSettable[Real] = Auto,
    ):
        super().__init__('LeakyReLU', width, dtype)
        self.negative_slope = negative_slope

    def model_data(self):
        return {'negative_slope': self.negative_slope}


class ELU(Activation):
    def __init__(
        self, a: float = 1.0, width: AutoSettable[int] = Auto, dtype: AutoSettable[Real] = Auto
    ):
        super().__init__('ELU', width, dtype)
        self.a = a

    def model_data(self):
        return {'a': self.a}


class SmeLU(Activation):
    def __init__(self, width: AutoSettable[int] = Auto, dtype: AutoSettable[Real] = Auto):
        super().__init__('SmeLU', width, dtype)


class Swish(Activation):
    def __init__(self, width: AutoSettable[int] = Auto, dtype: AutoSettable[Real] = Auto):
        super().__init__('Swish', width, dtype)


class Tanh(Activation):
    def __init__(self, width: AutoSettable[int] = Auto, dtype: AutoSettable[Real] = Auto):
        super().__init__('Tanh', width, dtype)


class Sigmoid(Activation):
    def __init__(self, width: AutoSettable[int] = Auto, dtype: AutoSettable[Real] = Auto):
        super().__init__('Sigmoid', width, dtype)


class Exp(Activation):
    def __init__(self, width: AutoSettable[int] = Auto, dtype: AutoSettable[Real] = Auto):
        super().__init__('Exp', width, dtype)


class ScaledSigmoid(Activation):
    def __init__(
        self, scale: float = 1.0, width: AutoSettable[int] = Auto, dtype: AutoSettable[Real] = Auto
    ):
        super().__init__('ScaledSigmoid', width, dtype)
        self.scale = scale

    def model_data(self):
        return {'scale': self.scale}
