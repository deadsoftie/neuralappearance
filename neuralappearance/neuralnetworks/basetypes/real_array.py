# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum

from slangpy import TypeReflection
from slangpy.reflection import ArrayType, ScalarType, SlangType, VectorType

from .coop_vec_type import CoopVecType
from .real import Real


class ArrayKind(Enum):
    """Storage form of a fixed-width, real-valued Slang sequence.

    ``array`` maps to types such as ``float[10]``, ``vector`` to ``float2`` or
    ``vector<float, N>``, and ``coopvec`` to ``CoopVec<float, N>``.
    """

    array = 0
    vector = 1
    coopvec = 2

    def __str__(self):
        return self._name_


class RealArray:
    """Python description of a fixed-width, real-valued Slang sequence type.

    Models use it to construct input types and to normalize reflected arrays,
    vectors, and ``CoopVec`` values into three common properties.
    """

    def __init__(self, kind: ArrayKind, dtype: Real, length: int):
        super().__init__()
        self.kind = kind
        self.dtype = dtype
        self.length = length

    def name(self):
        """Return the equivalent Slang type declaration."""
        if self.kind == ArrayKind.array:
            return f'{self.dtype}[{self.length}]'
        elif self.kind == ArrayKind.vector:
            if self.length <= 4:
                return f'{self.dtype}{self.length}'
            else:
                return f'vector<{self.dtype}, {self.length}>'
        else:
            return f'CoopVec<{self.dtype}, {self.length}>'

    def __str__(self):
        """Return the equivalent Slang type declaration."""
        return self.name()

    @staticmethod
    def from_slangtype(st: SlangType) -> RealArray:
        """Parse a ``RealArray`` from a reflected ``SlangType``.

        Raise an exception unless ``st`` is a one-dimensional array, vector,
        or cooperative vector with a supported real element type.
        """
        kind: ArrayKind | None = None
        if isinstance(st, ArrayType):
            kind = ArrayKind.array
        elif isinstance(st, VectorType):
            kind = ArrayKind.vector
        elif isinstance(st, CoopVecType):
            kind = ArrayKind.coopvec

        if kind == ArrayKind.coopvec:
            coopvec = CoopVecType.from_slangtype(st)
            assert coopvec is not None
            element_type = coopvec.element_type
            shape = (coopvec.dims,)
        else:
            element_type = st.element_type
            shape = st.shape

        if kind is None or len(shape) != 1 or element_type is None:
            raise ValueError(
                'Expected a 1D array-like input type (vector, array, coopvec, etc.), '
                f"received '{st.full_name}' instead"
            )

        dtype: Real | None = None
        if isinstance(element_type, ScalarType):
            scalar = element_type.slang_scalar_type
            if scalar == TypeReflection.ScalarType.float16:
                dtype = Real.half
            elif scalar == TypeReflection.ScalarType.float32:
                dtype = Real.float
            elif scalar == TypeReflection.ScalarType.float64:
                dtype = Real.double

        if dtype is None:
            raise ValueError(
                'Expected an input with a Real element type (half, float or double). '
                f"Received '{element_type.full_name}' instead"
            )

        return RealArray(kind, dtype, shape[0])
