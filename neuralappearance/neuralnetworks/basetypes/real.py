# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum

import numpy as np
from slangpy import DataType, TypeReflection
from slangpy.reflection import ScalarType, SlangType


class Real(Enum):
    """Scalar precision shared by the Python and Slang model APIs.

    ``Real`` maps NumPy, Slang reflection, SGL, and cooperative-vector type
    names. Its string form is a valid Slang type name used when Python builds
    reflected generic model types.
    """

    half = 1
    float = 2
    double = 3

    def __str__(self):
        """Return the Slang scalar name for this ``Real``."""
        return self._name_

    def numpy(self):
        """Return the equivalent ``numpy.dtype``."""
        if self is Real.half:
            return np.float16
        elif self is Real.float:
            return np.float32
        elif self is Real.double:
            return np.float64
        else:
            raise ValueError(f"Invalid Real type '{self}'")

    def slang(self):
        """Return the equivalent ``TypeReflection.ScalarType``."""
        if self is Real.half:
            return TypeReflection.ScalarType.float16
        elif self is Real.float:
            return TypeReflection.ScalarType.float32
        elif self is Real.double:
            return TypeReflection.ScalarType.float64
        else:
            raise ValueError(f"Invalid Real type '{self}'")

    def sgl(self):
        """Return the equivalent ``sgl.DataType``."""
        if self is Real.half:
            return DataType.float16
        elif self is Real.float:
            return DataType.float32
        elif self is Real.double:
            return DataType.float64
        else:
            raise ValueError(f"Invalid Real type '{self}'")

    def coopvec(self):
        """Return the equivalent Slang ``CoopVecComponentType`` name."""
        if self is Real.half:
            return 'CoopVecComponentType::Float16'
        elif self is Real.float:
            return 'CoopVecComponentType::Float32'
        elif self is Real.double:
            return 'CoopVecComponentType::Float64'
        else:
            raise ValueError(f"Invalid Real type '{self}'")

    def size(self):
        """Return this scalar type's size in bytes."""
        if self is Real.half:
            return 2
        elif self is Real.float:
            return 4
        elif self is Real.double:
            return 8
        else:
            raise ValueError(f"Invalid Real type '{self}'")

    @staticmethod
    def from_slangtype(st: SlangType | None) -> Real | None:
        """Convert a reflected Slang type to ``Real``, if supported."""
        if not isinstance(st, ScalarType):
            return None

        if st.slang_scalar_type == TypeReflection.ScalarType.float16:
            return Real.half
        elif st.slang_scalar_type == TypeReflection.ScalarType.float32:
            return Real.float
        elif st.slang_scalar_type == TypeReflection.ScalarType.float64:
            return Real.double

        return None
