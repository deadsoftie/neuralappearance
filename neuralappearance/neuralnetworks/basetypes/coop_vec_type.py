# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from slangpy import TypeReflection
from slangpy.reflection import SlangProgramLayout, SlangType


class _CoopVecTypeMeta(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance, SlangType) and instance.name == 'CoopVec'


class CoopVecType(metaclass=_CoopVecTypeMeta):
    """Parse reflected ``CoopVec<T, N>`` types.

    This wrapper extracts the element type and width for Python initialization.
    """

    def __init__(self, program: SlangProgramLayout, refl: TypeReflection | SlangType):
        args = program.get_resolved_generic_args(refl)
        assert args is not None
        assert len(args) == 2
        assert isinstance(args[0], SlangType)
        assert isinstance(args[1], int)
        self.element_type: SlangType = args[0]
        self._dims = args[1]

    @classmethod
    def from_slangtype(cls, slang_type: SlangType) -> CoopVecType | None:
        if not isinstance(slang_type, cls):
            return None
        return cls(slang_type.program, slang_type)

    @property
    def dims(self) -> int:
        return self._dims

    @property
    def dtype(self) -> SlangType:
        return self.element_type
