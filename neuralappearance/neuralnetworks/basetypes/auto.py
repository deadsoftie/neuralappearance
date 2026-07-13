# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TypeVar


class AutoType:
    """Sentinel type for model properties inferred during initialization.

    ``Auto`` defers properties that depend on a reflected input type. For
    example, a linear layer can infer its input width and precision from the
    preceding model during initialization.
    """

    def __str__(self):
        return 'Auto'


Auto = AutoType()
T = TypeVar('T')
# The caller can supply this value or defer it until model initialization.
AutoSettable = T | AutoType


def resolve_auto[T](auto_settable: AutoSettable[T], default: T) -> T:
    """Resolve an ``Auto`` value to the inferred default."""
    if auto_settable is Auto:
        return default
    assert not isinstance(auto_settable, AutoType)
    return auto_settable
