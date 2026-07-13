# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python views of the generic dimensions in the shared Slang model inputs.

These classes do not hold runtime shader data. They parse reflected Slang types
so Python model stages can validate their inputs and obtain dimensions such as
the latent-channel or shading-frame count.
"""

from __future__ import annotations

from dataclasses import dataclass

from slangpy.reflection import SlangType


@dataclass(frozen=True)
class EncoderInput:
    num_params: int
    num_latents: int

    @classmethod
    def from_slangtype(cls, slang_type: SlangType) -> EncoderInput | None:
        if not _slang_type_matches(slang_type, 'EncoderInput'):
            return None
        args = _slang_type_int_args(slang_type, 2)
        if args is None:
            return None
        return cls(num_params=args[0], num_latents=args[1])


@dataclass(frozen=True)
class BsdfDecoderInputWithFrames:
    num_latents: int
    num_frames: int

    @classmethod
    def from_slangtype(cls, slang_type: SlangType) -> BsdfDecoderInputWithFrames | None:
        if _slang_type_matches(slang_type, 'BsdfDecoderInputWithFrames'):
            args = _slang_type_int_args(slang_type, 2)
            if args is None:
                return None
            return cls(num_latents=args[0], num_frames=args[1])

        if _slang_type_matches(slang_type, 'BsdfDecoderInput'):
            # ``base_types.slang`` defines ``BsdfDecoderInput<N>`` as shorthand
            # for ``BsdfDecoderInputWithFrames<N, 1>``.
            args = _slang_type_int_args(slang_type, 1)
            if args is None:
                return None
            return cls(num_latents=args[0], num_frames=1)

        return None


@dataclass(frozen=True)
class AuxDecoderInput:
    num_latents: int

    @classmethod
    def from_slangtype(cls, slang_type: SlangType) -> AuxDecoderInput | None:
        if not _slang_type_matches(slang_type, 'AuxDecoderInput'):
            return None
        args = _slang_type_int_args(slang_type, 1)
        if args is None:
            return None
        return cls(num_latents=args[0])


@dataclass(frozen=True)
class SamplerInput:
    num_latents: int

    @classmethod
    def from_slangtype(cls, slang_type: SlangType) -> SamplerInput | None:
        if not _slang_type_matches(slang_type, 'SamplerInput'):
            return None
        args = _slang_type_int_args(slang_type, 1)
        if args is None:
            return None
        return cls(num_latents=args[0])


def _slang_type_matches(slang_type: SlangType, name: str) -> bool:
    # Accept both the base name and a generic specialization of it.
    return slang_type.name == name or slang_type.full_name.startswith(f'{name}<')


def _slang_type_int_args(slang_type: SlangType, count: int) -> tuple[int, ...] | None:
    # Array sizes such as the latent and frame counts are generic arguments in
    # Slang and become ordinary integers on the Python side.
    args = slang_type.program.get_resolved_generic_args(slang_type)
    if args is None:
        return None
    args = tuple(args)
    if len(args) != count or not all(isinstance(arg, int) for arg in args):
        return None
    return args
