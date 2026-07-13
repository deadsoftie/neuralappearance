# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib

from slangpy import Tensor

from .basetypes.real import Real


def slang_include_paths() -> list[pathlib.Path]:
    return [pathlib.Path(__file__).parent]


def merge_tensors(tensors: list[Tensor], alignment: int = 0) -> list[Tensor]:
    """
    Transforms list of input tensors into views of a larger merged tensor.

    This is helpful for reducing the number of dispatches the optimizer needs
    to do by merging a potentially large number of parameter buffers allocated
    by a model into a small number of combined tensors.

    Tensors are grouped by element type, and for each group, a
    1D tensor is allocated to hold the combined input tensors.
    The input tensors are then turned into views of this larger tensor.
    The original data is copied over, and the merged tensors are returned.

    ``alignment`` specifies a required byte alignment, for example for
    cooperative-vector matrices.

    The input tensors are modified to point into the merged tensor. They become
    dense views, so custom strides are not retained. Gradient tensors are also
    merged and zeroed.
    """
    tensors_by_dtype: dict[Real, list[Tensor]] = {
        Real.half: [],
        Real.float: [],
        Real.double: [],
    }

    for i, tensor in enumerate(tensors):
        dtype = Real.from_slangtype(tensor.dtype)
        if dtype is None:
            raise ValueError(
                f"Unsupported element type '{tensor.dtype.full_name}' "
                f'of tensor {i}: Must be half, float or double'
            )

        tensors_by_dtype[dtype].append(tensor)

    result: list[Tensor] = []

    for dtype, dtype_tensor in tensors_by_dtype.items():
        if len(dtype_tensor) == 0:
            continue

        # Convert byte alignment to element alignment.
        dtype_size = dtype.size()
        if alignment < dtype_size:
            element_alignment = 1
        else:
            element_alignment = alignment // dtype_size
        if (element_alignment * dtype_size % alignment) != 0:
            raise ValueError(f"Requested alignment of {alignment} can't be satisfied with {dtype}")

        # Compute aligned offsets and the total combined size.
        offsets = []
        offset = 0
        for param_tensor in dtype_tensor:
            misalignment = offset % element_alignment
            if misalignment != 0:
                offset += element_alignment - misalignment

            offsets.append(offset)
            offset += param_tensor.element_count

        total_count = offset

        # Allocate the merged tensor.
        merged_params = Tensor.empty(dtype_tensor[0].device, (total_count,), dtype_tensor[0].dtype)
        merged_params = merged_params.with_grads(zero=True)

        # Point each input tensor at a slice of the merged tensor.
        for offset, param_tensor in zip(offsets, dtype_tensor, strict=True):
            param_slice = merged_params.view(param_tensor.shape, offset=offset)
            param_slice.copy_from_numpy(param_tensor.to_numpy())
            param_tensor.point_to(param_slice)

            if param_tensor.grad_out is not None:
                grad_slice = merged_params.grad_out.view(param_tensor.shape, offset=offset)
                param_tensor.grad_out.point_to(grad_slice)

        result.append(merged_params)

    return result
