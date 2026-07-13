// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing neural_networks;

__include imodel;

// Trivial model that just returns its input unmodified.
public struct Identity<T : IDifferentiable> : IModel<T, T>
{
    [BackwardDifferentiable]
    public IOType forward(IOType x) { return x; }
}
