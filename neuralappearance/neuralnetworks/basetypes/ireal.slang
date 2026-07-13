// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing neural_networks;

// Short name for Slang's differentiable scalar constraint, which currently
// accepts ``half``, ``float``, and ``double``. Model implementations use this
// alias to keep generic declarations readable.
public typealias IReal = __BuiltinFloatingPointType;
