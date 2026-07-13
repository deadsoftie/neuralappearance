// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing neural_networks;

__include losses.l1;
__include losses.l1_with_power_log;
__include losses.l1_with_power_root;
__include losses.l1_with_safe_log;
__include losses.l2;
__include losses.relative_l1;
__include losses.relative_l2;
__include losses.smape;

public interface ILossFunction<int N>
{
    // Return one differentiable scalar for a prediction/reference pair.
    // Implementations own channel weighting and normalization.
    [BackwardDifferentiable]
    public float eval(float pred[N], float[N] ref);
}
