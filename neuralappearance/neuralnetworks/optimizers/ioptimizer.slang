// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing neural_networks;

__include basetypes.ireal;

// State stored for each scalar parameter, such as Adam's moments. The host
// constructs one ``State`` value per element from its initial value.
public interface IOptimizerState<T : IReal>
{
    public __init(T param);
}

// GPU optimizer contract. ``step()`` defines the scalar update, while
// ``batch_step()`` accesses batches assembled by ``optimizer.py``.
public interface IOptimizer<T : IReal>
{
    public associatedtype State : IOptimizerState<T>;
    // Bundle of one parameter tensor, its gradients, and its ``State`` buffer.
    public associatedtype Batch;

    public void step(inout State state, inout T param, inout T grad);
    public void batch_step(ITensor<Batch, 1> batches, int2 batch_index);
}
