// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing neural_networks;

__include ioptimizer;

/*
    Runs a nested optimizer, such as Adam, on a full-precision parameter copy,
    then converts it to half precision after each ``step()``. This avoids
    accumulating half-precision rounding errors during training.

    Float parameters are optimized directly. Half parameters retain a float
    copy.
*/
public struct FullPrecisionOptimizer<
    T : IReal,
    NestedOptimizer : IOptimizer<float>
>
{
    public float gradient_scale;
    public NestedOptimizer nested_optim;
}

// Full-precision parameters pass directly to the nested optimizer.
public extension<NestedOptimizer : IOptimizer<float>>
FullPrecisionOptimizer<float, NestedOptimizer> : IOptimizer<float>
{
    public typealias State = NestedOptimizer::State;
    public typealias Batch = NestedOptimizer::Batch;

    public void step(inout State state, inout float param, inout float grad)
    {
        nested_optim.step(state, param, grad);
    }
    public void batch_step(ITensor<Batch, 1> batches, int2 batch_index)
    {
        nested_optim.batch_step(batches, batch_index);
    }
}

// Half-precision parameters retain and optimize a full-precision copy, then
// convert it back after each update.
public extension<NestedOptimizer : IOptimizer<float>>
FullPrecisionOptimizer<half, NestedOptimizer> : IOptimizer<half>
{
    public typealias State = FullPrecisionParameterHalfState<NestedOptimizer>;
    public typealias Batch = FullPrecisionParameterHalfBatch<NestedOptimizer>;

    public void step(inout State state, inout half param_h, inout half grad_h)
    {
        float grad_f = float(grad_h) / gradient_scale;
        nested_optim.step(state.nested_state, state.param_f, grad_f);
        grad_h = half(grad_f);
        param_h = half(state.param_f);
    }

    public void batch_step(ITensor<Batch, 1> batches, int2 batch_index)
    {
        Batch batch = batches[batch_index.x];

        int i = batch_index.y;

        half param = batch.params[i];
        half grad = batch.grads[i];
        State state = batch.states[i];

        step(state, param, grad);

        batch.params[i] = param;
        batch.grads[i] = grad;
        batch.states[i] = state;
    }
}

// This struct definition should ideally move into the extension for T==half,
// (and avoid the NestedOptimizer generic param), but this triggers two separate
// reflection bugs (slang issue #6544, #6546)
public struct FullPrecisionParameterHalfState<
    NestedOptimizer : IOptimizer<float>
> : IOptimizerState<half>
{
    public float param_f;

    public NestedOptimizer::State nested_state;

    public __init(half param_h)
    {
        param_f = float(param_h);
        nested_state = NestedOptimizer::State(param_f);
    }
}
public struct FullPrecisionParameterHalfBatch<
    NestedOptimizer : IOptimizer<float>
>
{
    RWStructuredBuffer<half>.Handle params;
    RWStructuredBuffer<half>.Handle grads;
    RWStructuredBuffer<FullPrecisionParameterHalfState<NestedOptimizer>>.Handle states;
}
