// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing neural_networks;

// Common interface for every stage that can be composed by Python's IModel.
// Implementations provide a differentiable forward transformation. They do not
// need to own trainable parameters.
public interface IModel<InputT : IDifferentiable, OutputT : IDifferentiable>
{
    [BackwardDifferentiable]
    public OutputT forward(InputT x);
}

// Make an implementation's declared input and output types available by name.
public extension<InputT : IDifferentiable, OutputT : IDifferentiable, Model : IModel<InputT, OutputT>> Model
{
    public typealias InputType = InputT;
    public typealias OutputType = OutputT;
}
// In-place transformations can use IOType when their input and output match.
public extension<T : IDifferentiable, Model : IModel<T, T>> Model
{
    public typealias IOType = T;
}

// ``imodel.py`` uses this witness to resolve overloaded ``forward()`` return
// types.
// Application shaders do not call it directly.
namespace impl
{
    public void return_type_witness<
        InputT : IDifferentiable,
        OutputT : IDifferentiable,
        Model : IModel<InputT, OutputT>
    >() {}
}

// Per-invocation context shared by neural network components. Training kernels
// set the batch index before model evaluation; layers use the derived gradient
// index for accumulation, while debug recording uses the original sample index.
public struct NeuralConfig
{
    static public constexpr uint k_warp_size = 32;
    static uint global_gradient_index = 0;
    static uint sample_index = 0;

    static public void set_batch_index(int batch_sample_index)
    {
        sample_index = batch_sample_index;
        global_gradient_index = batch_sample_index / k_warp_size;
    }
}
