// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing neural_networks;

__include basetypes.imodel;

/*
    Linear neural-network layer computing ``A * x + b``.

    Takes an input with NumInputs elements, and returns NumOutputs elements.

    LinearLayer implements the multiply in software for plain arrays,
    and CoopVecLinearLayer uses hardware acceleration using cooperative vectors.
*/
public struct LinearLayer<
    T : IReal,
    int NumInputs,
    int NumOutputs,
    int UseBiases = 1
> : IRecordableModel<T[NumInputs], T[NumOutputs]> where T.Differential : IArithmeticAtomicable
{
    public int model_id;
    public StructuredBuffer<T>.Handle forward_weights, backward_weights, biases;
    public uint forward_weight_offset, backward_weight_offset, bias_offset;
    public RWStructuredBuffer<Atomic<T.Differential>>.Handle weight_grads, bias_grads;
    public uint weight_grad_offset, bias_grad_offset;
    public uint weight_grad_stride, bias_grad_stride;

    [BackwardDerivative(model_backward)]
    public OutputType model_forward(InputType x)
    {
        OutputType y;
        for (int row = 0; row < NumOutputs; ++row)
        {
            var sum = UseBiases == 1 ? biases[bias_offset + row] : T(0);
            [ForceUnroll]
            for (int col = 0; col < NumInputs; ++col)
                sum += forward_weights[forward_weight_offset + row * NumInputs + col] * x[col];

            y[row] = sum;
        }

        return y;
    }

    public void model_backward(inout DifferentialPair<InputType> x, OutputType.Differential grad)
    {
        uint gradient_index = NeuralConfig::global_gradient_index;
        uint sample_weight_grad_offset = weight_grad_offset + weight_grad_stride * gradient_index;
        uint sample_bias_grad_offset = bias_grad_offset + bias_grad_stride * gradient_index;

        var dx = InputType.dzero();
        for (int row = 0; row < NumOutputs; ++row)
        {
            if (UseBiases == 1)
                bias_grads[sample_bias_grad_offset + row].add(grad[row]);

            [ForceUnroll]
            for (int col = 0; col < NumInputs; ++col)
            {
                weight_grads[sample_weight_grad_offset + row * NumInputs + col].add(
                    __slang_noop_cast<T.Differential>(x.p[col] * __slang_noop_cast<T>(grad[row]))
                );
                dx[col] = __slang_noop_cast<T.Differential>(
                    __slang_noop_cast<T>(dx[col])
                    + backward_weights[backward_weight_offset + row * NumInputs + col]
                        * __slang_noop_cast<T>(grad[row])
                );
            }
        }
        x = diffPair(x.p, dx);
    }
}

public struct CoopVecLinearLayer<
    T : IReal,
    int NumInputs,
    int NumOutputs,
    CoopVecComponentType ForwardComponentType,
    int UseBiases = 1
> : ICoopRecordableModel<T, NumInputs, NumOutputs>
{
    // Forward weights may use FP8. Backpropagation reads separate trainable
    // weights with ``component_type`` so input gradients do not use FP8 values.
    static const CoopVecComponentType component_type =
        (T is half) ? CoopVecComponentType.Float16 :
        (T is float) ? CoopVecComponentType.Float32 :
        CoopVecComponentType.Float64;

    public int model_id;
    public ByteAddressBuffer.Handle forward_weights, backward_weights, biases;
    public uint forward_weight_offset, backward_weight_offset, bias_offset;
    public RWByteAddressBuffer.Handle weight_grads, bias_grads;
    public uint weight_grad_offset, bias_grad_offset;
    public uint weight_grad_stride, bias_grad_stride;

    [BackwardDerivative(model_backward)]
    public OutputType model_forward(InputType x)
    {
        // ``inputInterpretation`` must match ``matrixInterpretation``. The
        // compiler converts the float16 input to the requested format.
        if (UseBiases == 1) {
            return coopVecMatMulAdd<T, NumOutputs>(
                x,
                ForwardComponentType,
                *forward_weights,
                forward_weight_offset * sizeof(T),
                ForwardComponentType,
                *biases,
                bias_offset * sizeof(T),
                component_type,
                CoopVecMatrixLayout.TrainingOptimal,
                false,
                0
            );
        } else {
            return coopVecMatMul<T, NumOutputs>(
                x,
                ForwardComponentType,
                *forward_weights,
                forward_weight_offset * sizeof(T),
                ForwardComponentType,
                CoopVecMatrixLayout.TrainingOptimal,
                false,
                0
            );
        }
    }

    public void model_backward(inout DifferentialPair<InputType> x, OutputType grad)
    {
        uint gradient_index = NeuralConfig::global_gradient_index;
        uint sample_weight_grad_offset = sizeof(T) * (weight_grad_offset + weight_grad_stride * gradient_index);
        uint sample_bias_grad_offset = sizeof(T) * (bias_grad_offset + bias_grad_stride * gradient_index);

        coopVecOuterProductAccumulate(
            grad,
            x.p,
            *weight_grads,
            sample_weight_grad_offset,
            0,
            CoopVecMatrixLayout.TrainingOptimal,
            component_type
        );
        if (UseBiases == 1)
            coopVecReduceSumAccumulate(grad, *bias_grads, sample_bias_grad_offset);

        let d_x = coopVecMatMul<T, NumInputs>(
            grad,
            component_type,
            *backward_weights,
            backward_weight_offset * sizeof(T),
            component_type,
            CoopVecMatrixLayout.TrainingOptimal,
            true,
            0
        );

        x = diffPair(x.p, d_x);
    }
}
