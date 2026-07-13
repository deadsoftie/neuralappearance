// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing neural_networks;

public enum DebugRecordKind : int {
    Input,
    Output,
    InputGrad,
    OutputGrad,
    Count,
}

struct DebugRecord
{
    // The buffer is untyped because models expose different value types. The
    // host validates ``entry_size`` and records the reflected layout in JSON.
    uint entry_size;
    RWByteAddressBuffer.Handle data;
}

struct DebugRecorderInfo
{
    uint num_records;
    uint batch_size;

    StructuredBuffer<int> model_record_base;
    StructuredBuffer<DebugRecord> records;
}

DebugRecorderInfo g_recorder_info;
extern static const int k_debug_recording_enabled = 0;

public void record_value<T>(int model_id, int record_index, T x)
{
    if (k_debug_recording_enabled == 1) {
        int sample_index = NeuralConfig::sample_index;
        if (sample_index >= g_recorder_info.batch_size)
            return;
        int record_base = g_recorder_info.model_record_base[model_id];
        int record_count = g_recorder_info.model_record_base[model_id + 1] - record_base;
        if (record_count <= 0 || record_index >= record_count)
            return;
        let record = g_recorder_info.records[record_base + record_index];
        if (record.entry_size != sizeof(T))
            return;
        record.data.Store(sample_index * sizeof(T), x);
    }
}

public interface IRecordableModel<InputT : IDifferentiable, OutputT : IDifferentiable>
{
    // Implement ``model_forward()`` rather than ``forward()``. The extension
    // wraps it with operations that record values and gradients.
    public property int model_id { get; }

    [BackwardDifferentiable]
    public OutputT model_forward(InputT x);
}

public extension<
    InputT : IDifferentiable,
    OutputT : IDifferentiable,
    Model : IRecordableModel<InputT, OutputT>
> Model : IModel<InputT, OutputT>
{
    [BackwardDifferentiable]
    public OutputT forward(InputT x)
    {
        x = record_differentiable_value(model_id, DebugRecordKind::Input, DebugRecordKind::InputGrad, x);
        let y = model_forward(x);
        return record_differentiable_value(model_id, DebugRecordKind::Output, DebugRecordKind::OutputGrad, y);
    }

    [BackwardDerivative(record_differentiable_value_bwd)]
    public static T record_differentiable_value<T : IDifferentiable>(int model_id, int fwd_record, int bwd_record, T x)
    {
        if (k_debug_recording_enabled == 1)
            record_value(model_id, fwd_record, x);

        return x;
    }
    public static void record_differentiable_value_bwd<T : IDifferentiable>(
        int model_id,
        int fwd_record,
        int bwd_record,
        inout DifferentialPair<T> x,
        T.Differential grad
    )
    {
        if (k_debug_recording_enabled == 1)
            record_value(model_id, bwd_record, grad);

        x = diffPair(x.p, grad);
    }
}

public interface ICoopRecordableModel<T : IReal, int NumIn, int NumOut>
{
    // Cooperative vectors use a separate interface because debug buffers store
    // them as ordinary arrays.
    public property int model_id { get; }

    [BackwardDifferentiable]
    public CoopVec<T, NumOut> model_forward(CoopVec<T, NumIn> x);
}

public extension<
    T : IReal,
    int NumIn,
    int NumOut,
    Model : ICoopRecordableModel<T, NumIn, NumOut>
> Model : IModel<CoopVec<T, NumIn>, CoopVec<T, NumOut>>
{
    [BackwardDifferentiable]
    public OutputType forward(InputType x)
    {
        x = record_differentiable_value(model_id, DebugRecordKind::Input, DebugRecordKind::InputGrad, x);
        let y = model_forward(x);
        return record_differentiable_value(model_id, DebugRecordKind::Output, DebugRecordKind::OutputGrad, y);
    }

    [BackwardDerivative(record_differentiable_value_bwd)]
    public static CoopVec<ElemT, N> record_differentiable_value<ElemT : IReal, int N>(
        int model_id,
        int fwd_record,
        int bwd_record,
        CoopVec<ElemT, N> x
    )
    {
        if (k_debug_recording_enabled == 1)
            record_value(model_id, fwd_record, coop_vec_to_array(x));

        return x;
    }
    public static void record_differentiable_value_bwd<ElemT : IReal, int N>(
        int model_id,
        int fwd_record,
        int bwd_record,
        inout DifferentialPair<CoopVec<ElemT, N>> x,
        CoopVec<ElemT, N>.Differential grad
    )
    {
        if (k_debug_recording_enabled == 1)
            record_value(model_id, bwd_record, coop_vec_to_array(grad));

        x = diffPair(x.p, grad);
    }
}

namespace impl
{

void recordable_witness<
    InputT : IDifferentiable,
    OutputT : IDifferentiable,
    Model : IRecordableModel<InputT, OutputT>
>() {}

void recordable_witness<
    T : IReal,
    int NumIn,
    int NumOut,
    Model : ICoopRecordableModel<T, NumIn, NumOut>
>() {}

}
