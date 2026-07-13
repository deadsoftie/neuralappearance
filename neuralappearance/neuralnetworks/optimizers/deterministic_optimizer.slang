// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing neural_networks;

public struct AccumulateTask
{
    // One task describes every fixed-size block in a parameter tensor. Each
    // block reduces ``accum_count`` independently accumulated gradient slices.
    public RWByteAddressBuffer.Handle src;
    public RWByteAddressBuffer.Handle dst;
    public uint group_base;

    public uint src_offset;
    public uint dst_offset;
    public uint accum_stride;
    public uint accum_count;

    [mutating]
    void finalize_offsets<T : IReal, int BlockSize>(uint global_block_index)
    {
        uint group_index = (global_block_index - group_base) * BlockSize;
        src_offset = (src_offset + group_index) * sizeof(T);
        dst_offset = (dst_offset + group_index) * sizeof(T);
        accum_stride *= sizeof(T);
    }
};

// Serial reduction avoids scheduling-dependent atomics and suits a small
// slice count.
public void single_threaded_accumulate<T : IReal, int BlockSize>(
    StructuredBuffer<AccumulateTask> tasks,
    StructuredBuffer<uint> task_map,
    uint call_id
)
{
    typealias CV = CoopVec<T, BlockSize>;

    uint block_index = call_id;
    uint task_index = task_map[block_index];
    var task = tasks[task_index];
    task.finalize_offsets<T, BlockSize>(block_index);

    uint offset = task.src_offset;
    var x = CV::load(task.src, offset);
    for (uint item = 0; item < task.accum_count - 1; ++item) {
        offset += task.accum_stride;
        x = x + CV::load(task.src, offset);
    }
    x.store(task.dst, task.dst_offset);
}

// Cooperative reduction distributes more slices over a wave, then combines
// lanes without unordered global atomic additions.
public void wave_cooperative_accumulate<T : IReal, int BlockSize>(
    StructuredBuffer<AccumulateTask> tasks,
    StructuredBuffer<uint> task_map,
    uint call_id
)
{
    typealias CV = CoopVec<T, BlockSize>;
    constexpr uint wave_size = WaveGetLaneCount();

    uint block_index = call_id / wave_size;
    uint task_index = task_map[block_index];
    var task = tasks[task_index];
    task.finalize_offsets<T, BlockSize>(block_index);

    const int thread = WaveGetLaneIndex();
    uint items_per_thread = (task.accum_count + wave_size - 1) / wave_size;

    uint offset = task.src_offset + thread * task.accum_stride;
    var x = CV::load(task.src, offset);
    for (uint item = 0; item < items_per_thread - 1; ++item) {
        offset += task.accum_stride * wave_size;
        x = x + CV::load(task.src, offset);
    }
    if (thread < (BlockSize * sizeof(T)) / 4)
        task.dst.Store(task.dst_offset + thread * 4, 0, 4);
    coopVecReduceSumAccumulate(x, task.dst, task.dst_offset);
}
