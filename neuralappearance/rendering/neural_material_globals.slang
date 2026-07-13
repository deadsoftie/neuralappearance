// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
module neural_material_globals;

// Per-shading-point values supplied by the renderer when constructing a neural
// material instance. They control stochastic texture filtering and either
// dynamic footprint-based LOD or an explicitly selected checkpoint mip.
public struct NeuralMaterialGlobals
{
    public float2 duv_dx = float2(0.f);
    public float2 duv_dy = float2(0.f);
    public int fixed_mip_level = 0;
    public float mip_sample = 0.f;
    public float texel_sample = 0.f;
};
public static NeuralMaterialGlobals g_neural_material_globals = {};
