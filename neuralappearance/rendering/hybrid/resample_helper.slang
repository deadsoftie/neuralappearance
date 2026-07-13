// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
import slangpy;

float3 tonemap_aces_film(float3 input)
{
    float3 x = input;
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

void blit_tensor_to_texture(
    int2 pixel,
    int2 size,
    int2 offset,
    float scale,
    bool tone_map,
    Tensor<float4, 2> input,
    RWTexture2D<float4> output
)
{
    uint[2] shape = input.shape;
    float2 uv = (float2(pixel) + 0.5f) / float2(size);

    uint2 src_pixel = uint2(
        min(shape[1] - 1, uint(shape[1] * uv.x)),
        min(shape[0] - 1, uint(shape[0] * uv.y))
    );

    float4 color = input.load(src_pixel);
    color.xyz *= scale;
    if (tone_map)
        color.xyz = tonemap_aces_film(max(color.xyz, float3(0.f)));

    if (any(isnan(color.xyz)) || any(isinf(color.xyz)))
        color = float4(1.f, 0.f, 1.f, 1.f);

    output[offset + pixel] = color;
}
