// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off

// Differentiable BC6H mode-10 decode matching fixed-function texture decoding.
// Mode 10 uses one subset and absolute endpoints, avoiding discrete partition
// choices and mode 11's limited endpoint separation.
//
// The integer forward pass has zero automatic derivative almost everywhere, so
// the backward pass uses a straight-through surrogate that treats quantization
// as identity. At precision 10, unquantize is affine away from saturation:
// 64*|x| + 32. Signed endpoints saturate at |x| >= 511; unsigned endpoints
// saturate at 1023. Saturation masks therefore produce genuine zero gradients.
//
// finish_unquantize also caps the output at the largest finite half, so bit
// reinterpretation only needs normal and subnormal cases. Its slope is
// discontinuous at half-exponent boundaries, and unquantize has a step at zero.
//
// Endpoints use endpoint-integer units: [-512, 511] signed or [0, 1023]
// unsigned. Python stores normalized parameters and scales them before decode.
// Initialize away from saturation to avoid zero endpoint gradients. Position
// gradients scale with endpoint separation; Adam largely absorbs this difference.

static const int BC6H_WEIGHT4[16] = {
    0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64
};

// Mode 10 constants, named so the arithmetic below reads against the spec.
static const int BC6H_MODE10_PRECISION = 10;
static const int BC6H_MODE10_INDEX_LEVELS = 15;   // 4-bit indices, 16 levels
static const int BC6H_MODE10_SIGNED_MIN = -512;
static const int BC6H_MODE10_SIGNED_MAX = 511;
static const int BC6H_MODE10_UNSIGNED_MAX = 1023;
// |value| at or above this saturates SIGNED `unquantize`, killing its gradient.
static const int BC6H_MODE10_SIGNED_SATURATION = 511;  // (1 << (precision - 1)) - 1

// BPTC `unquantize`, signed, precision 10. Exactly `64*|x| + 32` away from the
// saturation clamp -- see the header note. Written in the spec's shift form
// anyway, because that is the form the verification gate checked.
int bc6h_unquantize_signed(int value)
{
    let negative = value < 0;
    let magnitude = negative ? -value : value;
    int result;
    if (magnitude == 0)
        result = 0;
    else if (magnitude >= BC6H_MODE10_SIGNED_SATURATION)
        result = 0x7FFF;
    else
        result = ((magnitude << 15) + 0x4000) >> (BC6H_MODE10_PRECISION - 1);
    return negative ? -result : result;
}

// BPTC `unquantize`, unsigned, precision 10. Present because Stage 0 validated
// both formats bit-exactly, so switching the texture format stays a config
// change rather than new verification work. `value` is NOT sign-extended here.
int bc6h_unquantize_unsigned(int value)
{
    if (value == 0)
        return 0;
    if (value == (1 << BC6H_MODE10_PRECISION) - 1)
        return 0xFFFF;
    return ((value << 15) + 0x4000) >> (BC6H_MODE10_PRECISION - 1);
}

int bc6h_unquantize(int value, bool is_signed)
{
    return is_signed ? bc6h_unquantize_signed(value) : bc6h_unquantize_unsigned(value);
}

// Saturation masks the local gradient. Signed values saturate at |value| >= 511;
// unsigned values retain slope through 1022 and saturate only at 1023. Zero is
// not masked because unquantize has a step discontinuity there, not a flat spot.
bool bc6h_unquantize_is_saturated(int value, bool is_signed)
{
    return is_signed ? (abs(value) >= BC6H_MODE10_SIGNED_SATURATION)
                     : (value >= BC6H_MODE10_UNSIGNED_MAX);
}

// BPTC `interpolate`: one line segment, 16 positions along it.
int bc6h_interpolate(int a, int b, int weight)
{
    return (a * (64 - weight) + b * weight + 32) >> 6;
}

// BPTC `finish_unquantize`: scale the magnitude by 31/32 signed, 31/64
// unsigned. This is also what keeps the result inside the half's finite range.
int bc6h_finish_unquantize(int value, bool is_signed)
{
    if (!is_signed)
        return (value * 31) >> 6;
    if (value < 0)
        return -(((-value) * 31) >> 5);
    return (value * 31) >> 5;
}

// The real-number scale `finish_unquantize` applies, for the backward pass.
float bc6h_finish_scale(bool is_signed)
{
    return is_signed ? (31.f / 32.f) : (31.f / 64.f);
}

// Bit-pattern-reinterpret the final intermediate as an IEEE-754 half, returned
// widened to float (lossless -- every half is exactly representable).
//
// Decoded explicitly rather than through a cast intrinsic, for three reasons:
// it does not depend on which of `asfloat16` / `bit_cast` this Slang version
// exposes, it is auditable against the IEEE-754 layout, and the backward pass
// needs the exponent field anyway so it is computed here regardless.
//
// The signed case assembles magnitude and sign separately. Reinterpreting a
// negative two's-complement integer directly would produce a completely
// different float, and it is the kind of error that still yields
// plausible-looking images.
//
// No inf/nan branch: `finish_unquantize` caps the exponent field at 30. See the
// header note.
float bc6h_bits_to_float(int value, bool is_signed)
{
    let bits = is_signed ? ((value < 0 ? -value : value) & 0x7FFF) : (value & 0xFFFF);
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = bits & 0x3FF;
    float magnitude;
    if (exponent == 0)
        magnitude = float(mantissa) * exp2(-24.f);          // subnormal
    else
        magnitude = exp2(float(exponent - 15)) * (1.f + float(mantissa) * (1.f / 1024.f));
    return (is_signed && value < 0) ? -magnitude : magnitude;
}

// `d(bc6h_bits_to_float) / d(value)`, the only genuinely non-linear step in the
// chain. Within one exponent bin the half is linear in its mantissa, so the
// slope is piecewise constant and doubles at every exponent boundary -- exactly
// the "almost everywhere differentiable" the BCf paper describes.
//
// Positive for both signs of `value`: for `value < 0` the output is
// `-h(|value|)`, and increasing `value` toward zero decreases `|value|` and so
// increases the output. The pipeline is monotonic in `value` throughout.
float bc6h_bits_to_float_slope(int value)
{
    let bits = (value < 0 ? -value : value) & 0x7FFF;
    let exponent = (bits >> 10) & 0x1F;
    // Subnormals are linear with slope 2^-24 across the whole bin; normals step
    // to 2^(e-25). Using the normal formula at e == 0 would be wrong by 2x.
    return (exponent == 0) ? exp2(-24.f) : exp2(float(exponent - 25));
}

// Round and clamp to the signed [-512, 511] or unsigned [0, 1023] endpoint
// lattice. The Python packer mirrors these format-dependent ranges.
int bc6h_quantize_endpoint(float endpoint, bool is_signed)
{
    let lo = is_signed ? float(BC6H_MODE10_SIGNED_MIN) : 0.f;
    let hi = is_signed ? float(BC6H_MODE10_SIGNED_MAX) : float(BC6H_MODE10_UNSIGNED_MAX);
    return int(clamp(round(endpoint), lo, hi));
}

// The real format gives every texel a 4-bit index (16 levels) EXCEPT the
// block's anchor -- the texel at local raster position 0 (top-left of the
// 4x4 block) -- which stores only 3 bits (0-7) because the format reuses its
// top bit to save one bit per block; hardware reads that stored value and
// implicitly zero-extends it, so the anchor can only reach the LOWER half of
// the same 16-entry weight table every other texel indexes into.
//
// Use seven levels for the anchor and fifteen for every other texel.
// `is_anchor` must be true for exactly one texel per 4x4 block.
int bc6h_quantize_index(float position, bool is_anchor)
{
    let levels = is_anchor ? 7.f : float(BC6H_MODE10_INDEX_LEVELS);
    return int(clamp(round(position * levels), 0.f, levels));
}

// Sign-extend the low `BC6H_MODE10_PRECISION` bits. Idempotent for values
// already in range, so it is safe to apply to an already-signed parameter.
int bc6h_sign_extend_endpoint(int value)
{
    let mask = 1 << (BC6H_MODE10_PRECISION - 1);
    return ((value & ((1 << BC6H_MODE10_PRECISION) - 1)) ^ mask) - mask;
}

// One channel of the exact decode, from already-quantized integer endpoints.
// No autodiff: this is the ground-truth path, shared by the forward pass below
// and by any verification that compares against packed bytes.
float bc6h_decode_channel_exact(int endpoint0, int endpoint1, int index, bool is_signed)
{
    let ep0 = is_signed ? bc6h_sign_extend_endpoint(endpoint0) : endpoint0;
    let ep1 = is_signed ? bc6h_sign_extend_endpoint(endpoint1) : endpoint1;
    let a = bc6h_unquantize(ep0, is_signed);
    let b = bc6h_unquantize(ep1, is_signed);
    let interpolated = bc6h_interpolate(a, b, BC6H_WEIGHT4[index]);
    return bc6h_bits_to_float(bc6h_finish_unquantize(interpolated, is_signed), is_signed);
}

// ---------------------------------------------------------------------------
// The differentiable entry point
// ---------------------------------------------------------------------------
//
// `endpoint0`, `endpoint1`: endpoint-integer units, per channel. `position`:
// where this texel sits along the block's line segment, in [0, 1], shared by
// all three channels because BC6H stores ONE index per texel rather than one
// per channel. That sharing is a real constraint of the format and a real
// constraint on what the latent can represent, not a simplification here.
[Differentiable]
float3 bc6h_decode_mode10(float3 endpoint0, float3 endpoint1, float position,
                          no_diff bool is_signed, no_diff bool is_anchor)
{
    let index = bc6h_quantize_index(position, is_anchor);
    float3 result;
    [ForceUnroll]
    for (int c = 0; c < 3; ++c)
    {
        result[c] = bc6h_decode_channel_exact(
            bc6h_quantize_endpoint(endpoint0[c], is_signed),
            bc6h_quantize_endpoint(endpoint1[c], is_signed),
            index, is_signed);
    }
    return result;
}

// Analytic backward pass. Derivation, per channel, with `s0`/`s1` the
// saturation masks and `w` the index weight:
//
//   out = reinterpret(V),  dout/dV  = bc6h_bits_to_float_slope(V)
//   V   = finish(v),       dV/dv    = 31/32 signed, 31/64 unsigned
//   v   = (a*(64-w) + b*w)/64
//                          dv/da    = (64-w)/64
//                          dv/db    = w/64
//                          dv/dw    = (b-a)/64
//   a   = 64*ep0 + 32,     da/dep0  = 64 * s0
//   b   = 64*ep1 + 32,     db/dep1  = 64 * s1
//   w   = W(round(p*15)),  dw/dp    = (64/15) * 15 = 64        [non-anchor]
//
// The endpoint gradients never depend on how many INDEX LEVELS exist --
// `da/dep0`/`db/dep1` are always exactly 64, regardless of anchor status --
// so `d_endpoint0`/`d_endpoint1` are identical in both cases:
//
//   dout/dep0 = dout/dV * finish_scale * (64 - w) * s0
//   dout/dep1 = dout/dV * finish_scale * w * s1
//
// `dout/dp`, however, DOES depend on it, through `dw/dp`. The weight table's
// TOP reachable value is what the 64 in the non-anchor case actually is: at
// p=1, index=15, weight=64, so the continuous map p->w spans [0,64] and
// dw/dp=64 drops the "/64" from `dv/dw` entirely (`(b-a)/64 * 64 = b-a`,
// "the 64s cancel"). At the ANCHOR (see `bc6h_quantize_index`), p=1 maps to
// index=7, weight=30 (`BC6H_WEIGHT4[7]`) -- the map spans only [0,30], so
// dw/dp=30, and the cancellation is partial: `(b-a)/64 * 30 = (b-a) * 30/64`.
// Getting this wrong would not break correctness in any way a forward-value
// test could catch -- it would only make the anchor's position gradient
// about 2.1x too large, a purely-magnitude error in exactly the texel this
// file exists to treat specially. The anchor's own top-weight VALUE is
// looked up live via `BC6H_WEIGHT4[quantize_index(1.0, true)]` rather than
// hand-copied as a literal 30, so a future change to the anchor's level
// count cannot silently leave this scale stale.
//
// Unlike mode 11 the two endpoints are independent -- there is no delta
// transform, so neither collects the other's path. `position` accumulates over
// all three channels, since one index serves them all.
[BackwardDerivativeOf(bc6h_decode_mode10)]
void bc6h_decode_mode10_bwd(
    inout DifferentialPair<float3> endpoint0,
    inout DifferentialPair<float3> endpoint1,
    inout DifferentialPair<float> position,
    bool is_signed,
    bool is_anchor,
    float3 d_out)
{
    let index = bc6h_quantize_index(position.p, is_anchor);
    let weight = BC6H_WEIGHT4[index];
    let finish_scale = bc6h_finish_scale(is_signed);
    // See the derivation above: 1.0 when the weight table's reachable top is
    // 64 (non-anchor), BC6H_WEIGHT4[7]/64.0 (0.46875) at the anchor.
    let position_scale = is_anchor
        ? float(BC6H_WEIGHT4[bc6h_quantize_index(1.f, true)]) / 64.f
        : 1.f;

    float3 d_endpoint0 = float3(0.f);
    float3 d_endpoint1 = float3(0.f);
    float d_position = 0.f;

    [ForceUnroll]
    for (int c = 0; c < 3; ++c)
    {
        let quantized0 = bc6h_quantize_endpoint(endpoint0.p[c], is_signed);
        let quantized1 = bc6h_quantize_endpoint(endpoint1.p[c], is_signed);
        let ep0 = is_signed ? bc6h_sign_extend_endpoint(quantized0) : quantized0;
        let ep1 = is_signed ? bc6h_sign_extend_endpoint(quantized1) : quantized1;
        let a = bc6h_unquantize(ep0, is_signed);
        let b = bc6h_unquantize(ep1, is_signed);
        let final_value = bc6h_finish_unquantize(bc6h_interpolate(a, b, weight), is_signed);

        // A saturated endpoint has zero local gradient. This is a genuine dead
        // zone, not a numerical artifact: an endpoint pushed into the clamp
        // stops learning rather than being pulled back, which is why
        // initialisation must stay well inside the range. The threshold is
        // format-dependent -- see `bc6h_unquantize_is_saturated`.
        let s0 = bc6h_unquantize_is_saturated(ep0, is_signed) ? 0.f : 1.f;
        let s1 = bc6h_unquantize_is_saturated(ep1, is_signed) ? 0.f : 1.f;

        let d_value = bc6h_bits_to_float_slope(final_value) * finish_scale * d_out[c];

        d_endpoint0[c] = d_value * float(64 - weight) * s0;
        d_endpoint1[c] = d_value * float(weight) * s1;
        d_position += d_value * float(b - a) * position_scale;
    }

    endpoint0 = diffPair(endpoint0.p, endpoint0.d + d_endpoint0);
    endpoint1 = diffPair(endpoint1.p, endpoint1.d + d_endpoint1);
    position = diffPair(position.p, position.d + d_position);
}
