// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off

// Differentiable BC6H mode-10 decode. rtbcnam/PLAN.md Stage 1.
//
// This is the core of RTBCNAM: a forward pass that produces EXACTLY what the
// fixed-function texture unit will produce from the same parameters, with a
// usable gradient back to those parameters. Because the forward pass is the
// real format, the trained features export to a real BC6H texture with no
// separate compression step and nothing to repair -- which is the whole
// difference from RTCNAM's post-training 4-bit grid round plus a 20k-iteration
// `decoder_repair` phase.
//
// ---------------------------------------------------------------------------
// Why mode 10, and why not mode 11
// ---------------------------------------------------------------------------
//
// Both are one-subset modes with 4-bit indices and NO partition mask, which is
// the property that matters: there is nothing discrete to choose, so the
// forward pass is differentiable from iteration zero. BCf has to fix each
// block's partition before its main training phase precisely because that
// choice is unreachable by gradient descent -- a 5k unconstrained warmup, one
// real BC6 compress, then 200k constrained iterations
// (docs/deep-dives/dd1-block-compressed-features.md).
//
// They differ in how the SECOND endpoint is stored, and that decided it:
//
//   - **Mode 10 stores both endpoints absolutely**, 10 bits each, signed range
//     [-512, 511]. A block can span the entire representable range.
//   - **Mode 11 stores a base plus a 9-bit signed delta** at 11-bit precision.
//     Finer endpoint placement, but endpoint separation is capped at 256 of
//     1024 endpoint units, so a block can only span A QUARTER of the range.
//
// PLAN.md originally chose mode 11 for its extra precision bit. That was
// wrong, and Stage 1 caught it by trying to fit a block rather than only
// checking local gradient correctness. On a 4x4 linear ramp spanning
// [-0.6, 0.7], mode 11's delta needed 967 endpoint units and could only supply
// 256, so its ceiling was 7.67 dB -- the segment physically could not reach.
// Mode 10 fit the same block at 9.44 dB. Across four synthetic blocks mode 10
// won by 1.40 to 3.08 dB and tied on the fourth (near-constant, where span
// does not matter). See rtbcnam/results/STAGE0.md.
//
// Mode 10 is also simpler: no delta transform means no modular wrap and no
// re-sign-extension, so the gradient chain loses a coupled term and the two
// endpoints are independent.
//
// Mode 11 remains implemented and hardware-verified in the numpy reference and
// in `verify_bc6h_reference.py`'s gate, so revisiting the trade needs no new
// verification work.
//
// The integer helpers below are verified bit-exact against DirectXTex AND the
// real GPU texture unit, in both signed and unsigned formats, over thousands of
// blocks including constructed saturation cases
// (rtbcnam/harness/verify_bc6h_reference.py). Do not "simplify" them against
// intuition; re-run that gate instead.
//
// ---------------------------------------------------------------------------
// Why the derivative is hand-written rather than synthesised
// ---------------------------------------------------------------------------
//
// The forward pass is an integer pipeline, so it is piecewise CONSTANT in its
// continuous parameters and automatic differentiation of it yields exactly
// zero almost everywhere. Training on that gradient would not move at all.
//
// So `bc6h_decode_mode10` carries a custom `[BackwardDerivative]` that
// differentiates a smooth surrogate coinciding with the integer pipeline at
// integer inputs. That is what BCf means by its equations 7-9 being "almost
// everywhere differentiable": you take the real-valued analogue of each step
// and treat the roundings as identity (a straight-through estimator).
//
// Two measured facts make the surrogate EXACT rather than approximate in the
// interior, both checked in Stage 0 and asserted by Stage 1's tests:
//
//   - **`unquantize` at precision 10 is exactly affine**: `64*|x| + 32`, with
//     no rounding whatsoever, because `x << 15` is divisible by `1 << 9` and
//     the `+0x4000` rounding term is exactly `32 << 9`. SIGNED saturates to
//     `0x7FFF` at `|x| >= 511`; UNSIGNED saturates to `0xFFFF` only at the
//     single value 1023. Both are genuine dead gradients -- see
//     `bc6h_unquantize_is_saturated` and the note on initialisation below.
//   - **The output never reaches the half's inf/nan exponent.** `unquantize`
//     saturates at `0x7FFF`, and `finish_unquantize` then scales by 31/32 to
//     `0x7BFF` = 65504, the largest finite half. So the exponent field spans
//     0..30 and the reinterpretation needs only the normal and subnormal
//     cases. This is what `finish_unquantize` is for.
//
// Measured agreement with finite differences of the real function: median
// relative error exactly 0.0, maximum 1.6%, over the interior. The surrogate
// is inexact only on two measure-zero sets, both genuine format
// discontinuities rather than slack:
//
//   - a step crossing a half EXPONENT BOUNDARY, where the reinterpretation's
//     slope doubles, so one analytic value cannot match a secant spanning two
//     slopes;
//   - a step where an endpoint passes through ZERO, because BPTC's
//     `unquantize` special-cases `magnitude == 0` to return 0 and so breaks
//     the affine form with a jump of 32 at the origin.
//
// ---------------------------------------------------------------------------
// Parameter units and initialisation
// ---------------------------------------------------------------------------
//
// `endpoint0` and `endpoint1` are in ENDPOINT-INTEGER units, not normalised:
// signed they span [-512, 511], unsigned [0, 1023]. That keeps this file in
// the spec's own units so it can be read against the spec, and it is why the
// Python side is responsible for storing a normalised parameter and scaling
// into these units -- a raw Adam step of 1e-3 in these units would need a
// thousand steps to cross one quantization level.
//
// Initialise `|endpoint|` well below the format's saturation point (511 signed,
// 1023 unsigned). At saturation `d(unquantize)/d(endpoint)` is zero, so an
// endpoint pushed into the clamp stops learning permanently rather than being
// pulled back.
//
// `position` and the endpoints also do not share a natural learning rate: the
// ratio of their gradients is the endpoint separation, measured at three to
// four orders of magnitude for a wide segment. That is physical -- moving
// `position` from 0 to 1 traverses the whole segment while moving an endpoint
// by one unit shifts only one end -- and Adam's per-parameter normalisation
// largely absorbs it.

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

// Is `unquantize` on its flat clamp at this endpoint, so that the local
// gradient really is zero?
//
// THE THRESHOLD DIFFERS BY FORMAT and getting it wrong is silent. Signed
// saturates once `|value| >= 511`, which the quantizer's own [-512, 511] range
// makes a boundary-only condition. Unsigned saturates ONLY at the single value
// 1023: everything from 511 to 1022 still has the full slope of 64.
//
// Using the signed threshold for both was a real bug, caught by code review
// rather than by the test suite, because every backward test ran signed only.
// It zeroed the gradient across [511, 1022] -- half the unsigned range -- so an
// unsigned endpoint that drifted past the midpoint would stop learning for no
// reason. The forward pass was unaffected, which is why bit-exactness tests
// stayed green.
//
// Zero is deliberately NOT masked. `unquantize(0)` short-circuits to 0 while
// `unquantize(1)` is 96 rather than 64, so the origin is a step
// discontinuity, not a flat spot; the affine slope is the right surrogate
// there, and it is one of the two measure-zero sets the accuracy bound already
// excludes.
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

// Clamp a continuous parameter into mode 10's representable integer range and
// round to the lattice. Separated out so the forward pass and the Python-side
// exporter cannot disagree about what a given parameter quantizes to.
//
// The range DEPENDS ON SIGNEDNESS: a signed endpoint is 10-bit two's
// complement, [-512, 511]; an unsigned one is the plain 10-bit range
// [0, 1023]. Hardcoding the signed range silently halved the unsigned usable
// range, which is a real bug Stage 1's
// `test_continuous_path_matches_exact_path[False]` caught.
int bc6h_quantize_endpoint(float endpoint, bool is_signed)
{
    let lo = is_signed ? float(BC6H_MODE10_SIGNED_MIN) : 0.f;
    let hi = is_signed ? float(BC6H_MODE10_SIGNED_MAX) : float(BC6H_MODE10_UNSIGNED_MAX);
    return int(clamp(round(endpoint), lo, hi));
}

int bc6h_quantize_index(float position)
{
    return int(clamp(round(position * float(BC6H_MODE10_INDEX_LEVELS)),
                     0.f, float(BC6H_MODE10_INDEX_LEVELS)));
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
                          no_diff bool is_signed)
{
    let index = bc6h_quantize_index(position);
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
//   w   = W(round(p*15)),  dw/dp    = (64/15) * 15 = 64
//
// The 64s cancel, leaving:
//
//   dout/dep0 = dout/dV * finish_scale * (64 - w) * s0
//   dout/dep1 = dout/dV * finish_scale * w * s1
//   dout/dp   = dout/dV * finish_scale * (b - a)
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
    float3 d_out)
{
    let index = bc6h_quantize_index(position.p);
    let weight = BC6H_WEIGHT4[index];
    let finish_scale = bc6h_finish_scale(is_signed);

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
        d_position += d_value * float(b - a);
    }

    endpoint0 = diffPair(endpoint0.p, endpoint0.d + d_endpoint0);
    endpoint1 = diffPair(endpoint1.p, endpoint1.d + d_endpoint1);
    position = diffPair(position.p, position.d + d_position);
}
