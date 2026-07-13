// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// clang-format off
__exported import falcor2.utils;

/**
 * Generate a vector that is orthogonal to the input vector.
 * This can construct a tangent frame for meshes without tangent data.
 * @param[in] u Unit vector.
 * @return A unit vector orthogonal to ``u``.
 */
float3 perp_stark(float3 u)
{
    float3 a = abs(u);
    uint uyx = (a.x - a.y) < 0 ? 1 : 0;
    uint uzx = (a.x - a.z) < 0 ? 1 : 0;
    uint uzy = (a.y - a.z) < 0 ? 1 : 0;
    uint xm = uyx & uzx;
    uint ym = (1 ^ xm) & uzy;
    uint zm = 1 ^ (xm | ym); // 1 ^ (xm & ym)
    float3 v = normalize(cross(u, float3(xm, ym, zm)));
    return v;
}

/**
 * Uniform sampling of the unit sphere using spherical coordinates.
 * @param[in] u Uniform random numbers in [0,1)^2.
 * @return Sampled point on the unit sphere.
 */
float3 sample_sphere(float2 u)
{
    float phi = float.TWO_PI * u.y;
    float cos_theta = 1.0f - 2.0f * u.x;
    float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));
    return float3(sin_theta * cos(phi), sin_theta * sin(phi), cos_theta);
}

/**
 * Uniform sampling of the unit hemisphere using sphere sampling.
 * @param[in] u Uniform random numbers in [0,1)^2.
 * @return Sampled point on the unit hemisphere.
 */
float3 sample_hemisphere(float2 u)
{
    float3 w = sample_sphere(u);
    w.z = abs(w.z);
    return w;
}

/**
 * Uniform sampling of the unit disk using Shirley's concentric mapping.
 * @param[in] u Uniform random numbers in [0,1)^2.
 * @return Sampled point on the unit disk.
 */
float2 sample_disk_concentric(float2 u)
{
    u = 2.f * u - 1.f;
    if (u.x == 0.f && u.y == 0.f)
        return u;
    float phi, r;
    if (abs(u.x) > abs(u.y))
    {
        r = u.x;
        phi = (u.y / u.x) * float.PI_OVER_FOUR;
    }
    else
    {
        r = u.y;
        phi = float.PI_OVER_TWO - (u.x / u.y) * float.PI_OVER_FOUR;
    }
    return r * float2(cos(phi), sin(phi));
}

/**
 * Sample the hemisphere with Shirley's cosine-weighted concentric mapping.
 * @param[in] u Uniform random numbers in [0,1)^2.
 * @param[out] pdf Probability density of the sampled direction.
 * @return Sampled direction in the local frame (+z axis up).
 */
float3 sample_cosine_hemisphere_concentric(float2 u, out float pdf)
{
    float2 d = sample_disk_concentric(u);
    float z = sqrt(max(0.f, 1.f - dot(d, d)));
    pdf = z * float.INV_PI;
    return float3(d, z);
}

float pdf_cosine_hemisphere_concentric(float3 w)
{
    return w.z * float.INV_PI;
}

bool sample_uniform_rusinkiewicz(float4 u, out float3 wi, out float3 wo)
{
    float theta_h_warp_power = 1.f;
    float theta_h = pow(u.x, theta_h_warp_power) * float.PI * 0.5f;
    float theta_d = u.y * float.PI * 0.5f;
    float phi_h = u.z * float.PI * 2.f;
    float phi_d = u.w * float.PI * 2.f;
    rusinkiewicz_to_cartesian(float2(theta_h, phi_h), float2(theta_d, phi_d), wi, wo);
    return wi.z >= k_min_cos && wo.z >= 0.f;
}

/**
 * Sampling of a 2D Gaussian distribution using the Box-Muller transform.
 * @param[in] u Uniform random numbers in [0,1)^2.
 * @param[in] mu Mean of the Gaussian.
 * @param[in] sigma Isotropic standard deviation of the Gaussian.
 * @return Sampled 2D point from the Gaussian distribution.
 */
float2 sample_2d_gaussian(float2 u, float2 mu, float sigma)
{
    float r = sqrt(-2.0f * log(u.x));
    float phi = float.TWO_PI * u.y;
    float2 xy = r * float2(cos(phi), sin(phi));
    return mu + sigma * xy;
}

/** Probability density function of a 2D Gaussian.
 * @param[in] x Point to evaluate the PDF at.
 * @param[in] mu Mean of the Gaussian.
 * @param[in] sigma Isotropic standard deviation of the Gaussian.
 * @return Probability density at point x.
 */
float pdf_2d_gaussian(float2 x, float2 mu, float sigma)
{
    float sigma2 = sigma * sigma;
    float2 d = x - mu;
    return exp(-0.5f * dot(d, d) / sigma2) / (float.TWO_PI * sigma2);
}

[Differentiable]
float max_float(float a, float b)
{
    // Propagate NaNs explicitly because the built-in ``max()`` may hide them.
    if (isnan(a))
        return a;
    if (isnan(b))
        return b;
    return max(a, b);
}

[Differentiable]
float safe_sqrt(float x, float epsilon = 1e-4f)
{
    return sqrt(max_float(epsilon, x));
}

[Differentiable]
float3 safe_normalize(float3 x, float epsilon = 1e-4f)
{
    float safe_norm = max_float(epsilon, length(x));
    return x / safe_norm;
}

[Differentiable]
float3 halfvector(float3 wi, float3 wo)
{
    float3 wh = safe_normalize(wi + wo);
    return wh * sign(wh.z);
}

[Differentiable]
float tanh_approx(float x)
{
    return x / sqrt(1.f + x * x);
}

[Differentiable]
float sinh_approx(float x)
{
    return x * sqrt(1.f + x * x);
}

[BackwardDifferentiable]
void get_local_coords(float2 mu, out float3 n, out float3 s, out float3 t)
{
    // Apply normal mapping based on learned slope.
    n = safe_normalize(float3(-mu.x, -mu.y, 1.f));
    const float3 u = float3(1.f, 0.f, 0.f);
    s = safe_normalize(u - n * dot(n, u));
    t = cross(n, s);
}

[BackwardDifferentiable]
float3 transform_to_local(float3 v, float3 n, float3 s, float3 t)
{
    return float3(dot(v, s), dot(v, t), dot(v, n));
}

[Differentiable]
float ndf_ggx(no_diff float3 wh, float2 alpha, float rho, float2 mu)
{
    float sqrt_one_minus_rho_sqr = safe_sqrt(1.f - rho * rho);

    // Move to slope-space and revert any slope offset.
    float2 slope = -wh.xy / max_float(1e-4f, wh.z);
    slope -= mu;

    // Unstretch slope into alpha=1.0 configuration.
    float sx_std = slope.x / alpha.x;
    float normalization = 1.f / max_float(1e-6f, alpha.x * alpha.y * sqrt_one_minus_rho_sqr);
    float sy_std = (alpha.x * slope.y - rho * alpha.y * slope.x) * normalization;

    // Evaluate basic slope distribution.
    float r2 = sx_std * sx_std + sy_std * sy_std;
    float p22_std = 1.f / max_float(1e-4f, float.PI * (1.f + r2) * (1.f + r2));
    float p22 = p22_std * normalization;

    float ndf_h = p22 / max_float(1e-4f, wh.z * wh.z * wh.z * wh.z);
    return ndf_h;
}

[Differentiable]
float pdf_ggx(no_diff float3 wi, no_diff float3 wo, float2 alpha, float rho, float2 mu)
{
    if (wi.z <= 0 || wo.z <= 0)
    {
        return 0.0;
    }

    // Construct half-vector.
    float3 wh = halfvector(wi, wo);

    float ndf_h = ndf_ggx(wh, alpha, rho, mu);
    float pdf_h = ndf_h * wh.z; // p22 / max_float(1e-4f, wh.z * wh.z * wh.z);

    // Account for specular reflection Jacobian.
    float jh = 4 * abs(dot(wi, wh));
    return pdf_h / max_float(1e-4f, jh);
}

float3 sample_ggx(float3 wi, float2 sample, float2 alpha, float rho, float2 mu)
{
    float sqrt_one_minus_rho_sqr = safe_sqrt(1.f - rho * rho);

    // Sample basic slope.
    float s = sqrt(sample.x) / sqrt(1.f - sample.x);
    float phi = 2.f * float.PI * sample.y;
    float2 slope_std = s * float2(cos(phi), sin(phi));

    // Stretch slope.
    float2 slope = { alpha.x * slope_std.x, alpha.y * (rho * slope_std.x + sqrt_one_minus_rho_sqr * slope_std.y) };

    // Apply slope offset and convert to unit vector.
    slope += mu;
    float3 wh = safe_normalize(float3(-slope.x, -slope.y, 1.f));

    // Reflect along the sampled microfacet.
    return 2.f * dot(wi, wh) * wh - wi;
}

[Differentiable]
float pdf_diffuse(no_diff float3 wo, float2 mu)
{
    if (wo.z < 0)
    {
        return 0.0f;
    }
    float3 n = safe_normalize(float3(-mu.x, -mu.y, 1.f));
    float woz = max_float(0.f, dot(wo, n));
    return woz / float.PI;
}

float3 sample_diffuse(float2 sample, float2 mu)
{
    // Sample cosine distribution.
    float unused;
    float3 wo = sample_cosine_hemisphere_concentric(sample, unused);

    // Apply normal mapping based on learned slope.
    float3 n = safe_normalize(float3(-mu.x, -mu.y, 1.f));
    const float3 u = float3(1.f, 0.f, 0.f);
    float3 s = safe_normalize(u - n * dot(n, u));
    float3 t = cross(n, s);
    return wo.x * s + wo.y * t + wo.z * n;
}

static const float k_min_cos = 1e-3f;

// Convert Rusinkiewicz coordinates to Cartesian directions.
[Differentiable]
void rusinkiewicz_to_cartesian(const float2 h_sph, const float2 d_sph, out float3 wi, out float3 wo)
{
    float3 d = spherical_to_cartesian_rad(d_sph);
    float3 temp = rotate_y(d, h_sph.x);
    wi = rotate_z(temp, h_sph.y);
    float3 half = spherical_to_cartesian_rad(h_sph);
    wo = normalize(2 * dot(wi, half) * half - wi);
}

/**
 * Converts Cartesian coordinates to spherical coordinates (radians).
 * ``theta`` is the polar angle from +z in the range [0, pi].
 * ``phi`` is the azimuth from +x in the range [0, 2pi].
 * @param[in] p Cartesian coordinates (x,y,z).
 * @return Spherical coordinates (theta,phi).
 */
[Differentiable]
float2 cartesian_to_spherical_rad(float3 p)
{
    p = normalize(p);
    float2 sph;
    sph.x = acos(p.z);
    sph.y = atan2(-p.y, -p.x) + float.PI;
    return sph;
}

/**
 * Converts spherical coordinates (radians) to Cartesian coordinates.
 * Inverse of cartesian_to_spherical_rad.
 * @param[in] sph Spherical coordinates (theta,phi).
 * @return Cartesian coordinates (x,y,z).
 */
[Differentiable]
float3 spherical_to_cartesian_rad(float2 sph)
{
    float3 p;
    p.x = -cos(sph.y - float.PI) * sin(sph.x);
    p.y = -sin(sph.y - float.PI) * sin(sph.x);
    p.z = cos(sph.x);
    return p;
}

/**
 * Rotate a 3D vector counterclockwise around the X-axis.
 * The rotation angle is positive for a rotation that is counterclockwise when
 *  looking along the x-axis towards the origin.
 * @param[in] v Original vector.
 * @param[in] angle Angle in radians.
 * @return Transformed vector.
 */
[Differentiable]
float3 rotate_x(const float3 v, const float angle)
{
    float c, s;
    sincos(angle, s, c);
    return float3(v.x, v.y * c - v.z * s, v.y * s + v.z * c);
}

/**
 * Rotate a 3D vector counterclockwise around the Y-axis.
 * The rotation angle is positive for a rotation that is counterclockwise when
 * looking along the y-axis towards the origin.
 * @param[in] v Original vector.
 * @param[in] angle Angle in radians.
 * @return Transformed vector.
 */
[Differentiable]
float3 rotate_y(const float3 v, const float angle)
{
    float c, s;
    sincos(angle, s, c);
    return float3(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
}

/**
 * Rotate a 3D vector counterclockwise around the Z-axis.
 * The rotation angle is positive for a rotation that is counterclockwise when
 * looking along the z-axis towards the origin.
 * @param[in] v Original vector.
 * @param[in] angle Angle in radians.
 * @return Transformed vector.
 */
[Differentiable]
float3 rotate_z(const float3 v, const float angle)
{
    float c, s;
    sincos(angle, s, c);
    return float3(v.x * c - v.y * s, v.x * s + v.y * c, v.z);
}
