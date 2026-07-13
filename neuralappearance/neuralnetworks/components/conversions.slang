// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

// clang-format off
implementing neural_networks;

__include basetypes.ireal;
__include basetypes.imodel;

// Utilities cast element types or convert between arrays, vectors, and
// cooperative vectors without otherwise changing their contents.

// ``__BuiltinFloatingPointType`` guarantees conversion only through ``float``.
// When the source and destination match, compare the types first to avoid that
// unnecessary conversion.
[BackwardDifferentiable, ForceInline]
To convert_real<To : IReal, From : IReal>(From x)
{
    if (let z = x as To)
        return z;
    return To(x.toFloat());
}

[BackwardDerivative(convert_coop_vec_precision_bwd)]
public CoopVec<OutElem, Width> convert_coop_vec_precision<
    InElem : IReal,
    OutElem : IReal,
    int Width
>(CoopVec<InElem, Width> x)
{
    CoopVec<OutElem, Width> y = no_diff CoopVec<OutElem, Width>(OutElem(0.0));
    [ForceUnroll]
    for (int i = 0; i < Width; ++i)
        y[i] = convert_real<OutElem>(x[i]);
    return y;
}

public void convert_coop_vec_precision_bwd<
    InElem : IReal,
    OutElem : IReal,
    int Width
>(
    inout DifferentialPair<CoopVec<InElem, Width>> x,
    CoopVec<OutElem, Width> grad
)
{
    CoopVec<InElem, Width> d_x = CoopVec<InElem, Width>(InElem(0.0));
    [ForceUnroll]
    for (int i = 0; i < Width; ++i)
        d_x[i] = convert_real<InElem>(grad[i]);
    x = diffPair(x.p, d_x);
}

[BackwardDerivative(coop_vec_to_array_bwd)]
public T[Width] coop_vec_to_array<T : IReal, int Width>(CoopVec<T, Width> x)
{
    T y[Width];
    [ForceUnroll]
    for (int i = 0; i < Width; ++i)
        y[i] = x[i];
    return y;
}

public void coop_vec_to_array_bwd<T : IReal, int Width>(
    inout DifferentialPair<CoopVec<T, Width>> x,
    T.Differential[Width] grad
)
{
    CoopVec<T, Width> d_x = CoopVec<T, Width>(T(0.0));
    [ForceUnroll]
    for (int i = 0; i < Width; ++i)
        d_x[i] = __slang_noop_cast<T>(grad[i]);
    x = diffPair(x.p, d_x);
}

[BackwardDerivative(coop_vec_to_vector_bwd)]
public vector<T, Width> coop_vec_to_vector<T : IReal, int Width>(CoopVec<T, Width> x)
{
    vector<T, Width> y;
    [ForceUnroll]
    for (int i = 0; i < Width; ++i)
        y[i] = x[i];
    return y;
}

public void coop_vec_to_vector_bwd<T : IReal, int Width>(
    inout DifferentialPair<CoopVec<T, Width>> x,
    vector<T, Width> grad
)
{
    CoopVec<T, Width> d_x = CoopVec<T, Width>(T(0.0));
    [ForceUnroll]
    for (int i = 0; i < Width; ++i)
        d_x[i] = grad[i];
    x = diffPair(x.p, d_x);
}

[BackwardDerivative(array_to_coop_vec_bwd)]
public CoopVec<T, Width> array_to_coop_vec<T : IReal, int Width>(T x[Width])
{
    CoopVec<T, Width> y = no_diff CoopVec<T, Width>(T(0.0));
    [ForceUnroll]
    for (int i = 0; i < Width; ++i)
        y[i] = x[i];
    return y;
}

public void array_to_coop_vec_bwd<T : IReal, int Width>(
    inout DifferentialPair<T[Width]> x,
    CoopVec<T, Width> grad
)
{
    T.Differential d_x[Width];
    [ForceUnroll]
    for (int i = 0; i < Width; ++i)
        d_x[i] = __slang_noop_cast<T.Differential>(grad[i]);
    x = diffPair(x.p, d_x);
}

[BackwardDerivative(vector_to_coop_vec_bwd)]
public CoopVec<T, Width> vector_to_coop_vec<T : IReal, int Width>(vector<T, Width> x)
{
    CoopVec<T, Width> y = no_diff CoopVec<T, Width>(T(0.0));
    [ForceUnroll]
    for (int i = 0; i < Width; ++i)
        y[i] = x[i];
    return y;
}

public void vector_to_coop_vec_bwd<T : IReal, int Width>(
    inout DifferentialPair<vector<T, Width>> x,
    CoopVec<T, Width> grad
)
{
    vector<T, Width> d_x;
    [ForceUnroll]
    for (int i = 0; i < Width; ++i)
        d_x[i] = grad[i];
    x = diffPair(x.p, d_x);
}

public struct ConvertArrayPrecision<
    InElem : IReal,
    OutElem : IReal,
    int Width
> : IModel<InElem[Width], OutElem[Width]>,
    IModel<vector<InElem, Width>, vector<OutElem, Width>>,
    IModel<CoopVec<InElem, Width>, CoopVec<OutElem, Width>>
{
    [BackwardDifferentiable]
    public OutElem[Width] forward(InElem x[Width])
    {
        OutElem[Width] y;
        [ForceUnroll]
        for (int i = 0; i < Width; ++i)
            y[i] = convert_real<OutElem>(x[i].toFloat());
        return y;
    }

    [BackwardDifferentiable]
    public vector<OutElem, Width> forward(vector<InElem, Width> x)
    {
        return vector<OutElem, Width>(x);
    }

    [BackwardDifferentiable]
    public CoopVec<OutElem, Width> forward(CoopVec<InElem, Width> x)
    {
        return convert_coop_vec_precision<InElem, OutElem, Width>(x);
    }
}

public struct ConvertToArray<T : IReal, int Width>
    : IModel<T[Width], T[Width]>,
      IModel<vector<T, Width>, T[Width]>,
      IModel<CoopVec<T, Width>, T[Width]>
{
    [BackwardDifferentiable]
    public T[Width] forward(T x[Width]) { return x; }

    [BackwardDifferentiable]
    public T[Width] forward(vector<T, Width> x)
    {
        T[Width] y;
        [ForceUnroll]
        for (int i = 0; i < Width; ++i)
            y[i] = x[i];
        return y;
    }

    [BackwardDifferentiable]
    public T[Width] forward(CoopVec<T, Width> x) { return coop_vec_to_array<T, Width>(x); }
}

public struct ConvertToVector<T : IReal, int Width>
    : IModel<vector<T, Width>, vector<T, Width>>,
      IModel<T[Width], vector<T, Width>>,
      IModel<CoopVec<T, Width>, vector<T, Width>>
{
    [BackwardDifferentiable]
    public vector<T, Width> forward(vector<T, Width> x) { return x; }

    [BackwardDifferentiable]
    public vector<T, Width> forward(CoopVec<T, Width> x) { return coop_vec_to_vector<T, Width>(x); }

    [BackwardDifferentiable]
    public vector<T, Width> forward(T x[Width])
    {
        vector<T, Width> y;
        [ForceUnroll]
        for (int i = 0; i < Width; ++i)
            y[i] = x[i];
        return y;
    }
}

public struct ConvertToCoopVec<T : IReal, int Width>
    : IModel<CoopVec<T, Width>, CoopVec<T, Width>>,
      IModel<T[Width], CoopVec<T, Width>>,
      IModel<vector<T, Width>, CoopVec<T, Width>>
{
    [BackwardDifferentiable]
    public CoopVec<T, Width> forward(CoopVec<T, Width> x) { return x; }

    [BackwardDifferentiable]
    public CoopVec<T, Width> forward(T x[Width]) { return array_to_coop_vec<T, Width>(x); }

    [BackwardDifferentiable]
    public CoopVec<T, Width> forward(vector<T, Width> x) { return vector_to_coop_vec<T, Width>(x); }
}
