/**
 * Low-level math primitives.
 *
 * Copyright: 2017 Netflix, Inc.
 * License: $(LINK2 http://www.apache.org/licenses/LICENSE-2.0, Apache License Version 2.0)
 */
module vectorflow.math;

private{
    import vectorflow.neurallayer : SparseF;
    import std.math : log1p;
}
version(LDC)
{
    private{
    import ldc.attributes;
    import ldc.intrinsics;
    }

    alias exp = llvm_exp;
    alias fabs = llvm_fabs;
    alias fmax = llvm_maxnum;
    alias log = llvm_log;
    alias round = llvm_round;
    alias sqrt = llvm_sqrt;
}
else
{
    public import std.math : sqrt, exp, fmax, log, round, sqrt;
    private import std.math : abs;
    alias fabs = abs;
}

mixin template Functions()
{
    static float dotProd(float[] x, float[] y) pure @nogc
    {
        float res = 0;
        for(int i = 0; i < x.length; ++i)
            res += x[i] * y[i];
        return res;
    }

    static void relu(float[] x, float[] y) pure @nogc
    {
        for(int i = 0; i < x.length; ++i)
            y[i] = fmax(0, x[i]);
    }

    static void axpy(float a, float[] x, float[] y) pure @nogc
    {
        for(int i = 0; i < x.length; ++i)
            y[i] += a * x[i];
    }

    static void axpy(float a, SparseF[] x, float[] y) pure @nogc
    {
        foreach(ref f; x)
            y[f.id] += a * f.val;
    }

    static void tanh(float[] x, float[] y) pure @nogc
    {
        for(int i = 0; i < x.length; ++i)
        {
            if(x[i] > 20)
                y[i] = 1;
            else
            {
                y[i] = exp(2 * x[i]);
                y[i] = (y[i] - 1) / (y[i] + 1);
            }
        }
    }

    static double log1expp(double x) pure @nogc
    {
        if(-x > 60)
            return x;
        return log1p(exp(x));
    }

    static double logistic(double x) pure @nogc
    {
        return 1.0 / (1 + exp(-x));
    }
}

version(LDC)
{
    pragma(inline, true)
    {
        @fastmath
        {
            mixin Functions!();
        }
    }
}
else
{
    mixin Functions!();
}
