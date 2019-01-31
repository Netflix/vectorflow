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

version (X86)
    version = X86_Any;
else
version (X86_64)
    version = X86_Any;

version (X86_Any)
static this()
{
    // Enable flushing denormals to zero
    enum FTZ_BIT = 15;
    enum DAZ_BIT = 6;

    // Manually align to 16 bytes - https://issues.dlang.org/show_bug.cgi?id=16098
    uint[128 + 4] buf;
    auto state = cast(uint*)((cast(size_t)buf.ptr + 0xF) & ~size_t(0xF));
    version (X86_64)
        asm { mov RAX, state; fxsave 0[RAX]; }
    else
        asm { mov EAX, state; fxsave 0[EAX]; }
    uint mxcsr = state[6];
    mxcsr |= 1 << FTZ_BIT;
    mxcsr |= 1 << DAZ_BIT;
    asm { ldmxcsr mxcsr; }
}
