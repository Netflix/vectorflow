module tests.common;

import std.conv : to;
import std.random : uniform;
import std.traits : isFloatingPoint, isNumeric;

import vectorflow;
import vectorflow.math : fabs;
import vectorflow.dataset;
import vectorflow.utils;

bool fequal(T, S)(T a, S b, double tol = 1e-6)
    if((isFloatingPoint!T && isNumeric!S) ||
       (isFloatingPoint!S && isNumeric!T))
{
    return fabs(a - b) < tol;
}

struct ObsD {
    float label;
    float[] features;
}

struct ObsS {
    float label;
    SparseF[] features;
}

ObsD[] fake_data_dense(size_t dim = 20, size_t samples = 100)
{
    auto d = new ObsD[samples];
    foreach(i; 0..samples)
    {
        auto x = new float[dim];
        x[0..dim/2] = 0.3;
        x[dim/2..$] = -0.2;
        auto lab = i % 2 == 0 ? 0.0 : 1.0;

        d[i] = ObsD(lab, x);
    }

    return d;
}

ObsS[] fake_data_sparse(size_t dim = 20, size_t samples = 100)
{
    auto res = new ObsS[samples];
    foreach(i; 0..samples)
    {
        auto d = i % 2 == 0 ? dim : dim/2;
        auto x = new SparseF[d];
        foreach(j; 0..d)
        {
            auto ind = uniform(0, d, RAND_GEN);
            x[j] = SparseF(ind.to!uint, j + 0.1);
        }
        auto lab = i % 3 == 0 ? 0.0 : 1.0;

        res[i] = ObsS(lab, x);
    }

    return res;
}

// override default unit-tester. See core.runtime source code for details
version(unittest)
{
    import core.exception;
    import core.runtime;
    import std.range : retro;
    import std.format : format;
    import std.stdio;

    bool customModuleUnitTester()
    {
        string[] messages;

        size_t failed = 0;
        foreach (m; ModuleInfo)
        {
            if (m)
            {
                auto fp = m.unitTest;
                if (fp)
                {
                    try
                    {
                        fp();
                    }
                    catch (Throwable e)
                    {
                        if(failed == 0)
                        {
                            messages ~= format(
                                "TEST FAILURE | %s | %s",
                                m.name, e);
                        }
                        else
                        {
                            messages ~= format(
                                "TEST FAILURE | %s | %s | line=%d | %s",
                                m.name, e.file, e.line, e.msg);
                        }
                        failed++;
                        break;
                    }
                }
            }
        }
        writeln("----------");
        if(failed > 0)
        {
            writefln("At least %d test(s) failed.", failed);
            foreach(m; retro(messages))
                writeln(m);
        }
        return failed == 0;
    }

    shared static this()
    {
        Runtime.moduleUnitTester = &customModuleUnitTester;
    }
}
