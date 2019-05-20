/**
 * Utility functions used across the library.
 *
 * Copyright: 2017 Netflix, Inc.
 * License: $(LINK2 http://www.apache.org/licenses/LICENSE-2.0, Apache License Version 2.0)
 */
module vectorflow.utils;

private{
import std.ascii : isDigit;
import std.random;
import std.stdio : stdout;
import std.traits : isSomeString;

extern(C) int isatty(int);
}

/// RNG used across the library
auto static RAND_GEN = Xorshift(42); // way faster than Mersenne-Twister

auto init_matrix_rand(T)(T[][] M, double rand_scale)
{
    foreach(i; 0..M.length)
        foreach(j; 0..M[i].length)
            M[i][j] = rand_scale * (2 * uniform01(RAND_GEN) - 1);
    return M;
}

auto allocate_matrix_zero(T)(size_t num_row, size_t num_col)
{
    T[][] M;
    M.length = num_row;
    foreach(i; 0..num_row)
    {
        M[i].length = num_col;
        foreach(j; 0..num_col)
            M[i][j] = 0;
    }
    return M;
}

/**
* Fast but unsafe function to parse a string into a long.
*/
long to_long(T)(in T str) pure
    if(isSomeString!T)
{
    long val = 0;
    foreach(c; str)
        val = 10 * val + (c - '0');
    return val;
}

/**
* Fast but unsafe function to parse a string into a float.
* 
* If you trust your input, this is much faster than to!float.
* Doesn't handle Inf numbers nor Nan, and doesn't throw exceptions.
* Adapted from Phobos std.conv source code. See NOTICE for licence details.
*/
float to_float(T)(in T p) pure
    if(isSomeString!T)
{
    static immutable real[14] negtab =
        [ 1e-4096L,1e-2048L,1e-1024L,1e-512L,1e-256L,1e-128L,1e-64L,1e-32L,
                1e-16L,1e-8L,1e-4L,1e-2L,1e-1L,1.0L ];
    static immutable real[13] postab =
        [ 1e+4096L,1e+2048L,1e+1024L,1e+512L,1e+256L,1e+128L,1e+64L,1e+32L,
                1e+16L,1e+8L,1e+4L,1e+2L,1e+1L ];

    int ind = 0;
    ulong len = p.length;

    float ldval = 0.0;
    char dot = 0;
    int exp = 0;
    long msdec = 0, lsdec = 0;
    ulong msscale = 1;

    char sign = 0;
    switch (p[ind])
    {
        case '-':
            sign++;
            ind++;
            break;
        case '+':
            ind++;
            break;
        default: {}
    }

    while (ind < len)
    {
        int i = p[ind];
        while (isDigit(i))
        {
            if (msdec < (0x7FFFFFFFFFFFL-10)/10)
                msdec = msdec * 10 + (i - '0');
            else if (msscale < (0xFFFFFFFF-10)/10)
            {
                lsdec = lsdec * 10 + (i - '0');
                msscale *= 10;
            }
            else
                exp++;

            exp -= dot;
            ind++;
            if (ind == len)
                break;
            i = p[ind];
        }
        if (i == '.' && !dot)
        {
            ind++;
            dot++;
        }
        else
            break;
    }
    if (ind < len && (p[ind] == 'e' || p[ind] == 'E'))
    {
        char sexp;
        int e;

        sexp = 0;
        ind++;
        switch (p[ind])
        {
            case '-':    sexp++;
                         goto case;
            case '+':    ind++;
                         break;
            default: {}
        }
        e = 0;
        while (ind < len && isDigit(p[ind]))
        {
            if (e < 0x7FFFFFFF / 10 - 10)   // prevent integer overflow
            {
                e = e * 10 + p[ind] - '0';
            }
            ind++;
        }
        exp += (sexp) ? -e : e;
    }

    ldval = msdec;
    if (msscale != 1)
        ldval = ldval * msscale + lsdec;
    if (ldval)
    {
        uint u = 0;
        int pow = 4096;

        while (exp > 0)
        {
            while (exp >= pow)
            {
                ldval *= postab[u];
                exp -= pow;
            }
            pow >>= 1;
            u++;
        }
        while (exp < 0)
        {
            while (exp <= -pow)
            {
                ldval *= negtab[u];
                exp += pow;
            }
            pow >>= 1;
            u++;
        }
    }
    return (sign) ? -ldval : ldval;
}


final class Hasher {
    // MurmurHash3 was written by Austin Appleby, and is placed in the public
    // domain. The author hereby disclaims copyright to this source code.
    // Original C++ source code at: https://code.google.com/p/smhasher/

    private static uint _rotl32(uint x, int r) pure nothrow @safe
    {
        return (x << r) | (x >> (32 - r));
    }

    // Finalization mix - force all bits of a hash block to avalanche
    private static uint fmix32(uint h) pure nothrow @safe
    {
      h ^= h >> 16;
      h *= 0x85ebca6b;
      h ^= h >> 13;
      h *= 0xc2b2ae35;
      h ^= h >> 16;

      return h;
    }

    public static uint MurmurHash3(T)(in T key, uint seed = 42) pure nothrow
        if(isSomeString!T)
    {
      const ubyte * data = cast(const(ubyte*))key;
      uint len = cast(uint)key.length;
      const int nblocks = len / 4;

      uint h1 = seed;

      const uint c1 = 0xcc9e2d51;
      const uint c2 = 0x1b873593;

      // body
      const uint * blocks = cast(const (uint *))(data + nblocks*4);

      for(int i = -nblocks; i; i++)
      {
        uint k1 = blocks[i];

        k1 *= c1;
        k1 = _rotl32(k1,15);
        k1 *= c2;
        
        h1 ^= k1;
        h1 = _rotl32(h1,13); 
        h1 = h1*5+0xe6546b64;
      }

      // tail
      const ubyte * tail = cast(const (ubyte*))(data + nblocks*4);

      uint k1 = 0;

      switch(len & 3)
      {
        case 3: k1 ^= tail[2] << 16; goto case 2;
        case 2: k1 ^= tail[1] << 8; goto case 1;
        case 1: k1 ^= tail[0];
                k1 *= c1;
                k1 = _rotl32(k1,15);
                k1 *= c2;
                h1 ^= k1; goto default;
        default:
            break;
      }

      // finalization
      h1 ^= len;
      h1 = fmix32(h1);

      return h1;
    }
}

package mixin template ct_msg(string msg)
{
    pragma(msg, "[VECTORFLOW-COMPILE] " ~ msg);
}

package bool isTerminal()
{
    return isatty(stdout.fileno) == 1;
}

package mixin template opCallNew()
{
    static auto opCall(T...)(T args)
    {
        return new typeof(this)(args);
    }
}
