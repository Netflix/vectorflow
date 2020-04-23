/**
 * Internal serialization functions for automatic neural net ser/deser.
 *
 * Copyright: 2017 Netflix, Inc.
 * License: $(LINK2 http://www.apache.org/licenses/LICENSE-2.0, Apache License Version 2.0)
 */
module vectorflow.serde;

private
{
import std.conv : to;
import std.stdio;
import std.traits;

import vectorflow.neurallayer;
}

/**
* Utility class used by vectorflow to serialize a net.
* @TODO: handle endianness properly for cross-arch compat.
*/
class Serializer {

    File* _f;

    this(File* f)
    {
        _f = f;
    }

    class EOFException : Exception{this(string msg){super(msg);}}

    void write(T)(T data) if(isSomeString!T || isBasicType!T || isUnsigned!T)
    {
        static if(isSomeString!T)
        {
            write_str(data.to!string);
        }
        else
        {
            _f.rawWrite((&data)[0..1]);
        }
    }

    private void write_str(string s)
    {
        write(s.length.to!ulong);
        auto x = cast(ubyte[])s;
        _f.rawWrite(x);
    }

    final T read(T)() if(isSomeString!T || isBasicType!T || isUnsigned!T)
    {
        scope(failure){ if(_f.eof) throw new EOFException("");}
        version(Windows){} // Windows requires special treatment with EOF
        else scope(exit){ if(_f.eof) throw new EOFException(""); }
        static if(isSomeString!T)
        {
            // On Windows EOF is not set before you try to read past the end, so it requires some special handling
            // Original code was causing runtime crashes rendering deserialization impossible.
            // If we use more low level approach and manually throw EOFException if we can't read elemnts it works.
            version(Windows)
            {
                import core.stdc.stdio : fread;
                ulong str_sz;
                auto sz = fread(&str_sz, str_sz.sizeof, 1, _f.getFP);
                if(sz < 1) throw new EOFException("");
                auto str = new ubyte[str_sz.to!size_t];
                sz = fread(str.ptr, ubyte.sizeof, str_sz, _f.getFP);
                if(sz < str_sz) throw new EOFException("");
                return cast(string)str;
            }
            else
            {
                ulong str_sz;
                _f.rawRead((&str_sz)[0..1]);
                auto str = new ubyte[str_sz.to!size_t];
                str = _f.rawRead(str);
                return cast(string)str;
            }
        }
        else
        {
            T d;
            _f.rawRead((&d)[0..1]);
            return d;
        }
    }

    void write_vec(T)(T vec) if(isArray!T)
    {
        write(vec.length.to!ulong);
        foreach(v; vec)
            write(v);
    }

    T[] read_vec(T)()
    {
        auto len = read!ulong().to!size_t;
        auto res = new T[len];
        foreach(i; 0..len)
            res[i] = read!T();

        return res;
    }

    NeuralLayer[] deserialize_layers()
    {
        NeuralLayer[] layers;

        while(!_f.eof)
        {
            string layer_type;
            try{ layer_type = read!string(); }
            catch(EOFException e){ break; } 

            auto l = Object.factory(layer_type).to!NeuralLayer;
            l.deser(this);
            layers ~= l;
            writeln("Deserialized ", l.to!string);
        }

        return layers;
    }
}
