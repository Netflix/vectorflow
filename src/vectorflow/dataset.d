/**
 This module provides utility classes to iterate over data.

 It is not mandatory to use them when using vectorflow, but you might find them
 useful and slightly more intuitive to use than the built-in range mechanism if
 you're a beginner with D.

 When creating a dataset for vectorflow, it is important for the data sharding
 to be thread-safe if learning over multiple cores is considered, as data
 parallelism with Hogwild is the main strategy used.
 By default, vectorflow will try to shard the forward range provided with
 std.range.evenChunks, which might or might not work depending on your
 specific reader. To explicitly shard the data, just specify an `evenChunks`
 function in your reader implementation (see MultiFilesReader for an example).

 Copyright: 2017 Netflix, Inc.
 License: $(LINK2 http://www.apache.org/licenses/LICENSE-2.0, Apache License Version 2.0)
 */
module vectorflow.dataset;

private
{
import std.conv : to;
import std.stdio;
import std.traits : isBuiltinType;
import std.range : _evenChunks = evenChunks;
}

class DataFileReader(T) : DataReader!T
{
    protected File _f;
    protected string _path;
    protected bool _binary;
    public @property path(){return _path;}

    this(string path, bool binary)
    {
        _path = path;
        _binary = binary;
        rewind();
    }

    protected override abstract bool read_next();
    override abstract @property DataFileReader!T save();

    override @property bool empty()
    {
        return _f.eof;
    }

    ~this()
    {
        _f.close();
    }

    override void rewind()
    {
        _f.close();
        if(_binary)
            _f.open(_path, "rb");
        else
            _f.open(_path, "r");
    }

    protected override void share_save_params(DataReader!T e)
    {
        super.share_save_params(e);
        _f.seek(e.to!(DataFileReader!T)._f.tell);
    }
}


class MultiFilesReader(T)
{
    DataFileReader!(T)[] readers;
    protected size_t _currInd;
    public @property ulong currentFileIndex(){return _currInd;}

    private bool _cached;

    this(DataFileReader!(T)[] readers_)
    {
        readers = readers_;
        _currInd = 0;
        _cached = false;
    }

    int opApply(scope int delegate(ref T) dg)
    {
        int result = 0;
        if(readers[0]._cache.length == 0)
        {
            foreach(r; readers)
            {
                foreach(ref T obs; r)
                {
                    result = dg(obs);
                    if (result)
                        break;
                }
            }
        }
        else
        {
            foreach(r; readers)
            {
                foreach(ref T obs; r._cache)
                {
                    result = dg(obs);
                    if(result)
                        break;
                }
            }
        }
        rewind();
        return result;
    }

    @property bool empty()
    {
        return _currInd == readers.length - 1 && readers[_currInd].empty;
    }

    @property ref T front()
    {
        return readers[_currInd].front;
    }

    void popFront()
    {
        if(readers[_currInd].empty)
            _currInd++;
        readers[_currInd].popFront();
    }

    void rewind()
    {
        foreach(r; readers)
            r.rewind();
        _currInd = 0;
    }

    @property MultiFilesReader!(T) save()
    {
        DataFileReader!(T)[] cps;
        foreach(r; readers)
            cps ~= r.save();
        auto cp = new MultiFilesReader!T(cps);
        if(_cached)
            cp.cache();

        return cp;
    }

    @property ulong length()
    {
        ulong s = 0;
        foreach(r; readers)
            s += r.length;
        return s;
    }

    /// Split reader into num_chunks readers based on even number of
    /// physical files
    MultiFilesReader!(T)[] evenChunks(uint num_chunks)
    {
        MultiFilesReader!(T)[] res;
        DataFileReader!(T)[] cps;
        foreach(r; readers)
            cps ~= r.save();
        auto files_chunks = _evenChunks(cps, num_chunks);
        foreach(c; files_chunks)
        {
            auto r = new MultiFilesReader!(T)(c);
            if(_cached)
                r.cache();
            res ~=r;
        }
        return res;
    }

    MultiFilesReader!T cache()
    {
        _cached = true;
        foreach(r; readers)
            r.cache();
        return this;
    }
}


class DataReader(T)
{
    protected T _obs;
    protected size_t _length;

    T[] _cache;
    protected bool _start_cache;

    /// sum in bytes of the size of the dataset elements in memory
    protected ulong _memory_size;

    this()
    {
        _length = -1;
        _start_cache = false;
        _memory_size = 0;
    }

    protected abstract bool read_next();

    int opApply(scope int delegate(ref T) dg)
    {
        int result = 0;
        static if(__traits(compiles, _obs.dup) || isBuiltinType!T)
        {
            if(_start_cache && _cache.length == 0)
            {
                while(read_next())
                {
                    static if(__traits(compiles, _obs.dup))
                        _cache ~= _obs.dup;
                    else
                        _cache ~= _obs;
                    result = dg(_obs);
                    if (result)
                        break;
                }
                _length = _cache.length;
                _start_cache = false;
            }
            else if(_cache.length != 0)
            {
                foreach(ref T obs; _cache)
                {
                    result = dg(obs);
                    if(result)
                        break;
                }
            }
            else
            {
                while(read_next())
                {
                    result = dg(_obs);
                    if (result)
                        break;
                }
            }
        }
        else
        {
            while(read_next())
            {
                result = dg(_obs);
                if (result)
                    break;
            }
        }
        rewind();
        return result;
    }

    abstract @property bool empty();

    @property ref T front()
    {
        return _obs;
    }

    void popFront()
    {
        read_next();
    }

    abstract void rewind();

    @property size_t length()
    {
        if(_length == -1)
        {
            rewind();
            _length = 0;
            while(read_next())
                _length++;
        }
        return _length;
    }

    abstract @property DataReader!T save();

    protected void share_save_params(DataReader!T e)
    {
        _length = e._length;
        _cache = e._cache;
    }

    DataReader!T cache()
    {
        static if(__traits(compiles, _obs.dup) || isBuiltinType!T)
        {
            if(_cache.length == 0)
                _start_cache = true;
            return this;
        }
        else
        {
            throw new Exception(
                "Support of dataset caching requires a `.dup` " ~
                "function on the elements of your dataset in order to " ~
                "be able to store an immutable copy of them in memory");
        }
    }

    @property float memory_size(string unit)()
        if(unit == "B" || unit == "KB" || unit == "MB" || unit == "GB")
    {
        if(_memory_size == 0)
            throw new Exception("Dataset didn't populate memory size field.");
        auto bytes = _memory_size.to!double;
        static if(unit == "B")
            return bytes;
        static if(unit == "KB")
            return bytes / 1_024;
        static if(unit == "MB")
            return bytes / (1_024 * 1_024);
        static if(unit == "GB")
            return bytes / (1_024 * 1_024 * 1_024);
    }
}
