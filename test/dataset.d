module tests.dataset;

import std.exception;
import std.stdio;
import std.range : enumerate;

import tests.common;

import vectorflow.dataset;


unittest {
    class D : DataReader!ulong{

        static ulong[] _data = [1, 2, 3];
        size_t _cnt = 0;

        this(){_memory_size = 3 * 4;}

        override bool read_next()
        {
            if(empty)
                return false;
            _obs = _data[_cnt];
            _cnt++;
            return true;
        }

        override @property bool empty()
        {
            return _cnt >= _data.length;
        }

        override void rewind(){_cnt = 0;}
        
        override @property D save()
        {
            auto cp = new D();
            cp.share_save_params(this);
            cp._cnt = _cnt;
            return cp;
        }
    }

    auto d = new D();
    assert(d.memory_size!"B".fequal(12));
    ulong sum = 0;
    foreach(v; d)
        sum += v;
    assert(sum == 6);

    d.rewind();
    sum = 0;
    foreach(v; d)
        sum += v;
    assert(sum == 6);

    d.cache();
    
    sum = 0;
    foreach(v; d)
        sum += v;
    assert(sum == 6);

    sum = 0;
    foreach(i, ref v; d.enumerate)
        sum += v;
    assert(sum == 6);

    assert(d.length == 3);
}
