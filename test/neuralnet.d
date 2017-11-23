module tests.neuralnet;

import core.exception : AssertError;
import std.exception;
import std.stdio;

import tests.common;

import vectorflow;
import vectorflow.neuralnet;
import vectorflow.layers;
import vectorflow.neurallayer;

/// Test dup of neural net topology
unittest {
    void same_topology(NeuralNet a, NeuralNet b, bool shallow)
    {
        import std.conv : to;

        assert(a.layers.length == b.layers.length);
        assert(a.leaves.length == b.leaves.length);
        foreach(i, ref l1; b.layers)
        {
            auto l2 = a.layers[i];
            assert(l1.dim_in == l2.dim_in);
            assert(l1.dim_out == l2.dim_out);
            assert(l1.name == l2.name);
            assert(l1.parents.length == l2.parents.length);
            assert(l1.children.length == l2.children.length);
            if(!shallow)
            {
                assert(l1.num_params == l2.num_params);
                assert(l1.backgrads.length == l2.backgrads.length);
                foreach(j; 0..l1.parents.length)
                    assert(l2.backgrads[j].length == l1.backgrads[j].length);
                if(auto l1l = cast(Linear)l1)
                {
                    auto l2l = l2.to!Linear;
                    assert(l1l.grad.length == l2l.grad.length);
                    assert(l1l.W.length == l2l.W.length);
                }
            }
        }
    }

    auto nn = NeuralNet()
        .stack(DenseData(8))
        .stack(DropOut(0.1))
        .stack(ReLU())
        .stack(Linear(20))
        .stack(Linear(3));
    same_topology(nn, nn.dup, false);
    same_topology(nn, nn.dup(true), true);

    nn = NeuralNet()
        .stack(SparseData(15))
        .stack(Linear(2))
        .stack(Linear(5))
        .stack(SeLU())
        .stack(DropOut(0.2))
        .stack(Linear(7));
    same_topology(nn, nn.dup, false);
    same_topology(nn, nn.dup(true), true);

    nn = NeuralNet()
        .add_root("a", SparseData(5))
        .add_root("b", DenseData(4))
        .add("c", Linear(3));
    nn.wire("a", "c");
    nn.wire("b", "c");
    same_topology(nn, nn.dup, false);
    same_topology(nn, nn.dup(true), true);

    nn = NeuralNet()
        .add_root("a", DenseData(1))
        .add("b", Linear(3, false))
        .add("c", SeLU());
    nn.wire("a", "b");
    nn.wire("a", "c");
    same_topology(nn, nn.dup, false);
    same_topology(nn, nn.dup(true), true);

    nn = NeuralNet()
        .add_root("a", SparseKernelExpander(5, ""))
        .add_root("b", DenseData(4))
        .add("c", Linear(3))
        .add("d", DropOut(2));
    nn.wire("a", "c");
    nn.wire("b", "d");
    same_topology(nn, nn.dup, false);
    same_topology(nn, nn.dup(true), true);

    nn = NeuralNet()
        .add_root("a", SparseKernelExpander(5, ""))
        .add_root("b", DenseData(4))
        .add("c", Linear(3))
        .add("d", DropOut(2))
        .add("e", Linear(1));
    nn.wire("a", "c");
    nn.wire("b", "c");
    nn.wire("c", "d");
    nn.wire("c", "e");
    same_topology(nn, nn.dup, false);
    same_topology(nn, nn.dup(true), true);
    same_topology(nn, nn.dup.dup(true), true);

    nn = NeuralNet()
            .add_root("a", DenseData(1))
            .add_root("b", DenseData(1))
            .add("c", Linear(1));
    nn.wire("a", "c");
    nn.wire("b", "c");
    assertNotThrown!Exception({
        nn.stack("d", Linear(1));
    }());
    nn.add("e", Linear(1));
    nn.wire("a", "e");
    assert(nn.leaves.length == 2);
    // this net is not a stack, should thrown if we're trying to stack:
    assertThrown!Exception({
        nn.stack(Linear(1));
    }());

    // roots are not learnable
    assertThrown!Exception({
        NeuralNet().stack(DenseData(1), AdaGrad(10, 0.1, 10));
    }());

}

/// Test predict
unittest {
    auto nn = NeuralNet()
        .stack(DenseData(2))
        .stack(Linear(3));
    nn.initialize(0.0);

    struct O{
        float[] features;
    }

    auto preds = nn.predict(O([1.3f, 2.5f]));
    assert(preds.length == 3);

    // multi roots
    nn = NeuralNet()
        .add_root("a", DenseData(1))
        .add_root("b", SparseData(2));
    auto l = Linear(1, false);
    nn.add("c", l);
    nn.wire("a", "c");
    nn.wire("b", "c");
    nn.initialize(0.0);
    l.W[0][] = 1.0;

    auto p = nn.predict([1.0f], [SparseF(1, 0.2f), SparseF(0, -0.7f)]);
    assert(p.length == 1);
    assert(fequal(p[0], 1.0 + 0.2 - 0.7));

    // empty input
    SparseF[] d2;
    assertNotThrown!Exception(nn.predict([0.3f], d2));

    // wrong roots order
    assertThrown!AssertError(
        nn.predict([SparseF(0, 0.1f)], [1.0f]));

    // wrong number of roots, will throw in non-release mode
    assertThrown!AssertError(nn.predict([0.9f]));
}

/// Test leaves
unittest {
    auto nn = NeuralNet()
        .stack("123", DenseData(3));
    assert(nn.leaves.length == 1);
    nn.stack("abc", TanH());

    assert(nn.leaves.length == 1);
    assert(nn.leaves[0].name == "abc");

    nn.add("foo", Linear(2));
    nn.add("bar", Linear(3));
    assert(nn.leaves.length == 3);

    nn.wire("123", "foo");
    assert(nn.leaves.length == 3);

    nn.wire("abc", "foo");
    assert(nn.leaves.length == 2);
    assert(nn.leaves[0].name == "foo");
    assert(nn.leaves[1].name == "bar");

    nn.wire("foo", "bar");
    assert(nn.leaves.length == 1);
    assert(nn.leaves[0].name == "bar");

    nn.add("blah", Linear(5));
    nn.add("blob", Linear(3));
    nn.add("plop", ReLU());

    assert(nn.leaves.length == 4);
    nn.wire("blob", "plop");
    assert(nn.leaves.length == 3);
    assert(nn.leaves[1].name == "blah");
    assert(nn.leaves[2].name == "plop");
}
