module tests.layers;

import core.exception : AssertError;
import std.algorithm : all, count;
import std.conv : to;
import std.exception;
import std.math : cos;
import std.stdio;

import tests.common;

import vectorflow.math;
import vectorflow.neuralnet;
import vectorflow.layers;
import vectorflow.neurallayer;

unittest {
    auto nn = NeuralNet();
    nn.stack(DenseData(4));
    auto l1 = Linear(12);
    nn.stack(l1);
    nn.initialize(0.0);
    l1.W[5][0..3] = [2.3, 4.8, -9.1];
    auto x = [0.2f, -7.5f, 0.1f, 6.6f];
    auto pred1 = nn.predict(x);

    scope(exit) remove("tmp_test");
    nn.serialize("tmp_test");
    auto nn2 = NeuralNet.deserialize("tmp_test");

    auto l2 = nn2.layers[1].to!Linear;

    // assert that all weights are equal
    assert(l1.W.length == l2.W.length);
    foreach(k; 0..l1.W.length)
        foreach(i; 0..5)
            assert(fequal(l1.W[k][i], l2.W[k][i]));

    auto pred2 = nn2.predict(x);
    foreach(k; 0..l1.W.length)
        assert(fequal(pred2[k], pred1[k]));
}

unittest {
    auto nn = NeuralNet()
        .stack(SparseData(123))
        .stack(Linear(1));
    nn.initialize(0.0);

    assert(nn.layers.length == 2);
    auto l = nn.layers[$-1].to!Linear;
    assert(l.W.length == 1);
    assert(l.W[0].length == 123 + 1);
}

unittest {
    auto nn = NeuralNet()
        .stack(DenseData(20))
        .stack(Linear(10))
        .stack(ReLU())
        .stack(Linear(2));
    nn.initialize(0.0);

    assert(nn.layers.length == 4);
    auto l = nn.layers[$-1].to!Linear;
    assert(l.W.length == 2);
    assert(l.W[1].length == 10 + 1);

    l = nn.layers[1].to!Linear;
    assert(l.W.length == 10);
    assert(l.W[4].length == 20 + 1);
}

unittest {
    auto nn = NeuralNet()
        .stack(SparseData(3))
        .stack(Linear(1, false)); //  no intercept
    nn.initialize(0.0);

    assert(nn.layers.length == 2);
    auto l = nn.layers[$-1].to!Linear;
    assert(l.W.length == 1);
    assert(l.W[0].length == 3);
}

/// Test of SparseKernelExpander layer
unittest {
    assertNotThrown!Exception(NeuralNet()
        .stack(SparseKernelExpander(50, "1^2,3^5^1")));

    assertThrown!Exception(NeuralNet()
        .stack(SparseKernelExpander(50, "1^2,3^500^1")));

    assertNotThrown!Exception(NeuralNet()
        .stack(SparseKernelExpander(50, "1^2,3^500^1", 500)));

    assertThrown!Exception({
        auto l = SparseKernelExpander(1000, "1^2", 1_000_000);
    }());


    auto nn = NeuralNet();
    auto l = SparseKernelExpander(10_000, "1^2,3^5^1", 888, 999);
    nn.stack(l)
      .stack(Linear(1))
      .initialize(0.0);

    // cf order 2
    nn.predict([SparseFG(123, 1.3f, 1), SparseFG(456, -2.7f, 2)]);
    assert(l.out_s.length == 3);
    assert(l.out_s[$-1].id == (123 ^ 456));
    assert(fequal(l.out_s[$-1].val, 1.3 * (-2.7)));

    // cf order 2, reversed order
    nn.predict([SparseFG(123, 1.3f, 2), SparseFG(456, -2.7f, 1)]);
    assert(l.out_s.length == 3);
    assert(l.out_s[$-1].id == (123 ^ 456));
    assert(fequal(l.out_s[$-1].val, 1.3 * (-2.7)));

    // not all hashes seen, no cf generated
    nn.predict([SparseFG(123, 1.3f, 1), SparseFG(456, -2.7f, 3)]);
    assert(l.out_s.length == 2);

    // cf order 3
    nn.predict([SparseFG(123, 1.3f, 5), SparseFG(456, -2.7f, 1), SparseFG(789, 0.2f, 3)]);
    assert(l.out_s.length == 4);
    assert(l.out_s[$-1].id == (123 ^ 456 ^ 789));
    assert(fequal(l.out_s[$-1].val, 1.3 * (-2.7) * 0.2));
    
    // bag cf order 2: 3 hashes for bag 1
    nn.predict([SparseFG(123, 1.3f, 1), SparseFG(456, -2.7f, 2), SparseFG(789, 0.4f, 1), SparseFG(333, -9.1f, 1)]);
    assert(l.out_s.length == 7);

    // wrong type, should throw in non-release mode
    assertThrown!AssertError({
        nn.predict([SparseF(123, 1.3f)]);
    }());

    // test serialization
    scope(exit) remove("tmp_test");
    nn.serialize("tmp_test");
    auto l2 = NeuralNet.deserialize("tmp_test").layers[0].to!SparseKernelExpander;
    assert(l._cross_features_str == l2._cross_features_str);
    assert(l._max_group_id == l2._max_group_id && l2._max_group_id == 888);
    assert(l._buff_single_feats_sz == l2._buff_single_feats_sz && l2._buff_single_feats_sz == 999);

    // test that assert is thron in debug mode if adding groups that overflow
    assertThrown!AssertError({
        auto l = SparseKernelExpander(100, "1^2,3^1", 20);
        l.reset();
        l.monitor(23, 12, 1.2f);
    }());
}


/// Test of TanH layer
unittest {
    auto l = TanH();

    auto nn = NeuralNet()
        .stack(DenseData(3))
        .stack(l);
    nn.initialize(0.0);
    
    // forward prop test
    auto x = [1.3f, -2.7f, 0.1f];
    auto preds = nn.predict(x);
    foreach(i; 0..x.length)
    {
        auto truth = (exp(2 * x[i]) - 1) / (exp(2 * x[i]) + 1);
        assert(fequal(preds[i], truth));
    }
    // force non-empty backgrad for test purpose
    l.backgrads.length = 1;
    l.backgrads[0].length = 3;

    // backward prop test
    auto g = [1.0f, 0.52f, 10.9f];
    l.accumulate_grad(g);

    foreach(i; 0..g.length)
    {
        auto tanh = (exp(2 * g[i]) - 1) / (exp(2 * g[i]) + 1);
        auto truth = (1 - tanh * tanh) * g[i];

        assert(fequal(l.backgrads[0][i], truth));
    }
}

/// Test of DropOut layer
unittest {
    auto x = new float[10];
    foreach(i; 0..10)
        x[i] = fabs(cos(i.to!float) + 3.2);

    // drop rate = 1.0, should filter out everything
    auto nn = NeuralNet()
        .stack(DenseData(10))
        .stack(DropOut(1.0));
    nn.initialize(0.0);
    foreach(l; nn.layers)
        l.pre_learning();

    bool all_zeros = true;
    foreach(it; 0..50)
        all_zeros &= nn.predict(x).all!(x => fequal(x, 0.0));
    assert(all_zeros);

    // drop rate = 0.3, should filter out 30%
    nn = NeuralNet()
        .stack(DenseData(10))
        .stack(DropOut(0.3));
    nn.initialize(0.0);
    foreach(l; nn.layers)
        l.pre_learning();

    float num_filtered = 0.0;
    foreach(it; 0..1_000)
        num_filtered += nn.predict(x).count!(x => fequal(x, 0.0));
    num_filtered /= 1_000;
    assert(fequal(num_filtered, 3.0, 0.2)); //loose tol to avoid test flakiness

    // test serialization
    scope(exit) remove("tmp_test");
    nn.serialize("tmp_test");
    auto l2 = NeuralNet.deserialize("tmp_test").layers[1].to!DropOut;
    assert(fequal(l2._drop_rate, 0.3));

    //backward prop for dense case
    auto ldrop = DropOut(0.4);
    nn = NeuralNet()
        .stack(DenseData(10))
        .stack(Linear(5))
        .stack(ldrop);
    nn.initialize(0.001);
    foreach(l; nn.layers)
        l.pre_learning();
    auto x2 = new float[10];
    x2[] = 1.0f;
    auto g2 = new float[5];
    g2[0..2] = 2.0f;
    g2[2..5] = -0.7f;
    double num_pos = 0.0;
    foreach(it; 0..1_000)
    {
        nn.reset();
        nn.predict(x2);
        ldrop.accumulate_grad(g2);
        foreach(i; 0..g2.length)
        {
            num_pos += !fequal(ldrop.backgrads[0][i], 0.0);
        }
    }
    num_pos /= 1_000;
    assert(fequal(num_pos, 3.0, 0.2)); //loose tol to avoid test flakiness
}
