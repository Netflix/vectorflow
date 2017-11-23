module tests.learning;

import std.algorithm : canFind;
import std.exception;
import std.range;

import tests.common;

import vectorflow;
import vectorflow.math;

unittest {

    auto nn = () => NeuralNet()
        .stack(DenseData(20))
        .stack(Linear(1));

    auto data = fake_data_dense(20, 76);

    // no optimizer specified
    assertThrown!Throwable(nn().learn(data, "logistic"));

    auto base = (bool sparse) => sparse ?
        NeuralNet().stack(SparseData(30)) :
        NeuralNet().stack(DenseData(20));

    auto nets = (bool sparse) => [
        () => base(sparse)
            .stack(Linear(1)),
        () => base(sparse)
            .stack(Linear(44))
            .stack(ReLU())
            .stack(Linear(1)),
        () => base(sparse)
            .stack(Linear(44))
            .stack(ReLU())
            .stack(Linear(23), AdaGrad(5, 0.02, 1))
            .stack(TanH())
            .stack(Linear(1))
    ];

    auto dataS = fake_data_sparse(30, 83);

    foreach(net; nets(false))
    { 
        auto calls = [
            delegate void() {net().learn(data, "logistic", AdaGrad(3, 0.1, 1));},
            delegate void() {net().learn(data, "logistic", AdaGrad(3, 0.1, 1), false, 2);},
            delegate void() {net().learn(data, "logistic", ADAM(1, 0.01, 2));}
        ];

        foreach(c; calls)
            assertNotThrown!Throwable(c());
    }

    foreach(net; nets(true))
    { 
        auto calls = [
            delegate void() {net().learn(dataS, "logistic", AdaGrad(3, 0.1, 25));},
            delegate void() {net().learn(dataS, "logistic", AdaGrad(2, 0.1, 1), false, 2);},
            delegate void() {net().learn(dataS, "logistic", ADAM(1, 0.01, 2));}
        ];

        foreach(c; calls)
            assertNotThrown!Throwable(c());
    }


    // if each layer has an optimizer, optimization should run
    assertNotThrown!Throwable({
        auto n = NeuralNet()
            .stack(DenseData(20))
            .stack(Linear(1), AdaGrad(2, 0.3, 10));
        n.initialize(0.5);
        n.learn(data, "logistic");
    }());
}

/// Test sparse backgrad callback
unittest {
    auto dataS = fake_data_sparse(10, 50);

    auto l = Linear(8);
    auto nn = NeuralNet()
        .stack(SparseData(10))
        .stack(l);
    nn.initialize(0.0);

    // random sparse gradient function
    auto loss = delegate float(float[] nn_out, ref ObsS o, ref SparseF[] grads)
    {
        if(o.label > 0)
        {
            grads.length = 2;
            grads[0] = SparseF(5, 3.2f * (nn_out[6] + 1.4));
            grads[1] = SparseF(7, -0.1f * (0.012 * nn_out[1] - 0.08));
        }
        else
        {
            grads.length = 1;
            grads[0] = SparseF(3, 1.8f * (nn_out[0] + 0.39));
        }
        return 0.0f;
    };

    assertNotThrown!Throwable({
        nn.learn(dataS, loss, AdaGrad(2, 0.1, 5));
    }());

    foreach(k; 0..8)
        foreach(i; 0..l.W[k].length)
            if([5, 7, 3].canFind(k))
                assert(!fequal(l.W[k][i], 0.0));
            else
                assert(fequal(l.W[k][i], 0.0));
}

/// Test sparse backgrad callback : dense and sparse should give same results
unittest{
    auto sparse_logistic = delegate int(float[] nn_out, ref ObsS o, ref SparseF[] grads)
    {
        auto label = o.label > 0 ? 1.0 : -1.0;
        auto expp = exp(-label * nn_out[0]);
        auto pr = 1.0/(1.0 + expp);
        grads.length = 1;
        grads[0] = SparseF(0, - label * (1.0 - pr));
        return 0;
    };

    auto dataS = fake_data_sparse(6, 50);

    // first net: trained through dense callback
    auto l1 = Linear(1);
    auto nn1 = NeuralNet()
        .stack(SparseData(10))
        .stack(l1);
    nn1.initialize(0.0);

    nn1.learn(dataS, "logistic", AdaGrad(2, 0.1, 5), false, 1);

    // second net: trained through sparse callback
    auto l2 = Linear(1);
    auto nn2 = NeuralNet()
        .stack(SparseData(10))
        .stack(l2);
    nn2.initialize(0.0);

    nn2.learn(dataS, sparse_logistic, AdaGrad(2, 0.1, 5), false, 1);

    // weights should be identical
    auto w1 = l1.W[0];
    auto w2 = l2.W[0];

    foreach(i; 0..w1.length)
        assert(fequal(w1[i], w2[i]));
}

/// Test generic callback
unittest{
    auto nn = NeuralNet()
        .stack(DenseData(4))
        .stack(Linear(2))
        .stack(ReLU())
        .stack(Linear(1));
    nn.initialize(0.1);

    // random callback using internal layer state
    // it's important to use `net` here and not a reference to `nn`, because
    // there is hogwild going on so we need to work on the thread-specific
    // shallow copy of the net, which is what is passed as `net` argument.
    auto callback = delegate float(NeuralNet net, ref ObsD o, ref float[] grads)
    {
        import std.conv : to;

        auto pred = net.layers[$-1].out_d[0];
        auto W_inner = net.layers[1].to!Linear.W;
        assert(W_inner.length == 2);
        if(o.label > 0)
            grads[0] = W_inner[1][2] * W_inner[0][3];
        else
            grads[0] = -pred;

        return 0.0;
    };

    auto data = fake_data_dense(4, 30);

    assertNotThrown!Throwable({
        nn.learn(data, callback, AdaGrad(2, 0.01, 6), false, 2);
    }());
}

/// Multi roots learning
unittest{
    
    struct O1 {
        float label;
        float[] features;
        SparseF[] features2;
    }
    auto data = [O1(1.0, [0.1f], [SparseF(1, -4.8f)])];

    auto nn = NeuralNet()
        .add_root("a", DenseData(1))
        .add_root("b", SparseData(2))
        .add("c", Linear(1));
    nn.wire("a", "c");
    nn.wire("b", "c");
    nn.initialize(0.0);

    assertNotThrown!Throwable({
        nn.learn(data, "square", AdaGrad(2, 0.1, 10));
    }());

    // reverse fields definition but good lexicographic order, should work
    struct O2 {
        float label;
        SparseF[] features2;
        float[] features;
    }
    auto data2 = [O2(1.0, [SparseF(0, -1.7f)], [3.1f])];

    assertNotThrown!Throwable({
        nn.learn(data2, "square", AdaGrad(2, 0.1, 10));
    }());

    // wrong lexicographic order, should throw
    struct O3 {
        float label;
        SparseF[] features;
        float[] features_blah;
    }
    auto data3 = [O3(1.0, [SparseF(1, 6.5f)], [0.6f])];

    assertThrown!Throwable({
        nn.learn(data3, "square", AdaGrad(2, 0.1, 10), false);
    }());

    // wrong number of roots, should throw
     struct O4 {
        float label;
        float[] features_blaaah;
    }
    auto data4 = [O4(1.0, [0.6f])];

    assertThrown!Throwable({
        nn.learn(data4, "square", AdaGrad(2, 0.1, 10), false);
    }());
}
