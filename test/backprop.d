module tests.backprop;

import std.exception;
import std.stdio;

import tests.common;

import vectorflow.math;
import vectorflow.neuralnet;
import vectorflow.layers;
import vectorflow.neurallayer;


unittest {

    auto l1 = Linear(20, false);
    auto l2 = Linear(1, false);

    auto nn = NeuralNet()
        .stack(DenseData(2))
        .stack(l1)
        .stack(TanH())
        .stack(l2);
    nn.initialize(0.0);

    // all weights at 1 for l1
    foreach(i; 0..l1.W.length)
        foreach(j; 0..l1.W[i].length)
            l1.W[i][j] = 1.0;

    // all weights at 2 for l2
    foreach(i; 0..l2.W.length)
        foreach(j; 0..l2.W[i].length)
            l2.W[i][j] = 2.0;


    nn.predict([1.0f, 1.0f]);
    double tanh2 = (exp(4.0) - 1)/(exp(4.0) + 1);
    assert(fequal(nn.output[0], 40 * tanh2, 1e-5));

    nn.backward_prop([1.0f]);
    assert(fequal(l2.grad[0][0], tanh2, 1e-5));

    assert(fequal(l1.grad[11][1], 2 * (1 - tanh2 * tanh2), 1e-5));
}
