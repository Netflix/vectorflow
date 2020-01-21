module tests.serde;

import tests.common;

import vectorflow;

// ser round-tripping with multi-parents arch
unittest {
    auto nn1 = NeuralNet();
    // two roots
    auto root1 = DenseData(2);
    nn1.add_root(root1);
    auto root2 = DenseData(1);
    nn1.add_root(root2);

    auto l = Linear(1);
    nn1.add(l);

    nn1.wire(root1, l);
    nn1.wire(root2, l);

    nn1.initialize(1.0);

    nn1.serialize("tmp.vf");

    auto nn2 = NeuralNet.deserialize("tmp.vf");

    auto p1 = nn1.predict([0.1f, -0.3f], [0.3f])[0];
    auto p2 = nn2.predict([0.1f, -0.3f], [0.3f])[0];

    assert(fequal(p1, p2));
}
