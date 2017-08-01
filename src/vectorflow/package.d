/**
 Vectorflow is a lightweight neural network library for sparse data.

 Copyright: 2017 Netflix, Inc.
 License: $(LINK2 http://www.apache.org/licenses/LICENSE-2.0, Apache License Version 2.0)
 Authors: Benoit Rostykus
*/
module vectorflow;

public import vectorflow.neuralnet;
public import vectorflow.layers;
public import vectorflow.optimizers;
public import vectorflow.regularizers;
public import vectorflow.neurallayer : type = LayerT, SparseF, SparseFG;
