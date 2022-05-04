/+ dub.json:
{
    "name": "mnist",
    "dependencies": {"vectorflow": "*"}
}
+/

import std.stdio;
import std.algorithm;

import vectorflow;
import vectorflow.math : fabs, round;

static auto data_dir = "mnist_data/";

struct Obs {
    float label;
    float[] features;
}

auto load_data()
{
    import std.file;
    import std.typecons;
    if(!exists(data_dir))
    {
        auto root_url = "http://yann.lecun.com/exdb/mnist/";
        mkdir(data_dir);
        import std.net.curl;
        import std.process;
        writeln("Downloading training set...");
        download(
            root_url ~ "train-images-idx3-ubyte.gz",
            data_dir ~ "train.gz");
        download(
            root_url ~ "train-labels-idx1-ubyte.gz",
            data_dir ~ "train_labels.gz");
        writeln("Downloading test set...");
        download(
            root_url ~ "t10k-images-idx3-ubyte.gz",
            data_dir ~ "test.gz");
        download(
            root_url ~ "t10k-labels-idx1-ubyte.gz",
            data_dir ~ "test_labels.gz");

        wait(spawnShell(`gunzip ` ~ data_dir ~ "train.gz"));
        wait(spawnShell(`gunzip ` ~ data_dir ~ "train_labels.gz"));
        wait(spawnShell(`gunzip ` ~ data_dir ~ "test.gz"));
        wait(spawnShell(`gunzip ` ~ data_dir ~ "test_labels.gz"));
    }
    return tuple(load_data(data_dir ~ "train"), load_data(data_dir ~ "test"));
}

Obs[] load_data(string prefix)
{
    import std.conv;
    import std.bitmanip;
    import std.exception;
    import std.array;
    auto fx = File(prefix, "rb");
    auto fl = File(prefix ~ "_labels", "rb");
    scope(exit)
    {
        fx.close();
        fl.close();
    }

    T to_native(T)(T b)
    {
        return bigEndianToNative!T((cast(ubyte*)&b)[0..b.sizeof]);
    }

    Obs[] res;
    int n;
    fx.rawRead((&n)[0..1]);
    enforce(to_native(n) == 2051, "Wrong MNIST magic number. Corrupted data");
    foreach(_; 0..3)
        fx.rawRead((&n)[0..1]);
    foreach(_; 0..2)
        fl.rawRead((&n)[0..1]);

    if(prefix == data_dir ~ "train")
        n = 60_000;
    else
        n = 10_000;

    res.length = n;
    ubyte[] pxls = new ubyte[28 * 28];
    foreach(i; 0..n)
    {
        ubyte label;
        fl.rawRead((&label)[0..1]);

        fx.rawRead(pxls);
        res[i] = Obs(label.to!float, pxls.to!(float[]));
    }

    return res;
}


void main(string[] args)
{
    writeln("Hello world!");

    auto nn = NeuralNet()
        .stack(DenseData(28 * 28)) // MNIST is of dimension 28 * 28 = 784
        .stack(Linear(200)) // one hidden layer
        .stack(DropOut(0.3))
        .stack(SeLU()) // non-linear activation
        .stack(Linear(10)); // 10 classes for 10 digits
    nn.initialize(0.0001);

    auto data = load_data();
    auto train = data[0];
    auto test = data[1];

    nn.learn(train, "multinomial",
            new ADAM(
                15, // number of passes
                0.0001, // learning rate
                200 // mini-batch-size
                ),
            true, // verbose
            4 // number of cores
    );

    // if you want to save the model locally, do this:
    // nn.serialize("dump_model.vf");
    // if you want to load a serialized from disk, do that:
    // auto nn = NeuralNet.deserialize("mnist_model.vf");

    double err = 0;
    foreach(ref o; test)
    {
        auto pred = nn.predict(o);
        float max_dp = -float.max;
        ulong ind = 0;
        foreach(i, f; pred)
            if(f > max_dp)
            {
                ind = i;
                max_dp = f;
            }
        if(fabs(o.label - ind) > 1e-3)
            err++;
    }
    err /= test.length;
    writeln("Classification error: ", err);
}
