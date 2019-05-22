/+ dub.json:
{
    "name": "rcv1",
    "dependencies": {"vectorflow": "*"}
}
+/

import std.conv : to;
import std.stdio;
import std.algorithm;
import std.algorithm.searching : countUntil;
import std.algorithm.iteration : splitter;

import vectorflow;
import vectorflow.math : fabs, round;
import vectorflow.dataset : DataFileReader, MultiFilesReader;
import vectorflow.utils : to_long, to_float;

static auto data_dir = "rcv1_data/";

struct Obs {
    float label;
    SparseF[] features;
    Obs dup()
    {
        return Obs(label, features.dup);
    }
}

auto load_data()
{
    // For details on the original dataset, see:
    // Lewis, David D., et al. "Rcv1: A new benchmark collection for text
    // categorization research." Journal of machine learning research
    // 5.Apr (2004): 361-397.
    import std.file;
    import std.typecons;
    if(!exists(data_dir))
    {
        auto root_url = "http://ae.nflximg.net/vectorflow/";
        auto url_data = root_url ~ "lyrl2004_vectors_";
        auto url_topics = root_url ~ "rcv1v2.topics.qrels.gz";
        mkdir(data_dir);
        import std.net.curl;
        import std.process;
        writeln("Downloading data...");
        download(url_data ~ "test_pt0.dat.gz", data_dir ~ "test0.gz");
        download(url_data ~ "test_pt1.dat.gz", data_dir ~ "test1.gz");
        download(url_data ~ "test_pt2.dat.gz", data_dir ~ "test2.gz");
        download(url_data ~ "test_pt3.dat.gz", data_dir ~ "test3.gz");
        download(url_data ~ "train.dat.gz", data_dir ~ "train.gz");
        download(url_topics, data_dir ~ "topics.gz");
        wait(spawnShell(`gunzip ` ~ data_dir ~ "test0.gz"));
        wait(spawnShell(`gunzip ` ~ data_dir ~ "test1.gz"));
        wait(spawnShell(`gunzip ` ~ data_dir ~ "test2.gz"));
        wait(spawnShell(`gunzip ` ~ data_dir ~ "test3.gz"));
        wait(spawnShell(`gunzip ` ~ data_dir ~ "train.gz"));
        wait(spawnShell(`gunzip ` ~ data_dir ~ "topics.gz"));
    }
    // Following Bottou's construction, we use `test{0,1,2,3}` as training set
    // and `train` as test set and build a binary classification
    // dataset to predict whether or not an article has the tag CCAT
    auto labels = load_labels("CCAT");
    writeln("Number of positives: ", labels.sum);
    return tuple(
            new MultiFilesReader!(Obs)(
                [new RCV1Reader(data_dir ~ "test0", labels),
                 new RCV1Reader(data_dir ~ "test1", labels),
                 new RCV1Reader(data_dir ~ "test2", labels),
                 new RCV1Reader(data_dir ~ "test3", labels)]),
            new RCV1Reader(data_dir ~ "train", labels));
}

bool[] load_labels(string cat_name)
{
    auto labels = new bool[816_000];
    labels[] = false;

    auto f = File(data_dir ~ "topics", "r");
    scope(exit) f.close();

    char[] buff;
    while(f.readln(buff))
    {
        auto toks = splitter(buff, " ");
        if(toks.front == cat_name)
        {
            toks.popFront();
            auto ind = to_long(toks.front);
            labels[ind.to!size_t] = true;
        }
    }
    return labels;
}

/*
    Data reader : iterable of
    `Obs` == (label, array of (feature_id == uint, feature_value == float))
*/
class RCV1Reader : DataFileReader!(Obs) {

    private char[] buff;
    private SparseF[] features_buff;
    bool[] labels;

    uint mask = (1 << 16) - 1;

    this(string path, bool[] labels_)
    {
        super(path, false);
        labels = labels_;
        features_buff.length = 1_500;
        _obs = Obs(0, null);
    }

    override bool read_next()
    {
        if(_f.eof)
            return false;

        _f.readln(buff);
        auto lab_end = countUntil(buff, "  ");
        if(lab_end == -1)
            return false;
        auto label = labels[to_long(buff[0..lab_end]).to!size_t];
        _obs.label = label;
        size_t cnt = 0;

        foreach(t; splitter(buff[lab_end+2..$], ' '))
        {
            auto feat_id_end = countUntil(t, ':');
            if(feat_id_end < 1)
                continue;
            auto feat_id = to_long(t[0..feat_id_end]).to!uint & mask; // hashing trick
            auto feat_val = to_float(t[feat_id_end+1..$]);
            features_buff[cnt++] = SparseF(feat_id, feat_val);
        }

        _obs.features = features_buff[0..cnt];
        return true;
    }

    override @property RCV1Reader save()
    {
        auto cp = new RCV1Reader(_path, labels);
        cp.share_save_params(this);
        return cp;
    }
}

void main(string[] args)
{
    writeln("Hello world!");
    auto data = load_data();
    auto train = data[0];
    auto test = data[1];

    // simple sparse linear model with L2 regularization:
    auto nn = NeuralNet()
        .stack(SparseData(1 << 16)) // 2^16 = 65536 > 47236 features
        .stack(Linear(1)
                .prior(L2Prior(0.001)) // L2 regularization with lambda=0.001
                );
    nn.initialize(0.0); // all weights at 0

    train.cache(); // lazily load the dataset in memory

    nn.learn(train, "logistic",
        new AdaGrad(
            4, // number of passes
            0.1, // learning rate
            1_000 // mini-batch size
            ),
        true, // verbose
        4 // number of cores
    );

    ulong cnt = 0;
    double err = 0;
    foreach(ref Obs o; test)
    {
        auto pred = nn.predict(o)[0];
        auto lab = o.label > 0 ? 1 : -1;
        if(lab * pred < 0)
            err++;
        cnt++;
    }
    writeln("Classification error: ", err / cnt);
}
