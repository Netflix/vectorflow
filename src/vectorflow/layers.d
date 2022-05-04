/**
 * Implementation of the various computational layers.
 *
 * Copyright: 2017 Netflix, Inc.
 * License: $(LINK2 http://www.apache.org/licenses/LICENSE-2.0, Apache License Version 2.0)
 */
module vectorflow.layers;

private
{
import std.algorithm : all, any, map, sum;
import std.array;
import std.conv : text, to;
import std.random : uniform01;
import std.string : format, split;

import vectorflow.neurallayer;
import vectorflow.optimizers;
import vectorflow.regularizers;
import vectorflow.serde;
import vectorflow.utils;
import vectorflow.math;
}


/**
 * Linear layer accepting sparse or dense parents and outputing a dense vector.
Examples:
-----------------
auto l1 = Linear(10); // 10 dense output neurons, each with an intercept
auto l2 = Linear(5, false); // 5 dense output neurons, without intercepts
-----------------
 *
 */
class Linear : NeuralLayer {

    float[][] W;
    float[][] grad;
    protected size_t _with_intercept;
    final @property bool with_intercept(){return _with_intercept == 1;}

    AdditiveLinearPrior[] priors;
    ProxyLinearPrior prox;

    this(){super();}
    mixin opCallNew;

    this(ulong dim_out, bool with_intercept_ = true)
    {
        super(dim_out, LayerT.DENSE);
        _learnable = true;
        _with_intercept = with_intercept_;
    }

    override void init(double rand_scale)
    {
        init_matrix_rand(W, rand_scale);
        if(_with_intercept)
        {
            // initialize intercept at 0
            foreach(k; 0..dim_out)
                W[k][0] = 0;
        }
        out_d[] = 0;
    }

    override void allocate_params()
    {
        auto din = dim_in + _with_intercept;
        W = allocate_matrix_zero!float(dim_out, din.to!size_t);
    }

    override void allocate_grad_params()
    {
        auto din = dim_in + _with_intercept;
        grad = allocate_matrix_zero!float(dim_out, din.to!size_t);
    }

    override void predict()
    {
        foreach(k; 0..dim_out)
        {
            auto row = W[k];
            float dp = _with_intercept * row[0];
            auto offset = _with_intercept;
            foreach(l; parents)
            {
                final switch(l.type)
                {
                    case LayerT.DENSE:
                        dp += dotProd(row[offset..offset+l.dim_out], l.out_d);
                        break;

                    case LayerT.SPARSE:
                        foreach(ref f; l.out_s)
                            dp += row[offset + f.id] * f.val;
                        break;
                }
                offset += l.dim_out;
            }
            out_d[k] = dp;
        }
    }

    override void accumulate_grad(float[] ext_grad)
    {
        // gradient
        foreach(k; 0..dim_out)
        {
            auto row = grad[k];
            float g = ext_grad[k];
            _accumulate_grad_row(row, g, k);
        }

        // now backprop gradient
        auto offset = _with_intercept;
        foreach(i, ref b; backgrads)
        {
            if(parents[i].type == LayerT.SPARSE) // TODO: fixme
                continue;
            if(b.length == 0)
                continue;
            foreach(j; 0..dim_out)
                axpy(ext_grad[j], W[j][offset..offset+b.length], b);
            offset += b.length;
        }
    }

    override void accumulate_grad(SparseF[] ext_grad)
    {
        // gradient
        foreach(ref SparseF fg; ext_grad)
        {
            auto row = grad[fg.id];
            float g = fg.val;
            _accumulate_grad_row(row, g, fg.id);
        }

        // now backprop gradient
        auto offset = _with_intercept;
        foreach(i, ref b; backgrads)
        {
            if(parents[i].type == LayerT.SPARSE) // TODO: fixme
                continue;
            if(b.length == 0)
                continue;
            foreach(ref SparseF fg; ext_grad)
                axpy(fg.val, W[fg.id][offset..offset+b.length], b);
            offset += b.length;
        }
    }

    final protected void _accumulate_grad_row(float[] row, float g, ulong index)
    {
        auto offset = _with_intercept;
        foreach(l; parents)
        {
            final switch(l.type)
            {
                case LayerT.DENSE:
                    axpy(g, l.out_d, row[offset..offset+l.dim_out]);
                    break;
                case LayerT.SPARSE:
                    axpy(g, l.out_s, row[offset..offset+l.dim_out]);
                    break;
            }
            offset += l.dim_out;
        }
        row[0] += _with_intercept * g;
    }

    override void serialize(Serializer s)
    {
        s.write(_with_intercept.to!ulong);
        s.write(W.length.to!ulong);
        foreach(i; 0..W.length)
            s.write_vec(W[i]);
    }

    override void deserialize(Serializer s)
    {
        _with_intercept = s.read!ulong().to!size_t;
        W.length = s.read!ulong().to!size_t;
        foreach(i; 0..W.length)
            W[i] = s.read_vec!float();
    }

    override NeuralLayer dup()
    {
        auto cp = new Linear(dim_out, _with_intercept == 1);
        cp.set_name(name);
        foreach(p; priors)
            cp.priors ~= p.dup;
        if(prox !is null)
            cp.prox = prox.dup;

        return cp;
    }

    override void share_params(NeuralLayer l)
    {
        auto c = l.to!Linear;
        W = c.W;
        _with_intercept = c.with_intercept;
    }

    override @property ulong num_params()
    {
        if(W.length == 0)
            return 0;
        return W.length * W[0].length;
    }

    Linear prior(AdditiveLinearPrior prior)
    {
        priors ~= prior;
        prior.register(this);
        return this;
    }

    Linear prior(ProxyLinearPrior prior)
    {
        if(prox !is null)
            throw new Exception("Single proxy prior supported for now.");
        prox = prior;
        prior.register(this);
        return this;
    }
}

/**
 * DropOut layer accepting all dense or all sparse parents.
 *
 * Features rescaling happens automatically at training time.
 * Example:
 * ---
 * auto l = DropOut(0.3); // Drop 30% of the input neurons at random.
 * ---
 */
class DropOut : NeuralLayer {

    float _drop_rate;
    float _scale_ratio;
    protected void delegate() _predict;
    protected void delegate(float[]) _acc_grad;

    this(){super();}
    mixin opCallNew;

    this(float drop_rate)
    {
        super(0, LayerT.DENSE);
        _learnable = false;
        set_drop_rate(drop_rate);
    }

    override void recompute_topology()
    {
        super.recompute_topology();
        dim_out = dim_in;
        auto all_dense = parents.all!(x => x.type == LayerT.DENSE);
        auto all_sparse = parents.all!(x => x.type == LayerT.SPARSE);
        if(!all_dense && !all_sparse)
            throw new Exception(
                "DropOut layer parents have all to be of the same kind " ~
                "(sparse or dense outputs).");
        if(all_dense)
        {
            set_type(LayerT.DENSE);
            _predict = &_predict_dense;
            _acc_grad = &_acc_grad_dense;
        }
        else
        {
            set_type(LayerT.SPARSE);
            _predict = &_predict_sparse;
            _acc_grad = &_acc_grad_sparse;
        }
    }

    override void predict()
    {
        _predict();
    }

    void _predict_sparse()
    {
        // @TODO: this is currently very slow because of allocations
        out_s.length = 0;
        out_s ~= parents[0].out_s; // no need to re-index
        if(parents.length > 1)
        {
            ulong offset = parents[0].dim_out;
            foreach(p; parents[1..$])
            {
                foreach(ref f; p.out_s)
                    out_s ~= SparseF((f.id + offset).to!uint, f.val);
                offset += p.dim_out;
            }
        }
    }

    void _predict_dense()
    {
        size_t offset = 0;
        foreach(p; parents)
        {
            out_d[offset.. offset + p.dim_out] = p.out_d[];
            offset += p.dim_out;
        }
    }

    void _predict_train_sparse()
    {
        // @TODO: this is currently very slow because of allocations
        out_s.length = 0;
        ulong offset = 0;
        foreach(p; parents)
        {
            foreach(ref f; p.out_s)
            {
                if(uniform01(RAND_GEN) > _drop_rate)
                    out_s ~= SparseF((f.id + offset).to!uint, f.val * _scale_ratio);
            }
            offset += p.dim_out;
        }
    }

    void _predict_train_dense()
    {
        size_t offset = 0;
        foreach(p; parents)
        {
            foreach(i; 0..p.dim_out)
            {
                if(uniform01(RAND_GEN) > _drop_rate)
                    out_d[offset + i] = p.out_d[i] * _scale_ratio;
                else
                    out_d[offset + i] = 0.0f;
            }
            offset += p.dim_out;
        }
    }

    override void accumulate_grad(float[] grad)
    {
        _acc_grad(grad);
    }

    void _acc_grad_sparse(float[] grad)
    {
        if(grad.length == 0)
            return;
        throw new NotImplementedException("");
    }

    void _acc_grad_dense(float[] grad)
    {
        if(grad.length == 0)
            return;
        size_t offset = 0;
        foreach(ip, ref p; parents)
        {
            if(parents[ip].type == LayerT.SPARSE)
                throw new NotImplementedException("");
            // for dense parents:
            foreach(i; 0..p.dim_out)
            {
                if(fabs(out_d[offset + i]) > 1e-11)
                    backgrads[ip][i] += grad[offset + i];
            }
            offset += p.dim_out;
        }
    }

    override void pre_learning()
    {
        if(type == LayerT.DENSE)
            _predict = &_predict_train_dense;
        else
        {
            _predict = &_predict_train_sparse;
        }
    }

    override void post_learning()
    {
        if(type == LayerT.DENSE)
            _predict = &_predict_dense;
        else
            _predict = &_predict_sparse;
    }

    protected void set_drop_rate(float rate)
    {
        _drop_rate = rate;
        _scale_ratio = 1.0 / (1.0 - rate);
    }

    override void set_optimizer(Optimizer opt_)
    {
        // nothing to optimize
        optimizer = null;
    }

    override void serialize(Serializer s)
    {
        s.write(_drop_rate);
    }

    override void deserialize(Serializer s)
    {
        _drop_rate = s.read!float();
    }

    override NeuralLayer dup()
    {
        auto cp = new DropOut();
        cp.set_name(name);
        cp.set_drop_rate(_drop_rate);
        return cp;
    }

    override @property ulong num_params()
    {
        return 0;
    }
}


/**
 * ReLU activation layer accepting dense parents.
 */
class ReLU : NeuralLayer {

    this()
    {
        super(0, LayerT.DENSE);
        _learnable = false;
    }
    mixin opCallNew;

    override void recompute_topology()
    {
        super.recompute_topology();
        dim_out = dim_in;
        if(!parents.all!(x => x.type == LayerT.DENSE))
            throw new Exception("ReLU layer only supports dense parents.");
    }

    override void predict()
    {
        size_t offset = 0;
        foreach(p; parents)
        {
            relu(p.out_d, out_d[offset..offset+p.dim_out]);
            offset += p.dim_out;
        }
    }

    version(LDC)
    {
        import ldc.attributes;

        pragma(inline, true)
        @fastmath static void _relu_op(float[] b, float[] o, float[] g) pure
        {
            for(int i = 0; i < o.length; ++i)
            {
                if(o[i] > 0)
                    b[i] += g[i];
            }
        }
    }

    override void accumulate_grad(V)(V[] grad)
        if ((is(V == float) || is(V == SparseF)))
    {
        ulong offset = 0;
        foreach(ip, ref p; parents)
        {
            version(LDC)
            {
                _relu_op(backgrads[ip], p.out_d, grad[offset..offset+p.dim_out]);
            }
            else
            {
                foreach(i; 0..p.dim_out)
                {
                    if(p.out_d[i] > 0)
                        backgrads[ip][i] += grad[offset + i];
                }
            }
            offset += p.dim_out;
        }
    }

    override void set_optimizer(Optimizer opt_)
    {
        // nothing to optimize
        optimizer = null;
    }

    override NeuralLayer dup()
    {
        auto cp = new ReLU();
        cp.set_name(name);
        return cp;
    }

    override @property ulong num_params()
    {
        return 0;
    }
}

/**
 * TanH activation layer accepting dense parents.
 */
class TanH : NeuralLayer {

    this()
    {
        super(0, LayerT.DENSE);
        _learnable = false;
    }
    mixin opCallNew;

    override void recompute_topology()
    {
        super.recompute_topology();
        dim_out = dim_in;
        if(!parents.all!(x => x.type == LayerT.DENSE))
            throw new Exception("TanH layer only supports dense parents.");
    }

    override void predict()
    {
        size_t offset = 0;
        foreach(p; parents)
        {
            tanh(p.out_d, out_d[offset..offset + p.dim_out]);
            offset += p.dim_out;
        }
    }

    override void accumulate_grad(float[] grad)
    {
        size_t offset = 0;
        foreach(ip, ref p; parents)
        {
            // todo: fixme when I have multiple children, should accumulate, not override
            tanh(grad[offset..offset+p.dim_out], backgrads[ip]);
            foreach(i; 0..p.dim_out)
                backgrads[ip][i] = (1 - backgrads[ip][i] * backgrads[ip][i]) * grad[offset + i];

            offset += p.dim_out;
        }
    }

    override void set_optimizer(Optimizer opt_)
    {
        // nothing to optimize
        optimizer = null;
    }

    override NeuralLayer dup()
    {
        auto cp = new TanH();
        cp.set_name(name);
        return cp;
    }

    override @property ulong num_params()
    {
        return 0;
    }
}


/**
 * Scaled Exponential Linear Unit activation layer accepting dense parents.
 * See $(LINK2 https://arxiv.org/pdf/1706.02515.pdf, Self-Normalizing Neural Networks)
 * for details.
 */
class SeLU : NeuralLayer {

    float _alpha;
    float _lambda;

    this()
    {
        super(0, LayerT.DENSE);
        _learnable = false;

        _alpha = 1.67326324235;
        _lambda = 1.05070098736;
    }
    mixin opCallNew;

    override void recompute_topology()
    {
        super.recompute_topology();
        dim_out = dim_in;
        if(!parents.all!(x => x.type == LayerT.DENSE))
            throw new Exception("SeLU layer only supports dense parents.");
    }

    override void predict()
    {
        size_t offset = 0;
        foreach(p; parents)
        {
            foreach(j; 0..p.dim_out)
                if(p.out_d[j] > 0)
                    out_d[offset + j] = _lambda * p.out_d[j];
                else
                    out_d[offset + j] = _lambda * (_alpha * exp(p.out_d[j]) - _alpha);
            offset += p.dim_out;
        }
    }

    override void accumulate_grad(float[] grad)
    {
        size_t offset = 0;
        foreach(ip, ref p; parents)
        {
            foreach(j; 0..p.dim_out)
                if(p.out_d[j] > 0)
                    backgrads[ip][j] += _lambda * grad[offset + j];
                else
                    backgrads[ip][j] += _lambda * _alpha * exp(p.out_d[j]) * grad[offset + j];
            offset += p.dim_out;
        }
    }

    override void set_optimizer(Optimizer opt_)
    {
        // nothing to optimize
        optimizer = null;
    }

    override NeuralLayer dup()
    {
        auto cp = new SeLU();
        cp.set_name(name);
        return cp;
    }

    override @property ulong num_params()
    {
        return 0;
    }
}



/**
 * On-the-fly polynomial kernel expansion of sparse input.
 *
 * This will perform polynomial kernel expansion of a set of sparse features
 * based on a group attribute of this features.
 * The features fed to this layer need to be a SparseFG[].
 * It assumes that the feature ids are uniform random numbers (hashes)
 * so that we can efficiently generate a cross-feature hash by just XOR-ing
 * together the single hashes of the monomial.
 * This layer is meant to be used as part of a NeuralNet() topology at test
 * time, but it's preferable to run the expansion outside the net at training
 * time so that it can be run only once while building the dataset. This will
 * avoid rebuilding the cross-features at every pass during training.
 * Example:
 * ---
 * // polynomial kernel (x_1 * x_3, x_1 * x_2 * x_4) in a 1k dimensional space:
 * auto l = SparseKernelExpander(1_000, "1^3,1^2^4");
 * ---
 */
class SparseKernelExpander : InputLayer
{
    short[] here;
    uint[][] single_hashes;
    float[][] single_vals;
    ushort[][] cross2build;

    uint[] hash_buff;
    float[] val_buff;

    string _cross_features_str;
    uint _max_group_id;
    uint _buff_single_feats_sz;

    this(){super();}
    mixin opCallNew;

    /**
    * Instantiate a new SparseKernelExpander layer.
    *
    * Params:
    *   dim_out = total dimensionality of the input data
    *   cross_feats_str = a string of the form `1^3,2^4^1` specifying which
    *                     groups need to be crossed. The commas delimit the
    *                     groups, the carets delimit the group ids present in
    *                     the monomial
    *   max_group_id = maximum group id present in the data, 1-indexed
    *   buff_single_feats_sz = upper bound of the maximum number of features
    *                          per row post expansion
    */
    this(ulong dim_out, string cross_feats_str, uint max_group_id = 100u,
            uint buff_single_feats_sz = 50_000)
    {
        super(dim_out, LayerT.SPARSE);
        _learnable = false;

        if(max_group_id > ushort.max)
            throw new Exception(
                "Doesn't support group ids bigger than %d".format(ushort.max));
        _cross_features_str = cross_feats_str;
        _max_group_id = max_group_id;
        _buff_single_feats_sz = buff_single_feats_sz;

        _init();
    }

    protected void _init()
    {
        here.length = _max_group_id + 1;
        single_hashes.length = _max_group_id + 1;
        single_vals.length = _max_group_id + 1;
        foreach(k; 0.._max_group_id + 1)
        {
            single_hashes[k].length = _buff_single_feats_sz;
            single_vals[k].length = _buff_single_feats_sz;
        }
        hash_buff.length = _buff_single_feats_sz;
        val_buff.length = _buff_single_feats_sz;
        reset();

        auto cfs = _cross_features_str.split(',');
        foreach(cf; cfs)
        {
            auto ids = cf.split('^').map!(to!ushort).array;
            if(ids.any!(g => g > _max_group_id))
                throw new Exception(
                    ("One group id for cross-feature `%s` is too large. " ~
                     "Maximum group id specified is `%s`").format(
                        cf, _max_group_id));
            cross2build ~= ids;
        }
    }

    override void reset() pure
    {
        here[] = 0;
    }

    override void predict()
    {
        assert(input.convertsTo!(SparseFG[]), (
            "Wrong type: you need to feed features of type `SparseFG[]` to " ~
            "SparseKernelExpander layer `%s`, not `%s`.").format(
                name, input.type));

        auto feats = input.get!(SparseFG[]);
        out_s.length = feats.length;
        foreach(i, ref f; feats)
        {
            monitor(f.group, f.id, f.val);
            out_s[i] = SparseF(f.id, f.val);
        }
        expand!(SparseF[])(out_s);
    }

    final void monitor(ushort id, uint hash, float val) pure
    {
        assert(id > 0, "Group ids are 1-indexed.");
        assert(id <= _max_group_id, text(
                "Group-id bigger than the maximum group-id specified when ",
                "instantiating SparseKernelExpander."));
        short cnt_hashes = here[id];
        single_hashes[id][cnt_hashes] = hash;
        single_vals[id][cnt_hashes] = val;
        here[id] += 1;
    }

    // @TODO: extend to non-hashed ids by extra hashing.
    // Support of bags with bags crossing not supported.
    final void expand(T)(ref T buff)
    {
        foreach(cf; cross2build)
        {
            bool all_here = true;
            ulong ind_bag_feat = -1;
            size_t size_bag = 1;
            foreach(ind_cf; 0..cf.length)
            {
                auto num_hashes = here[cf[ind_cf]];
                if(num_hashes == 0)
                {
                    all_here = false;
                    break;
                }
                if(num_hashes > 1)
                {
                    ind_bag_feat = ind_cf;
                    size_bag = num_hashes;
                }
            }
            if(!all_here)
                continue;
            hash_buff[0..size_bag] = 0;
            val_buff[0..size_bag] = 1.0;
            foreach(ind_cf; 0..cf.length)
            {
                auto cf_id = cf[ind_cf];
                short num_hashes = here[cf_id];

                if(ind_cf != ind_bag_feat)
                {
                    foreach(j; 0..size_bag)
                    {
                        hash_buff[j] ^= single_hashes[cf_id][0];
                        val_buff[j] *= single_vals[cf_id][0];
                    }
                }
                else
                {
                    foreach(j; 0..size_bag)
                    {
                        hash_buff[j] ^= single_hashes[cf_id][j];
                        val_buff[j] *= single_vals[cf_id][j];
                    }
                }
            }
            // add to buffer the expanded CF
            foreach(j; 0..size_bag)
                buff ~= SparseF(hash_buff[j], val_buff[j]);
        }
    }

    override void serialize(Serializer s)
    {
        s.write(_cross_features_str);
        s.write(_max_group_id);
        s.write(_buff_single_feats_sz);
    }

    override void deserialize(Serializer s)
    {
        _cross_features_str = s.read!string();
        _max_group_id = s.read!uint();
        _buff_single_feats_sz = s.read!uint();
        _learnable = false;
        _init();
    }

    override NeuralLayer dup()
    {
        auto cp = new SparseKernelExpander(dim_out, _cross_features_str);
        cp.set_name(name);
        return cp;
    }

    override void share_params(NeuralLayer l)
    {
        auto c = l.to!SparseKernelExpander;
        input = c.input;
    }
}


class Data(alias TYPE) : InputLayer
{
    this(){super();}

    this(ulong dim_out)
    {
        super(dim_out, TYPE);
    }

    override void predict()
    {
        static if(TYPE == LayerT.DENSE)
        {
            assert(input.convertsTo!(float[]), (
            "Wrong type: you need to feed features of type `float[]` to " ~
            "DenseData layer `%s`, not `%s`.").format(name, input.type));

            out_d = input.get!(float[]);
        }
        else
        {
            assert(input.convertsTo!(SparseF[]), (
            "Wrong type: you need to feed features of type `SparseF[]` to " ~
            "SparseData layer `%s`, not `%s`.").format(name, input.type));

            out_s = input.get!(SparseF[]);
        }
    }

    override NeuralLayer dup()
    {
        auto cp = new Data!TYPE(dim_out);
        cp.set_name(name);
        return cp;
    }

    override void share_params(NeuralLayer l)
    {
        auto c = l.to!(Data!TYPE);
        input = c.input;
    }
}

/**
 * Input layer representing a dense float[]
Example:
-----------------
auto l = DenseData(50); // this layer will feed a 50-dimension dense float[] to its children.
-----------------
 *
 */
class DenseData : Data!(LayerT.DENSE)
{
    this(){super();}
    this(ulong dim_out)
    {
        super(dim_out);
    }
    mixin opCallNew;
}

/**
 * Input layer representing a sparse array SparseF[] of (uint, float) pairs
Example:
-----------------
auto l = SparseData(100); // 100 is the total dimensionality of the input space,
// which means that the indices of the pairs SparseF are <= 100. For example,
// [(13, 4.7), (2, -0.12), (87, 0.6)]
-----------------
 *
 */
class SparseData : Data!(LayerT.SPARSE)
{
    this(){super();}
    this(ulong dim_out)
    {
        super(dim_out);
    }
    mixin opCallNew;
}

class NotImplementedException : Exception
{
    this(string msg)
    {
        super(msg);
    }
}
