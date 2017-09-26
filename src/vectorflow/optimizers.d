/**
 Implementation of different stochastic optimizers.

 The default parallelization strategy over the cores is
 $(LINK2 https://people.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf, Hogwild!).
 This is a lock-free strategy where race conditions will occur. This means that
 the library is non-deterministic when training a network as soon as there is
 more than one core involved.
 Hogwild! will work as long as the data access pattern is sparse enough, which
 means that if you have too few dense parameters to learn and too many cores,
 the optimization can fail.

 * Copyright: 2017 Netflix, Inc.
 * License: $(LINK2 http://www.apache.org/licenses/LICENSE-2.0, Apache License Version 2.0)
 */
module vectorflow.optimizers;


private
{
import core.thread : Thread;
import core.time : MonoTime;
import std.algorithm : map, max, reduce;
import std.conv : text, to;
import std.format;
import std.meta : Filter, staticSort;
import std.parallelism : TaskPool;
import std.random;
import std.range : evenChunks;
import std.range.interfaces : ForwardRange;
import std.range.primitives : ElementType, isForwardRange;
import std.stdio;
import std.string : lastIndexOf;
import std.variant;
import std.traits : isNumeric;

import vectorflow.monitor;
import vectorflow.neurallayer;
import vectorflow.neuralnet;
import vectorflow.layers;
import vectorflow.regularizers;
import vectorflow.utils;
import vectorflow.math : fabs, dotProd, sqrt;
}


interface Optimizer {

   void register(NeuralLayer layer);

   void update(NeuralLayer layer, float[] ext_grad);
   void update(NeuralLayer layer, SparseF[] ext_grad);

   Optimizer dup();
}


class SGDOptimizer : Optimizer {

    ulong num_epochs;
    ulong mini_batch_sz;
    float lr;
    ulong cnt;

    AdditiveLinearPrior[] priors;
    ProxyLinearPrior prox;

    this(ulong epochs, ulong mini_batch_sz_, float lr_)
    {
        num_epochs = epochs;
        mini_batch_sz = mini_batch_sz_;
        lr = lr_;
    }

    void learn(D, T, V, R, S)(NeuralNet nn, D data,
                     S delegate(R net_out, ref T ex, ref V[] grad) grad_f,
                     bool verbose, uint num_cores)
        if(isForwardRange!D
                && (is(V == float) || is(V == SparseF))
                && (is(R == float[]) || is(R == NeuralNet))
                && (isNumeric!S || is(S == void)))
    {
        enum Comp(string F1, string F2) = F1 < F2;
        alias feats_fields = staticSort!(
            Comp, Filter!(isFeaturesField, __traits(allMembers, T)));

        auto start_time = MonoTime.currTime;

        auto monitor = new SGDMonitor(verbose, num_epochs, num_cores,
                start_time, isNumeric!S);
        
        void _learn(U)(NeuralNet net, U d, ulong n_passes, uint core_id)
        {
            foreach(l; net.layers)
                l.pre_learning();

            // seed rng per thread with a different seed
            RAND_GEN.seed(42 + cast(uint)Thread.getThis.id);
            static if(is(V == float))
            {
                auto grads = new V[nn.output.length];
                grads[] = 0;
            }
            static if(is(V == SparseF))
            {
                V[] grads;
                grads.length = nn.output.length;
            }

            ulong sum_num_features = 0;
            ulong tmp_cnt = 0;
            double sum_loss = 0.0;
            foreach(p; 0..num_epochs)
            {
                cnt = 0;
                foreach(v; d)
                {
                    auto preds = net.predict(v);
                    foreach(root_id, field; feats_fields)
                        sum_num_features += mixin("v." ~ field ~ ".length");
                    tmp_cnt++;

                    static if(is(R == float[]))
                    {
                        static if(isNumeric!S)
                            sum_loss += grad_f(preds, v, grads);
                        else
                            grad_f(preds, v, grads);
                    }
                    else
                    {
                        static if(isNumeric!S)
                            sum_loss += grad_f(net, v, grads);
                        else
                            grad_f(net, v, grads);
                    }

                    cnt++;
                    net.backward_prop(grads);

                    if(cnt % 100_000 == 0)
                    {
                        synchronized(this)
                            monitor.progress_callback(
                                core_id, p, tmp_cnt, sum_num_features, sum_loss);
                        sum_num_features = 0;
                        tmp_cnt = 0;
                        static if(isNumeric!S)
                            sum_loss = 0;
                    }

                }
                // flush last mini batch
                if(cnt % mini_batch_sz != 0)
                {
                    cnt = mini_batch_sz;
                    net.backward_prop(grads);
                }
                synchronized(this)
                    monitor.progress_callback(
                        core_id, p+1, tmp_cnt, sum_num_features, sum_loss);
                sum_num_features = 0;
                tmp_cnt = 0;
                static if(isNumeric!S)
                    sum_loss = 0;
            }
            foreach(l; nn.layers)
                l.post_learning();

        }

        if(num_cores == 1)
            _learn(nn, data, num_epochs, 0);
        else
        {
            auto chunks = data.evenChunks(num_cores);
            NeuralNet[] nets;
            foreach(i; 0..num_cores)
            {
                auto cp = nn.dup(true);
                foreach(l; cp.layers)
                {
                    l.allocate_interface();
                    l.allocate_grad_params();
                }
                cp.share_params(nn);
                foreach(l; cp.layers)
                    if(l.optimizer)
                        l.optimizer.register(l);
                nets ~= cp;
            }

            auto pool = new TaskPool(num_cores);
            foreach(i, chunk; pool.parallel(chunks))
            {
                auto net = nets[i];
                _learn(net, chunk, num_epochs, i.to!uint);
            }
            pool.stop();
        }
        monitor.wrap_up();
    }

    abstract float current_lr(size_t k, size_t j);

    abstract void register(NeuralLayer layer);

    abstract void update(NeuralLayer layer, float[] ext_grad);
    abstract void update(NeuralLayer layer, SparseF[] ext_grad);
    abstract Optimizer dup();

    override string toString()
    {
        auto fullname = this.classinfo.name;
        auto classname = fullname[fullname.lastIndexOf('.')+1..$];
        return "opt." ~ classname ~
            "[epochs:" ~ num_epochs.to!string ~
            ", mini_batch_sz:" ~ mini_batch_sz.to!string ~
            ", lr:" ~ lr.to!string ~
            "]";
    }
}


/**
 AdaGrad stochastic optimizer.

 See $(LINK2 http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf, 
 Adaptive Subgradient Methods for Online Learning and Stochastic Optimization)
Examples:
-----------------

auto nn = NeuralNet()
    .stack(SparseData(1000))
    .stack(Linear(1));

// Adagrad learning with 5 epochs, learning rate 0.1, mini-batch size of 100:
nn.learn(data, "square", AdaGrad(5, 0.1, 100));
-----------------
 *
 */
class AdaGrad : SGDOptimizer {

    // references
    float[][] W;
    float[][] grad;

    // local variables
    float eps;
    float[][] acc_grad; 

    void delegate(NeuralLayer, float[]) _update;

    this(ulong num_epochs, float lr, ulong mini_batch_sz = 1, float eps_ = 1e-8)
    {
        super(num_epochs, mini_batch_sz, lr);
        eps = eps_;
        _update = &update_general!float;
    }
    mixin opCallNew;

    override void register(NeuralLayer layer)
    {
        if(auto l = cast(Linear)layer)
        {
            params_to_optimize(l.W, l.grad);
            priors = l.priors;
            foreach(p; priors)
                p.register(l);
            prox = l.prox;
            if(prox !is null)
                prox.register(l);
        }
        else if(layer.learnable)
            throw new Exception(
                "AdaGrad solver cannot optimize this layer type: " ~
                layer.to!string);
    }

    void params_to_optimize(float[][] W_, float[][] grad_)
    {
        W = W_;
        grad =grad_;
        acc_grad.length = W.length;
        foreach(k; 0..acc_grad.length)
        {
            acc_grad[k].length = W[k].length;
            acc_grad[k][] = 0;
            grad[k][] = 0;
        }
    }

    override void update(NeuralLayer layer, float[] ext_grad)
    {
        _update(layer, ext_grad);
    }

    override void update(NeuralLayer layer, SparseF[] ext_grad)
    {
        update_general!SparseF(layer, ext_grad);
    }

    void update_general(V)(NeuralLayer layer, V[] ext_grad)
        if ((is(V == float) || is(V == SparseF)))
    {
        layer.accumulate_grad(ext_grad);
        cnt++;
        if(cnt % mini_batch_sz == 0)
        {
            foreach(p; priors)
                p.accumulate_grad();
            update_matrix(W, grad, acc_grad, lr, eps);
            if(prox !is null)
                prox.proxy_step();
            foreach(k; 0..grad.length)
                grad[k][] = 0;
            cnt = 0;
        }
    }

    pragma(inline, true)
    override float current_lr(size_t k, size_t j)
    {
        return lr / sqrt(eps + acc_grad[k][j]);
    }

    version(LDC)
    {
        private import ldc.attributes;

        pragma(inline, true)
        @fastmath static void ada_op1(float[] row, float[] g) pure
        {
            for(int i = 0; i < row.length; ++i)
                row[i] += g[i] * g[i];
        }

        pragma(inline, true)
        @fastmath static void ada_op2(float[] row, float[] g, float[] sg,
                float lr_, float eps_) pure
        {
            for(int i = 0; i < row.length; ++i)
            {
                // reduces cache invalidation across cores
                // since we're writing back to the shared weights
                if(fabs(g[i]) > 0) //LLVM is smart and will vec this to ucomiss
                    row[i] -= lr_ * g[i] / sqrt(eps_ + sg[i]);
            }
        }
    }

    pragma(inline, true)
    final static void update_matrix(
            float[][] w, float[][] g, float[][] sum_g_sq,
            float lr_, float eps_) pure
    {
        foreach(k; 0..w.length)
        {
            auto row_W = w[k];
            auto row_g = g[k];
            auto row_sum_g_sq = sum_g_sq[k];

            version(LDC)
            {
                ada_op1(row_sum_g_sq, row_g);
                ada_op2(row_W, row_g, row_sum_g_sq, lr_, eps_);
            }
            else
            {
                foreach(i; 0..row_W.length)
                {
                    row_sum_g_sq[i] += row_g[i] * row_g[i];
                    row_W[i] -= lr_ * row_g[i] / sqrt(eps_ + row_sum_g_sq[i]);
                }
            }
        }
    }

    override AdaGrad dup()
    {
        return new AdaGrad(num_epochs, lr, mini_batch_sz, eps);
    }

    override string toString()
    {
        return super.toString ~ "[eps:" ~ eps.to!string ~ "]";
    }
}


/**
 ADAM stochastic optimizer.

 See $(LINK2 https://arxiv.org/pdf/1412.6980.pdf, ADAM: A method for stochastic optimization)
Examples:
-----------------

auto nn = NeuralNet()
    .stack(DenseData(200))
    .stack(Linear(10));

// ADAM learning with 5 epochs, learning rate 0.1, mini-batch size of 100:
nn.learn(data, "multinomial", ADAM(5, 0.1, 100));
-----------------
 *
 */
class ADAM : SGDOptimizer {

    // references
    float[][] W;
    float[][] grad;

    // local variables
    float eps;

    float beta1_0;
    float beta2_0;
    float beta1;
    float beta2;

    float[][] M;
    float[][] S;

    this(ulong num_epochs, float lr, ulong mini_batch_sz = 1,
         float eps_ = 1e-8, float beta1_0_ = 0.9, float beta2_0_ = 0.999)
    {
        super(num_epochs, mini_batch_sz, lr);
        // See original ADAM paper for definition of these parameters:
        eps = eps_;
        beta1_0 = beta1_0_;
        beta2_0 = beta2_0_;
    }
    mixin opCallNew;

    override void register(NeuralLayer layer)
    {
        if(auto l = cast(Linear)layer)
        {
            params_to_optimize(l.W, l.grad);
            priors = l.priors;
            foreach(p; priors)
                p.register(l);
            prox = l.prox;
            if(prox !is null)
                prox.register(l);
        }
    }

    void params_to_optimize(float[][] W_, float[][] grad_)
    {
        W = W_;
        grad =grad_;
        M.length = W.length;
        S.length = W.length;
        foreach(k; 0..W.length)
        {
            grad[k][] = 0;
            M[k].length = W[k].length;
            M[k][] = 0;
            S[k].length = W[k].length;
            S[k][] = 0;
        }
        beta1 = beta1_0;
        beta2 = beta2_0;
    }

    override void update(NeuralLayer layer, float[] ext_grad)
    {
        update_general!float(layer, ext_grad);
    }

    override void update(NeuralLayer layer, SparseF[] ext_grad)
    {
        update_general!SparseF(layer, ext_grad);
    }

    void update_general(V)(NeuralLayer layer, V[] ext_grad)
        if ((is(V == float) || is(V == SparseF)))
    {
        layer.accumulate_grad(ext_grad);
        cnt++;
        if(cnt % mini_batch_sz == 0)
        {
            foreach(p; priors)
                p.accumulate_grad();
            update_matrix();            
            if(prox !is null)
                prox.proxy_step();
            beta1 *= beta1_0;
            beta2 *= beta2_0;
            foreach(k; 0..grad.length)
                grad[k][] = 0;
            cnt = 0;
        }
    }

    pragma(inline, true)
    override float current_lr(size_t k, size_t j)
    {
        return lr / (eps + sqrt(S[k][j] / (1.0 - beta2)));
    }

    version(LDC)
    {
        import ldc.attributes;
        pragma(inline, true)
        @fastmath static void adam_op(float[] row, float beta, float[] g) pure
        {
            for(int i = 0; i < row.length; ++i)
                row[i] = beta * row[i] + (1.0 - beta) * g[i];
        }

        pragma(inline, true)
        @fastmath static void adam_op2(float[] row, float beta, float[] g) pure
        {
            for(int i = 0; i < row.length; ++i)
                row[i] = beta * row[i] + (1.0 - beta) * g[i] * g[i];
        }

        pragma(inline, true)
        @fastmath static void adam_op3(
                float[] row_W, float[] row_S, float[] row_M,
                float beta1_, float beta2_, float eps_, float lr_) pure
        {
            float k1 = 1.0/(1.0 - beta1_);
            float k2 = 1.0/(1.0 - beta2_);
            for(int i = 0; i < row_W.length; ++i)
                row_W[i] -= lr_ * k1 * row_M[i] / (sqrt(k2 * row_S[i]) + eps_);
        }
    }

    final void update_matrix()
    {
        foreach(k; 0..W.length)
        {
            auto row_grad = grad[k];
            auto row_W = W[k];
            auto row_M = M[k];
            auto row_S = S[k];

            version(LDC)
            {
                adam_op(row_M, beta1_0, row_grad);
                adam_op2(row_S, beta2_0, row_grad);
                adam_op3(row_W, row_S, row_M, beta1, beta2, eps, lr);
            }
            else
            {
                foreach(i; 0..row_W.length)
                {
                    auto g = row_grad[i];

                    row_M[i] = beta1_0 * row_M[i] + (1.0 - beta1_0) * g;
                    row_S[i] = beta2_0 * row_S[i] + (1.0 - beta2_0) * g * g;

                    auto gt = row_M[i] / (1.0 - beta1);
                    auto st = row_S[i] / (1.0 - beta2);

                    row_W[i] -= lr * gt / (sqrt(st) + eps);
                }
            }
        }
    }

    override ADAM dup()
    {
        return new ADAM(num_epochs, lr, mini_batch_sz, eps, beta1_0, beta2_0);
    }

    override string toString()
    {
        return text(
            super.toString,
            "[eps:", eps, " beta1:", beta1_0, ", beta2:", beta2_0, "]");
    }
}


class ShadowSGDOptimizer : SGDOptimizer {

    NeuralNet _net;

    this(NeuralNet net)
    {
        _net = net;
        num_epochs = 0;
        foreach(l; net.layers)
        {
            if(!l.learnable)
                continue;
            if(auto o = cast(SGDOptimizer)(l.optimizer))
            {
                if(o.num_epochs > num_epochs)
                    num_epochs = o.num_epochs;
            }
        }
        super(num_epochs, 1, 0.0);
    }

    override float current_lr(size_t k, size_t j){return 0.0f;}
    override void register(NeuralLayer layer){}
    override void update(NeuralLayer layer, float[] ext_grad){}
    override void update(NeuralLayer layer, SparseF[] ext_grad){}

    override Optimizer dup()
    {
        return new ShadowSGDOptimizer(_net);
    }
}
