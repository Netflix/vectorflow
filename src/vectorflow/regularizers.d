/**
 * Implementation of standard regularizers for Linear layer.
 *
 * Copyright: 2017 Netflix, Inc.
 * License: $(LINK2 http://www.apache.org/licenses/LICENSE-2.0, Apache License Version 2.0)
 */
module vectorflow.regularizers;

private
{
import std.random : uniform;

import vectorflow.layers;
import vectorflow.math;
import vectorflow.neurallayer;
import vectorflow.optimizers;
import vectorflow.utils;
}


abstract class LinearPrior
{
    /// pointer to the layer weight matrix
    float[][] W; 
    /// pointer to the layer gradient matrix
    float[][] grad;
    /// pointer to the optimizer for this layer
    SGDOptimizer opt;

    void register(Linear layer)
    {
        W = layer.W;
        grad = layer.grad;
        if(layer.optimizer is null)
           return; 
        if(auto o = cast(SGDOptimizer)(layer.optimizer))
            opt = o;
        else
            throw new Exception("Only support SGDOptimizer type.");
    }

    LinearPrior dup();
}

abstract class ProxyLinearPrior : LinearPrior
{
    abstract void proxy_step();

    override ProxyLinearPrior dup();
}

abstract class AdditiveLinearPrior : LinearPrior
{
    abstract void accumulate_grad();

    override AdditiveLinearPrior dup();
}


/***
Adds L2 regularization to a Linear layer.
Example:
-------------
// basic L2 regularization: loss = (0.03 / 2) * || W ||^2
auto l1 = Linear(5).prior(L2Prior(0.03));

// same, but centered around a non-zero matrix: loss = (0.03 / 2) * || W - W_p ||^2
auto l2 = Linear(5).prior(L2Prior(0.03, W_p));

// L2 regularization with 1 lambda per feature (diagonal Hessian 0-centered prior).
// Example when input dim is 5:
auto l3 = Linear(10).prior(L2Prior([0.01f, 0.02f, 0.01f, 0.015f, 0.005f]));

// L2 regularization with 1 lambda per feature, centered around a non-zero matrix.
// Example when input dim is 5:
auto l4 = Linear(10).prior(L2Prior([0.01f, 0.02f, 0.01f, 0.015f, 0.005f], W_p));
-------------
*/
class L2Prior : AdditiveLinearPrior
{
    float _lambda;
    float[] _lambdas;
    float[][] W_prior;

    protected size_t _ind_start;

    void delegate() _accumulate_grad;

    this(float lambda)
    {
        _lambda = lambda;
        _ind_start = 0;
    }

    this(float lambda, float[][] W_prior_)
    {
        _lambda = lambda;
        W_prior = W_prior_;
        _ind_start = 0;
    }

    this(float[] lambdas)
    {
        _lambdas = lambdas;
        _ind_start = 0;
    }

    this(float[] lambdas, float[][] W_prior_)
    {
        _lambdas = lambdas;
        W_prior = W_prior_;
        _ind_start = 0;
    }

    mixin opCallNew;

    override void register(Linear layer)
    {
        super.register(layer);
        if(W.length > 0 && W_prior.length == 0)
        {
            // default to prior centered on 0
            W_prior = allocate_matrix_zero!float(W.length, W[0].length);
        }
        if(layer.with_intercept)
            _ind_start = 1; // don't regularize intercept

        if(_lambdas.length == 0)
            _accumulate_grad = &_acc_grad_scal;
        else
            _accumulate_grad = &_acc_grad_vec;
    }

    version(LDC)
    {
        private import ldc.attributes;

        pragma(inline, true)
        @fastmath static void l2op_scal(float a, float[] x, float[] x2,
                float[] y, size_t start) pure
        {
            for(auto i = start; i < x.length; ++i)
                y[i] += a * (x[i] - x2[i]);
        }

        pragma(inline, true)
        @fastmath static void l2op_vec(float[] a, float[] x, float[] x2,
                float[] y, size_t start) pure
        {
            for(auto i = start; i < x.length; ++i)
                y[i] += a[i] * (x[i] - x2[i]);
        }

        void _acc_grad_scal()
        {
            foreach(k; 0..W.length)
                l2op_scal(_lambda, W[k], W_prior[k], grad[k], _ind_start);
        }
        
        void _acc_grad_vec()
        {
            foreach(k; 0..W.length)
                l2op_vec(_lambdas, W[k], W_prior[k], grad[k], _ind_start);
        }
    }
    else
    {
        void _acc_grad_scal()
        {
            foreach(k; 0..W.length)
                for(auto i = _ind_start; i < W[k].length; ++i)
                    grad[k][i] += _lambda * (W[k][i] - W_prior[k][i]);
        }

        void _acc_grad_vec()
        {
            foreach(k; 0..W.length)
                for(auto i = _ind_start; i < W[k].length; ++i)
                    grad[k][i] += _lambdas[i] * (W[k][i] - W_prior[k][i]);
        }
    }

    override void accumulate_grad()
    {
        _accumulate_grad();
    }

    override AdditiveLinearPrior dup()
    {
        L2Prior cp;
        if(_lambdas.length == 0)
            cp = new L2Prior(_lambda, W_prior);
        else
            cp = new L2Prior(_lambdas, W_prior);
        cp._ind_start = _ind_start;
        return cp;
    }
}

/***
Adds L1 regularization to a Linear layer.

This is implemented as a proximal operator during SGD.

Example:
-------------
// basic L1 regularization: loss = 0.03 * | W |
auto l1 = Linear(5).prior(L1Prior(0.03));

// same, but centered around a non-zero matrix: loss = 0.03 * | W - W_p |
auto l2 = Linear(5).prior(L1Prior(0.03, W_p));
-------------
*/
class L1Prior : ProxyLinearPrior
{
    float _lambda;
    float[][] W_prior;

    protected size_t _ind_start;

    this(float lambda)
    {
        _lambda = lambda;
        _ind_start = 0;
    }

    this(float lambda, float[][] W_prior_)
    {
        _lambda = lambda;
        W_prior = W_prior_;
        _ind_start = 0;
    }

    mixin opCallNew;

    override void register(Linear layer)
    {
        super.register(layer);
        if(W.length > 0 && W_prior.length == 0)
        {
            // default to prior centered on 0
            W_prior = allocate_matrix_zero!float(W.length, W[0].length);
        }
        if(layer.with_intercept)
            _ind_start = 1; // don't regularize intercept
    }

    override void proxy_step()
    {
        foreach(k; 0..W.length)
        {
            for(auto i = _ind_start; i < W[k].length; ++i)
            {
                if(W[k][i] > W_prior[k][i])
                    W[k][i] = fmax(0, W[k][i] - W_prior[k][i] - _lambda * opt.current_lr(k,i));
                else
                    W[k][i] = -fmax(0, -W[k][i] + W_prior[k][i] - _lambda * opt.current_lr(k,i));
            }
        }
    }

    override ProxyLinearPrior dup()
    {
        auto cp = new L1Prior(_lambda, W_prior);
        cp._ind_start = _ind_start;
        return cp;
    }
}


/***
Adds a positive constraint on the weights of a Linear layer.

If the Linear layer has intercepts, the constraint won't apply to them.

This is implemented as a proximal operator during SGD.
Example:
-------------
// force weights to be positive:
auto l1 = Linear(5).prior(PositivePrior());

// force weights to be above 1e-3:
auto l2 = Linear(10).prior(PositivePrior(1e-3));
-------------
*/
class PositivePrior : ProxyLinearPrior
{
    protected size_t _ind_start;
    protected float _eps;

    this()
    {
        _ind_start = 0;
        _eps = 0;
    }

    this(float eps)
    {
        _ind_start = 0;
        _eps = eps;
    }

    mixin opCallNew;

    override void register(Linear layer)
    {
        super.register(layer);
        if(layer.with_intercept)
            _ind_start = 1; // don't regularize intercept
    }

    override void proxy_step()
    {
        foreach(k; 0..W.length)
            for(auto i = _ind_start; i < W[k].length; ++i)
                W[k][i] = fmax(_eps, W[k][i]);
    }

    override ProxyLinearPrior dup()
    {
        auto cp = new PositivePrior();
        cp._eps = _eps;
        cp._ind_start = _ind_start;
        return cp;
    }
}



class RotationPrior : AdditiveLinearPrior
{
    float _lambda;
    ulong _num_draws;

    this(float lambda, ulong num_draws = 1)
    {
        _lambda = lambda;
        _num_draws = num_draws;
    }

    mixin opCallNew;

    override void accumulate_grad()
    {
        foreach(_; 0.._num_draws)
        {
            size_t i = uniform(0, W.length, RAND_GEN);
            size_t j = uniform(0, W.length, RAND_GEN);

            auto ri = W[i];
            auto rj = W[j];
            float g = dotProd(ri, rj);
            if(i != j)
            {
                foreach(u; 0..W[i].length)
                {
                    grad[i][u] += _lambda * g * rj[u];
                    grad[j][u] += _lambda * g * ri[u];
                }
            }
            else
            {
                g -= 1.0;
                foreach(u; 0..W[i].length)
                    grad[i][u] += 2 * _lambda * g * ri[u];
            }
        }
    }

    override AdditiveLinearPrior dup()
    {
        return new RotationPrior(_lambda, _num_draws);
    }
}

