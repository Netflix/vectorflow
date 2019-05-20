/**
 The library supports already implemented loss functions, as well as
 a callback-based way to specify a custom loss.
 
 The losses pre-implemented are: `logistic`, `square`, `multinomial`.

 For these losses, if an attribute `.weight` is found in the row, it will
 be used to weight the loss during MLE.

 If one wants to specify a custom loss function, one has to implement a gradient
 callback of the form `S delegate(R net_out, ref T ex, ref V[] grad)` which
 is expected to populate in `grad` the gradient of the loss on datapoint `ex`
 with respect to the output of the net `net_out`.

 `S` is `void` or numeric (float, double, int...).
    If numeric, the callback is expected to return the loss value on
    training sample `ex` for monitoring purposes.
    
 `R` is `float[]` or `NeuralNet`. If `float[]`, the net is expected to have
    a single leaf and the callback receives the predictions of the leaf after
    forward-prop. If `NeuralNet`, the callback receives a reference of the net
    after forward-prop. Useful in case the loss function depends on multiple
    layers values.

 `T` is the templatized row. This row needs at minimum to have an attribute
    starting with the name `feature` to be able to perform forward-prop.

 `V` is `float[]` or `SparseF[]`. If `float`, the backpropagation will be
    ran densely. If `SparseF[]`, the last layer will be sparsely backpropagated.
    More efficient when the gradient is sparse and the output dimension large.

 Examples:
 ---
 // median (L1) loss: minimize absolute differences
 auto loss_grad = float delegate(float[] nn_out, ref Obs o, ref float[] grads)
 {
    auto pred = nn_out[0]; // this is the predictions of the net
    // after forward-prop
    if(pred > o.label) // gradient of |pred - label| with respect to pred
        grads[0] = 1.0f;
    else
        grads[0] = -1.0f;
    
    return fabs(pred - o.label); // return loss value so it's monitored
    // during training
 }
 net.learn(data, loss_grad, ...);
 ---


 Copyright: 2017 Netflix, Inc.
 License: $(LINK2 http://www.apache.org/licenses/LICENSE-2.0, Apache License Version 2.0)
*/
module vectorflow.losses;

private{
import vectorflow.math;
import vectorflow.utils : ct_msg;
}

auto get_grad(T, alias WITH_VAL, V...)(string loss, V args)
{
    static if(!__traits(hasMember, T, "label"))
        static assert(0,
            "When using a predefined loss, your row needs to have a `label`" ~
            " attribute.");

    static if(__traits(hasMember, T, "weight"))
        mixin ct_msg!("Using `weight` attribute to perform weighted MLE inference");
    switch(loss)
    {
        case "logistic":
            static if(WITH_VAL == true)
            {
                return delegate float(float[] nn_out, ref T o, ref float[] grads)
                {
                    auto label = o.label > 0 ? 1.0 : -1.0;
                    auto expp = exp(-label * nn_out[0]);
                    auto pr = 1.0/(1.0 + expp);
                    grads[0] = - label * (1.0 - pr);
                    float loss = log(1.0 + expp);
                    static if(__traits(hasMember, T, "weight"))
                    {
                        grads[0] *= o.weight;
                        loss *= o.weight;
                    }
                    return loss;
                };
            }
            else
            {
                return delegate void(float[] nn_out, ref T o, ref float[] grads)
                {
                    auto label = o.label > 0 ? 1.0 : -1.0;
                    auto expp = exp(-label * nn_out[0]);
                    auto pr = 1.0/(1.0 + expp);
                    grads[0] = - label * (1.0 - pr);
                    static if(__traits(hasMember, T, "weight"))
                        grads[0] *= o.weight;
                };
            }
        case "square":
            static if(WITH_VAL == true)
            {
                return delegate float(float[] nn_out, ref T o, ref float[] grads)
                {
                    auto diff = o.label - nn_out[0];
                    grads[0] = - diff;
                    static if(__traits(hasMember, T, "weight"))
                    {
                        grads[0] *= o.weight;
                        return 0.5 * diff * diff * o.weight;
                    }
                    else
                        return 0.5 * diff * diff;
                };
            }
            else
            {
            return delegate void(float[] nn_out, ref T o, ref float[] grads)
            {
                auto diff = o.label - nn_out[0];
                grads[0] = - diff;
                static if(__traits(hasMember, T, "weight"))
                    grads[0] *= o.weight;
            };

            }
        case "multinomial":
            static if(WITH_VAL == true)
            {
                return delegate float(float[] nn_out, ref T o, ref float[] grads)
                {
                    double normalizer = 0;
                    foreach(i; 0..nn_out.length) // number of classes
                    {
                        auto expp = exp(nn_out[i]);
                        normalizer += expp;
                        grads[i] = expp;
                    }
                    double loss = 0;
                    foreach(i; 0..nn_out.length)
                    {
                        auto r = grads[i] / normalizer;
                        double lab = round(o.label - i) == 0 ? 1.0 : 0.0;
                        if(lab > 0)
                            loss += log(r + 1e-9);
                        grads[i] = r - lab;
                        static if(__traits(hasMember, T, "weight"))
                            grads[i] *= o.weight;
                    }
                    static if(__traits(hasMember, T, "weight"))
                        loss *= o.weight;
                    return -loss;
                };
            }
            else
            {
                return delegate void(float[] nn_out, ref T o, ref float[] grads)
                {
                    double normalizer = 0;
                    foreach(i; 0..nn_out.length) // number of classes
                    {
                        auto expp = exp(nn_out[i]);
                        normalizer += expp;
                        grads[i] = expp;
                    }
                    foreach(i; 0..nn_out.length)
                    {
                        auto r = grads[i] / normalizer;
                        double lab = round(o.label - i) == 0 ? 1.0 : 0.0;
                        grads[i] = r - lab;
                        static if(__traits(hasMember, T, "weight"))
                            grads[i] *= o.weight;
                    }
                };
            }
        default:
            throw new Exception("Unknown loss function: " ~ loss ~ ". You " ~
                    "have to compute the gradient of your loss yourself.");
    }
}
