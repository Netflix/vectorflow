/**
 * Base class for all layers.
 *
 * Copyright: 2017 Netflix, Inc.
 * License: $(LINK2 http://www.apache.org/licenses/LICENSE-2.0, Apache License Version 2.0)
 */
module vectorflow.neurallayer;

private
{
import std.algorithm : canFind, map, sum;
import std.conv : to;
import std.string : lastIndexOf;
import std.variant;

import vectorflow.optimizers;
import vectorflow.serde;
import vectorflow.utils;
}

enum LayerT {
    DENSE,
    SPARSE
}

struct SparseF {
    uint id;
    float val;
}

struct SparseFG {
    uint id;
    float val;
    ushort group;
}

/**
* Abstract base class of all layers in the net.
*/
abstract class NeuralLayer {

    ///
    string name;
    ///
    LayerT type;

    /// total input dimension of this layer (sum of output dimensions of its parents)
    size_t dim_in;
    /// total output dimension of this layer
    size_t dim_out;

    /// array referencing all the children of this layer
    NeuralLayer[] children;
    /// array referencing all the parents of this layer
    NeuralLayer[] parents;

    protected bool _learnable;
    /// whether or not this layer has any parameters to be learnt
    final @property bool learnable(){return _learnable;}

    private ushort num_parents_seen;

    /// dense output vector of this layer (might be unused)
    float[] out_d;
    /// sparse output vector of this layer (might be unused)
    SparseF[] out_s;
    /// array of gradients to backpropagate to parents
    float[][] backgrads; // total sum of sizes should be dim_in

    void init(double random_scale){}
    abstract void predict();

    void accumulate_grad(float[] grad){}
    void accumulate_grad(SparseF[] grad){}

    abstract @property ulong num_params();
    NeuralLayer dup(){return null;}
    void allocate_interface()
    {
        if(type == LayerT.DENSE)
            out_d.length = dim_out;
        backgrads.length = 0;
        foreach(p; parents)
        {
            if(!not_learnable_branch(p))
                backgrads ~= new float[p.dim_out];
            else
                backgrads ~= new float[0];
        }        
    }
    void allocate_params(){}
    void allocate_grad_params(){}

    Optimizer optimizer;
    final @property optimizer_set(){return optimizer !is null;}

    this(){}
    
    this(ulong dim_out_, LayerT type_)
    {
        dim_out = dim_out_.to!size_t;
        dim_in = 0;
        set_type(type_);
    }

    protected void set_type(LayerT t)
    {
        type = t;
        if(type == LayerT.DENSE)
            out_d.length = dim_out;
        else if(type == LayerT.SPARSE)
        {
            out_s.length = 0;
            out_s.reserve(100_000); // non-zero per row pre-alloc, to avoid reallocations
        }
    }

    final void forward_prop()
    {
        num_parents_seen++;
        if(num_parents_seen >= parents.length) // ready to compute value
        {
            num_parents_seen = 0;
            // compute node prediction, using parents (if any) data
            predict();

            // propagate message to children
            foreach(c; children)
                c.forward_prop();
        }
    }

    void backward_prop(V)(V[] grad)
        if ((is(V == float) || is(V == SparseF)))
    {
        if(optimizer !is null)
            optimizer.update(this, grad);
        else
            accumulate_grad(grad);
        foreach(ind, ref p; parents)
            p.backward_prop(backgrads[ind]);
    }

    void set_optimizer(Optimizer opt_)
    {
        optimizer = opt_;
    }

    void reset()
    {
        foreach(b; backgrads)
            b[] = 0;
    }

    void set_name(string name_)
    {
        if(name_ !is null && name.length > 0)
            name = name_;
    }

    void ser(Serializer s)
    {
        s.write(this.classinfo.name);
        s.write(name);
        s.write(dim_in.to!ulong);
        s.write(dim_out.to!ulong);
        s.write(type.to!string);
        serialize(s);
    }

    void deser(Serializer s)
    {
        name = s.read!string();
        dim_in = s.read!ulong().to!size_t;
        dim_out = s.read!ulong().to!size_t;
        type = s.read!string().to!LayerT;
        
        deserialize(s);
    }

    protected void serialize(Serializer s){}
    protected void deserialize(Serializer s){}

    void pre_learning(){}
    void post_learning(){}

    // discard local parameters, and use the one from the argument instead:
    void share_params(NeuralLayer layer){}

    void recompute_topology()
    {
        dim_in = parents.map!(x => x.dim_out).sum;
    }

    static bool not_learnable_branch(NeuralLayer layer)
    {
        bool not_learnable = !layer.learnable;
        foreach(p; layer.parents)
            not_learnable &= not_learnable_branch(p);
        return not_learnable;
    }

    override string toString()
    {
        auto fullname = this.classinfo.name;
        auto classname = fullname[fullname.lastIndexOf('.')+1..$];
        return "layer." ~ classname ~
            "[dim_in:" ~ dim_in.to!string ~
            ", dim_out:" ~ dim_out.to!string ~
            "]";
    }
}

/**
* Base class for all roots of the net.
*/
abstract class InputLayer : NeuralLayer
{
    Variant input;

    this(){super();}

    this(ulong dim_out, LayerT type)
    {
        super(dim_out, type);
        _learnable = false;
    }

    final void forward_prop(T)(T obs)
    {
        static if(!is(T == Variant))
        {
            Variant v = obs;
            input = v;
        }
        else
            input = obs;
        super.forward_prop(); // propagate forward in the graph
    }

    abstract override void predict();
    
    override void accumulate_grad(V)(V[] grad) pure
        if ((is(V == float) || is(V == SparseF))) {}

    override void backward_prop(V)(V[] grad) pure
        if ((is(V == float) || is(V == SparseF))) {}

    override @property ulong num_params(){return 0;}

    override void recompute_topology(){}
    override void allocate_interface(){}
}
