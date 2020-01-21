/**
 * NeuralNet is the main abstraction of vectorflow.
 *
 * Copyright: 2017 Netflix, Inc.
 * License: $(LINK2 http://www.apache.org/licenses/LICENSE-2.0, Apache License Version 2.0)
 */
module vectorflow.neuralnet;

private
{
import std.algorithm : canFind, countUntil, map, sort, startsWith, sum;
import std.array;
import std.conv : text, to;
import std.file : exists, FileException, remove;
import std.format : format;
import std.meta : anySatisfy, Filter, staticSort;
import std.stdio : File, writeln;

import std.range.primitives : isForwardRange, isInputRange, ElementType;
import std.traits : isAggregateType, isNumeric;
import std.variant;

import vectorflow.layers;
import vectorflow.neurallayer;
import vectorflow.serde;
import vectorflow.optimizers;
import vectorflow.losses;
import vectorflow.utils : ct_msg, opCallNew;
}


/***
 * Neural-network abstraction.
Example:
-----------------
auto nn = NeuralNet()
    .stack(DenseData(400))
    .stack(Linear(10));
// nn is a network working on 400-dimensions dense vectors and predicting
// a 10-dimensions vector
-----------------
*/
class NeuralNet {

    /// array of all the roots
    InputLayer[] roots;
    /// all nodes of the computational graph
    NeuralLayer[] layers;
    /// map: name --> layer
    NeuralLayer[string] layers_map;
    /// edges of the graph: src -> [dst1, ..., dstk]
    string[][string] edges;
    /// array of all the leaves
    NeuralLayer[] leaves;
    /// reference to the leaf of the net
    @property NeuralLayer out_layer(){return leaves[0];}

    private bool _ever_initialized;

    this()
    {
        _ever_initialized = false;
    }
    mixin opCallNew;

    /**
    * Name and add a root to the net.
    *
    * Params:
    *   name_ = name to give to the layer.
    *   layer = input layer to add as root to the net.
    *
    * Returns: current neural network with the newly added layer.
    */
    NeuralNet add_root(string name_, InputLayer layer)
    {
        check_name(name_);
        layer.name = name_;
        return add_root(layer);
    }

    /**
    * Add a root to the net.
    *
    * Params:
    *   root_ = input layer to add as root to the net.
    *
    * Returns: current neural network with the newly added layer.
    */
    NeuralNet add_root(InputLayer root_)
    {
        roots ~= root_;
        add(root_);
        return this;
    }

    /**
    * Name and add a layer to the net, without wiring it.
    *
    * Params:
    *   name_ = name to give to the layer.
    *   layer = which layer to add to the net.
    *   opt = optional optimizer to use for this layer.
    *
    * Returns: current neural network with the newly added layer.
    */
    NeuralNet add(string name_, NeuralLayer layer, Optimizer opt = null)
    {
        check_name(name_);
        layer.name = name_;
        return add(layer, opt);
    }

    /**
    * Add a layer to the net, without wiring it.
    *
    * Params:
    *   layer = which layer to add to the net.
    *   opt = optional optimizer to use for this layer.
    *
    * Returns: current neural network with the newly added layer.
    */
    NeuralNet add(NeuralLayer layer, Optimizer opt = null)
    {
        if(roots.length == 0)
        {
            if((cast(InputLayer)layer) is null)
                throw new Exception(
                    "First layer added has to be an InputLayer.");
            if(opt !is null)
                throw new Exception(
                    "A root is not learnable, it cannot have an optimizer.");
            add_root(layer.to!InputLayer);
        }
        else
        {
            if(layer.name == "")
                layer.name = generate_name();
            if(layer.name in layers_map)
                throw new Exception("A layer with the name `" ~ 
                        layer.name ~ "` already exist.");
            layers_map[layer.name] = layer;
            layers ~= layer;
            leaves ~= layer;
        }
        if(opt !is null)
            layer.set_optimizer(opt);

        return this;
    }

    /**
    * Stack a layer on top of the former leaf of the net.
    *
    * Params:
    *   layer = which layer to add to the net. It will be wired to the
    *           previous leaf.
    *   opt = optional optimizer to use for this layer.
    *
    * Returns: current neural network with the newly added layer.
    */
    NeuralNet stack(NeuralLayer layer, Optimizer opt = null)
    {
        if(leaves.length > 1)
            throw new Exception("Your current net is not a stack.");
        add(layer, opt);
        if(layers.length >= 2)
        {
            // wire to previous
            auto previous = layers[$-2];
            wire(previous, layer);
        }
        return this;
    }

    /**
    * Stack a layer on top of the former leaf of the net.
    *
    * Params:
    *   name_ = name to give to the layer
    *   layer = which layer to add to the net. It will be wired to the
    *           previous leaf.
    *   opt = optional optimizer to use for this layer.
    *
    * Returns: current neural network with the newly added layer.
    */
    NeuralNet stack(string name_, NeuralLayer layer, Optimizer opt = null)
    {
        check_name(name_);
        layer.name = name_;
        return stack(layer, opt);
    }

    /**
    * Compute the prediction of the net for $(PARAM v).
    * Runs forward-propagation and outputs the predicted vector.
    *
    * Params:
    *    v = observation with one or multiple `features*` attributes
    *        which have the types expected by the roots in proper order
    *        (i.e: float[], SparseF[], SparseFG[], custom roots types...)
    *
    * Returns: array of last layer neurons values 
    *
    * Example:
    * ---
    * struct O {
    *   float[] features_foo;
    * }
    * net.predict(O([1.2f, 0.7f]));
    * ---
    */
    float[] predict(T)(T v) if(isAggregateType!T && isLearnableRow!T)
    {
        enum Comp(string F1, string F2) = F1 < F2;
        alias feats_fields = staticSort!(
            Comp, Filter!(isFeaturesField, __traits(allMembers, T)));
        assert(feats_fields.length == roots.length,
            "Number of `features*` fields should match number of roots.");
        reset();
        foreach(root_id, field; feats_fields)
            roots[root_id].forward_prop(mixin("v." ~ field));
        return output;
    }

    /**
    * Compute the prediction of the net when passing the arguments to the
    * root(s) of the net.
    *
    * Params:
    *    args = the data to feed to the roots in proper order
    *
    * Returns: array of last layer neurons values
    *
    * Examples:
    * ---
    * // net with a single DenseData(2) root:
    * net.predict([3.2f, -1.5f]);
    * // net with a single SparseData(dim >= 34) root:
    * net.predict([SparseF(34, -0.7f), SparseF(3, 0.2f)]);
    * // net with one DenseData(1), one SparseData(dim >= 16) root:
    * net.predict([0.2f], [SparseF(16, -0.15f)]);
    * ---
    */
    float[] predict(T...)(T args)
    {
        assert(args.length == roots.length,
            "The number of arguments should match the number of roots.");
        reset();
        foreach(i, v; args)
            roots[i].forward_prop(v);
        return output;
    }

    /**
    * Create a directed edge between `parent` and `child` nodes.
    *
    * Params:
    *    parent = name of origin layer
    *    child = name of destination layer
    *    with_alloc = whether or not both layers should allocate internal
    *                 parameters
    */
    void wire(string parent, string child, bool with_alloc = true)
    {
        check_layer_here(parent);
        check_layer_here(child);

        auto p = layers_map[parent];
        auto c = layers_map[child];
        wire(p, c, with_alloc);
    }

    /**
    * Create a directed edge between `parent` and `child` nodes.
    *
    * Params:
    *    parent = origin layer
    *    child = destination layer
    *    with_alloc = whether or not both layers should allocate internal
    *                 parameters
    */    
    void wire(NeuralLayer parent, NeuralLayer child, bool with_alloc = true)
    {
        check_layer_here(parent.name);
        check_layer_here(child.name);
        if(parent.name in edges && edges[parent.name].canFind(child.name))
            throw new Exception(
                    "The edge `" ~
                    parent.name ~ "` -> `" ~ child.name ~
                    "` has already been added to the graph.");
        parent.children ~= child;
        child.parents ~= parent;
        foreach(l; layers)
            l.recompute_topology();
        if(with_alloc)
        {
            parent.allocate_interface();
            parent.allocate_params();
            parent.allocate_grad_params();
            child.allocate_interface();
            child.allocate_params();
            child.allocate_grad_params();
        }
        edges[parent.name] ~= child.name;

        // remove parent from the leaves array if it was already there:
        auto ind_leaf = leaves.countUntil!(l => l.name == parent.name);
        if(ind_leaf != -1)
        {
            if(leaves.length == 1)
                leaves.length = 0;
            if(ind_leaf == 0)
                leaves = leaves[1..$];
            else if(ind_leaf == leaves.length - 1)
                leaves = leaves[0..$-1];
            else
                leaves = leaves[0..ind_leaf] ~ leaves[ind_leaf+1..$];
        }
        
        optimize_graph(this);
    }

    protected void check_layer_here(string name)
    {
        if(name in layers_map)
            return;
        throw new Exception(text(
           "Layer `", name,
           "` is unknown. Add it to the net first if you ",
           "want to wire it.\nCurrent net: ", this));
    }

    /**
    * Initialize at random all the parameters of the net.
    *
    * Params:
    *    rand_scale = parameters values drawn in ]-rand_scale, rand_scale[
    */ 
    void initialize(double rand_scale)
    {
        _ever_initialized = true;
        foreach(l; layers)
            l.init(rand_scale);
    }

    /**
    * Return a reference to the dense output vector of the leaf of the net.
    */
    @property float[] output(){ return out_layer.out_d; }

    void backward_prop(V)(V[] output_grad)
        if ((is(V == float) || is(V == SparseF)))
    {
        out_layer.backward_prop(output_grad); // backpropagation
    }

    /**
    * Return the total number of learnable parameters in the net.
    */
    @property ulong num_params()
    {
        return layers.map!(l => l.num_params).sum;
    }

    /**
    * Reset any internal state variables of the net.
    */
    void reset()
    {
        foreach(l; layers)
            l.reset();
    }

    /**
    * Remove any optimizer defined on layers of the net.
    */
    void clear_opt()
    {
        foreach(l; layers)
            l.set_optimizer(null);
    }

    /**
    * Discard local weights and use those of the target net instead.
    * However, the net keeps its own internal state.
    * Useful for hogwild SGD implementation.
    *
    * Params:
    *    net = NeuralNet whose parameters should be used.
    */
    void share_params(NeuralNet net)
    {
        foreach(i, ref l; layers)
            if((cast(InputLayer)l) is null)
                l.share_params(net.layers[i]);
    }

    /**
    * Train neural network on some data, using specified gradient callback and
    * optimizer.
    *
    * Params:
    *    data = forward range of rows
    *    grad_f = gradient callback (see losses.d for details)
    *    opt = optimizer to use on all learnable layers for training
    *    verbose = whether or not to show progress during training
    *    num_cores = degree of Hogwild parallelism
    */
    void learn(D, T, V, R, S, O : Optimizer)(
            D data,
            S delegate(R net_out, ref T ex, ref V[] grad) grad_f,
            O opt, bool verbose = false, uint num_cores = 1)
        if(isForwardRange!D && is(ElementType!D == T) // dataset constraints
                && (is(V == float) || is(V == SparseF))
                && (is(R == float[]) || is(R == NeuralNet))
                && (isNumeric!S || is(S == void)))
    {
        static if(!isAggregateType!T || !isLearnableRow!T)
        {
            static assert(0, text(
                "Your rows are invalid. Rows should be of an aggregate type (",
                "struct, class, union or interface) and have at least one ",
                "attribute or property whose name starts with `features`: ",
                "that's the data that will be forward-propagated into the ",
                "computational graph. If your graph has multiple roots, the ",
                "lexicographic order of the attributes starting with ",
                "`features` will be used to map them to the roots of ",
                "the graph, in the original order these roots were added to ",
                "the graph."));
        }

        if(!_ever_initialized)
        {
            writeln("Net not initialized. Initializing all weights to 0.");
            initialize(0.0);
        }
        {
            foreach(l; layers)
            {
                if(!l.learnable)
                    continue;
                if(!l.optimizer_set)
                {
                    auto opt_cp = opt.dup;
                    l.set_optimizer(opt_cp);
                    opt_cp.register(l);
                }
                else
                {
                    l.set_optimizer(l.optimizer);
                    l.optimizer.register(l);
                }
            }
            // this is just to drive the learning, but each node
            // has its own copy and optimization variables in a SGD setting
        }

        auto cores_str = (
                num_cores == 1 ? "1 core." : "%d cores.".format(num_cores));
        writeln("Training net with ", num_params, " parameters on ", cores_str);
        foreach(l; layers)
            l.pre_learning();
        opt.learn(this, data, grad_f, verbose, num_cores);
        foreach(l; layers)
            l.post_learning();
    }

    /**
    * Train neural network on a dataset, using a predefined loss.
    *
    * Params:
    *    data = forward range of rows
    *    loss = one of the predefined loss functions
    *    opt = optimizer to use on all learnable layers for training
    *    verbose = whether or not to show progress during training
    *    num_cores = degree of Hogwild parallelism
    *    monitor_loss = whether or not loss value should be tracked during
    *    training for monitoring (slightly slower)
    */
    void learn(D, O : Optimizer)(D data, string loss,
            O opt, bool verbose = false, uint num_cores = 1,
            bool monitor_loss = true)
        if(isForwardRange!D)
    {
        if(monitor_loss)
        {
            learn(data, get_grad!(ElementType!D, true)(loss),
                    opt, verbose, num_cores);
        }
        else
        {
            learn(data, get_grad!(ElementType!D, false)(loss),
                    opt, verbose, num_cores);
        }
    }

    /**
    * Train neural network on some data, using a gradient callback.
    *
    * Assumes that an optimizer has already been specified on all learnable
    * layers.
    *
    * Params:
    *    data = forward range of rows
    *    grad_f = gradient callback (see losses.d for details)
    *    verbose = whether or not to show progress during training
    *    num_cores = degree of Hogwild parallelism
    */
    void learn(D, T, V, R, S)(
            D data,
            float delegate(R net_out, ref T ex, ref V[] grad) grad_f,
            bool verbose = false, uint num_cores = 1)
    {
        check_all_layers_have_optimizer();
        auto driver = new ShadowSGDOptimizer(this);

        learn(data, grad_f, driver, verbose, num_cores);
    }

    /**
    * Train neural network on some data, using a predefined loss.
    *
    * Assumes that an optimizer has already been specified on all learnable
    * layers.
    *
    * Params:
    *    data = forward range of rows
    *    loss = one of the predefined loss functions
    *    verbose = whether or not to show progress during training
    *    num_cores = degree of Hogwild parallelism
    */
    void learn(D)(D data, string loss, bool verbose = false, uint num_cores = 1)
    {
        check_all_layers_have_optimizer();
        auto driver = new ShadowSGDOptimizer(this);

        learn(data, loss, driver, verbose, num_cores);
    }

    override string toString()
    {
        string s = "NeuralNet[" ~ this.num_params.to!string ~ " parameters]\n";
        foreach(l; layers)
            s ~= (l.name ~ "|" ~ l.to!string ~ "\n");
        return s[0..$-1];
    }

    private void check_name(string name_)
    {
        if(name_.length == 0)
            throw new Exception("You must specify a non-empty name");
        else if(name_.canFind(','))
            throw new Exception(
                "Name of layers cannot contain commas: `" ~ name_ ~ "`.");
    }

    static bool is_upstream_stack(NeuralLayer layer)
    {
        bool is_stack = layer.parents.length <= 1 && layer.children.length <= 1;
        foreach(p; layer.parents)
            is_stack &= is_upstream_stack(p);
        return is_stack;
    }

    private void check_all_layers_have_optimizer()
    {
        string not_set;
        foreach(l; layers)
        {
            if(l.learnable && !l.optimizer_set())
                not_set ~= l.name ~ ",";
        }
        if(not_set != "")
            throw new Exception(
                "You haven't specified an optimizer for the following " ~
                "learnable layers: " ~ not_set[0..$-1]);
    }

    private string generate_name()
    {
        return "layer" ~ to!string(layers.length + 1);
    }

    /**
    * Dump the neural net (topology and weight values) to the specified path.
    *
    * Params:
    *    path = where to dump the neural net.
    */
    void serialize(string path)
    {
        auto f = File(path, "w");
        scope(exit) f.close();
        scope(failure)
        {
            f.close();
            try
            {
                writeln("Serialization failed.");
                remove(path);
            }
            catch(FileException e)
            {
                writeln("Couldn't cleanup `", path,
                        "` after serialization failure: ", e);
            }
        }

        auto ser = new Serializer(&f);

        // serialize root names
        ser.write(roots.length.to!ulong);
        foreach(r; roots)
            ser.write(r.name);

        // serialize edges
        ser.write(edges.length.to!ulong);
        foreach(p; edges.byKeyValue().array.sort!((x, y) => x.key < y.key))
        {
            ser.write(p.value.length.to!ulong);
            foreach(child; p.value)
            {
                ser.write(p.key ~ "," ~ child); // parent,child
            }
        }

        // now serialize layers
        foreach(l; layers)
            l.ser(ser);
    }

    /**
    * Deserialize the neural net from the specified path.
    *
    * Params:
    *    path = file path of the neural net to read.
    */
    static NeuralNet deserialize(string path)
    {
        if(!exists(path))
            throw new Exception("File does not exists: " ~ path);
        auto f = File(path, "r");
        scope(exit) f.close();
        scope(failure) f.close();

        auto nn = new NeuralNet();

        auto deser = new Serializer(&f);

        // deserialize root names
        bool[string] root_names;
        auto num_roots = deser.read!ulong();
        foreach(_; 0..num_roots)
            root_names[deser.read!string()] = true;

        // deserialize edges
        string[][string] edges;
        auto num_parents = deser.read!ulong();
        foreach(_; 0..num_parents)
        {
            auto num_children = deser.read!ulong();
            foreach(__; 0..num_children)
            {
                auto edge = deser.read!string();
                auto toks = edge.split(',');
                edges[toks[0]] ~= toks[1];
            }
        }

        // deserialize all layers
        auto layers = deser.deserialize_layers();
        foreach(l; layers)
        {
            if(l.name in root_names)
                nn.add_root(l.to!InputLayer);
            else
                nn.add(l);
        }

        foreach(p; edges.byKeyValue().array.sort!((x, y) => x.key < y.key))
            foreach(child; p.value)
                nn.wire(p.key, child, false);
        foreach(l; nn.layers)
        {
            if(l.type ==LayerT.DENSE)
                l.out_d.length = l.dim_out;
            l.allocate_interface();
        }

        return nn;
    }

    /**
    * Return a copy of the net.
    *
    * Params:
    *    topology_only = whether or not the copy should be shallow
    */
    NeuralNet dup(bool topology_only = false)
    {
        auto cp = new NeuralNet();

        bool[string] root_names;
        foreach(r; roots)
        {
            root_names[r.name] = true;
            cp.add_root(r.name, cast(InputLayer)r.dup);
        }
        foreach(l; layers)
        {
            if(l.name !in root_names)
            {
                auto lcp = l.dup;
                if(l.optimizer)
                    lcp.set_optimizer(l.optimizer.dup);
                cp.add(l.name, lcp);
            }
        }

        foreach(p; edges.byKeyValue())
            foreach(child; p.value)
                cp.wire(p.key, child, !topology_only);
        if(!topology_only)
            foreach(l; cp.layers)
            {
                l.allocate_interface();
                l.allocate_params();
            }
        return cp;
    }
}

package enum isFeaturesField(string s) = s.startsWith("features");
package enum isLearnableRow(T) = anySatisfy!(isFeaturesField, __traits(allMembers, T));


private void optimize_graph(NeuralNet net)
{
    foreach(layer; net.layers)
    {
        if(auto l = cast(Linear)layer)
        {
            foreach(p; l.priors)
                p.register(l);
            if(l.prox !is null)
                l.prox.register(l);
        }
    }
}

version(assert)
{
    mixin ct_msg!("Non-release build.");
}
