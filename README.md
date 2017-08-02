<img src="http://ae.nflximg.net/vectorflow/vectorflow_logo.png" width="200">

**Vectorflow** is a minimalist neural network library optimized for sparse data and single machine environments.

Original blog post [here](https://medium.com/@NetflixTechBlog/introducing-vectorflow-fe10d7f126b8).

### Installation

#### dub package
The library is distributed as a [`dub`](https://code.dlang.org/) package. Add `vectorflow` to the `dependencies` section of your `dub.json`:
```
"vectorflow": "~>1.0.0"
```

The library itself doesn't have any dependency. All you need is a recent D compiler.

**`LDC` is the recommended compiler** for the fastest runtime speed. 

Tested on:
- Linux, OSX
- LDC version: >= 1.1.1
- DMD version: >= 2.073.1

#### Setting up a D environment 
If you're new to [D](http://dlang.org/), keep reading. You will need `dub` (the D package manager) and `LDC` (the LLVM-based D compiler).
##### macOs
```
brew install dub
brew install ldc
```
##### Ubuntu
`dub` can be downloaded [here](https://code.dlang.org/download) (or follow instructions [on this page](http://blog.ljdelight.com/installing-dlang-dmd-dub-on-ubuntu/)).
`LDC` can be installed by running:
```
snap install --classic --edge ldc2
```

### Examples
To run the RCV1 example (sparse logistic regression):
```
cd examples && ./compile_run.sh rcv1.d
```

### Tests
To run the tests:
```
dub test
```

### Documentation
`vectorflow` is using [ddoc](https://dlang.org/spec/ddoc.html).
One way of building and serving locally the documentation (you will need `libevent` for serving) is:
```
dub build -b ddox && dub run -b ddox
```
Or use your favorite DDOC compiler.

Please also refer to the repo wiki.
