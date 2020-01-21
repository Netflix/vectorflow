<img src="http://ae.nflximg.net/vectorflow/vectorflow_logo.png" width="200">

**Vectorflow** is a minimalist neural network library optimized for sparse data and single machine environments.

Original blog post [here](https://medium.com/@NetflixTechBlog/introducing-vectorflow-fe10d7f126b8).

[![Build Status](https://travis-ci.org/Netflix/vectorflow.svg?branch=master)](https://travis-ci.org/Netflix/vectorflow)

### Installation

#### dub package
The library is distributed as a [`dub`](https://code.dlang.org/) package. Add `vectorflow` to the `dependencies` section of your `dub.json`:
```
"vectorflow": "~>1.0.2"
```

The library itself doesn't have any dependencies. All you need is a recent D compiler.

**`LDC` is the recommended compiler** for the fastest runtime speed. 

Tested on:
- Linux, OSX
- LDC version: >= 1.1.1
- DMD version: >= 2.073.1

#### Setting up a D environment 
If you're new to [D](http://dlang.org/), keep reading. You will need `dub` (the D package manager) and `LDC` (the LLVM-based D compiler).
##### macOS
```
brew install dub
brew install ldc
```
##### Ubuntu
```
apt-get install -y curl xz-utils
curl -fsS https://dlang.org/install.sh | bash -s ldc
source ~/dlang/ldc-{VERSION}/activate
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
`vectorflow` uses [ddoc](https://dlang.org/spec/ddoc.html).
One way of building and serving the documentation locally (you will need `libevent` for serving) is:
```
dub build -b ddox && dub run -b ddox
```
Or use your favorite DDOC compiler.

Please also refer to the repo wiki.
