# JuliaRBM
Restricted Boltzmann Machines neural network pre-training implemented in Julia.

The setup allows for testing the implementation on MNIST dataset. 

## Installation

1. Compile lodepng as a dynamic library. In OS X this would be:
```
gcc -dynamiclib -o lodepng.dylib lodepng.c
```
2. Download and install [Julia][http://julialang.org]. You may need to add it to your PATH
3. Download the MNIST training data:
```
./learningset.py
```
4. Run julia:
```
julia train_mnist.jl
```
