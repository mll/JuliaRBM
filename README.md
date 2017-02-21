# JuliaRBM
Restricted Boltzmann Machines neural network pre-training implemented in Julia.

The setup allows for testing the implementation on MNIST dataset. 

## Installation

1. Compile lodepng as a dynamic library. In OS X this would be:

gcc -dynamiclib -o lodepng.dylib lodepng.c

2. Download and install [Julia][http://julialang.org]

3. Prepare the learn-1.dat learning file. The file has the following format:
```
2 3 2

0.3 0.2 0.1
1 0

0.1 0.3 0.2
0 1
```

First numbers are number of cases (2), number of floats in each feature vector (3) and number of class labels (2).
Next, we see all the cases - first 3 floats, than 2 class label numbers.

The file we tested the code with was 50000 MNIST examples (the MNIST training set)

To download the MNIST test data, just run

./learningset.py

3. julia train_mnist.jl
