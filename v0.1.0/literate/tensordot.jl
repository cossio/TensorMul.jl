#=
In this example we describe the usage of `TensorMul.tensordot`.
First let's load the package.
=#

import TensorMul
nothing #hide

#=
Let's define the dimensions of the problem.
=#

N = (3,7)
M = (2,3)
B = 16
nothing #hide

#=
We will assume our data consists of `B` batch examples, where each example
consists of matrices `X` of size `N`, and matrices `Y` of size `M`.
=#

X = randn(N..., B)
Y = randn(M..., B)
nothing #hide

#=
These datasets are combined via some weight tensor of corresponding dimensions:
=#

W = randn(N..., M...)
nothing #hide

#=
Now we want to form a contraction of `X` and `Y` with `W`.
More precisely, we want to compute the quantity:
=#

C = [
    sum(
        X[i,b] * W[i,μ] * Y[μ,b]
        for i in CartesianIndices(N),
            μ in CartesianIndices(M)
    ) for b in 1:B
]
nothing #hide

#=
Here `C` is a vector with `B` elements.
=#

#=
The downside with the above approach is that we have to know which dimensions to reduce,
what dimensions correspond to batches, and dimensions of `X` correspond to those of `W`,
etc.
That's what `TensorMul.tensordot` computes.
In the following line, it will compute the same quantity `C`, but automatically figuring
out which dimensions to reduce.
=#

TensorMul.tensordot(X, W, Y) ≈ C
