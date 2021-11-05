module TensorMul
    using LinearAlgebra

    export tensormul_ff, tensormul_ll, tensormul_lf, tensormul_fl
    export tensordot

    include("tensor.jl")
end # module
