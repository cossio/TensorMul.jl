using Test, Random, LinearAlgebra
import TensorMul

@testset "tensordot" begin
    X = randn(3)
    Y = randn(7)
    W = randn(3, 7)
    @test size(TensorMul.tensordot(X, W, Y)) == ()
    @test only(TensorMul.tensordot(X, W, Y)) ≈ dot(X, W, Y)
    @inferred TensorMul.tensordot(X, W, Y)

    X = randn(3,4,5,6)
    Y = randn(7,2,5,6)
    W = randn(3,4,7,2)
    C = zeros(5,6)
    for b2=1:6, b1=1:5, j2=1:2, j1=1:7, i2=1:4, i1=1:3
        C[b1,b2] += X[i1,i2,b1,b2]*W[i1,i2,j1,j2]*Y[j1,j2,b1,b2]
    end
    @test size(TensorMul.tensordot(X, W, Y)) == (5, 6)
    @test TensorMul.tensordot(X, W, Y) ≈ C
    @inferred TensorMul.tensordot(X, W, Y)

    X = randn(7,2,5,6)
    Y = randn(3,4,5,6)
    W = randn(7,2,3,4)
    C = zeros(5,6)
    for b2=1:6, b1=1:5, j2=1:4, j1=1:3, i2=1:2, i1=1:7
        C[b1,b2] += X[i1,i2,b1,b2]*W[i1,i2,j1,j2]*Y[j1,j2,b1,b2]
    end
    @test size(TensorMul.tensordot(X, W, Y)) == (5, 6)
    @test TensorMul.tensordot(X, W, Y) ≈ C
    @inferred TensorMul.tensordot(X, W, Y)

    A = randn(10, 10)
    v = randn(10, 100)
    @test TensorMul.tensordot(v, A, v) ≈ diag(v' * A * v)
end

@testset "tensormul_ff" begin
    A = randn(5,6,3,4)
    B = randn(5,6,10,7)
    @test size(TensorMul.tensormul_ff(A, B, Val(2))) == (3,4,10,7)
    C = [sum(A[k,l,i1,i2]B[k,l,j3,j4] for k=1:5, l=1:6)
    for i1=1:3, i2=1:4, j3=1:10, j4=1:7]
    @test TensorMul.tensormul_ff(A, B, Val(2)) ≈ C
    @inferred TensorMul.tensormul_ff(A, B, Val(2))
end

@testset "tensormul_ll" begin
    A = randn(3,4,5,6)
    B = randn(7,5,6)
    @test size(TensorMul.tensormul_ll(A, B, Val(2))) == (3,4,7)
    C = [sum(A[i1,i2,k,l]B[j,k,l] for k=1:5, l=1:6)
    for i1=1:3, i2=1:4, j=1:7]
    @test TensorMul.tensormul_ll(A, B, Val(2)) ≈ C
    @inferred TensorMul.tensormul_ll(A, B, Val(2))
end

@testset "tensormul_lf" begin
    A = randn(3,4,5,6)
    B = randn(5,6,10,7)
    @test size(TensorMul.tensormul_lf(A, B, Val(2))) == (3,4,10,7)
    C = [sum(A[i1,i2,k,l]B[k,l,j3,j4] for k=1:5, l=1:6)
    for i1=1:3, i2=1:4, j3=1:10, j4=1:7]
    @test C ≈ TensorMul.tensormul_lf(A, B, Val(2))
    @inferred TensorMul.tensormul_lf(A, B, Val(2))
end

@testset "tensormul_fl" begin
    A = randn(5,6,3,4)
    B = randn(10,7,5,6)
    @test size(TensorMul.tensormul_fl(A, B, Val(2))) == (3,4,10,7)
    C = [sum(A[k,l,i1,i2]B[j3,j4,k,l] for k=1:5, l=1:6)
    for i1=1:3, i2=1:4, j3=1:10, j4=1:7]
    @test TensorMul.tensormul_fl(A, B, Val(2)) ≈ C
    @inferred TensorMul.tensormul_fl(A, B, Val(2))
end

@testset "tensormul fl & lf" begin
    A = randn(5,6,3,4)
    B = randn(10,7,5,6)
    Cab = @inferred TensorMul.tensormul_fl(A, B, Val(2))
    Cba = @inferred TensorMul.tensormul_lf(B, A, Val(2))
    @test Cab ≈ permutedims(Cba, (3,4,1,2))
end

@testset "broadlike" begin
    A = randn(1,3)
    B = randn(2,1)
    @test TensorMul.broadlike(A, B) ≈ A .+ B .- B
    @inferred TensorMul.broadlike(A, B)
end
