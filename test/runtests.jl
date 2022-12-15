using Test
using LinearAlgebra
using PRIMME
const Primme = PRIMME

@testset "eig trial" begin
    let n=200, k=2
        @info "testing with n,k=$n $k"
        q,_ = qr(randn(n,n))
        A = q * Diagonal(exp.(5*rand(n))) * q'
        A = 0.5 * (A + A')
        eigPrimme = Primme.eigs(A, k, verbosity=5)
        err = eigPrimme[3]
        @test err == 0
        if err == 0
            E = eigen(A)
            idx = sortperm(E.values, by=abs, rev=true)
            nconv = length(eigPrimme[1])
            valsLAPACK = [E.values[i] for i in idx[1:nconv]]
            vecsLAPACK = E.vectors[:,idx[1:nconv]]
            @test valsLAPACK ≈ eigPrimme[1]
            @test abs.(vecsLAPACK'eigPrimme[2]) ≈ Matrix{Float64}(I,nconv,nconv)
        end
  end

end
@testset "svd trial" begin
    let n=200, m=200, k=10, method=Primme.svds_op_AAt
        @info "testing with n,m,k=$n $m $k"
        A = randn(m, n)
        svdPrimme = Primme.svds(A, k, method = method, verbosity = 2)
        svdLAPACK = svd(A)
        @test svdLAPACK[2][1:k] ≈ svdPrimme[2]
        @test abs.(svdLAPACK[1][:, 1:k]'svdPrimme[1]) ≈ Matrix{Float64}(I,nconv,nconv)
  end

end

@testset "svds" begin
    @testset "svds m=$m, n=$n, k=$k" for n in [200, 400],
                                         m in [200, 400],
                                         k = [10,20],
                                         method = [Primme.svds_op_AAt, Primme.svds_op_AtA, Primme.svds_op_augmented]
        A = randn(m, n)
        svdPrimme = Primme.svds(A, k, method = method)
        nconv = size(svdPrimme[1],2)
        svdLAPACK = svd(A)
        @test svdLAPACK.S[1:nconv] ≈ svdPrimme[2]
        @test abs.(svdLAPACK.U[:, 1:nconv]'svdPrimme[1]) ≈ Matrix{Float64}(I,nconv,nconv)
    end
end
