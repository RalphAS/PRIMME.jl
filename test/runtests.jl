using Test
using LinearAlgebra
using PRIMME
const Primme = PRIMME

@testset "eig trial" begin
    let n=200, k=10
        @info "testing with n,k=$n $k"
        A = randn(n, n)
        eigPrimme = Primme.eigs(A, k, debuglevel=2)
        E = eigen(A)
        idx = sortperm(E.values, by=abs, rev=true)
        valsLAPACK = [E.values[i] for i in idx[1:k]]
        vecsLAPACK = E.vectors[:,idx[1:k]]
        @test valsLAPACK ≈ eigPrimme[2]
        @test abs.(vecsLAPACK'eigPrimme[1]) ≈ eye(k)
  end

end
@testset "svd trial" begin
    let n=200, m=200, k=10, method=Primme.svds_op_AAt
        @info "testing with n,m,k=$n $m $k"
        A = randn(m, n)
        svdPrimme = Primme.svds(A, k, method = method, debuglevel = 2)
        svdLAPACK = svd(A)
        @test svdLAPACK[2][1:k] ≈ svdPrimme[2]
        @test abs.(svdLAPACK[1][:, 1:k]'svdPrimme[1]) ≈ eye(k)
  end

end

@testset "svds" begin
    @testset "svds m=$m, n=$n, k=$k" for n in [200, 400],
                                         m in [200, 400],
                                         k = [10,20],
                                         method = [Primme.svds_op_AAt, Primme.svds_op_AtA, Primme.svds_op_augmented]
        A = randn(m, n)
        svdPrimme = Primme.svds(A, k, method = method)
        svdLAPACK = svd(A)
        @test svdLAPACK[2][1:k] ≈ svdPrimme[2]
        @test abs.(svdLAPACK[1][:, 1:k]'svdPrimme[1]) ≈ eye(k)
    end
end
