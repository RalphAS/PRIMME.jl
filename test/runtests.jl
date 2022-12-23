using Test
using LinearAlgebra
using PRIMME

# do some basic tests first to fail quickly and succinctly on simple errors

@testset "basic svds $T" for T in [Float64, ComplexF64]
    let n=200, m=200, k=10
        A = randn(T, m, n)
        svdPrimme = PRIMME.svds(A, k, verbosity = 1)
        nconv = size(svdPrimme[1],2)
        svdLAPACK = svd(A)
        @test svdLAPACK.S[1:k] ≈ svdPrimme[2]
        @test abs.(svdLAPACK.U[:, 1:k]'svdPrimme[1]) ≈ Matrix{Float64}(I,nconv,nconv)
  end

end
@testset "basic eigs $T" for T in [Float64, ComplexF64]
    let n=200, k=2
        q,_ = qr(randn(T, n, n))
        A = q * Diagonal(exp.(5*rand(n))) * q'
        A = 0.5 * (A + A')
        eigPrimme = PRIMME.eigs(A, k, verbosity=1)
        E = eigen(A)
        idx = sortperm(E.values, by=abs, rev=true)
        nconv = length(eigPrimme[1])
        valsLAPACK = [E.values[i] for i in idx[1:nconv]]
        vecsLAPACK = E.vectors[:,idx[1:nconv]]
        vecsPrimme = eigPrimme[2]
        @test vecsPrimme isa AbstractMatrix{T}
        @test valsLAPACK ≈ eigPrimme[1]
        @test abs.(vecsLAPACK' * vecsPrimme) ≈ Matrix{Float64}(I,nconv,nconv)
  end
end

@testset "eigs, 'which' specified" begin
    @testset "eigs n=$n which=$which" for n in [200],
                                          which in [:SA, :LA, :SM, :LM]
        k = 5
        q,_ = qr(randn(n,n))
        if which != :SM
            d = rand((-1,1),n) .* exp.(5*rand(n))
        else
            # interior ones from the above are usually too hard w/o cleverness
            # TODO: add cleverness (probably below)
            d = rand((-1,1),n) .* (1:n)
        end
        A = q * Diagonal(d) * q'
        A = 0.5 * (A + A')
        eigPrimme = PRIMME.eigs(A, k, verbosity=1, which=which)
        E = eigen(A)
        idx = sortperm(E.values, by= (which in (:SM,:LM) ? abs : identity),
                       rev=(which in (:LA,:LM)))
        nconv = length(eigPrimme[1])
        @test nconv == k
        valsLAPACK = [E.values[i] for i in idx[1:nconv]]
        vecsLAPACK = E.vectors[:,idx[1:nconv]]
        @test valsLAPACK ≈ eigPrimme[1]
        @test abs.(vecsLAPACK'eigPrimme[2]) ≈ Matrix{Float64}(I,nconv,nconv)
    end
end

@testset "svds, various methods" begin
    @testset "svds m=$m, n=$n, k=$k" for (m,n) in ((200,200),(200,400),(400,200)),
                                         k = [1,10],
                                         method = [PRIMME.svds_hybrid, PRIMME.svds_normalequations, PRIMME.svds_augmented]
        A = randn(m, n)
        svdPrimme = PRIMME.svds(A, k, method = method)
        nconv = size(svdPrimme[1],2)
        @test nconv == k
        svdLAPACK = svd(A)
        @test svdLAPACK.S[1:nconv] ≈ svdPrimme[2]
        @test abs.(svdLAPACK.U[:, 1:nconv]'svdPrimme[1]) ≈ Matrix{Float64}(I,nconv,nconv)
        @test abs.(svdLAPACK.V[:, 1:nconv]'svdPrimme[3]) ≈ Matrix{Float64}(I,nconv,nconv)
    end
end

@testset "svds, :SR $T" for T in [Float64, ComplexF64]
    @testset "svds m=$m, n=$n" for (m,n) in ((200,200),(200,400),(400,200))
        k = 2
        mn = min(m,n)
        A = randn(T, m, n)
        U,S,V,resids,stats = PRIMME.svds(A, k, verbosity = 1, which=:SR)
        nconv = size(U,2)
        @test nconv == k
        svalsLAPACK = svdvals(A)
        @test svalsLAPACK[mn:-1:mn-nconv+1] ≈ S
        @test norm(A * V - U * Diagonal(S)) < max(m,n) * 1e-6
    end
end

@testset "svds, :SM $T" for T in [Float64, ComplexF64]
    @testset "svds m=$m, n=$n"  for (m,n) in ((200,200),(200,400),(400,200))
        k = 2
        mn = min(m,n)
        # interior singular values of randn(m,n) are just TOO random
        u,_ = qr!(randn(T,m,m))
        v,_ = qr!(randn(T,n,n))
        d = collect(1:mn)
        A = u * diagm(m,n,d) * v'
        svalsLAPACK = svdvals(A)
        itgt = mn >> 1
        # bias to avoid ambiguous ordering
        tgt = (3//5) * svalsLAPACK[itgt] + (2//5) * svalsLAPACK[itgt+1]
        idxp = sortperm(abs.(svalsLAPACK .- tgt))
        U,S,V,resids,stats = PRIMME.svds(A, k, verbosity = 1, which=:SM, sigma=tgt)
        nconv = size(U,2)
        @test nconv == k
        @test svalsLAPACK[idxp[1:nconv]] ≈ S
        @test norm(A * V - U * Diagonal(S)) < max(m,n) * 1e-6
    end
end


@testset "eigs w/ $(length(tgts)) shift(s) $T" for T in [Float64, ComplexF64], tgts in (25.2, [26.1, 40.2])
    n = 50
    d = zero(T) .+ collect(1:n)
    A = Diagonal(d)
    k=2
    eigPrimme = PRIMME.eigs(A, k, verbosity=1, which=:SM, shifts=tgts)
    E = eigen(A)
    idx = sortperm(E.values, by= x->minimum([abs(x-tgt) for tgt in tgts]))
    nconv = length(eigPrimme[1])
    @test nconv == k
    #idx = sort(idx[1:nconv], by=x->E.values[x], rev=true)
    valsLAPACK = [E.values[i] for i in idx[1:nconv]]
    vecsLAPACK = E.vectors[:,idx[1:nconv]]
    @test valsLAPACK ≈ eigPrimme[1]
    @test abs.(vecsLAPACK'eigPrimme[2]) ≈ Matrix{Float64}(I,nconv,nconv)

end

@testset "eigs w/ preconditioning n=$n $T" for T in [Float64, ComplexF64], n in [25, 500]
    tgt = 30.4
    # n = 25# 500
    d = zero(T) .+ collect(1:n)
    q,_ = qr!(randn(T,n,n))
    A = q * Diagonal(d) * q'
    A = 0.5 * (A + A')
    k=2
    E = eigen(A)
    idx = sortperm(E.values, by= x->abs(x-tgt))

    # baseline
    eigPrimme = PRIMME.eigs(A, k, verbosity=1, which=:SM, shifts=tgt)
    stats = eigPrimme[4]
    @test stats.numPreconds == 0

    # preconditioner provided as matrix
    P = Diagonal(diag(A)) - tgt * I
    eigPrimme = PRIMME.eigs(A, k, verbosity=1, which=:SM, shifts=tgt, P=P)
    stats = eigPrimme[4]
    @test stats.numPreconds > 0
    nconv = length(eigPrimme[1])
    @test nconv == k
    valsLAPACK = [E.values[i] for i in idx[1:nconv]]
    vecsLAPACK = E.vectors[:,idx[1:nconv]]
    @test valsLAPACK ≈ eigPrimme[1]
    @test abs.(vecsLAPACK'eigPrimme[2]) ≈ Matrix{Float64}(I,nconv,nconv)

    # preconditioner provided as factorization
    P = Diagonal(diag(A)) - tgt * I
    pp = 0.001 * randn(n,n)
    pp = 0.5 * (pp + pp')
    P = bunchkaufman(P + pp)
    eigPrimme = PRIMME.eigs(A, k, verbosity=1, which=:SM, shifts=tgt, P=P)
    stats = eigPrimme[4]
    @test stats.numPreconds > 0
    nconv = length(eigPrimme[1])
    @test nconv == k
    valsLAPACK = [E.values[i] for i in idx[1:nconv]]
    vecsLAPACK = E.vectors[:,idx[1:nconv]]
    @test valsLAPACK ≈ eigPrimme[1]
    @test abs.(vecsLAPACK'eigPrimme[2]) ≈ Matrix{Float64}(I,nconv,nconv)
end
