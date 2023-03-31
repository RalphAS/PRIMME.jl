using Test
using LinearAlgebra
using PRIMME

# do some basic tests first to fail quickly and succinctly on simple errors

@testset "basic svds $T" for T in [Float64, ComplexF64, Float32, ComplexF32]
    let n=200, m=200, k=10
        A = randn(T, m, n)
        tol = sqrt(sqrt(eps(real(T)))^3)
        U, vals, V, resids, stats = PRIMME.svds(A, k, verbosity = 1, tol=tol)
        nconv = size(U,2)
        svd_ref = svd(A)
        @test U isa AbstractMatrix{T}
        @test V isa AbstractMatrix{T}
        @test vals isa AbstractVector{real(T)}
        @test svd_ref.S[1:k] ≈ vals
        @test abs.(svd_ref.U[:, 1:k]' * U) ≈ Matrix{Float64}(I,nconv,nconv)
  end

end
@testset "basic eigs $T" for T in [Float64, ComplexF64, Float32, ComplexF32]
    let n=200, k=2
        q,_ = qr(randn(T, n, n))
        A = q * Diagonal(exp.(5*rand(real(T),n))) * q'
        A = T(0.5) * (A + A')
        tol = sqrt(sqrt(eps(real(T)))^3)
        vals, vecs, resids, stats = PRIMME.eigs(A, k, verbosity=1, tol=tol)
        E = eigen(A)
        idx = sortperm(E.values, by=abs, rev=true)
        nconv = length(vals)
        vals_ref = [E.values[i] for i in idx[1:nconv]]
        vecs_ref = E.vectors[:,idx[1:nconv]]
        vecs = vecs
        @test vecs isa AbstractMatrix{T}
        @test vals isa AbstractVector{real(T)}
        @test vals_ref ≈ vals
        @test abs.(vecs_ref' * vecs) ≈ Matrix{Float64}(I,nconv,nconv)
  end
end

@testset "unconverged returns" begin
    let n=200, k=10
        A = randn(n, n)
        A = 0.5 * (A + A')
        U, vals, V, resids, stats = (@test_logs (:warn,r"did not converge") PRIMME.svds(A, k, verbosity = 1, maxMatvecs=2, check=:warn))
        nconv = size(U,2)
        @test nconv < k
        @test_throws PRIMME.PRIMMEException PRIMME.svds(A, k, verbosity = 1, maxMatvecs=2, check=:throw)
        vals, vecs, _, _ = (@test_logs (:warn,r"did not converge") PRIMME.eigs(A, k, verbosity = 1, maxMatvecs=2, check=:warn))
        nconv = size(vecs,2)
        @test nconv < k
        @test_throws PRIMME.PRIMMEException PRIMME.eigs(A, k, verbosity = 1, maxMatvecs=2, check=:throw)
    end
end

@testset "eigs, 'which' specified" begin
    @testset "eigs n=$n which=$which" for n in [200],
                                          which in [:SR, :LR, :SM, :LM]
        T = Float64
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
        tol = sqrt(sqrt(eps(real(T)))^3)
        vals, vecs, resids, stats = PRIMME.eigs(A, k, verbosity=1, which=which, tol=tol)
        E = eigen(A)
        idx = sortperm(E.values, by= (which in (:SM,:LM) ? abs : identity),
                       rev=(which in (:LR,:LM)))
        nconv = length(vals)
        @test nconv == k
        vals_ref = [E.values[i] for i in idx[1:nconv]]
        vecs_ref = E.vectors[:,idx[1:nconv]]
        @test vals_ref ≈ vals
        @test abs.(vecs_ref' * vecs) ≈ Matrix{Float64}(I,nconv,nconv)
    end
end

@testset "svds, various methods" begin
    @testset "svds m=$m, n=$n, k=$k" for (m,n) in ((200,200),(200,400),(400,200)),
                                         k = [1,10],
                                         method = [PRIMME.svds_hybrid, PRIMME.svds_normalequations, PRIMME.svds_augmented]
        T = Float64
        A = randn(m, n)
        tol = sqrt(sqrt(eps(real(T)))^3)
        U, vals, V, resids, stats = PRIMME.svds(A, k, method = method, tol=tol)
        nconv = size(U,2)
        @test nconv == k
        svd_ref = svd(A)
        @test svd_ref.S[1:nconv] ≈ vals
        @test abs.(svd_ref.U[:, 1:nconv]' * U) ≈ Matrix{Float64}(I,nconv,nconv)
        @test abs.(svd_ref.V[:, 1:nconv]' * V) ≈ Matrix{Float64}(I,nconv,nconv)
    end
end

@testset "svds, :SR $T" for T in [Float64, ComplexF64]
    @testset "svds m=$m, n=$n" for (m,n) in ((200,200),(200,400),(400,200))
        k = 2
        mn = min(m,n)
        A = randn(T, m, n)
        tol = sqrt(sqrt(eps(real(T)))^3)
        U,S,V,resids,stats = PRIMME.svds(A, k, verbosity = 1, which=:SR, tol=tol)
        nconv = size(U,2)
        @test nconv == k
        svals_ref = svdvals(A)
        @test svals_ref[mn:-1:mn-nconv+1] ≈ S
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
        svals_ref = svdvals(A)
        itgt = mn >> 1
        # bias to avoid ambiguous ordering
        tgt = (3//5) * svals_ref[itgt] + (2//5) * svals_ref[itgt+1]
        idxp = sortperm(abs.(svals_ref .- tgt))
        tol = sqrt(sqrt(eps(real(T)))^3)
        U,S,V,resids,stats = PRIMME.svds(A, k, verbosity = 1, which=:SM, sigma=tgt, tol=tol)
        nconv = size(U,2)
        @test nconv == k
        @test svals_ref[idxp[1:nconv]] ≈ S
        @test norm(A * V - U * Diagonal(S)) < max(m,n) * 1e-6
    end
end


@testset "eigs w/ $(length(tgts)) shift(s) $T" for T in [Float64, ComplexF64], tgts in (25.2, [26.1, 40.2])
    n = 50
    d = zero(T) .+ collect(1:n)
    A = Diagonal(d)
    k=2
    tol = sqrt(sqrt(eps(real(T)))^3)
    vals, vecs, resids, stats = PRIMME.eigs(A, k, verbosity=1, which=:SM, shifts=tgts, tol=tol)
    E = eigen(A)
    idx = sortperm(E.values, by= x->minimum([abs(x-tgt) for tgt in tgts]))
    nconv = length(vals)
    @test nconv == k
    #idx = sort(idx[1:nconv], by=x->E.values[x], rev=true)
    vals_ref = [E.values[i] for i in idx[1:nconv]]
    vecs_ref = E.vectors[:,idx[1:nconv]]
    @test vals_ref ≈ vals
    @test abs.(vecs_ref' * vecs) ≈ Matrix{Float64}(I,nconv,nconv)

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
    tol = sqrt(sqrt(eps(real(T)))^3)

    # baseline
    vals, vecs, resids, stats = PRIMME.eigs(A, k, verbosity=1, which=:SM, shifts=tgt, tol=tol)
    @test stats.numPreconds == 0

    # preconditioner provided as matrix
    P = Diagonal(diag(A)) - tgt * I
    vals, vecs, resids, stats = PRIMME.eigs(A, k, verbosity=1, which=:SM, shifts=tgt, P=P, tol=tol)
    @test stats.numPreconds > 0
    nconv = length(vals)
    @test nconv == k
    vals_ref = [E.values[i] for i in idx[1:nconv]]
    vecs_ref = E.vectors[:,idx[1:nconv]]
    @test vals_ref ≈ vals
    @test abs.(vecs_ref' * vecs) ≈ Matrix{Float64}(I,nconv,nconv)

    # preconditioner provided as factorization
    P = Diagonal(diag(A)) - tgt * I
    pp = 0.001 * randn(n,n)
    pp = 0.5 * (pp + pp')
    P = bunchkaufman(P + pp)
    vals, vecs, resids, stats = PRIMME.eigs(A, k, verbosity=1, which=:SM, shifts=tgt, P=P, tol=tol)
    @test stats.numPreconds > 0
    nconv = length(vals)
    @test nconv == k
    vals_ref = [E.values[i] for i in idx[1:nconv]]
    vecs_ref = E.vectors[:,idx[1:nconv]]
    @test vals_ref ≈ vals
    @test abs.(vecs_ref' * vecs) ≈ Matrix{Float64}(I,nconv,nconv)
end

# this is intended as a minimal operator type
struct MyLinOp{T,TA}
    A::TA
    hflag::Bool
    MyLinOp(A::TA) where {TA <: AbstractMatrix} = new{eltype(A),TA}(A,ishermitian(A))
end
LinearAlgebra.mul!(y,lop::MyLinOp,x) = mul!(y,lop.A,x)
Base.size(lop::MyLinOp) = size(lop.A)
Base.size(lop::MyLinOp,i::Integer) = size(lop.A,i)
Base.eltype(lop::MyLinOp{T}) where {T} = T
LinearAlgebra.adjoint(lop::MyLinOp) = lop.hflag ? lop : MyLinOp(lop.A')
LinearAlgebra.issymmetric(lop::MyLinOp{T}) where {T} = (T <: Real) ? lop.hflag : false
LinearAlgebra.ishermitian(lop::MyLinOp) = lop.hflag

@testset "linop svds $T" for T in [Float64, ComplexF64]
    let n=100, m=200, k=10
        Am = randn(T, m, n)
        A = MyLinOp(Am)
        tol = sqrt(sqrt(eps(real(T)))^3)
        U, vals, V, resids, stats = PRIMME.svds(A, k, verbosity = 1, tol=tol)
        nconv = size(U,2)
        svd_ref = svd(Am)
        @test U isa AbstractMatrix{T}
        @test V isa AbstractMatrix{T}
        @test vals isa AbstractVector{real(T)}
        @test svd_ref.S[1:k] ≈ vals
        @test abs.(svd_ref.U[:, 1:k]' * U) ≈ Matrix{Float64}(I,nconv,nconv)
  end

end
@testset "linop eigs $T" for T in [Float64, ComplexF64, Float32, ComplexF32]
    let n=200, k=2
        q,_ = qr(randn(T, n, n))
        A = q * Diagonal(exp.(5*rand(real(T),n))) * q'
        A = T(0.5) * (A + A')
        Aop = MyLinOp(A)
        tol = sqrt(sqrt(eps(real(T)))^3)
        vals, vecs, resids, stats = PRIMME.eigs(Aop, k, verbosity=1, tol=tol)
        E = eigen(A)
        idx = sortperm(E.values, by=abs, rev=true)
        nconv = length(vals)
        vals_ref = [E.values[i] for i in idx[1:nconv]]
        vecs_ref = E.vectors[:,idx[1:nconv]]
        vecs = vecs
        @test vecs isa AbstractMatrix{T}
        @test vals isa AbstractVector{real(T)}
        @test vals_ref ≈ vals
        @test abs.(vecs_ref' * vecs) ≈ Matrix{Float64}(I,nconv,nconv)
  end
end
