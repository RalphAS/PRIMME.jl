module PRIMME

using LinearAlgebra

# const libprimme = joinpath(dirname(@__FILE__()), "../deps/primme/lib/libprimme")

const libprimme = "/scratch/build/primme-3.2/lib/libprimme.so"

include("types.jl")


# Julia API

free(r::Ref{C_svds_params}) = ccall((:primme_svds_free, libprimme), Cvoid, (Ptr{C_svds_params},), r)

free(r::Ref{C_params}) = ccall((:primme_free, libprimme), Cvoid, (Ptr{C_params},), r)

function eigs_initialize()
    r = Ref{C_params}()
    ccall((:primme_initialize, libprimme), Cvoid, (Ptr{C_params},), r)
    return r
end

function svds_initialize()
    r = Ref{C_svds_params}()
    ccall((:primme_svds_initialize, libprimme), Cvoid, (Ptr{C_svds_params},), r)
    return r
end

# matrix-vector product, y = a * x (or y = a^t * x), where
# (void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize,
       # int *transpose, struct primme_svds_params *primme_svds, int *ierr);
const _A_ = Base.Ref{Any}()
function matrixMatvec(xp, ldxp, yp, ldyp, blockSizep, trp, parp, ierrp)
    ldx, ldy           = unsafe_load(ldxp), unsafe_load(ldyp)
    blockSize, tr, par = Int(unsafe_load(blockSizep)), unsafe_load(trp), unsafe_load(parp)
    x, y = unsafe_wrap(Array, xp, (ldx, blockSize)), unsafe_wrap(Array, yp, (ldy, blockSize))

    if tr == 0
        mul!( view(y, 1:par.mLocal, :), _A_[], view(x, 1:par.nLocal, :))
    else
        mul!(view(y, 1:par.nLocal, :), _A_[]', view(x, 1:par.mLocal, :))
    end
    unsafe_store!(ierrp, 0)
    return nothing
end

_print(r::Ref{C_params}) = ccall((:primme_display_params, libprimme), Cvoid, (C_params,), r[])

_print(r::Ref{C_svds_params}) = ccall((:primme_svds_display_params, libprimme), Cvoid, (C_svds_params,), r[])

function _svds(r::Ref{C_svds_params})
    m, n, k = r[].m, r[].n, r[].numSvals
    svals  = Vector{Float64}(undef, k)
    svecs  = rand(Float64, m + n, k)
    rnorms = Vector{Float64}(undef, k)

    err = ccall((:dprimme_svds, libprimme), Cint,
        (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{C_svds_params}),
         svals, svecs, rnorms, r)
    if err != 0
        @warn "svds returned err = $err"
    end

    nConv = Int(r[].initSize)
    if nConv < k
        svals = svals[1:nConv]
    end
    nConst = r[].numOrthoConst
    return (reshape(svecs[nConst*m .+ (1:(m*nConv))], m, nConv),
            svals,
            reshape(svecs[((nConst + nConv)*m + nConst*n) .+ (1:(n*nConv))], n, nConv),
            err)
end

function svds(A::AbstractMatrix, k = 5;
              tol = 1e-12,
              maxBlockSize = 2k,
              verbosity::Int = 0,
              method::Svds_operator = svds_op_AtA,
              maxMatvecs = 10000,
              which = :LR)
    mul_fp = @cfunction(matrixMatvec, Cvoid,
        (Ptr{Float64}, Ptr{Int}, Ptr{Float64}, Ptr{Int}, Ptr{Cint}, Ptr{Cint},
         Ptr{C_svds_params}, Ptr{Cint}))
    r = svds_initialize()

    _A_[]            = A
    r[:m]            = size(A, 1)
    r[:n]            = size(A, 2)
    r[:matrixMatvec] = mul_fp
    r[:numSvals]     = k
    if which in (:LR, :LM)
        r[:target] = svds_largest
    elseif which == (:SR, :SM)
        r[:target] = svds_smallest
    else
        throw(ArgumentError("target value logic not yet implemented"))
        r[:target] = svds_closest_abs
    end
    r[:printLevel]   = verbosity
    r[:eps]          = tol
    r[:maxBlockSize] = maxBlockSize
    r[:method]       = method

    r[:maxMatvecs] = maxMatvecs

    out = _svds(r)
    free(r)
    return out
end

function matrixMatvecE(xp::Ptr{Float64}, ldxp::Ptr{PRIMME_INT}, yp::Ptr{Float64}, ldyp::Ptr{PRIMME_INT}, blockSizep::Ptr{Cint}, parp::Ptr{C_params}, ierrp::Ptr{Cint})
    ldx, ldy           = unsafe_load(ldxp), unsafe_load(ldyp)
    blockSize, par = Int(unsafe_load(blockSizep)), unsafe_load(parp)
    x, y = unsafe_wrap(Array, xp, (ldx, blockSize)), unsafe_wrap(Array, yp, (ldy, blockSize))

    mul!( view(y, 1:par.nLocal, :), _A_[], view(x, 1:par.nLocal, :))
    unsafe_store!(ierrp, 0)
    return nothing
end

function _eigs(r::Ref{C_params})
    n, k = r[].n, r[].numEvals
    vals  = Vector{Float64}(undef, k)
    vecs  = rand(Float64, n, k)
    rnorms = Vector{Float64}(undef, k)

    err = ccall((:dprimme, libprimme), Cint,
        (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{C_params}),
         vals, vecs, rnorms, r)
    if err != 0
        @warn "dprimme returned err = $err"
    end

    nConv = Int(r[].initSize)
    if nConv < k
        vals = vals[1:nConv]
    end

    return vals, reshape(vecs[1:n*nConv],n, nConv), Int(err)
end

function eigs(A::AbstractMatrix, k = 5; tol = 1e-12, maxBlockSize = 2k, verbosity::Int = 0,
              maxOuterIterations = 100, maxMatvecs = 1000, which = :LR)
    mul_fp = @cfunction(matrixMatvecE, Cvoid,
        (Ptr{Float64}, Ptr{Int}, Ptr{Float64}, Ptr{Int}, Ptr{Cint},
         Ptr{C_params}, Ptr{Cint}))

    r = eigs_initialize()
    _A_[]            = A
    r[:n]            = size(A, 2)
    r[:matrixMatvec] = mul_fp
    r[:numEvals]     = k
    r[:printLevel]   = verbosity
    r[:eps]          = tol
    r[:maxBlockSize] = maxBlockSize

    r[:maxOuterIterations] = maxOuterIterations
    r[:maxMatvecs] = maxMatvecs
    if which == :LR
        r[:target] = largest
    elseif which == :SR
        r[:target] = smallest
    else
        throw(ArgumentError("which: only :LR and :SR are currently implemented"))
    end
    out = _eigs(r)
    free(r)
    return out
end

end # module
