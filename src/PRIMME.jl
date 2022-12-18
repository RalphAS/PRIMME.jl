module PRIMME

using LinearAlgebra

using PRIMME_jll

const libprimme = PRIMME_jll.libprimme
# const libprimme = "/scratch/build/primme-3.2/lib/libprimme.so"

include("types.jl")

macro _fnq(x)
    return Expr(:quote, x)
end

function free(r::Ref{C_svds_params})
    ccall((:primme_svds_free, libprimme), Cvoid, (Ptr{C_svds_params},), r)
end

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

function _print(r::Ref{C_params})
    ccall((:primme_display_params, libprimme), Cvoid, (C_params,), r[])
end

function _print(r::Ref{C_svds_params})
    ccall((:primme_svds_display_params, libprimme), Cvoid, (C_svds_params,), r[])
end

for (func, elty, relty) in
    ((:dprimme_svds, :Float64, :Float64),
     (:zprimme_svds, :ComplexF64, :Float64),
     )
    @eval begin
function _svds(r::Ref{C_svds_params}, ::Type{$elty})
    m, n, k = r[].m, r[].n, r[].numSvals
    svals  = Vector{$relty}(undef, k)
    svecs  = rand($elty, m + n, k)
    rnorms = Vector{$relty}(undef, k)

    err = ccall((@_fnq($func), libprimme), Cint,
        (Ptr{$relty}, Ptr{$elty}, Ptr{$relty}, Ptr{C_svds_params}),
         svals, svecs, rnorms, r)
    if err != 0
        free(r)
        throw(PRIMMEException(Int(err), :dprimme_svds))
    end

    nConv = Int(r[].initSize)
    if nConv < k
        svals = svals[1:nConv]
    end
    nConst = r[].numOrthoConst
    return (reshape(svecs[nConst*m .+ (1:(m*nConv))], m, nConv),
            svals,
            reshape(svecs[((nConst + nConv)*m + nConst*n) .+ (1:(n*nConv))], n, nConv),
            rnorms,
            r[].stats)
end
    end # eval block
end # type loop

function _wrap_matvec_svds(A::AbstractMatrix{T}) where {T}
    # matrix-vector product, y = a * x (or y = a^t * x), where
    # (ELT *x, PRIMME_INT *ldx, ELT *y, PRIMME_INT *ldy, int *blockSize,
    #  int *transpose, struct primme_svds_params *primme_svds, int *ierr);
    function mv(xp, ldxp, yp, ldyp, blockSizep, trp, parp, ierrp)
        ldx, ldy = unsafe_load(ldxp), unsafe_load(ldyp)
        blockSize = Int(unsafe_load(blockSizep))
        tr, par = unsafe_load(trp), unsafe_load(parp)
        x = unsafe_wrap(Array, xp, (ldx, blockSize))
        y = unsafe_wrap(Array, yp, (ldy, blockSize))

        if tr == 0
            mul!( view(y, 1:par.mLocal, :), A, view(x, 1:par.nLocal, :))
        else
            mul!(view(y, 1:par.nLocal, :), A', view(x, 1:par.mLocal, :))
        end
        unsafe_store!(ierrp, 0)
        return nothing
    end
    mul_fp = @cfunction($mv, Cvoid,
        (Ptr{T}, Ptr{Int}, Ptr{T}, Ptr{Int}, Ptr{Cint}, Ptr{Cint},
         Ptr{C_svds_params}, Ptr{Cint}))
end

function _wrap_matldiv_svds(P::Union{AbstractMatrix{T},Factorization{T}}) where {T}
    if P isa AbstractMatrix
        PF = factorize(P)
    else
        PF = P
    end
    function mvP(xp::Ptr{Tmv}, ldxp::Ptr{PRIMME_INT},
                 yp::Ptr{Tmv}, ldyp::Ptr{PRIMME_INT},
                 blockSizep::Ptr{Cint}, modep::Ptr{Cint}, parp::Ptr{C_params},
                 ierrp::Ptr{Cint}) where {Tmv}
        ldx, ldy = unsafe_load(ldxp), unsafe_load(ldyp)
        blockSize, par = Int(unsafe_load(blockSizep)), unsafe_load(parp)
        mode = unsafe_load(modep)
        x = unsafe_wrap(Array, xp, (ldx, blockSize))
        y = unsafe_wrap(Array, yp, (ldy, blockSize))
        copyto!(view(y, 1:par.nLocal, :), view(x, 1:par.nLocal, :))
        if mode == Cint(svds_op_AtA)
            ldiv!(PF, view(y, 1:par.nLocal, :))
            unsafe_store!(ierrp, 0)
        elseif mode == Cint(svds_op_AtA)
            ldiv!(PF, view(y, 1:par.nLocal, :))
            unsafe_store!(ierrp, 0)
        elseif mode == Cint(svds_op_AtA)
            ldiv!(PF, view(y, 1:par.nLocal, :))
            unsafe_store!(ierrp, 0)
        else
            unsafe_store!(ierrp, -1)
        end
        return nothing
    end
    ldiv_fp = @cfunction($mvP, Cvoid,
                           (Ptr{T}, Ptr{Int}, Ptr{T}, Ptr{Int}, Ptr{Cint},
                            Ptr{C_params}, Ptr{Cint}))
    return ldiv_fp
end

"""
    PRIMME.svds(A, k::Integer; kwargs...) => U,S,V,resids,stats

computes some (`k`) singular triplets (partial SVD) of the matrix `A`

Returns `S`, a vector of `k` singular values; `U`, a `k×n` matrix of left singular vectors;
`V`, a `k×n` matrix of right singular vectors; `resids`, a vector of residual norms;
and `stats`, a structure with an account of the work done.

# Keyword args
 (names in square brackes are analogues in PRIMME documentation)

- `which::Symbol`, indicating which triplets should be computed.
 - `:SA`: smallest algebraic
 - `:SM`: closest to `sigma` (or 0 if not provided)
 - `:LM`: largest magnitude (default)
"""
function svds(A::AbstractMatrix{T}, k = 5;
              tol = 1e-12,
              maxBasisSize = nothing,
              verbosity::Int = 0,
              method::Union{Nothing,SvdsPresetMethod} = nothing,
              method1::Union{Nothing,EigsPresetMethod} = nothing,
              method2::Union{Nothing,EigsPresetMethod} = nothing,
              maxMatvecs = 10000,
              which = :LR,
              shifts = nothing,
              sigma::Union{Nothing,Real} = nothing,
              P = nothing,
              ) where {T}
    RT = real(T)
    r = svds_initialize()
    mul_fp = _wrap_matvec_svds(A)
    r[:m]            = size(A, 1)
    r[:n]            = size(A, 2)
    r[:numSvals]     = k
    if which in (:LR, :LM)
        r[:target] = svds_largest
    elseif which == :SR
        r[:target] = svds_smallest
    elseif which == :SM
        r[:target] = svds_closest_abs
        if sigma !== nothing
            shifts = [RT(sigma)]
        end
        if shifts === nothing
            shifts = [zero(RT)]
        end
    else
        throw(ArgumentError("which must be one of (:LR, :LM, :SR, :SM)"))
    end

    if P !== nothing
        throw(ArgumentError("preconditioning is not yet implemented"))
    else
        precon_fp = Ptr{Cvoid}()
    end

    if shifts !== nothing
        nshifts = length(shifts)
        shiftsx = Vector{RT}(undef,nshifts)
        shiftsx .= shifts
        r[:numTargetShifts] = nshifts
    else
        shiftsx = Vector{RT}(undef,0)
    end

    r[:printLevel]   = verbosity
    r[:eps]          = tol
    r[:maxMatvecs] = maxMatvecs
    if (method !== nothing) || (method1 !== nothing) || (method2 != nothing)
        if method == nothing; method = svds_default; end
        if method1 == nothing; method1 = default_method; end
        if method2 == nothing; method2 = default_method; end
        err = ccall((:primme_svds_set_method, libprimme), Cint,
                    (SvdsPresetMethod, EigsPresetMethod, EigsPresetMethod,
                     Ptr{C_svds_params}),
                    method, method1, method2, r)
        if err != 0
            throw(ArgumentError("illegal value of preset method"))
        end
    end
    if maxBasisSize !== nothing
        r[:maxBasisSize] = maxBasisSize
    end

    @GC.preserve mul_fp precon_fp shiftsx begin
        r[:matrixMatvec] = Base.unsafe_convert(Ptr{Cvoid}, mul_fp)
        if shifts !== nothing
            r[:targetShifts] = pointer(shiftsx)
        end
        out = _svds(r, T)
    end
    free(r)
    return out
end

for (func, elty, relty) in
    ((:dprimme, :Float64, :Float64),
     (:zprimme, :ComplexF64, :Float64),
     )
    @eval begin
function _eigs(r::Ref{C_params},::Type{$elty})
    n, k = r[].n, r[].numEvals
    vals  = Vector{$relty}(undef, k)
    vecs  = rand($elty, n, k)
    rnorms = Vector{$relty}(undef, k)

    local err
    try
        err = ccall((@_fnq($func), libprimme), Cint,
                    (Ptr{$elty}, Ptr{$elty}, Ptr{$relty}, Ptr{C_params}),
                    vals, vecs, rnorms, r)
    catch JE
        # Julia errors are presumably from MV or callbacks, so it should be
        # safe to call free()
        free(r)
        rethrow(JE)
    end
    if err != 0
        free(r)
        if err == -3
            @show r[].stats
        end
        throw(PRIMMEException(Int(err), @_fnq($func)))
    end

    nConv = Int(r[].initSize)
    if nConv < k
        vals = vals[1:nConv]
    end

    stats = r[].stats
    return vals, reshape(vecs[1:n*nConv],n, nConv), rnorms, stats
end
    end # eval block
end # type loop

const witches = Dict( :LA => largest, :LM => largest_abs, :SA => smallest,
                      :CGT => closest_geq, :CLT => closest_leq, :SM => closest_abs )

function _wrap_matvec(A::AbstractMatrix{T}) where {T}
    function mv(xp::Ptr{Tmv}, ldxp::Ptr{PRIMME_INT},
                yp::Ptr{Tmv}, ldyp::Ptr{PRIMME_INT},
                blockSizep::Ptr{Cint}, parp::Ptr{C_params}, ierrp::Ptr{Cint}) where {Tmv}
        ldx, ldy = unsafe_load(ldxp), unsafe_load(ldyp)
        blockSize, par = Int(unsafe_load(blockSizep)), unsafe_load(parp)
        x = unsafe_wrap(Array, xp, (ldx, blockSize))
        y = unsafe_wrap(Array, yp, (ldy, blockSize))

        mul!( view(y, 1:par.nLocal, :), A, view(x, 1:par.nLocal, :))
        unsafe_store!(ierrp, 0)
        return nothing
    end
    mul_fp = @cfunction($mv, Cvoid,
                        (Ptr{T}, Ptr{Int}, Ptr{T}, Ptr{Int}, Ptr{Cint},
                         Ptr{C_params}, Ptr{Cint}))
    return mul_fp
end

# eventually we may want something like this:
#
# function _wrap_callable(A, x::T) where {T}
#     # note that T would be illegal as an argument, so we pass an examplar
#     mul_fp = @cfunction($A, Cvoid,
#                         (Ptr{T}, Ptr{Int}, Ptr{T}, Ptr{Int}, Ptr{Cint},
#                          Ptr{C_params}, Ptr{Cint}))
#     return mul_fp
# end


# Preconditioning
# With misgivings, we follow the pattern in the Matlab and Python wrappers
# and handle matrices rather than (only) factorizations.
#
function _wrap_matldiv(P::Union{AbstractMatrix{T},Factorization{T}}, P2=nothing) where {T}
    if P isa AbstractMatrix
        PF = factorize(P)
    else
        PF = P
    end
    if P2 isa AbstractMatrix
        PF2 = factorize(P2)
    else
        PF2 = P
    end
    # If we have an ldiv! method, it will avoid allocations.
    # AFAICT all LinearAlgebra factorizations (and types that are their own factorizations)
    # comply.  Otherwise we may need to implement traits.
    function mvP(xp::Ptr{Tmv}, ldxp::Ptr{PRIMME_INT},
                 yp::Ptr{Tmv}, ldyp::Ptr{PRIMME_INT},
                 blockSizep::Ptr{Cint}, parp::Ptr{C_params},
                 ierrp::Ptr{Cint}) where {Tmv}
        ldx, ldy = unsafe_load(ldxp), unsafe_load(ldyp)
        blockSize, par = Int(unsafe_load(blockSizep)), unsafe_load(parp)
        x = unsafe_wrap(Array, xp, (ldx, blockSize))
        y = unsafe_wrap(Array, yp, (ldy, blockSize))
        n = par.nLocal
        if size(PF) != (n,n)
            unsafe_store!(ierrp, -2)
            return nothing
        end
        copyto!(view(y, 1:n, :), view(x, 1:n, :))
        ldiv!(PF, view(y, 1:n, :))
        unsafe_store!(ierrp, 0)
        return nothing
    end
    function mvPx(xp::Ptr{Tmv}, ldxp::Ptr{PRIMME_INT},
                 yp::Ptr{Tmv}, ldyp::Ptr{PRIMME_INT},
                 blockSizep::Ptr{Cint}, parp::Ptr{C_params},
                 ierrp::Ptr{Cint}) where {Tmv}
        ldx, ldy = unsafe_load(ldxp), unsafe_load(ldyp)
        blockSize, par = Int(unsafe_load(blockSizep)), unsafe_load(parp)
        x = unsafe_wrap(Array, xp, (ldx, blockSize))
        y = unsafe_wrap(Array, yp, (ldy, blockSize))
        n = par.nLocal
        if size(PF) != (n,n)
            unsafe_store!(ierrp, -2)
            return nothing
        end
        y[1:n, :] .= PF \ view(x, 1:par.nLocal, :)
        unsafe_store!(ierrp, 0)
        return nothing
    end
    function mvP2(xp::Ptr{Tmv}, ldxp::Ptr{PRIMME_INT},
                 yp::Ptr{Tmv}, ldyp::Ptr{PRIMME_INT},
                 blockSizep::Ptr{Cint}, parp::Ptr{C_params},
                 ierrp::Ptr{Cint}) where {Tmv}
        ldx, ldy = unsafe_load(ldxp), unsafe_load(ldyp)
        blockSize, par = Int(unsafe_load(blockSizep)), unsafe_load(parp)
        x = unsafe_wrap(Array, xp, (ldx, blockSize))
        y = unsafe_wrap(Array, yp, (ldy, blockSize))
        n = par.nLocal
        if (size(PF) != (n,n)) || (size(PF2) != (n,n))
            unsafe_store!(ierrp, -2)
            return nothing
        end
        copyto!(view(y, 1:par.nLocal, :), view(x, 1:par.nLocal, :))
        ldiv!(PF, view(y, 1:par.nLocal, :))
        ldiv!(PF2, view(y, 1:par.nLocal, :))
        unsafe_store!(ierrp, 0)
        return nothing
    end
    if isdefined(LinearAlgebra, nameof(typeof(PF)))
        if P2 !== nothing
            ldiv_fp = @cfunction($mvP2, Cvoid,
                                 (Ptr{T}, Ptr{Int}, Ptr{T}, Ptr{Int}, Ptr{Cint},
                                  Ptr{C_params}, Ptr{Cint}))
        else
            ldiv_fp = @cfunction($mvP, Cvoid,
                                 (Ptr{T}, Ptr{Int}, Ptr{T}, Ptr{Int}, Ptr{Cint},
                                  Ptr{C_params}, Ptr{Cint}))
        end
    else
        # probably a sparse factorization, inplace is too complicated
        if P2 !== nothing
            throw(ArgumentError("two-part preconditioner is only constructed for LinearAlgebra types"))
        end
        ldiv_fp = @cfunction($mvPx, Cvoid,
                             (Ptr{T}, Ptr{Int}, Ptr{T}, Ptr{Int}, Ptr{Cint},
                              Ptr{C_params}, Ptr{Cint}))
    end
    return ldiv_fp
end

"""
    PRIMME.eigs(A, k::Integer; kwargs...) => Λ,V,resids,stats

computes some (`k`) eigen-pairs of the matrix `A`

Returns `Λ`, a vector of `k` eigenvalues; `V`, a `k×n` matrix of eigenvectors;
`resids`, a vector of residual norms; and `stats`, a structure with an account of the
 work done.

# Keyword args (names in square brackes are analogues in PRIMME documentation):

- `which::Symbol`, indicating which pairs should be computed. If a target `sigma` is provided, `:SM` and `:LM` refer to distance from the target.

 - `:SA`: smallest algebraic
 - `:SM`: smallest magnitude (default if `sigma` is provided)
 - `:LA`: largest algebraic
 - `:LM`: largest magnitude (default)
 - `:CGT`: closest greater than or equal to target
 - `:CLT`: closest less than or equal to target
- `sigma::Real`, a target value; see `which` above.
- `tol`: convergence criterion (residual norm relative to estimated norm of `A`), called [`eps`]
- `maxiter`: bound on outer iterations [`maxOuterIterations`]
- `maxMatvecs`: bound on calls to matrix-vector multiplication function
- `P`: preconditioning matrix or (preferably) factorization
- `B`: mass matrix
- `verbosity::Int`: detail of diagnostics to report to standard output [`reportLevel`]
"""
function eigs(A::AbstractMatrix{T}, k::Integer = 5;
              which::Symbol = :default,
              tol = 1e-12,
              sigma::Union{Nothing,Real}=nothing,
              maxBasisSize = nothing,
              verbosity::Integer = 0,
              maxiter = 1000,
              maxMatvecs = nothing,
              shifts=nothing,
              method::Union{Nothing,EigsPresetMethod}=nothing,
              P::Union{Nothing,AbstractMatrix,Factorization}=nothing,
              P2::Union{Nothing,AbstractMatrix,Factorization}=nothing,
              B::Union{Nothing,AbstractMatrix{T}}=nothing,
              skipchecks=false,
              ) where {T}
    RT = real(T)
    if (T <: Real && !issymmetric(A)) || !ishermitian(A)
        throw(ArgumentError("matrix/operator must be Hermitian"))
    end
    if k <= 0
        throw(ArgumentError("k must be positive"))
    end

    mul_fp = _wrap_matvec(A)

    if B !== nothing
        if (T <: Real && !issymmetric(B)) || !ishermitian(B)
            throw(ArgumentError("mass matrix/operator must be Hermitian"))
        end
        Bmul_fp = _wrap_matvec(B)
    else
        Bmul_fp = Ptr{Cvoid}()
    end

    if P !== nothing
        precon_fp = _wrap_matldiv(P, P2)
    else
        precon_fp = Ptr{Cvoid}()
    end

    r = eigs_initialize()
    r[:n]            = size(A, 2)
    r[:numEvals]     = k
    r[:eps]          = tol

    if sigma !== nothing
        shifts = [RT(sigma)]
        if which == :default
            which = :SM
        end
    elseif which == :default
        which = :LM
    end

    if (which in (:SM, :LM)) && (shifts === nothing)
        shifts = [zero(RT)]
    end
    if shifts !== nothing
        nshifts = length(shifts)
        shiftsx = Vector{RT}(undef,nshifts)
        shiftsx .= shifts
        r[:numTargetShifts] = nshifts
    else
        shiftsx = Vector{RT}(undef,0)
    end
    if which in keys(witches)
        r[:target] = witches[which]
    else
        throw(ArgumentError("which must be in $(keys(witches))"))
    end
    r[:printLevel]   = verbosity


    if maxBasisSize !== nothing
        r[:maxBasisSize] = maxBasisSize
    end
    if maxiter !== nothing
        r[:maxOuterIterations] = maxiter
    end
    if maxMatvecs !== nothing
        r[:maxMatvecs] = maxMatvecs
    end

    @GC.preserve mul_fp Bmul_fp precon_fp shiftsx begin
        r[:matrixMatvec] = Base.unsafe_convert(Ptr{Cvoid}, mul_fp)
        if B !== nothing
            r[:massMatrixMatvec] = Base.unsafe_convert(Ptr{Cvoid}, Bmul_fp)
        end
        if P !== nothing
            r[:applyPreconditioner] = Base.unsafe_convert(Ptr{Cvoid}, precon_fp)
            # p = @set p.correctionParams.precondition = 1
            cr = Ref(r[].correctionParams)
            cr[:precondition] = 1
            r[:correctionParams] = cr[]
        end
        if shifts !== nothing
            r[:targetShifts] = pointer(shiftsx)
        end
        if method !== nothing
            # following examples, we call this last
            # it seems to leave valid user settings alone
            err = ccall((:primme_set_method, libprimme), Cint, (EigsPresetMethod, Ptr{C_params}),
                    method, r)
            if err != 0
                throw(ArgumentError("illegal value of preset method"))
            end
        end

        out = _eigs(r, T)
    end
    free(r)
    return out
end

end # module
