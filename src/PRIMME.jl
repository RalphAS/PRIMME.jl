module PRIMME

using LinearAlgebra

using PRIMME_jll
const libprimme = PRIMME_jll.libprimme

# for debugging
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

"""
    PRIMME.svds(A, k::Integer; kwargs...) => U,S,V,resids,stats

computes some (`k`) singular triplets (partial SVD) of the matrix `A`
Returns `S`, a vector of `k` singular values; `U`, a `n×k` matrix of left singular vectors;
`V`, a `n×k` matrix of right singular vectors; `resids`, a vector of residual norms;
and `stats`, a structure with an account of the work done.

# Keyword args
 (names in square brackes are analogues in PRIMME documentation)

- `which::Symbol`, indicating which triplets should be computed.
   - `:SR`: smallest algebraic
   - `:SM`: closest to `sigma` (or 0 if not provided)
   - `:LR`: largest magnitude (default)
- `check::Symbol`: (`:throw,:warn,:quiet`) what to do on convergence failure
- `tol`
- `sigma`
- `maxBasisSize`
- `verbosity::Int`
- `method::SvdsPresetMethod`
- `method1::EigsPresetMethod`: algorithm for first stage
- `method2::EigsPresetMethod`: algorithm for seconde stage, if hybrid
- `maxMatvecs`
- `shifts`
- `P_AtA`: preconditioner
- `P_AAt`: preconditioner
- `P_B`: preconditioner
- `fullStats::Bool`: whether to return stats from eigensolver stage(s)
- (some others may be passed to the constructor for `C_svds_params`)
"""
function svds end

for (func, elty, relty) in
    ((:dprimme_svds, :Float64, :Float64),
     (:zprimme_svds, :ComplexF64, :Float64),
     (:sprimme_svds, :Float32, :Float32),
     (:cprimme_svds, :ComplexF32, :Float32),
     )
    @eval begin
function _svds(r::Ref{C_svds_params}, ::Type{$elty};
               v0=nothing, verbosity=1, check=:throw, fullStats=false)
    m, n, k = r[].m, r[].n, r[].numSvals
    nlocal = r[].nLocal
    if nlocal <= 0
        nlocal = n
    end
    mlocal = r[].mLocal
    if mlocal <= 0
        mlocal = m
    end
    nconstr = r[].numOrthoConst
    nvec = nconstr+k
    svals  = Vector{$relty}(undef, k)

    if v0 === nothing
        if nconstr > 0
            throw(ArgumentError("orthogonal constraint vectors must be provided in v0"))
        end
        # does this matter if we don't set initSize?
        localVecs  = rand($elty, (mlocal + nlocal) * k)
    else
        if eltype(v0) != $elty
            throw(ArgumentError("v0 must have eltype $elty"))
        end
        nmin = (mlocal + nlocal) * nvec
        if prod(size(v0)) < nmin
            throw(ArgumentError("initial vector set must have at least $nmin elements"))
        end
        localVecs = v0
    end

    rnorms = Vector{$relty}(undef, k)

    err = ccall((@_fnq($func), libprimme), Cint,
        (Ptr{$relty}, Ptr{$elty}, Ptr{$relty}, Ptr{C_svds_params}),
         svals, localVecs, rnorms, r)

    if err != 0
        infoRet = err in (-3, -103, -203)
        if r[].procID == 0
            if infoRet && (check == :warn)
                @warn "some triplets did not converge in allowed maxMatvecs"
                if verbosity > 0
                    show(r[].stats); println()
                end
            end
        end
        if check == :throw || !infoRet
            free(r)
            throw(PRIMMEException(Int(err), @_fnq($func)))
        end
    end

    nConv = Int(r[].initSize)
    if nConv < k
        svals = svals[1:nConv]
    end
    nConst = r[].numOrthoConst
    loffset = nConst*mlocal
    roffset = (nConst + nConv)*mlocal + nConst*nlocal
    if fullStats
        stats = (r[].stats, r[].primme.stats, r[].primmeStage2.stats)
    else
        stats = r[].stats
    end
    return (reshape(localVecs[loffset .+ (1:(mlocal*nConv))], mlocal, nConv),
            svals,
            reshape(localVecs[roffset .+ (1:(nlocal*nConv))], nlocal, nConv),
            rnorms,
            stats)
end
    end # eval block
end # type loop

_wrap_matvec_svds(A::Ptr{Cvoid},::Type{T}) where{T} = A
_wrap_matvec_svds(A::Base.CFunction,::Type{T}) where {T} = A

function _wrap_matvec_svds(A,::Type{T}) where {T}
    # matrix-vector product, y = a * x (or y = a^t * x), where
    # (ELT *x, PRIMME_INT *ldx, ELT *y, PRIMME_INT *ldy, int *blockSize,
    #  int *transpose, struct primme_svds_params *primme_svds, int *ierr);
    function mv(xp, ldxp, yp, ldyp, blockSizep, trp, parp, ierrp)
        ldx, ldy = unsafe_load(ldxp), unsafe_load(ldyp)
        blockSize = Int(unsafe_load(blockSizep))
        tr, par = unsafe_load(trp), unsafe_load(parp)
        x = unsafe_wrap(Array, xp, (ldx, blockSize))
        y = unsafe_wrap(Array, yp, (ldy, blockSize))
        ml = Int(par.mLocal)
        nl = Int(par.nLocal)
        if tr == 0
            mul!( view(y, 1:ml, :), A, view(x, 1:nl, :))
        else
            mul!(view(y, 1:nl, :), A', view(x, 1:ml, :))
        end
        unsafe_store!(ierrp, 0)
        return nothing
    end
    mul_fp = @cfunction($mv, Cvoid,
        (Ptr{T}, Ptr{PRIMME_INT}, Ptr{T}, Ptr{PRIMME_INT}, Ptr{Cint}, Ptr{Cint},
         Ptr{C_svds_params}, Ptr{Cint}))
end

function _wrap_matldivs_svds(
    P_AtA::Union{Nothing,AbstractMatrix{T},Factorization{T}},
    P_AAt::Union{Nothing,AbstractMatrix{T},Factorization{T}},
    P_B::Union{Nothing,AbstractMatrix{T},Factorization{T}}
) where {T}
    if P_AtA isa AbstractMatrix
        PF_AtA = factorize(P_AtA)
    else
        PF_AtA = P_AtA
    end
    if P_AAt isa AbstractMatrix
        PF_AAt = factorize(P_AAt)
    else
        PF_AAt = P_AAt
    end
    if P_B isa AbstractMatrix
        PF_B = factorize(P_B)
    else
        PF_B = P_B
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
        ml = Int(par.mLocal)
        nl = Int(par.nLocal)
        copyto!(view(y, 1:nl, :), view(x, 1:nl, :))
        if mode == Cint(svds_op_AtA)
            if PF_AtA !== nothing
                ldiv!(PF_AtA, view(y, 1:nl, :))
            end
            unsafe_store!(ierrp, 0)
        elseif mode == Cint(svds_op_AAt)
            if PF_AAt !== nothing
                ldiv!(PF_AAt, view(y, 1:nl, :))
            end
            unsafe_store!(ierrp, 0)
        elseif mode == Cint(svds_op_augmented)
            if PF_B !== nothing
                ldiv!(PF_B, view(y, 1:nl, :))
            end
            unsafe_store!(ierrp, 0)
        else
            unsafe_store!(ierrp, -1)
        end
        return nothing
    end
    ldiv_fp = @cfunction($mvP, Cvoid,
                           (Ptr{T}, Ptr{PRIMME_INT}, Ptr{T}, Ptr{PRIMME_INT}, Ptr{Cint},
                            Ptr{Cint}, Ptr{C_params}, Ptr{Cint}))
    return ldiv_fp
end


function svds(A, k::Integer = 5;
              elt = nothing, m = nothing, n = nothing, # usu. get from A
              which::Symbol = :LR,
              tol::Union{Nothing,Real} = nothing,
              sigma::Union{Nothing,Real} = nothing,
              maxBasisSize::Union{Nothing,Integer} = nothing,
              verbosity::Int = 0,
              method::Union{Nothing,SvdsPresetMethod} = nothing,
              method1::Union{Nothing,EigsPresetMethod} = nothing,
              method2::Union{Nothing,EigsPresetMethod} = nothing,
              maxMatvecs = 10000,
              shifts = nothing,
              P_AtA = nothing,
              P_AAt = nothing,
              P_B = nothing,
              fullStats::Bool = false,
              check::Symbol = :throw,
              kwargs...
              )
    if A isa Ptr
        if elt === nothing || n === nothing || m === nothing
            throw(ArgumentError("elt, m, and n must be specified if A is a pointer"))
        end
        T = elt
    else
        T = eltype(A)
        m, n = size(A)
    end
    RT = real(T)
    if tol === nothing
        tol = sqrt(eps(RT))
    end
    r = svds_initialize()
    mul_fp = _wrap_matvec_svds(A,T)
    r[:m]            = m
    r[:n]            = n
    r[:numSvals]     = k
    if which == :LR
        r[:target] = svds_largest
        # docs claim targetShifts are ignored in these cases, but if we call _print
        # for debugging the pointer dereferencing makes for confusion
        # so let's make it valid.
        shifts = [zero(RT)]
    elseif which == :SR
        r[:target] = svds_smallest
        shifts = [zero(RT)]
    elseif which == :SM
        r[:target] = svds_closest_abs
        if sigma !== nothing
            shifts = [RT(sigma)]
        end
        if shifts === nothing
            shifts = [zero(RT)]
        end
    else
        throw(ArgumentError("argument 'which' must be one of (:LR, :SR, :SM)"))
    end

    if (P_AtA !== nothing) || (P_AAt !== nothing) || (P_B !== nothing)
        precon_fp = _wrap_matldivs_svds(P_AtA, P_AAt, P_B)
    else
        precon_fp = Ptr{Cvoid}()
    end

    if shifts !== nothing
        nshifts = length(shifts)
        shiftsx = Vector{RT}(undef,nshifts)
        shiftsx .= shifts
        r[:numTargetShifts] = nshifts
        r[:targetShifts] = pointer(shiftsx)
    else
        r[:numTargetShifts] = 0
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

    # and now for a genuine footgun
    for (k,v) in kwargs
        if k in fieldnames(C_svds_params)
            r[k] = v
        else
            throw(ArgumentError("invalid keyword $k; not a field of C_svds_params"))
        end
    end
    @GC.preserve mul_fp precon_fp shiftsx begin
        r[:matrixMatvec] = Base.unsafe_convert(Ptr{Cvoid}, mul_fp)
        out = _svds(r, T; fullStats=fullStats, check=check, verbosity=verbosity)
    end
    free(r)
    return out
end


"""
    PRIMME.eigs(A, k::Integer; kwargs...) => Λ,V,resids,stats

computes some (`k`) eigen-pairs of the `n×n` Hermitian matrix `A`

Returns `Λ`, a vector of `k` eigenvalues; `V`, a `n×k` matrix of right eigenvectors;
`resids`, a vector of residual norms; and `stats`, a structure with an account of the
 work done.

# Keyword args
 (names in square brackes are analogues in PRIMME documentation)
- `which::Symbol`, indicating which pairs should be computed. If a target `sigma` is provided, `:SM` and `:LM` refer to distance from the target, otherwise 'sigma' is implicitly zero.

  - `:SR`: smallest algebraic
  - `:SM`: smallest magnitude (default if `sigma` is provided)
  - `:LR`: largest algebraic
  - `:LM`: largest magnitude (default)
  - `:CGT`: closest greater than or equal to target
  - `:CLT`: closest less than or equal to target
 - `sigma::Real`, a target value; see `which` above.
- `tol`: convergence criterion (residual norm relative to estimated norm of `A`) [`eps`]
- `maxiter`: bound on outer iterations [`maxOuterIterations`]
- `maxMatvecs`: bound on calls to matrix-vector multiplication function
- `P`: preconditioning matrix or (preferably) factorization
- `B`: mass matrix
- `verbosity::Int`: detail of diagnostics to report to standard output [`reportLevel`]
- `check::Symbol`: (`:throw,:warn,:quiet`) what to do on convergence failure
Some other keyword arguments, such as those passed to internal functions, might be
properly documented someday.
"""
function eigs end

for (func, elty, relty) in
    ((:dprimme, :Float64, :Float64),
     (:zprimme, :ComplexF64, :Float64),
     (:sprimme, :Float32, :Float32),
     (:cprimme, :ComplexF32, :Float32),
     )
    @eval begin
function _eigs(r::Ref{C_params},::Type{$elty};
                       v0=nothing, verbosity=1, check=:throw)
    n, k = r[].n, r[].numEvals
    nlocal = r[].nLocal
    if nlocal <= 0
        nlocal = n
    end
    nconstr = r[].numOrthoConst
    nvec = nconstr+k
    vals  = Vector{$relty}(undef, k)
    if v0 === nothing
        if nconstr > 0
            throw(ArgumentError("orthogonal constraint vectors must be provided in v0"))
        end
        localVecs  = rand($elty, nlocal, nvec)
    else
        if eltype(v0) != $elty
            throw(ArgumentError("v0 must have eltype $elty"))
        end
        if size(v0) != (nlocal, nconstr+k)
            throw(ArgumentError("initial vectors must have size ($nlocal,$nvec)"))
        end
        localVecs = v0
        if norm(view(localVecs,1:nlocal,nconstr+1:nvec)) == 0
            localVecs[1:nlocal,nconstr+1:nvec] .= rand($elty, nlocal, k)
        end
    end
    rnorms = Vector{$relty}(undef, k)

    local err
    try
        err = ccall((@_fnq($func), libprimme), Cint,
                    (Ptr{$elty}, Ptr{$elty}, Ptr{$relty}, Ptr{C_params}),
                    vals, localVecs, rnorms, r)
    catch JE
        # Julia errors are presumably from MV or callbacks, so it should be
        # safe to call free()
        free(r)
        rethrow(JE)
    end
    if err != 0
        infoRet = err == -3
        if r[].procID == 0
            if infoRet && (check == :warn)
                @warn "some pairs did not converge in allowed maxMatvecs"
                if verbosity > 0
                    show(r[].stats); println()
                end
            end
        end
        if check == :throw || !infoRet
            free(r)
            throw(PRIMMEException(Int(err), @_fnq($func)))
        end
    end

    nConv = Int(r[].initSize)
    # CHECKME: maybe optionally return partially converged results too
    nRet = nConv
    if nRet < k
        vals = vals[1:nRet]
    end

    stats = r[].stats

    return vals, localVecs[1:nlocal, nconstr+1:nconstr+nRet], rnorms, stats
end
    end # eval block
end # type loop

const witches = Dict( :LR => largest, :LM => largest_abs, :SR => smallest,
                      :CGT => closest_geq, :CLT => closest_leq, :SM => closest_abs )

_wrap_matvec(A::Ptr{Cvoid},::Type{T}) where {T} = A
_wrap_matvec(A::Base.CFunction,::Type{T}) where {T} = A

function _wrap_matvec(A,::Type{T}) where {T}
    function mv(xp::Ptr{Tmv}, ldxp::Ptr{PRIMME_INT},
                yp::Ptr{Tmv}, ldyp::Ptr{PRIMME_INT},
                blockSizep::Ptr{Cint}, parp::Ptr{C_params}, ierrp::Ptr{Cint}) where {Tmv}
        ldx, ldy = unsafe_load(ldxp), unsafe_load(ldyp)
        blockSize, par = Int(unsafe_load(blockSizep)), unsafe_load(parp)
        x = unsafe_wrap(Array, xp, (ldx, blockSize))
        y = unsafe_wrap(Array, yp, (ldy, blockSize))

        nl = Int(par.nLocal)
        mul!( view(y, 1:nl, :), A, view(x, 1:nl, :))
        unsafe_store!(ierrp, 0)
        return nothing
    end
    mul_fp = @cfunction($mv, Cvoid,
                        (Ptr{T}, Ptr{PRIMME_INT}, Ptr{T}, Ptr{PRIMME_INT}, Ptr{Cint},
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
        n = Int(par.nLocal)
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
        n = Int(par.nLocal)
        if size(PF) != (n,n)
            unsafe_store!(ierrp, -2)
            return nothing
        end
        y[1:n, :] .= PF \ view(x, 1:n, :)
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
        n = Int(par.nLocal)
        if (size(PF) != (n,n)) || (size(PF2) != (n,n))
            unsafe_store!(ierrp, -2)
            return nothing
        end
        copyto!(view(y, 1:n, :), view(x, 1:n, :))
        ldiv!(PF, view(y, 1:n, :))
        ldiv!(PF2, view(y, 1:n, :))
        unsafe_store!(ierrp, 0)
        return nothing
    end
    if isdefined(LinearAlgebra, nameof(typeof(PF)))
        if P2 !== nothing
            ldiv_fp = @cfunction($mvP2, Cvoid,
                                 (Ptr{T}, Ptr{PRIMME_INT}, Ptr{T}, Ptr{PRIMME_INT},
                                  Ptr{Cint}, Ptr{C_params}, Ptr{Cint}))
        else
            ldiv_fp = @cfunction($mvP, Cvoid,
                                 (Ptr{T}, Ptr{PRIMME_INT}, Ptr{T}, Ptr{PRIMME_INT},
                                  Ptr{Cint}, Ptr{C_params}, Ptr{Cint}))
        end
    else
        # probably a sparse factorization, inplace is too complicated
        if P2 !== nothing
            throw(ArgumentError("two-part preconditioner is only constructed for LinearAlgebra types"))
        end
        ldiv_fp = @cfunction($mvPx, Cvoid,
                             (Ptr{T}, Ptr{PRIMME_INT}, Ptr{T}, Ptr{PRIMME_INT}, Ptr{Cint},
                              Ptr{C_params}, Ptr{Cint}))
    end
    return ldiv_fp
end

function eigs(A, k::Integer = 5;
              elt = nothing, n = nothing, # usu. get from A
              which::Symbol = :default,
              tol::Union{Nothing,Real} = nothing,
              sigma::Union{Nothing,Real}=nothing,
              maxBasisSize = nothing,
              verbosity::Integer = 0,
              maxiter::Integer = 1000,
              maxMatvecs::Union{Nothing,Integer} = nothing,
              shifts=nothing,
              method::Union{Nothing,EigsPresetMethod}=nothing,
              P::Union{Nothing,AbstractMatrix,Factorization}=nothing,
              P2::Union{Nothing,AbstractMatrix,Factorization}=nothing,
              B::Union{Nothing,AbstractMatrix}=nothing,
              check::Symbol = :throw,
              kwargs...
              )
    if A isa Ptr
        if elt === nothing || n === nothing
            throw(ArgumentError("elt and n must be specified if A is a pointer"))
        end
        T = elt
    else
        T = eltype(A)
        n = size(A, 2)
        if size(A, 1) != n
            throw(ArgumentError("this is an eigensolver, and your matrix is not square."))
        end
        if (T <: Real && !issymmetric(A)) || !ishermitian(A)
            throw(ArgumentError("matrix/operator must be Hermitian"))
        end
    end
    RT = real(T)
    if tol === nothing
        tol = sqrt(eps(RT))
    end
    if k <= 0
        throw(ArgumentError("k must be positive"))
    end

    mul_fp = _wrap_matvec(A, T)

    if B !== nothing
        if (T <: Real && !issymmetric(B)) || !ishermitian(B)
            throw(ArgumentError("mass matrix/operator must be Hermitian"))
        end
        Bmul_fp = _wrap_matvec(B, T)
    else
        Bmul_fp = Ptr{Cvoid}()
    end

    if P !== nothing
        precon_fp = _wrap_matldiv(P, P2)
    else
        precon_fp = Ptr{Cvoid}()
    end

    r = eigs_initialize()
    r[:n]            = n
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
        r[:targetShifts] = pointer(shiftsx)
    else
        r[:numTargetShifts] = 0
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

    # and now for a genuine footgun
    for (k,v) in kwargs
        if k in fieldnames(C_params)
            r[k] = v
        else
            throw(ArgumentError("invalid keyword $k; not a field of C_params"))
        end
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
        if method !== nothing
            # following examples, we call this last
            # it seems to leave valid user settings alone
            err = ccall((:primme_set_method, libprimme), Cint, (EigsPresetMethod, Ptr{C_params}),
                    method, r)
            if err != 0
                throw(ArgumentError("illegal value of preset method"))
            end
        end
        # @show r[]
        out = _eigs(r, T; check=check, verbosity=verbosity)
    end
    free(r)
    return out
end

end # module
