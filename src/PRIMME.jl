module PRIMME

using LinearAlgebra

# const libprimme = joinpath(dirname(@__FILE__()), "../deps/primme/lib/libprimme")

const libprimme = "/scratch/build/primme-3.2/lib/libprimme.so"

const PRIMME_INT = Int # might be wrong. Should be detected.

# Note: we rely on the "fact" that enum in Julia and C have the same size
# (4 bytes on Linux x86_64, at least)

@enum(Target,
    smallest,        # leftmost eigenvalues */
    largest,         # rightmost eigenvalues */
    closest_geq,     # leftmost but greater than the target shift */
    closest_leq,     # rightmost but less than the target shift */
    closest_abs,     # the closest to the target shift */
    largest_abs      # the farthest to the target shift */
)

@enum(Init,         # Initially fill up the search subspace with: */
    init_default,
    init_krylov, # a) Krylov with the last vector provided by the user or random */
    init_random, # b) just random vectors */
    init_user    # c) provided vectors or a single random vector */
)

@enum(Projection,
    proj_default,
    proj_RR,          # Rayleigh-Ritz */
    proj_harmonic,    # Harmonic Rayleigh-Ritz */
    proj_refined      # refined with fixed target */
)
const C_projection_params = Projection

@enum(Restartscheme,
    thick,
    dtr
)

abstract type PrimmeCStruct end

struct C_restarting_params <: PrimmeCStruct
    # scheme::Restartscheme
    maxPrevRetain::Cint
end

struct JD_projectors
    LeftQ::Cint
    LeftX::Cint
    RightQ::Cint
    RightX::Cint
    SkewQ::Cint
    SkewX::Cint
end

@enum(Convergencetest,
    full_LTolerance,
    decreasing_LTolerance,
    adaptive_ETolerance,
    adaptive
)

@enum(Event,
   event_outer_iteration,    # report at every outer iteration
   event_inner_iteration,    # report at every QMR iteration
   event_restart,            # report at every basis restart
   event_reset,              # event launch if basis reset
   event_converged,          # report new pair marked as converged
   event_locked,             # report new pair marked as locked
   event_message,            # report warning
   event_profile             # report time from consumed by a function
)

@enum(Orth,
    orth_default,
    orth_implicit_I, # assume for search subspace V, V' B V = I
    orth_explicit_I  # compute V' B V
)

@enum(OpDatatype,
    op_default,
    op_half,
    op_float,
    op_double,
    op_quad,
    op_int
)

struct C_correction_params <: PrimmeCStruct
    precondition::Cint
    robustShifts::Cint
    maxInnerIterations::Cint
    projectors::JD_projectors
    convTest::Convergencetest
    relTolBase::Cdouble
end

struct C_stats <: PrimmeCStruct
    numOuterIterations::PRIMME_INT
    numRestarts::PRIMME_INT
    numMatvecs::PRIMME_INT
    numPreconds::PRIMME_INT
    numGlobalSum::PRIMME_INT         # times called globalSumReal
    volumeGlobalSum::PRIMME_INT      # number of SCALARs reduced by globalSumReal
    volumeBroadcast::PRIMME_INT      # number broadcast by broadcastReal
    flopsDense::Cdouble              # FLOPS done by Num_update_VWXR_Sprimme
    numOrthoInnerProds::Cdouble      # number of inner prods done by Ortho
    elapsedTime::Cdouble
    timeMatvec::Cdouble              # time expend by matrixMatvec
    timePrecond::Cdouble             # time expend by applyPreconditioner
    timeOrtho::Cdouble               # time expend by ortho
    timeGlobalSum::Cdouble           # time expend by globalSumReal
    timeBroadcast::Cdouble           # time expend by broadcastReal
    timeDense::Cdouble               # time expend by Num_update_VWXR_Sprimme
    estimateMinEVal::Cdouble         # the leftmost Ritz value seen
    estimateMaxEVal::Cdouble         # the rightmost Ritz value seen
    estimateLargestSVal::Cdouble     # absolute value of the farthest to zero Ritz value seen
    maxConvTol::Cdouble              # largest norm residual of a locked eigenpair
    estimateResidualError::Cdouble   # accumulated error in V and W
    lockingIssue::PRIMME_INT         # Some converged with a weak criterion
end

struct C_params <: PrimmeCStruct

    # The user must input at least the following two arguments
    n::PRIMME_INT
    matrixMatvec::Ptr{Cvoid}
    # void (*matrixMatvec)
       # ( void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize,
         # struct primme_params *primme, int *ierr);
    matrixMatvec_type::OpDatatype

    # Preconditioner applied on block of vectors (if available)
    applyPreconditioner::Ptr{Cvoid}
    # void (*applyPreconditioner)
       # ( void *x, PRIMME_INT *ldx,  void *y, PRIMME_INT *ldy, int *blockSize,
         # struct primme_params *primme, int *ierr);
    applyPreconditioner_type::OpDatatype

    # Matrix times a multivector for mass matrix B for generalized Ax = xBl
    massMatrixMatvec::Ptr{Cvoid}
    # void (*massMatrixMatvec)
       # ( void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize,
         # struct primme_params *primme, int *ierr);
    massMatrixMatvec_type::OpDatatype

    # input for the following is only required for parallel programs */
    numProcs::Cint
    procID::Cint
    nLocal::PRIMME_INT
    commInfo::Ptr{Cvoid}
    globalSumReal::Ptr{Cvoid}
    # void (*globalSumReal)
       # (void *sendBuf, void *recvBuf, int *count, struct primme_params *primme,
        # int *ierr );
    globalSumReal_type::OpDatatype
    broadcastReal::Ptr{Cvoid}
    # void (*broadcastReal)(
    #     void *buffer, int *count, struct primme_params *primme, int *ierr);
    broadcastReal_type::OpDatatype

    # Though Initialize will assign defaults, most users will set these
    numEvals::Cint
    target::Target
    numTargetShifts::Cint             # For targeting interior epairs,
    targetShifts::Ptr{Cdouble}        # at least one shift must also be set

    # the following will be given default values depending on the method
    dynamicMethodSwitch::Cint
    locking::Cint
    initSize::Cint
    numOrthoConst::Cint
    maxBasisSize::Cint
    minRestartSize::Cint
    maxBlockSize::Cint
    maxMatvecs::PRIMME_INT
    maxOuterIterations::PRIMME_INT
    iseed::NTuple{4,PRIMME_INT}
    aNorm::Cdouble
    bNorm::Cdouble
    invBNorm::Cdouble
    eps::Cdouble
    orth::Orth
    internalPrecision::OpDatatype

    printLevel::Cint
    outputFile::Ptr{Cvoid}

    matrix::Ptr{Cvoid}
    preconditioner::Ptr{Cvoid}
    massMatrix::Ptr{Cvoid}
    ShiftsForPreconditioner::Ptr{Cdouble}
    initBasisMode::Init
    ldevecs::PRIMME_INT
    ldOPs::PRIMME_INT

    projectionParams::C_projection_params
    restartingParams::C_restarting_params
    correctionParams::C_correction_params
    stats::C_stats

    convTestFun::Ptr{Cvoid}
    # void (*convTestFun)(double *eval, void *evec, double *rNorm, int *isconv, 
          # struct primme_params *primme, int *ierr);
    convTestFun_type::OpDatatype
    convtest::Ptr{Cvoid}
    monitorFun::Ptr{Cvoid}
    # void (*monitorFun)(void *basisEvals, int *basisSize, int *basisFlags,
       # int *iblock, int *blockSize, void *basisNorms, int *numConverged,
       # void *lockedEvals, int *numLocked, int *lockedFlags, void *lockedNorms,
       # int *inner_its, void *LSRes, primme_event *event,
       # struct primme_params *primme, int *err);
    monitorFun_type::OpDatatype
    monitor::Ptr{Cvoid}
    queue::Ptr{Cvoid} # magma device queue
    profile::Ptr{Cvoid} # char *profile; regex with functions to monitor times
end

@enum(Svds_target,
    svds_largest,
    svds_smallest,
    svds_closest_abs
)

@enum(Svds_operator,
    svds_op_none,
    svds_op_AtA,
    svds_op_AAt,
    svds_op_augmented
)

struct C_svds_stats <: PrimmeCStruct
    numOuterIterations::PRIMME_INT
    numRestarts::PRIMME_INT
    numMatvecs::PRIMME_INT
    numPreconds::PRIMME_INT
    numGlobalSum::PRIMME_INT         # times called globalSumR
    volumeGlobalSum::PRIMME_INT      # number of SCALARs reduced by globalSumReal
    volumeBroadcast::PRIMME_INT      # number broadcast by broadCastReal
    numOrthoInnerProds::Cdouble      # number of inner prods done by Ortho
    elapsedTime::Cdouble
    timeMatvec::Cdouble              # time expend by matrixMatvec
    timePrecond::Cdouble             # time expend by applyPreconditioner
    timeOrtho::Cdouble               # time expend by ortho
    timeGlobalSum::Cdouble           # time expend by globalSumReal
    timeBroadcast::Cdouble           # time expend by broadcastReal
    lockingIssue::PRIMME_INT         # Some converged with a weak criterion
end

struct C_svds_params <: PrimmeCStruct
    # Low interface: configuration for the eigensolver
    primme::C_params # Keep it as first field to access primme_svds_params from
                          # primme_params
    primmeStage2::C_params # other primme_params, used by hybrid

    # Specify the size of the rectangular matrix A
    m::PRIMME_INT # number of rows
    n::PRIMME_INT # number of columns

    # High interface: these values are transferred to primme and primmeStage2 properly
    matrixMatvec::Ptr{Cvoid}
    # void (*matrixMatvec)
    #    (void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize,
    #     int *transpose, struct primme_svds_params *primme_svds, int *ierr);
    matrixMatvec_type::OpDatatype
    applyPreconditioner::Ptr{Cvoid}
    # void (*applyPreconditioner)
       # (void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy, int *blockSize,
        # int *transpose, struct primme_svds_params *primme_svds, int *ierr);
    applyPrecondioner_type::OpDatatype

    # Input for the following is only required for parallel programs
    numProcs::Cint
    procID::Cint
    mLocal::PRIMME_INT
    nLocal::PRIMME_INT
    commInfo::Ptr{Cvoid}
    globalSumReal::Ptr{Cvoid}
    # void (*globalSumReal)
       # (void *sendBuf, void *recvBuf, int *count,
        # struct primme_svds_params *primme_svds, int *ierr);
    globalSumReal_type::OpDatatype
    broadcastReal::Ptr{Cvoid}
    broadcastReal_type::OpDatatype

    # Though primme_svds_initialize will assign defaults, most users will set these
    numSvals::Cint
    target::Svds_target
    numTargetShifts::Cint  # For primme_svds_augmented method, user has to
    targetShifts::Ptr{Cdouble} # make sure  at least one shift must also be set
    method::Svds_operator # one of primme_svds_AtA, primme_svds_AAt or primme_svds_augmented
    methodStage2::Svds_operator # hybrid second stage method; accepts the same values as method */

    # These pointers may be used for users to provide matrix/preconditioner
    matrix::Ptr{Cvoid}
    preconditioner::Ptr{Cvoid}

    # The following will be given default values depending on the method
    locking::Cint
    numOrthoConst::Cint
    aNorm::Cdouble
    eps::Cdouble

    precondition::Cint
    initSize::Cint
    maxBasisSize::Cint
    maxBlockSize::Cint
    maxMatvecs::PRIMME_INT
    iseed::NTuple{4,PRIMME_INT}
    printLevel::Cint
    internalPrecision::OpDatatype
    outputFile::Ptr{Cvoid}
    stats::C_svds_stats

    convTestFun::Ptr{Cvoid}
    convTestFun_type::OpDatatype
    convtest::Ptr{Cvoid}
    monitorFun::Ptr{Cvoid}
    # void (*monitorFun)(void *basisSvals, int *basisSize, int *basisFlags,
       # int *iblock, int *blockSize, void *basisNorms, int *numConverged,
       # void *lockedSvals, int *numLocked, int *lockedFlags, void *lockedNorms,
       # int *inner_its, void *LSRes, primme_event *event, int *stage,
       # struct primme_svds_params *primme_svds, int *err);
    monitorFun_Type::OpDatatype
    monitor::Ptr{Cvoid}
    queue::Ptr{Cvoid}
    profile::Ptr{Cvoid} # actually char * for regex
end


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
        A_mul_B!( view(y, 1:par.m, :), _A_[], view(x, 1:par.n, :))
    else
        Ac_mul_B!(view(y, 1:par.n, :), _A_[], view(x, 1:par.m, :))
    end
    unsafe_store!(ierrp, 0)
    return nothing
end

_print(r::Ref{C_params}) = ccall((:primme_display_params, libprimme), Cvoid, (C_params,), r[])

_print(r::Ref{C_svds_params}) = ccall((:primme_svds_display_params, libprimme), Cvoid, (C_svds_params,), r[])

function Base.setindex!(r::Ref{T}, x, sym::Symbol) where T<:PrimmeCStruct
    p  = Base.unsafe_convert(Ptr{T}, r)
    pp = convert(Ptr{UInt8}, p)
    i  = findfirst(t -> t == sym, fieldnames(T))
    if i === nothing
        throw(ArgumentError("no such field"))
    end
    o  = fieldoffset(T, i)
    # AN had this, but we need to set first field in C_params
    # if o == 0
    #     throw(ArgumentError("no such field"))
    # end
    S = fieldtype(T, i)
    unsafe_store!(convert(Ptr{S}, pp + o), x)
    return x
end

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

    return reshape(svecs[r[].numOrthoConst*m + (1:(m*nConv))], m, nConv),
        svals,
        reshape(svecs[(r[].numOrthoConst + nConv)*m + r[].numOrthoConst*n + (1:(n*nConv))], n, nConv)
end

function svds(A::AbstractMatrix, k = 5; tol = 1e-12, maxBlockSize = 2k, verbosity::Int = 0, method::Svds_operator = svds_op_AtA)
    mul_fp = @cfunction(matrixMatvec, Cvoid,
        (Ptr{Float64}, Ptr{Int}, Ptr{Float64}, Ptr{Int}, Ptr{Cint}, Ptr{Cint},
         Ptr{C_svds_params}, Ptr{Cint}))
    r = svds_initialize()
    if verbosity > 1
        @info "barely initialized config"
        _print(r)
    end
    _A_[]            = A
    r[:m]            = size(A, 1)
    r[:n]            = size(A, 2)
    r[:matrixMatvec] = mul_fp
    r[:numSvals]     = k
    r[:printLevel]   = verbosity
    r[:eps]          = tol
    r[:maxBlockSize] = maxBlockSize
    r[:method]       = method
    if verbosity > 0
        _print(r)
    end
    out = _svds(r)
    if verbosity > 0
        _print(r)
    end
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

function eigs(A::AbstractMatrix, k = 5; tol = 1e-12, maxBlockSize = 2k, verbosity::Int = 0)
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

    r[:maxOuterIterations] = 100
    r[:maxMatvecs] = 1000
    r[:target] = largest
    if verbosity > 0
        _print(r)
    end
    out = _eigs(r)
    if verbosity > 0
        _print(r)
    end
    free(r)
    return out
end

end # module
