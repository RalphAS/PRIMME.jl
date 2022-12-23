# types for PRIMME.jl

# FIXME: this should be exported from JLL
const PRIMME_INT = Int

"""
    PRIMMEException

Exception thrown when a call to the PRIMME library fails. The `error_code` field
may be looked up in the PRIMME documentation for further explanation.
"""
struct PRIMMEException <: Exception
    error_code::Int
    function_name::Symbol
end

abstract type PrimmeCStruct end

# This is shameful. To be replaced with lenses.
function Base.setindex!(r::Ref{T}, x, sym::Symbol) where T<:PrimmeCStruct
    p  = Base.unsafe_convert(Ptr{T}, r)
    pp = convert(Ptr{UInt8}, p)
    i  = findfirst(t -> t == sym, fieldnames(T))
    if i === nothing
        throw(ArgumentError("no such field"))
    end
    o  = fieldoffset(T, i)
    S = fieldtype(T, i)
    if x isa AbstractVector
        x = tuple(x...)
    end
    unsafe_store!(convert(Ptr{S}, pp + o), x)
    return x
end

@enum(Target,
    smallest,        # leftmost eigenvalues */
    largest,         # rightmost eigenvalues */
    closest_geq,     # leftmost but greater than the target shift */
    closest_leq,     # rightmost but less than the target shift */
    closest_abs,     # the closest to the target shift */
    largest_abs      # the farthest to the target shift */
)

@enum(Projection,
    proj_default,
    proj_RR,          # Rayleigh-Ritz */
    proj_harmonic,    # Harmonic Rayleigh-Ritz */
    proj_refined      # refined with fixed target */
)

@enum(Init,         # Initially fill up the search subspace with: */
    init_default,
    init_krylov, # a) Krylov with the last vector provided by the user or random */
    init_random, # b) just random vectors */
    init_user    # c) provided vectors or a single random vector */
)

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

struct C_stats <: PrimmeCStruct
    numOuterIterations::PRIMME_INT
    numRestarts::PRIMME_INT
    numMatvecs::PRIMME_INT
    numPreconds::PRIMME_INT
    numGlobalSum::PRIMME_INT         # times called globalSumReal
    numBroadcast::PRIMME_INT
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
    estimateBNorm::Cdouble
    estimateInvBNorm::Cdouble
    maxConvTol::Cdouble              # largest norm residual of a locked eigenpair
    estimateResidualError::Cdouble   # accumulated error in V and W
    lockingIssue::PRIMME_INT         # Some converged with a weak criterion
end

struct C_svds_stats <: PrimmeCStruct
    numOuterIterations::PRIMME_INT
    numRestarts::PRIMME_INT
    numMatvecs::PRIMME_INT
    numPreconds::PRIMME_INT
    numGlobalSum::PRIMME_INT         # times called globalSumR
    numBroadcast::PRIMME_INT         # times called broadcastR
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

const PrimmeStats = Union{C_stats, C_svds_stats}
# as member or in container
function Base.show(io::IO, s::PrimmeStats)
    print(io,s.numOuterIterations," outer it., ",
          s.numRestarts," restarts, ", s.numMatvecs, " MV, ", s.numPreconds, " preconds...")
end
# standalone
function Base.show(io::IO, ::MIME"text/plain", s::PrimmeStats)
    np = s.numPreconds
    ng = s.numGlobalSum
    print(io, "PRIMME statistics:\n  ",
          s.numOuterIterations," outer iter., ", s.numRestarts, " restarts\n  ",
          s.numMatvecs," matvecs, ", s.numPreconds, " preconds\n  ",
          s.numGlobalSum," global sum calls, ", s.numBroadcast, " broadcast calls  \n  "
          )
    if ng > 0
        print(io, s.volumeGlobalSum," summed, time: ", s.timeGlobalSum, "\n  ",
              s.volumeBroadcast, " broadcast, time: ", s.timeBroadcast, "\n  ")
    end
    print(io, s.numOrthoInnerProds, " orth. inner prod\n  ",
          "total time: ", s.elapsedTime, ", matvec time: ", s.timeMatvec,"\n  ",
          "precond time: ", s.timePrecond, ", orth. time: ", s.timeOrtho)
    if isa(s, C_params)
        println(io, "\n  VWXR time: ", s.timeDense,
                ", FLOPS (VWXR), ", s.flopsDense, "\n  ",
                "estimated min. Eval: ", s.estimateMinEVal,
                ", max. Eval: ", s.estimateMaxEVal,
                "\n  max. Sval: ", s.estimateLargestSVal,"\n  ",
                "estimated B norm: ", s.estimateBNorm,
                ", inv(B): ", s.estimateInvBNorm,"\n  ",
                "max resid: ", s.maxConvTol,
                "   accumulated: ", s.estimateResidualError)
    end
    if s.lockingIssue != 0
        print(io, "\n  some pairs did not fully converge (lockingIssue = ", s.lockingIssue,
              ")")
    end
end

struct JD_projectors
    LeftQ::Cint
    LeftX::Cint
    RightQ::Cint
    RightX::Cint
    SkewQ::Cint
    SkewX::Cint
end

struct C_projection_params <: PrimmeCStruct
    projection::Projection
end

struct C_correction_params <: PrimmeCStruct
    precondition::Cint
    robustShifts::Cint
    maxInnerIterations::Cint
    projectors::JD_projectors
    convTest::Convergencetest
    relTolBase::Cdouble
end

struct C_restarting_params <: PrimmeCStruct
    maxPrevRetain::Cint
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

"""
    EigsPresetMethod

method selector for PRIMME eigs functions

An `enum` with the following values (see PRIMME documentation for interpretation):
* default_method,
* dynamic_method,
* default_min_time,
* default_min_matvecs,
* Arnoldi_method,
* GD_methd,
* GD_plusK_method,
* GD_Olsen_plusK,
* JD_Olsen_plusK,
* RQI_method,
* JDQR_method,
* JDQMR_method,
* JDQMR_ETol,
* steepest_descent,
* LOBPCG_OrthoBasis,
* LOBPCG_OrthoBasis_Window
"""
@enum(EigsPresetMethod,
   default_method,
   dynamic_method,
   default_min_time,
   default_min_matvecs,
   Arnoldi_method,
   GD_methd,
   GD_plusK_method,
   GD_Olsen_plusK,
   JD_Olsen_plusK,
   RQI_method,
   JDQR_method,
   JDQMR_method,
   JDQMR_ETol,
   steepest_descent,
   LOBPCG_OrthoBasis,
   LOBPCG_OrthoBasis_Window
)

@enum(Svds_target,
    svds_largest,
    svds_smallest,
    svds_closest_abs
)

"""
    SvdsPresetMethod

method specifier for PRIMME svds functions

An `enum` with the following values:
   * `svds_default` set as `svds_hybrid`
   * `svds_normalequations` work with `A'*A` or `A*A'`
   * `svds_augmented` work with `[0 A'; A 0]`
   * `svds_hybrid` start with normal equations, then refine with augmented
"""
@enum(SvdsPresetMethod,
   svds_default,
   svds_hybrid,
   svds_normalequations, # At*A or A*At
   svds_augmented
)

@enum(Svds_operator,
    svds_op_none,
    svds_op_AtA,
    svds_op_AAt,
    svds_op_augmented
)

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
