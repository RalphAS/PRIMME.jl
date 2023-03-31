# example of PRIMME usage with MPI: Eigensystem

# This works on a dense matrix without preconditioning for simplicity.

using LinearAlgebra
using PRIMME
using PRIMME: C_params, C_svds_params, PRIMME_INT

using MPI
MPI.Init()

const commW = MPI.COMM_WORLD

# For simplicity we just store the minimal MPI context in commInfo.
# It could be a pointer to a struct including some problem-specific information.
function getcomm(par::Union{C_params, C_svds_params})
    comm = MPI.Comm(unsafe_load(Base.unsafe_convert(Ptr{MPI.MPI_Comm}, par.commInfo)))
    return comm
end

const ROOT_PROC = 0

# if we made this a variable, we would need to use more closures
const AELT = Float64
#const AELT = ComplexF64

# for this toy problem all procs get the whole matrix and extract their own part
const Aglobal = Ref(zeros(AELT,1,1))
const Alocal = Ref(zeros(AELT,1,1))

# keep track of which rows belong to current proc
const rCounts = Ref([0])
const rIstarts = Ref([0])


function mkmat(n)
    q,_ = qr(randn(AELT,n,n))
    # trivial case for now
    d = 1:n
    A = q * Diagonal(d) * q'
    A = 0.5 * (A + A')
    return A
end

# each process deals with rows istart:istart+nrows-1
function xmat(istart,nrows,n)
    Asub = Matrix{AELT}(undef,nrows,n)
    iend = istart + nrows - 1
    copyto!(Asub, view(Aglobal[], istart:iend, :))
end

# There are 3 Julia functions which must be defined and wrapped so that
# they can be called by the PRIMME library code.
# (Others would be needed for preconditioning and mass operators, but the Julia API
# for those is not ready yet.)


# 1. distributed matrix-vector product
function mv(xp::Ptr{Tmv}, ldxp::Ptr{PRIMME_INT},
            yp::Ptr{Tmv}, ldyp::Ptr{PRIMME_INT},
            blockSizep::Ptr{Cint}, parp::Ptr{C_params}, ierrp::Ptr{Cint}) where {Tmv}

    ldx, ldy = unsafe_load(ldxp), unsafe_load(ldyp)
    blockSize, par = Int(unsafe_load(blockSizep)), unsafe_load(parp)
    x = unsafe_wrap(Array, xp, (ldx, blockSize))
    y = unsafe_wrap(Array, yp, (ldy, blockSize))

    comm = getcomm(par)

    nproc0 = MPI.Comm_size(comm)
    iproc0 = MPI.Comm_rank(comm)

    nproc = par.numProcs
    iproc = par.procID
    idxProc = iproc + 1
    ok = true
    if nproc != nproc0
        @warn "bad nproc"
        ok = false
    end
    if iproc != iproc0
        @warn "bad iproc"
        ok = false
    end
    n = par.n
    counts = rCounts[]
    istarts = rIstarts[]
    if par.nLocal != counts[idxProc]
        @warn "bad nLocal"
        ok = false
    end
    if !ok
        unsafe_store!(ierrp, 1)
        return nothing
    end

    n = size(Aglobal[], 2)
    xAll = zeros(AELT, n, blockSize)
    for jp in 1:nproc
        if jp == idxProc
            xTmp = copy(x)
        else
            xTmp = similar(x)
        end
        MPI.Bcast!(xTmp, jp-1, comm)
        i0 = istarts[jp]
        i1 = i0 + counts[jp] - 1
        xAll[i0:i1,:] .= view(xTmp, 1:counts[jp], :)
    end

    ierr = 0
    try
        mul!( view(y, 1:par.nLocal, :), Alocal[], xAll)
    catch JE
        @warn "[$(iproc)] mv threw $JE"
        ierr = 1
    end
    unsafe_store!(ierrp, ierr)
    return nothing
end

# This is how we make our Julia function available to the C library.
# It returns a "mul! function pointer".
function _wrap_mv(x::T) where {T}
    mul_fp = @cfunction(mv, Cvoid,
                        (Ptr{T}, Ptr{Int}, Ptr{T}, Ptr{Int}, Ptr{Cint},
                         Ptr{C_params}, Ptr{Cint}))
    return mul_fp
end

# 2. broadcast from root process
function bcast(bufferp, countp, parp, ierrp)
    count = unsafe_load(countp)
    par = unsafe_load(parp)
    comm = getcomm(par)
    iproc = par.procID
    buffer = unsafe_wrap(Array, bufferp, (count,))
    ierr = 0
    try
        MPI.Bcast!(buffer, comm; root=ROOT_PROC)
    catch JE
        @warn "[$(iproc)] bcast threw $JE"
        ierr = 1
    end
    unsafe_store!(ierrp, ierr)
    nothing
end

function _wrap_bcast(x::T) where {T}
    bcast_fp = @cfunction(bcast, Cvoid, (Ptr{T}, Ptr{Cint}, Ptr{C_params}, Ptr{Cint}))
    return bcast_fp
end
# some signatures differ for use in `svds`.
function _wrap_bcast_svds(x::T) where {T}
    bcast_fp = @cfunction(bcast, Cvoid, (Ptr{T}, Ptr{Cint}, Ptr{C_svds_params}, Ptr{Cint}))
    return bcast_fp
end

# 3. reduction from root process
function global_sum(sendbufp, recvbufp, countp, parp, ierrp)
    count = unsafe_load(countp)
    # @show sendbufp
    par = unsafe_load(parp)
    comm = getcomm(par)
    iproc = par.procID
    sendbuf = unsafe_wrap(Array, sendbufp, (count,))
    ierr = 0
    plus = MPI.SUM
    try
        if recvbufp == sendbufp
            MPI.Allreduce!(sendbuf, plus, comm)
        else
            recvbuf = unsafe_wrap(Array, recvbufp, (count,))
            MPI.Allreduce!(sendbuf, recvbuf, plus, comm)
        end
    catch JE
        @warn "[$(iproc)] reduce threw $JE"
        ierr = 1
    end
    unsafe_store!(ierrp, ierr)
    nothing
end

function _wrap_global_sum(x::T) where {T}
    fp = @cfunction(global_sum, Cvoid,
                    (Ptr{T}, Ptr{T}, Ptr{Cint}, Ptr{C_params}, Ptr{Cint}))
    return fp
end
function _wrap_global_sum_svds(x::T) where {T}
    fp = @cfunction(global_sum, Cvoid,
                    (Ptr{T}, Ptr{T}, Ptr{Cint}, Ptr{C_svds_params}, Ptr{Cint}))
    return fp
end

function runme(n=1000)
    LinearAlgebra.BLAS.set_num_threads(2)
    comm = MPI.COMM_WORLD
    nproc = MPI.Comm_size(comm)
    iproc = MPI.Comm_rank(comm)
    println("rank $iproc of $nproc started")
    MPI.Barrier(comm)

    istarts = ones(Int,nproc)
    counts = zeros(Int,nproc)
    d,r = divrem(n, nproc)
    for i in 1:nproc
        n1 = (i <= r) ? d+1 : d
        counts[i] = n1
        if i < nproc
            istarts[i+1] = istarts[i] + n1
        end
    end
    rCounts[] = counts
    rIstarts[] = istarts

    idxProc = iproc+1
    istart = istarts[idxProc]
    nlocal = counts[idxProc]

    # just give everybody the full matrix for simplicity
    if iproc == ROOT_PROC
        A = mkmat(n)
    else
        A = Matrix{AELT}(undef, n, n)
    end
    MPI.Bcast!(A, ROOT_PROC, comm)
    Aglobal[] = A

    Alocal[] = xmat(istart, nlocal, n)

    fp = _wrap_mv(zero(AELT))
    sump = _wrap_global_sum(zero(AELT))
    bcp = _wrap_bcast(zero(AELT))
    nev = 5
    commp = Base.unsafe_convert(Ptr{MPI.MPI_Comm}, comm)
    println("[$(iproc)] nlocal=$nlocal n=$n")

    evals, evecsLocal, resids, stats = PRIMME.eigs(fp, nev; elt=AELT, n=n,
                                                   numProcs=nproc, procID=iproc,
                                                   nLocal=nlocal,
                                                   globalSumReal = sump,
                                                   broadcastReal = bcp,
                                                   commInfo = commp,
                                                   verbosity=2)

    nconv = length(evals)
    evecs = zeros(AELT, n, nconv)
    for jp in 1:nproc
        if jp == idxProc
            xTmp = copy(evecsLocal)
        else
            xTmp = similar(evecsLocal)
        end
        MPI.Bcast!(xTmp, jp-1, comm)
        i0 = istarts[jp]
        i1 = i0 + counts[jp] - 1
        evecs[i0:i1,:] .= view(xTmp, 1:counts[jp], :)
    end
    MPI.Barrier(comm)

    # check the results
    if iproc == 0
        A = Aglobal[]
        nconv = length(evals)
        res1 = zeros(nconv)
        for i in 1:nconv
            res1[i] = norm(A*evecs[:,i] - evals[i] * evecs[:,i])
        end
        @show evals
        println("computed/reported residuals:")
        display(hcat(res1, resids[1:nconv]))
        println()
    end
end

runme(1000)
