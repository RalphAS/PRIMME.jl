var documenterSearchIndex = {"docs":
[{"location":"mpi/#Distributed-computation","page":"Distributed Computation","title":"Distributed computation","text":"","category":"section"},{"location":"mpi/","page":"Distributed Computation","title":"Distributed Computation","text":"For sufficiently large problems, the runtime is dominated by matrix-vector multiplications (or operator applications). PRIMME allows for these to be distributed over multiple processes using MPI. This package allows for MPI usage, but at a fairly low level. Appropriate usage is problem-specific, but an illustration of the API is included in the examples folder.","category":"page"},{"location":"svds/#Partial-Singular-Value-Decompositions","page":"Partial Singular Value Decompositions","title":"Partial Singular Value Decompositions","text":"","category":"section"},{"location":"svds/","page":"Partial Singular Value Decompositions","title":"Partial Singular Value Decompositions","text":"PRIMME.svds","category":"page"},{"location":"svds/#PRIMME.svds","page":"Partial Singular Value Decompositions","title":"PRIMME.svds","text":"PRIMME.svds(A, k::Integer; kwargs...) => U,S,V,resids,stats\n\ncomputes some (k) singular triplets (partial SVD) of the matrix A Returns S, a vector of k singular values; U, a n×k matrix of left singular vectors; V, a n×k matrix of right singular vectors; resids, a vector of residual norms; and stats, a structure with an account of the work done.\n\nKeyword args\n\n(names in square brackes are analogues in PRIMME documentation)\n\nwhich::Symbol, indicating which triplets should be computed.\n:SR: smallest algebraic\n:SM: closest to sigma (or 0 if not provided)\n:LR: largest magnitude (default)\ncheck::Symbol: (:throw,:warn,:quiet) what to do on convergence failure\ntol\nsigma\nmaxBasisSize\nverbosity::Int\nmethod::SvdsPresetMethod\nmethod1::EigsPresetMethod: algorithm for first stage\nmethod2::EigsPresetMethod: algorithm for seconde stage, if hybrid\nmaxMatvecs\nshifts\nP_AtA: preconditioner\nP_AAt: preconditioner\nP_B: preconditioner\nfullStats::Bool: whether to return stats from eigensolver stage(s)\n(some others may be passed to the constructor for C_svds_params)\n\n\n\n\n\n","category":"function"},{"location":"eigs/#Hermitian-Eigensystems","page":"Hermitian Eigensystems","title":"Hermitian Eigensystems","text":"","category":"section"},{"location":"eigs/","page":"Hermitian Eigensystems","title":"Hermitian Eigensystems","text":"PRIMME.eigs","category":"page"},{"location":"eigs/#PRIMME.eigs","page":"Hermitian Eigensystems","title":"PRIMME.eigs","text":"PRIMME.eigs(A, k::Integer; kwargs...) => Λ,V,resids,stats\n\ncomputes some (k) eigen-pairs of the n×n Hermitian matrix A\n\nReturns Λ, a vector of k eigenvalues; V, a n×k matrix of right eigenvectors; resids, a vector of residual norms; and stats, a structure with an account of the  work done.\n\nKeyword args\n\n(names in square brackes are analogues in PRIMME documentation)\n\nwhich::Symbol, indicating which pairs should be computed. If a target sigma is provided, :SM and :LM refer to distance from the target, otherwise 'sigma' is implicitly zero.\n:SR: smallest algebraic\n:SM: smallest magnitude (default if sigma is provided)\n:LR: largest algebraic\n:LM: largest magnitude (default)\n:CGT: closest greater than or equal to target\n:CLT: closest less than or equal to target\nsigma::Real, a target value; see which above.\ntol: convergence criterion (residual norm relative to estimated norm of A) [eps]\nmaxiter: bound on outer iterations [maxOuterIterations]\nmaxMatvecs: bound on calls to matrix-vector multiplication function\nP: preconditioning matrix or (preferably) factorization\nB: mass matrix\nverbosity::Int: detail of diagnostics to report to standard output [reportLevel]\ncheck::Symbol: (:throw,:warn,:quiet) what to do on convergence failure\n\nSome other keyword arguments, such as those passed to internal functions, might be properly documented someday.\n\n\n\n\n\n","category":"function"},{"location":"#man-primme","page":"Home","title":"PRIMME.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package provides Julia wrappers for the PRIMME library, which implements iterative algorithms to compute some eigenpairs of Hermitian matrices (or linear operators) and some singular value triples of matrices or linear operators. For extreme values it should be competitive with other schemes for moderately large matrices. For interior values it is appropriate for very large matrices where full factorizations for shift-invert methods are impractical. It is more reliable and/or efficient for very small singular values than most competitors, especially when preconditioners are available.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For information on PRIMME itself, please refer to its homepage or Github site.","category":"page"},{"location":"","page":"Home","title":"Home","text":"note: Note\nMuch of the Julia API is not yet properly documented. See the test suite for usage examples.","category":"page"}]
}
