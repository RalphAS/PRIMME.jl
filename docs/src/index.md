# [PRIMME.jl](@id man-primme)

This package provides Julia wrappers for the
[PRIMME](https://www.cs.wm.edu/~andreas/software/) library,
which implements iterative algorithms to compute
some eigenpairs of Hermitian matrices (or linear operators) and some singular value triples
of matrices or linear operators. For extreme values it should be competitive with
other schemes for moderately large matrices. For interior values it is appropriate
for very large matrices where full factorizations for shift-invert methods are impractical.
It is more reliable and/or efficient for very small singular values than most competitors,
especially when preconditioners are available.

For information on `PRIMME` itself, please refer to its
[homepage](https://www.cs.wm.edu/~andreas/software/)
or [Github site](https://github.com/primme/primme.git).

!!! note
	Much of the Julia API is not yet properly documented. See the test suite for usage examples.
