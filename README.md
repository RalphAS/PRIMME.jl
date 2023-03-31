# PRIMME.jl
![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
[![GitHub CI Build Status](https://github.com/RalphAS/PRIMME.jl/workflows/CI/badge.svg)](https://github.com/RalphAS/PRIMME.jl/actions)
[![codecov.io](http://codecov.io/github/RalphAS/PRIMME.jl/coverage.svg?branch=master)](http://codecov.io/github/RalphAS/PRIMME.jl?branch=master)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://RalphAS.github.io/PRIMME.jl/dev)

# Introduction

This package provides Julia wrappers for `PRIMME`, a library which
implements iterative solvers for large scale Hermitian eigenproblems
and singular value decompositions (SVD).

For information on `PRIMME` itself, please refer to its
[homepage](https://www.cs.wm.edu/~andreas/software/)
or [Github site](https://github.com/primme/primme.git).

## Status

Basic functionality (`eigs`, `svds`) is implemented. Documentation is sparse.
Preconditioners and mass matrices are not yet handled as flexibly as they should be.

# Usage

The Julia API largely resembles that of the Python and MATLAB versions in the
official PRIMME distribution. For the time being, users are referred to the associated
documentation for many details. Docstrings for `eigs` and `svds` will emphasise the
peculiarities of the Julia wrappers, and the test suite should provide useful examples.

## Eigen-pairs

As in some other iterative eigensolvers, a symbol is used to select which pairs are wanted,
e.g. `:LR` (`:SR`) for largest (smallest) real value.

If `A` is a matrix and `k` is the number of desired pairs, a typical call is
```julia
w, V, resids, stats = PRIMME.eigs(A, k, which=:LM)
```

## Singular-triples (partial SVD)

If `A` is a matrix and `k` is the number of desired triples, a typical call is
```julia
U, s, V, resids, stats = PRIMME.svds(A, k, which=:LM)
```

# Installation

Until this package has been registered, it must be added to your depot verbosely:

`]add https://github.com/RalphAS/PRIMME.jl.git`

For the time being, PRIMME_jll is a formal dependency; this may change if the author
learns of real-world use cases where it is inadequate.

# References
The algorithms are described in
[A.Stathopoulos and J.McCombs, ACM ToMS 37, 21 (2010)](https://doi.org/10.1145/1731022.1731031)
and [L.Wu et al., SIAM J.Sci.Comput. 39, S248 (2017)](https://doi.org/10.1137/16M1082214).

# Acknowledgements

Although this wrapper package is currently developed separately from the PRIMME
team, it obviously depends on their outstanding work.  No endorsement of this package
by them is implied.

This package builds on substantial earlier work by Andreas Noack.
