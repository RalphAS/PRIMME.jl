# PRIMME.jl
[![GitHub CI Build Status](https://github.com/RalphAS/PRIMME.jl/workflows/CI/badge.svg)](https://github.com/RalphAS/PRIMME.jl/actions)

# Introduction

This package provides Julia wrappers for routines in the PRIMME library, which
implements state-of-the-art iterative solvers for large scale Hermitian eigenproblems
and singular value decompositions (SVD).

PRIMME is hosted at https://github.com/primme/primme.git

# Usage

For the most part the Julia API resembles that of the Python and MATLAB versions in the
official PRIMME distribution. Users are referred to the associated documentation for
details. Docstrings for `eigs` and `svds` emphasise the peculiarities of the Julia
wrappers.

## Eigen-pairs

As in some other iterative eigensolvers, a symbol is used to select which pairs are wanted,
e.g. `:LM` for largest magnitude.

If `A` is a matrix and `k` is the number of desired pairs, a typical call is
```julia
w, V, resids, stats = PRIMME.eigs(A, k, which=:LM)
```

## Singular-triples (partial SVD)

If `A` is a matrix and `k` is the number of desired pairs, a typical call is
```julia
U, s, V, resids, stats = PRIMME.svds(A, k, which=:LM)
```

# Installation

Until this package has been registered, it must be added to your depot verbosely:

`]add https://github.com/RalphAS/PRIMME.jl.git`

One will (obviously) need a compatibly built version of the PRIMME library, currently
version v3.2. With luck, you can get one with this Pkg command:

`]add https://github.com/RalphAS/PRIMME_jll.jl.git`

For the time being, PRIMME_jll is a formal dependency; this may change if the author
learns of real-world use cases where it is inadequate.

# Acknowledgements

Although this package is currently separate from the PRIMME development team, it obviously
depends on their outstanding work.  No endorsement of this package by them is implied.

This package builds on substantial earlier work by Andreas Noack.