## Distributed computation

For sufficiently large problems, the runtime is dominated by matrix-vector multiplications
(or operator applications). `PRIMME` allows for these to be distributed over multiple
processes using MPI. This package allows for MPI usage, but at a fairly low level.
Appropriate usage is problem-specific, but an illustration of the API
is included in the `examples` folder.
