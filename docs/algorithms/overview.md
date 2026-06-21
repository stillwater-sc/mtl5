# Linear Algebra Algorithms

A reference series on the **core algorithms** behind MTL5's solvers — what they
do, why they perform the way they do, and the implementation pitfalls that
determine whether a from-scratch version is competitive.

## Why this section exists

The hard-won knowledge of *why* a numerical algorithm is fast — the data
structures, the asymptotic traps, the constant-factor techniques — tends to
evaporate into a library's source code, where it is invisible to users and to
the next maintainer. Re-deriving it later is expensive and error-prone.

This section captures that knowledge as durable, implementation-independent
documentation: the algorithm, its complexity, *why* a good implementation
reaches that complexity, and the specific weaknesses a naive implementation
exhibits. It is written to outlive any particular version of the code.

Each page is a characterization, not a status page — it describes the algorithm
and the engineering that makes it fast, using well-known reference
implementations (e.g. SuiteSparse) as the performance yardstick. Where MTL5
provides its own implementation, the page links to it and to the engineering
effort tracking its performance, but the characterization stands on its own.

## Pages

- **[Measuring Solver Accuracy](measuring-solver-accuracy.md)** — the residual,
  norms, absolute vs relative error, and backward vs forward error linked by the
  condition number. The foundation for how every solver's correctness is judged.
- **[KLU](klu.md)** — sparse LU for circuit-simulation matrices: BTF +
  block-wise Gilbert–Peierls LU with partial pivoting. A scalar, non-supernodal
  solver, and a case study in where sparse-LU implementations lose performance.

## Planned

Future characterizations for the other core solvers — sparse Cholesky (up/left
looking, supernodal), sparse QR, the fill-reducing orderings (AMD/COLAMD,
nested dissection), and the Krylov methods (CG, GMRES, BiCGSTAB) — following the
same template: algorithm, complexity, why-it-is-fast, implementation pitfalls.
