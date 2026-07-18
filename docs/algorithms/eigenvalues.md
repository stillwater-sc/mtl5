# Eigenvalues and Eigenvectors

MTL5 provides a **dense** eigen suite in namespace `mtl` (headers under
`include/mtl/operation/`) and a set of **matrix-free iterative** eigensolvers in
namespace `mtl::itl` (headers under `include/mtl/itl/eigen/`). This page covers
the routines available today and how to choose between them.

Use the dense suite for small/medium matrices where you want the whole spectrum;
use the iterative solvers for large or matrix-free operators where you want a few
eigenpairs and can only apply `A * x`.

## Which routine do I call?

| Function | Matrix | Output | Backend |
|---|---|---|---|
| `eigenvalue(A)` | general (non-symmetric) | eigenvalues only, `dense_vector<complex>` | LAPACK `geev` if available, else C++ Francis QR |
| `eigen(A)` | general (non-symmetric) | eigenvalues **and** right eigenvectors (`complex`) | LAPACK `geev` if available, else C++ QR + inverse iteration |
| `eigenvalue_symmetric(A)` | symmetric | eigenvalues, ascending | LAPACK `syev` if available, else C++ |
| `eigenvalue_symmetric_generic(A)` | symmetric | eigenvalues, ascending | pure C++ |
| `eigen_symmetric(A)` | symmetric | eigenvalues **and** eigenvectors | pure C++ |

The LAPACK `geev` fast path for the general problem engages when built with
`-DMTL5_WITH_LAPACK=ON` and the matrix is a **column-major** `dense2D<float/double>`
(mirroring the symmetric `syev` dispatch). Row-major matrices and custom number
types use the in-house path, which is always available and numerically identical
up to sign/phase conventions.

All routines take an optional `tol` and `max_iter`; the pure-C++ paths also work
with custom number types (posits, LNS, ...) since they do not depend on LAPACK.

## General eigenvalues and eigenvectors

```cpp
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/eigenvalue.hpp>

mtl::mat::dense2D<double> A(4, 4);
// ... fill A ...

// Eigenvalues only.
auto lambda = mtl::eigenvalue(A);            // dense_vector<complex<double>>

// Eigenvalues and right eigenvectors.
auto [eigs, V] = mtl::eigen(A);              // structured binding
// Column k of V is the right eigenvector for eigs(k): A * V(:,k) = eigs(k) * V(:,k)
```

`eigen` returns a struct `{ eigenvalues, eigenvectors }` (structured-bindable),
mirroring `eigen_symmetric`. Eigenvalue and eigenvector are paired by column.
Eigenvectors are returned with unit 2-norm and a canonical phase (the
largest-magnitude entry is rotated to be real and positive), so results are
deterministic. Real and complex-conjugate eigenpairs are handled uniformly in
complex arithmetic.

### Algorithm

`eigen` computes the spectrum with the tested general QR path (`eigenvalue`,
Hessenberg reduction followed by Francis QR iteration), then recovers each
eigenvector by **inverse iteration** on `A - lambda_k * I`. A partial-pivot
complex LU with a pivot floor keeps the (near-singular) system solvable — which
is exactly the regime inverse iteration exploits.

### Accuracy note

`eigen`'s accuracy is bounded by the eigenvalues that `eigenvalue` produces:
inverse iteration recovers the eigenvector of the true eigenvalue nearest each
computed `lambda_k`. `eigenvalue` uses a **Francis double-shift** QR, so complex
spectra of strongly non-normal matrices (e.g. companion / Forsythe matrices) are
resolved accurately, and the eigenvectors follow to near machine precision. A
LAPACK `geev` acceleration for the general eigenproblem is tracked separately.

## Symmetric problems

For symmetric matrices, prefer the symmetric routines — they exploit real
spectra and orthogonal eigenvectors, and `eigenvalue_symmetric` transparently
dispatches to LAPACK `syev` when built with `-DMTL5_WITH_LAPACK=ON` and the
matrix is a column-major `dense2D<float/double>`.

```cpp
#include <mtl/operation/eigenvalue_symmetric.hpp>

auto s = mtl::eigenvalue_symmetric(S);       // ascending eigenvalues
auto [eigs, Q] = mtl::eigen_symmetric(S);     // A = Q * diag(eigs) * Q^T
```

## Iterative (matrix-free) eigensolvers

For large matrices where a full dense factorization is infeasible, use the
Krylov eigensolvers in `mtl::itl`. They operate through the `LinearOperator`
concept — only `A * x` is required — so they apply to `dense2D`, `compressed2D`,
and user-supplied matrix-free operators alike.

| Function | Operator | Returns |
|---|---|---|
| `itl::power_iteration(A, v0)` | any (real dominant) | dominant eigenpair (`eigenpair<T>`) |
| `itl::lanczos(A, v0, k, which)` | **symmetric** | k extremal Ritz pairs (`ritz_pairs<T>`) |
| `itl::arnoldi(A, v0, k, which)` | general | k Ritz pairs (`ritz_pairs<complex<T>>`) |

`which` is an `itl::eigen_which` selector — `largest_magnitude`,
`smallest_magnitude`, `largest_algebraic`, `smallest_algebraic`. The optional
`subspace` argument sets the Krylov dimension (default: a modest multiple of `k`,
capped at `n`); a larger subspace converges more eigenpairs to tighter residuals.
Each solver projects onto its Krylov subspace and solves the small projected
problem (tridiagonal for Lanczos, Hessenberg for Arnoldi) with the dense
routines above — so accuracy of the projected solve rides on the same code.

```cpp
#include <mtl/itl/eigen/eigensolvers.hpp>

// Dominant eigenpair of a symmetric/SPD operator.
auto p = mtl::itl::power_iteration(A, v0);
// p.value, p.vector, p.converged

// The 5 largest eigenpairs of a large symmetric operator (matrix-free ok).
auto r = mtl::itl::lanczos(A, v0, 5, mtl::itl::eigen_which::largest_algebraic);
// r.values (dense_vector), r.vectors (columns), r.converged

// A few eigenpairs of a nonsymmetric operator (complex Ritz values).
auto g = mtl::itl::arnoldi(A, v0, 4, mtl::itl::eigen_which::largest_magnitude);
```

These are single-shot (non-restarted) builds with full reorthogonalization —
robust and simple. Implicit/thick restarting is a planned follow-up.

## Sparse eigensolver (shift-invert)

For sparse matrices (`compressed2D`), `mtl::sparse` builds on the iterative
solvers above:

| Function | Returns |
|---|---|
| `sparse::sparse_eigs(A, k, which)` | k eigenpairs by Arnoldi applied directly to the sparse operator (best for largest-magnitude) |
| `sparse::sparse_eigs_shift_invert(A, sigma, k)` | k eigenpairs **nearest `sigma`** (interior / smallest) |

Largest-magnitude eigenvalues come from running Arnoldi directly on `A` (a sparse
matrix is already a `LinearOperator`). Interior or smallest eigenvalues use
**shift-invert**: `(A - sigma*I)` is factored once with the sparse LU direct
solver, and its inverse is applied inside Arnoldi. An eigenpair `(theta, y)` of
`(A - sigma*I)^{-1}` maps to `(lambda, y)` of `A` with `lambda = sigma + 1/theta`
and the *same* eigenvector, so "nearest `sigma`" becomes "largest `|theta|`" —
which Arnoldi converges to quickly.

```cpp
#include <mtl/sparse/eigen/shift_invert.hpp>

// The 4 eigenvalues of A closest to sigma = 2.5, with eigenvectors.
auto near = mtl::sparse::sparse_eigs_shift_invert(A, 2.5, 4);
// near.values (complex, = sigma + 1/theta), near.vectors, near.converged

// The reusable operator, if you want to drive the Krylov solver yourself:
mtl::sparse::shift_invert_operator<double> op(A, 2.5);   // factor once
auto y = op * x;                                         // apply (A - 2.5 I)^{-1}
```

The factorization is computed once and applied on every Arnoldi step; tiny
pivots are perturbed so a shift landing (near-)exactly on an eigenvalue stays
solvable. Generalized problems `A x = lambda B x` are a future extension.
