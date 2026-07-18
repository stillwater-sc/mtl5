# Eigenvalues and Eigenvectors

MTL5 provides a dense eigen suite in namespace `mtl` (headers under
`include/mtl/operation/`). This page covers the routines available today and how
to choose between them. It is the seed of a fuller guide that will grow as the
iterative and sparse eigensolvers land.

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
