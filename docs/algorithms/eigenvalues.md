# Eigenvalues and Eigenvectors

MTL5 provides a **dense** eigen suite in namespace `mtl` (headers under
`include/mtl/operation/`) and a set of **matrix-free iterative** eigensolvers in
namespace `mtl::itl` (headers under `include/mtl/itl/eigen/`). This page covers
the routines available today and how to choose between them.

Use the dense suite for small/medium matrices where you want the whole spectrum;
use the iterative solvers for large or matrix-free operators where you want a few
eigenpairs and can only apply `A * x`.

## Choosing a solver

Answer four questions:

1. **Dense or sparse/large?** Small/medium and you want all (or most)
   eigenvalues → the **dense suite** below. Large, sparse, or matrix-free and you
   want a few → the **iterative** (`mtl::itl`) or **sparse** (`mtl::sparse`)
   solvers.
2. **Symmetric or general?** Symmetric → the `*_symmetric` dense routines or
   `itl::lanczos`; general (non-symmetric) → `eigenvalue`/`eigen` or
   `itl::arnoldi` (expect complex eigenvalues).
3. **Values only, or values + vectors?** Value-only routines are cheaper; the
   `eigen*` routines and the iterative solvers also return eigenvectors.
4. **Which eigenvalues?** All → dense. Largest → iterative directly on the
   operator. Smallest / interior / nearest a target `sigma` →
   `sparse::sparse_eigs_shift_invert`.

## Which routine do I call?

| Function | Matrix | Output | Backend |
|---|---|---|---|
| `eigenvalue(A)` | general (non-symmetric) | eigenvalues only, `dense_vector<complex>` | LAPACK `geev` if available, else C++ Francis QR |
| `eigen(A)` | general (non-symmetric) | eigenvalues **and** right eigenvectors (`complex`) | LAPACK `geev` if available, else C++ QR + inverse iteration |
| `eigenvalue_symmetric(A)` | symmetric | eigenvalues, ascending | LAPACK `syev` if available, else C++ |
| `eigenvalue_symmetric_generic(A)` | symmetric | eigenvalues, ascending | pure C++ |
| `eigen_symmetric(A)` | symmetric | eigenvalues **and** eigenvectors | pure C++ |
| `eigen_symmetric_jacobi(A)` | symmetric, real, small | eigenvalues **and** eigenvectors, with small eigenvalues relatively accurate | pure C++ |

The LAPACK `geev` fast path for the general problem engages when built with
`-DMTL5_WITH_LAPACK=ON` and the matrix is a **column-major** `dense2D<float/double>`
(mirroring the symmetric `syev` dispatch). Row-major matrices and custom number
types use the in-house path, which is always available and numerically identical
up to sign/phase conventions.

All routines take an optional `tol` and `max_iter`; the pure-C++ paths also work
with custom number types (posits, LNS, ...) since they do not depend on LAPACK.

For the symmetric routines, `tol` defaults to the machine epsilon of the
eigenvalue type (#565). A subdiagonal entry e between diagonal entries d₀ and d₁ is
dropped once |e| ≤ tol·√|d₀|·√|d₁|, LAPACK `dsteqr`'s test, which scales with the
smaller of the two eigenvalues it separates. A larger `tol` trades accuracy for
fewer iterations. Passing `1e-10` gives the earlier default tolerance, though not
the earlier results, since the deflation test also changed.

Tridiagonal QR is accurate to about ε‖A‖ in every eigenvalue, so an eigenvalue much
smaller than ‖A‖ carries a large relative error, and so does its eigenvector when
the gap is small. A normal matrix AᵀA with badly scaled columns is the common case:
measured on 9 × 9 and 12 × 12 null-vector problems with columns scaled across 10⁶,
the eigenvector of the smallest eigenvalue is off by about 1e-6 rad, while with
unscaled columns it is accurate to 1e-15.

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
resolved accurately, and the eigenvectors follow to near machine precision. When
LAPACK is available and the matrix qualifies, both routines dispatch to `geev`
instead (see the table above).

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

`eigenvalue_symmetric_generic` is the LAPACK-free reference path (Householder
tridiagonalization + implicit-QR with Wilkinson shifts); `eigenvalue_symmetric`
calls it when LAPACK is unavailable or the type does not qualify. Call it
directly to exercise the pure-C++ algorithm regardless of build flags, or for
custom number types.

### Small matrices and small eigenvalues: Jacobi

`eigen_symmetric_jacobi(A)` is a two-sided cyclic Jacobi solver (#564). It returns
the same `{eigenvalues, eigenvectors}` as `eigen_symmetric`, eigenvalues ascending:

```cpp
#include <mtl/operation/eigenvalue_symmetric_jacobi.hpp>

auto [eigs, V] = mtl::eigen_symmetric_jacobi(S);   // A = V * diag(eigs) * V^T
```

Tridiagonal QR is accurate to about ε‖A‖ in every eigenvalue. An eigenvalue much
smaller than ‖A‖ therefore has a large relative error, and so does its eigenvector
when the gap to the next eigenvalue is small. For a positive (semi)definite matrix,
Jacobi computes small eigenvalues and their eigenvectors to high **relative**
accuracy (Demmel & Veselić, 1992). The typical case is the null vector of a normal
matrix AᵀA whose columns are badly scaled, as in the 8-point algorithm, DLT pose
estimation, and plane or line fits. Measured on 200 random null-vector problems per
size, with the columns scaled across 10⁶:

| N | `eigen_symmetric` | `eigen_symmetric_jacobi` |
|---|---|---|
| 9 | 6.6e-7 rad | 4.9e-15 rad |
| 12 | 1.9e-6 rad | 5.4e-15 rad |

On graded positive definite matrices with eigenvalues spanning 8 to 24 decades,
Jacobi's worst relative eigenvalue error was 6.6e-16. `eigen_symmetric`'s was
between 2e-13 and 8e-11.

How it works:

- It sweeps the off-diagonal pairs row by row and rotates each one to zero with
  the stable rotation θ = (a_qq − a_pp)/(2a_pq), t = sign(θ)/(|θ| + √(θ² + 1)).
- It skips a pair once |a_pq| ≤ tol·√|a_pp|·√|a_qq|, and stops after a sweep that
  rotates nothing. `tol` defaults to machine ε, and `max_sweeps` caps the sweeps
  (default 60).
- The per-pair test is what keeps the relative accuracy. A stop on
  off-diagonal mass ≤ ε²‖A‖²_F can leave a tiny entry that couples two small
  eigenvalues: diag(1, 1e-10, 2e-10) with a₁₂ = 1.5e-16 would come back
  2.25e-12 relatively wrong.
- No entry is ever squared, so matrices scaled from 1e+300 down to subnormal
  values converge.
- Random normal matrices of size 9 and 12 converge in under 20 sweeps.

Jacobi costs more than tridiagonal QR. The `eig-small` suite of
`benchmarks/bench_all` (`bench_all --suite eig-small`) measured the following
medians in µs, from two runs of a Release build on a shared x86-64 machine, so the
ranges are noisy:

| n | `eigen_symmetric` | `eigen_symmetric_jacobi` | `svd` |
|---|---|---|---|
| 3 | 0.55 | 0.59 | 0.99 |
| 6 | 1.1–1.8 | 1.7–3.3 | 6.3–6.5 |
| 9 | 4.1–5.1 | 11–12 | 14–23 |
| 16 | 7.7–8.0 | 29–31 | 69–82 |

Use Jacobi for small matrices whose small eigenvalues or null vectors matter, and
`eigen_symmetric` (or LAPACK) for larger matrices and when only absolute accuracy
is needed. It takes real element types only, custom reals such as posits
included; a complex Hermitian matrix is rejected at compile time.

### Null vectors: `null_vector` and the V-only `svd`

A null-space problem needs only the right singular vector of A's smallest
singular value, which is the last column of V. The full `svd(A, U, S, V)` also
builds an m × m U. At m = 400 that makes it take about 0.24 s, almost all of it
spent completing U to a full basis. Two calls skip U entirely (#566):

```cpp
#include <mtl/operation/svd.hpp>

mtl::vec::dense_vector<double> s;     // singular values, descending, length min(m, n)
mtl::mat::dense2D<double> V;          // n x n right singular vectors
mtl::svd(A, s, V);                    // never forms U

auto x = mtl::null_vector(A);         // unit x minimizing ||A x||: V's last column
```

- **Tall A (m > n):** A is first reduced by Householder QR, A = QR, and one-sided
  Jacobi then runs on the n × n triangle R. R has the same singular values and
  right singular vectors as A, so each sweep costs O(n³) instead of O(mn²).
- **Accuracy:** Householder QR is backward stable column by column, so it keeps
  Jacobi's relative accuracy when A's columns are badly scaled. Measured against
  the full `svd` on 10 × 3 up to 400 × 12 matrices, with rows or columns scaled
  across 10⁸, the singular values agree to 1.3e-15 relative and the columns of V
  to the same.
- **Square or wide A (m ≤ n):** Jacobi runs on A itself, and S and V match the
  full `svd` bit for bit.
- **Backend:** both calls always use the generic C++ path. LAPACK's `gesdd`, which
  the full `svd` uses, has no V-only mode.

A null vector of a tall 2k × 9 or 2k × 12 system can be found four ways. The
`null-vector` suite of `benchmarks/bench_all` measured these medians in µs, in a
Release build. The two AᵀA columns include the time to form AᵀA.

| m × N | `null_vector(A)` | `svd(A)` | `eigen_symmetric(AᵀA)` | `eigen_symmetric_jacobi(AᵀA)` |
|---|---|---|---|---|
| 16 × 9 | 6.6 | 13 | 3.5 | 6.6 |
| 64 × 9 | 9.7 | 448 | 4.8 | 9.1 |
| 400 × 9 | 27 | 238,590 | 15 | 18 |
| 16 × 12 | 13 | 18 | 6.5 | 15 |
| 400 × 12 | 46 | 238,824 | 27 | 36 |

The full `svd`'s time at m = 400 is almost all spent completing U to a full
m × m basis. At m = 400, `null_vector` spends most of its time in the QR.

The AᵀA routes are faster, but forming AᵀA squares A's condition number. The
null vector's error then grows with 1/gap² instead of 1/gap, where the gap is the
second-smallest singular value relative to the largest. Measured in `float` on
36 × 9 matrices with an exact null vector (50 trials each), as the mean angle:

| second-smallest σ | `null_vector(A)` | `eigen_symmetric_jacobi(AᵀA)` |
|---|---|---|
| 1e-1 | 2.4e-7 | 1.6e-6 |
| 1e-2 | 1.3e-6 | 1.9e-4 |
| 1e-3 | 1.3e-5 | 2.0e-2 |

In `double` the same loss is ε·κ², so it rarely matters for these sizes, but it
does in `float` or in a narrow custom type such as a posit.

### Thin SVD: `svd(A, U, s, V)`

When the left singular vectors are needed too, as in least squares, a
pseudoinverse or PCA, a tall A rarely needs all m of them. The thin overload
returns k = min(m, n) of each:

```cpp
mtl::mat::dense2D<double> U, V;       // U: m x k, V: n x k, orthonormal columns
mtl::vec::dense_vector<double> s;     // k singular values, descending
mtl::svd(A, U, s, V);                 // A = U * diag(s) * V^T
```

- **Tall A (m > n), generic path:** the same QR and Jacobi on R as the V-only
  `svd`, then U = Q [U_R; 0], formed by applying the n stored Householder
  reflectors to R's left singular vectors. That is O(mn²), with no m × m basis to
  complete.
- **Square or wide A (m ≤ n):** U, s and the k columns of V match the first k of
  the full `svd` bit for bit.
- **Zero singular values:** U's columns for them are completed to an orthonormal
  set, as in the full `svd`.
- **Backend:** with LAPACK, a `dense2D<float/double>` goes to `gesdd` in its thin
  mode, as the full `svd` goes to `gesdd` in its full mode.

Measured with the `null-vector` suite, in µs:

| m × n | thin `svd` | full `svd` |
|---|---|---|
| 64 × 9 | 13 | 459 |
| 400 × 9 | 49 | 238,000 |
| 400 × 12 | 85 | 238,000 |

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

## Worked examples

The `examples/phase06_eigenvalue_svd/` directory has runnable programs:

- `vibrating_string.cpp` — eigenvalues of discrete 1D/2D Laplacians as vibration
  modes; uses `eigenvalue_symmetric` and contrasts it with the general
  `eigenvalue` on a non-symmetric perturbation.
- `pca_svd.cpp` — relates the SVD to the symmetric eigenproblem (`singular
  values = |eigenvalues|` for a symmetric matrix).

## Custom number types

Every pure-C++ path (`eigenvalue`, `eigen`, `eigenvalue_symmetric_generic`,
`eigen_symmetric`, and the iterative solvers) works with custom scalar types
such as posits and LNS — they depend only on standard arithmetic and `mtl`
primitives, not on LAPACK. LAPACK dispatch is a `float`/`double`-only
accelerator that is bypassed automatically for other types.
