# Sylvester equations

```cpp
#include <mtl/operation/sylvester.hpp>

auto [X, perturbed] = mtl::sylvester(A, B, C);                  // A X + X B = C
auto [Y, p2] = mtl::sylvester(A, B, C, mtl::sylvester_sign::minus);  // A X - X B = C

// A and B already in Schur form: quasi-triangular (schur) or triangular (rsf2csf)
auto [Z, p3] = mtl::sylvester_triangular(TA, TB, F);
```

The **Sylvester equation** `A X + X B = C` asks for the `m × n` matrix `X`, given
square `A` (`m × m`) and `B` (`n × n`). It is linear in `X`, and has a unique
solution exactly when `A` and `-B` have no eigenvalue in common (with
`sylvester_sign::minus`, `A X - X B = C`, when `A` and `B` have none). Sylvester
studied it in 1884, the same decade he gave the first definition of a general
matrix function (see [Matrix functions](matrix-functions.md)).

It is the step that the Schur methods for matrix functions repeat:

- **`sqrtm` by the Schur method** (Björck & Hammarling 1983, Higham 1987) builds
  the square root of the triangular `T` one block at a time; each off-diagonal
  block is a Sylvester equation in the diagonal blocks.
- **The block Schur–Parlett method** for a general `f(A)` (Davies & Higham 2003)
  joins its diagonal blocks the same way.
- **Fréchet derivatives** of `sqrtm` solve one, and condition estimates for
  matrix functions solve many.

It also appears on its own in control theory: `A X + X Aᵀ = C` is the Lyapunov
equation, whose solution is a system's controllability or observability Gramian.

## Algorithm

`sylvester` is the **Bartels–Stewart** method (1972):

1. Reduce both matrices to real Schur form, `A = Q_A T_A Q_Aᵀ` and
   `B = Q_B T_B Q_Bᵀ`, with [`schur`](eigenvalues.md#schur-decomposition).
2. Transform the right-hand side: `F = Q_Aᵀ C Q_B`.
3. Solve the quasi-triangular equation `T_A Y + Y T_B = F` with
   `sylvester_triangular`.
4. Transform back: `X = Q_A Y Q_Bᵀ`.

`sylvester_triangular` is step 3 on its own, LAPACK's `trsyl` for
`op(A) = A`, `op(B) = B`. Because `T_A` and `T_B` are block upper triangular, `Y`
can be found one diagonal block at a time: block columns of `T_B` from the left,
and within each, block rows of `T_A` from the bottom. Each block `Y_kl` solves

```text
A_kk Y_kl + sgn Y_kl B_ll = F_kl - Σ_j A_kj Y_jl - sgn Σ_j Y_kj B_jl
```

where `A_kk` and `B_ll` are 1×1 or 2×2 diagonal blocks, so there are at most four
unknowns. That small system is solved by Gaussian elimination with complete
pivoting, as LAPACK's `dlasy2` does.

The same code handles a complex upper triangular `T_A` and `T_B`, from
[`rsf2csf`](eigenvalues.md#schur-decomposition), where every block is 1×1. A 2×2
block need not be in LAPACK's standard form; any upper quasi-triangular `A` and
`B` are accepted, and anything else is rejected with `std::invalid_argument`.

## Singular and nearly singular equations

When an eigenvalue of `A` is within roundoff of one of `-B`, the equation is
singular or nearly so. As in `trsyl`, a pivot no larger than

```text
smin = max(ε · max(max|a_ij|, max|b_ij|), smlnum)
```

is replaced by `smin`, and the result's `perturbed` flag is set. `X` then solves a
nearby equation; it is finite, but it may be large and has no accuracy guarantee.
`trsyl` reports the same condition as `info = 1`. `smlnum` is LAPACK's
safe-minimum floor, used only when `numeric_limits` gives one below ε, as in
`schur`.

## Accumulator

The sums on the right-hand side of each block are dot products. They are
accumulated in `Accumulator` (leading template argument, default `void`: plain
arithmetic in the element type), so `sylvester_triangular<double>(A, B, C)` on
`float` data sums in double, and a quire makes each sum exact. `sylvester`
forwards the accumulator to both Schur decompositions, to `sylvester_triangular`
and to the four matrix products.

## LAPACK dispatch

With `-DMTL5_WITH_LAPACK=ON` and the default accumulator,
`sylvester_triangular` calls `strsyl`/`dtrsyl` for dense `float`/`double`
matrices, and `ctrsyl`/`ztrsyl` for dense `std::complex<float/double>` matrices
whose `A` and `B` are triangular. The complex routines read only the upper
triangle, so a complex `A` or `B` with a 2×2 block takes the in-house path.
`sylvester` dispatches through `schur` (`gees`), `trsyl`, and `mult` (BLAS
`gemm` when BLAS is enabled).

## Accuracy

The measure is the relative residual

```text
‖A X + sgn X B − C‖_F / ((‖A‖_F + ‖B‖_F) ‖X‖_F + ‖C‖_F)
```

in units of each type's ε. Bartels–Stewart keeps it at a small multiple of ε
(Higham, *Accuracy and Stability of Numerical Algorithms*, chapter 16). The forward
error also depends on the separation of the two spectra and can be much larger.

The table was measured on random `A` and `B`, shifted so that the two spectra are
apart, with `m ∈ {3, 6, 10}`, `n ∈ {2, 5, 9}`, both signs and three repetitions:
54 equations per type. The residual is evaluated in the element type itself.
The Universal rows come from a one-off program built against the Universal
library; MTL5 does not depend on Universal, so they are not part of the test
suite.

| Type | `sylvester_triangular` | `sylvester` | Note |
|---|---|---|---|
| `float`, `double` | ≤ 0.41 | ≤ 2.1 | |
| `bfloat16` | 0.45 | 1.6 | |
| `cfloat<16,5>` (half) | 0.37 | 57 | one equation; see below |
| `cfloat<32,8>` | 0.31 | 1.7 | |
| `posit<16,1>` | 0.41 | 215 | two equations; see below |
| `posit<32,2>` | 0.30 | 6.1 | see below |
| `posit<64,3>` | 0.28 | 24 | `sqrt` is accurate only to double (stillwater-sc/universal#1846) |
| `takum<32>` | 0.44 | 2.7 | |
| `lns<32,22>` | 0.35 | 2.6 | |
| `dd` | 0.10 | 2.7 | |
| `qd` | 0.00 | 0.08 | Universal's `qd` epsilon is conservative |

The triangular solver is below half an ε for every type. The larger numbers in
the `sylvester` column come from the Schur decompositions it calls.

The half and `posit<16,1>` outliers come from the Householder reflectors of the
Hessenberg reduction, which `householder()` builds for QR and the SVD as well.
The worst case is a 3×3 `A` whose first column has a subdiagonal tail 2% the
size of its leading entry. The reflector's scalar is then formed from the square
of a value near 2⁻¹², about 2⁻²⁴. That is the bottom of half's subnormal range,
and `posit<16,1>` has only a few fraction bits there, so the scalar is wrong by
several percent and `Q` is not orthogonal. Computing the same scalar as
`−v₀ / ‖x‖`, which is equal in exact arithmetic and squares nothing small,
brings them to 1.6 ε for half, 4.7 ε for `posit<16,1>` (whose second-worst
equation, at 40 ε, has the same cause) and 2.1 ε for `posit<32,2>`. Every other
half equation is at most 1.6 ε.

## Limits

- **`op(A)` and `op(B)` are `A` and `B`.** The transposed forms of `trsyl`
  (`Aᵀ X + X B = C` and so on) are not offered yet.
- **No overflow guard.** LAPACK's `trsyl` returns a scale factor `≤ 1` and
  solves for `scale · X` when `X` would overflow. Here `X` is returned unscaled,
  so a solution beyond the type's range overflows.
- **`sylvester` takes real matrices.** For complex `A` and `B` it needs the
  complex Schur form of a complex matrix (issue #650); until then, reduce them
  yourself and call `sylvester_triangular` on the triangular factors.
- **Fixed-point types are not supported**, for the same reason as in `schur`:
  their precision is absolute, not relative.

## References

- R. H. Bartels and G. W. Stewart, "Algorithm 432: Solution of the matrix
  equation AX + XB = C", *Communications of the ACM* 15(9), 1972.
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM,
  2002, chapter 16.
- N. J. Higham, *Functions of Matrices: Theory and Computation*, SIAM, 2008,
  chapters 6 (the square root) and 9 (Schur–Parlett).
- P. I. Davies and N. J. Higham, "A Schur–Parlett algorithm for computing matrix
  functions", *SIAM J. Matrix Anal. Appl.* 25(2), 2003.
- LAPACK `xTRSYL` and `xLASY2`.
