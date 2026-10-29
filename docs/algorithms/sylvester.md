# Sylvester equations

```cpp
#include <mtl/operation/sylvester.hpp>

auto [X, perturbed] = mtl::sylvester(A, B, C);                  // A X + X B = C
auto [Y, p2] = mtl::sylvester(A, B, C, mtl::sylvester_sign::minus);  // A X - X B = C

// A and B already in Schur form: quasi-triangular (schur) or triangular (rsf2csf)
auto [Z, p3] = mtl::sylvester_triangular(TA, TB, F);

// Scaled against overflow, as LAPACK trsyl: TA W + W TB = scale * F, scale <= 1
auto [W, scale, p4] = mtl::sylvester_triangular_scaled(TA, TB, F);
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

## Overflow

A solution can be too large for the element type even when the inputs are
not: `X = C / (a + b)` with `a + b` near roundoff, or a nearly singular
equation in half precision, where `X ≈ C · 10³` overflows once `C` is
about 100. An IEEE type would return infinities, and a posit, which does not
overflow, would saturate at maxpos and return a wrong finite value with no sign
of it.

- **`sylvester_triangular` and `sylvester`** return the true `X` or throw
  `std::overflow_error`. They reject NaN and infinite input with
  `std::invalid_argument`. Overflow then shows in `X` itself, as an infinite or
  NaN entry or an entry at the type's largest value, and is reported. The
  in-house path does not scale, so the small entries of an `X` with a wide
  range are kept.
- **`sylvester_triangular_scaled`** is LAPACK `trsyl`'s contract: it returns
  `{X, scale, perturbed}` with `A X + sgn X B = scale · C` and `0 < scale ≤ 1`.
  The solve scales the right-hand side by powers of two once a block's solution
  would exceed `max · ε / (m n)`. For double and for posits that is trsyl's
  `ε / (safmin m n)`, and it also suits half, whose safe minimum is far from
  `1 / max`. It is for callers that need only the direction or relative size of
  `X`, such as condition estimators, which solve many nearly singular
  equations. A type without `numeric_limits` has no threshold and always gets
  `scale = 1`. Scaling can push entries of `X` much smaller than its largest
  into the subnormals or to zero.

With LAPACK, `trsyl` scales by its own rule in both calls.
`sylvester_triangular` divides its `X` by `scale` when the result is
representable, and throws otherwise.

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
| `float`, `double` | ≤ 0.41 | ≤ 2.7 | |
| `bfloat16` | 0.45 | 1.8 | |
| `cfloat<16,5>` (half) | 0.39 | 1.6 | |
| `cfloat<32,8>` | 0.33 | 2.7 | |
| `posit<16,1>` | 0.38 | 4.7 | one equation; the next worst is 1.8 |
| `posit<32,2>` | 0.26 | 2.1 | |
| `posit<64,3>` | 0.31 | 2.5 | was 29 before reflectors could fall back to the identity; see below |
| `takum<32>` | 0.75 | 4.0 | |
| `lns<32,22>` | 0.35 | 2.2 | |
| `dd` | 0.11 | 3.2 | |
| `qd` | 0.00 | 0.08 | Universal's `qd` epsilon is conservative |

The triangular solver stays below one ε for every type. The larger numbers in
the `sylvester` column come from the Schur decompositions it calls.

The Hessenberg reduction inside `schur` uses the Householder reflectors that
`householder()` also builds for QR and the SVD. Their scalar used to be formed
from the square of the leading entry of `v`. When the subdiagonal tail is small,
for instance 2% of the leading entry in a 3×3 half matrix, that square is about
2⁻²⁴: the bottom of half's subnormal range, and a region where `posit<16,1>`
has only a few fraction bits. The scalar was then wrong by several percent and
`Q` was not orthogonal, which put half at 57 ε and `posit<16,1>` at 215 ε.
The scalar is now computed as `−v₀ / ‖x‖`, which is equal in exact arithmetic
and squares nothing small (#658); the table above was measured with it.

`householder()` also checks the reflector it computed. An exact one has
`β vᵀv = 2`, and a computed one is kept only if its distance from that, which is
how far `H` is from orthogonal, is less than what leaving the column alone
costs, `‖tail‖ / ‖x‖`. Otherwise it returns the identity. In `posit<64,3>`,
Universal's `sqrt` is accurate only to double (stillwater-sc/universal#1846),
so for a tail too small to matter the reflector was further from orthogonal
than the tail was large. That put `posit<64,3>` at 29 ε, and the check brings it
to 2.5 ε. The same check keeps 8-bit and fixed-point types finite: a reflector
they cannot represent becomes the identity, as it did before #658.

## Limits

- **`op(A)` and `op(B)` are `A` and `B`.** The transposed forms of `trsyl`
  (`Aᵀ X + X B = C` and so on) are not offered yet.
- **Saturation in an intermediate sum.** For a posit, a sum of products can
  saturate at maxpos while every entry of `X` stays below it. The overflow check
  sees only `X`. It takes products `|a_ij| · |x_jl|` near maxpos.
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
