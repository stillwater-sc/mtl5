# Matrix Functions and the Matrix Exponential

A *matrix function* applies a scalar function such as exp, log or the square
root to a matrix as a whole, not to each of its entries. MTL5 has the first of
these, `expm(A)`, the matrix exponential. This page covers four things:

- what a matrix function is;
- why the matrix exponential matters;
- how MTL5 computes it, and how the computation adapts to the precision of the
  element type;
- how the subject developed, from Cayley's 1858 memoir to the algorithm used
  in MATLAB and SciPy today.

The square root and the logarithm come later. [Issue #650](https://github.com/stillwater-sc/mtl5/issues/650)
has the plan.

## What a matrix function is

For a scalar function given by a power series, the matrix function substitutes
the matrix:

```
exp(A) = I + A + A^2/2! + A^3/3! + ...
```

That definition only covers power series. The general one goes through the
eigenvalues. For a diagonalizable `A = V diag(λ_i) V^-1`, `f(A) = V diag(f(λ_i)) V^-1`.
The Jordan canonical form, Hermite interpolation or a Cauchy integral extend
this to every matrix, and all three definitions agree. Higham (2008), Chapter 1,
sets this out.

Some consequences of the definition:

- **Not element-wise.** `mtl::exp(A)` in `operation/exp.hpp` returns the matrix
  of `e^(a_ij)`. `mtl::expm(A)` is the matrix exponential. The two coincide only
  for a diagonal matrix. The names follow MATLAB and SciPy (`exp` and `expm`).
- **Eigenvalues map through f.** The eigenvalues of `f(A)` are `f(λ_i)`. A
  similarity carries through, `f(X A X^-1) = X f(A) X^-1`, and `f(A)` commutes
  with `A`.
- **Products don't factor.** `exp(A + B) = exp(A) exp(B)` only when `A` and `B`
  commute. The Baker–Campbell–Hausdorff formula gives the correction in terms of
  commutators. That correction is what makes the exponential the bridge between
  a Lie algebra and its Lie group.
- **Multivalued functions need a branch.** The square root and the logarithm
  have more than one value. A library returns the *principal* one, which exists
  when no eigenvalue lies on the closed negative real axis.

## Why the matrix exponential matters

- **Linear differential equations.** `x'(t) = A x(t)` has the solution
  `x(t) = exp(tA) x(0)`. The exponential is the propagator of every linear
  constant-coefficient system, and exponential integrators build on it for
  nonlinear and stiff problems.
- **Control.** Discretizing a continuous-time system `x' = Ax + Bu` with a
  zero-order hold over a step `h` gives `A_d = exp(hA)`. `B_d` comes from the
  exponential of a block matrix, `exp(h [A B; 0 0])`.
- **Markov chains.** A continuous-time chain with generator `Q` has transition
  matrix `P(t) = exp(tQ)`.
- **Networks.** Communicability and centrality measures are entries of
  `exp(A)` for an adjacency matrix `A`.
- **Lie groups.** The exponential map from the Lie algebra to a matrix Lie group
  (rotations SO(3), rigid motions SE(3), and the rest) is `expm`. Robotics and
  geometry code builds on it. [The assessment of quaternions and Lie algebras](https://github.com/stillwater-sc/mtl5/blob/main/docs/other/quaternions-lie-algebras-matrix-functions.md)
  explains why MTL5 provides the general function rather than group-specific
  classes.

## How MTL5 computes it

### Why the obvious methods fail

Moler and Van Loan's "Nineteen dubious ways to compute the exponential of a
matrix" (1978, revisited in 2003) is the standard survey, and its title is the
point: most methods that look reasonable fail on some matrices.

- **Summing the Taylor series** suffers catastrophic cancellation when `A` has
  large negative eigenvalues. Even for a scalar, summing the series for `e^-x`
  in double gives:

  | x | Taylor sum | e^-x | relative error |
  |---|---|---|---|
  | 10 | 4.539992967e-5 | 4.539992976e-5 | 2.0e-9 |
  | 20 | 6.15e-9 | 2.06e-9 | 2.0 |
  | 30 | 6.10e-6 | 9.36e-14 | 6.5e7 |

  The terms grow to about `x^x / x!` before they shrink, and their rounding
  errors swamp the result.
- **Diagonalizing,** `V diag(e^λ_i) V^-1`, is exact in exact arithmetic. In
  floating point the error grows with the condition number of `V`, which is
  unbounded for a nearly defective matrix and undefined for a defective one.
- **ODE solvers** are expensive and give no guarantee about the error in `exp(A)`.

### Scaling and squaring

The method in MATLAB, SciPy and now MTL5 rests on the identity

```
exp(A) = exp(2^-s A)^(2^s).
```

1. **Scale.** Choose `s` so that `2^-s A` has small norm.
2. **Approximate.** Approximate `exp(2^-s A)` by a diagonal Padé approximant
   `r_m(X) = q_m(X)^-1 p_m(X)`, a ratio of two degree-`m` polynomials. For the
   exponential, `q_m(X) = p_m(-X)`. A Padé approximant is far more accurate near
   zero than a Taylor polynomial of the same cost. Computing it takes a few
   matrix products and one linear solve.
3. **Square.** Square the result `s` times.

Ward (1977) made this a practical algorithm. The open question was how to
choose `m` and `s`. Higham (2005) answered it with a backward error analysis:

- `r_m(X) = exp(X + E)` exactly, where `E = h_{2m+1}(X)` is a power series in
  `X` whose coefficients are known.
- So for `||X|| <= θ_m`, the relative backward error `||E|| / ||X||` is at most
  the unit roundoff `u`.
- `θ_m` is the largest norm for which that holds. Higham computed it for IEEE
  double: θ_3 = 0.015, θ_5 = 0.25, θ_7 = 0.95, θ_9 = 2.10 and θ_13 = 5.37.
- Weighing accuracy against cost, `m = 13` came out best for double.

The approximation is then as accurate as the data, whatever the matrix:
`r_m(2^-s A)^(2^s)` is `exp(A + ΔA)` with `||ΔA|| <= u ||A||`, apart from the
rounding errors in evaluating it and in squaring.

### Overscaling, and the 2009 refinement

Higham (2005) chooses `s` from `||A||` alone. For a nonnormal matrix that can be
far too large. Take

```
A = [1  0]      A^2 = I,      exp(A) = [e        0  ]
    [b -1]                             [b sinh 1  1/e].
```

Here `||A||_1 = 1 + b`, but all of `A`'s powers are bounded. Choosing `s` from
the norm means about `log2(b / 5.4)` squarings. Each squaring amplifies the
rounding errors of the step before, and the result loses digits that were never
at risk. This is *overscaling*.

Al-Mohy and Higham (2009) fixed it with a sharper bound. The backward error
series contains only even powers `X^j` with `j >= 2m`. Any such power is a
product of lower powers, so its norm is at most `α_p^j`, where

```
α_p = max(||A^(2p)||^(1/(2p)), ||A^(2p+2)||^(1/(2p+2))).
```

This holds for any `p` with `2m >= 2p(p-1)`. For a normal matrix `α_p` equals
`||A||`. For the matrix above it is 1, independent of `b`, and no scaling is
needed at all.

The 2009 algorithm adds two further guards:

- **The ℓ correction.** It adds `ℓ(A, m)` squarings when the leading error term,
  evaluated on `|A|`, says the bound is optimistic.
- **Exact entries for triangular matrices.** For a triangular `A`, every
  squaring step recomputes the diagonal and the first superdiagonal exactly
  (their Code Fragment 2.1). Errors there would otherwise propagate into
  everything above.

Measured in MTL5, double precision, on the matrix above:

| b | s chosen from ‖A‖ | relative error with that s | relative error of `expm` |
|---|---|---|---|
| 1e2 | 5 | 6.0e-15 | 4.8e-16 |
| 1e4 | 11 | 3.4e-13 | 6.2e-16 |
| 1e6 | 18 | 2.4e-6 | 4.0e-16 |
| 1e8 | 25 | 4.0e-2 | 3.8e-16 |
| 1e12 | 38 | overflow | 6.2e-16 |

The middle column scales by `2^-s`, calls `expm` on the scaled matrix and squares
`s` times. The scaled matrix's `expm` is itself accurate to `u`. At `b = 1e12`,
its upper-right entry is 6e-17 where the exact value is 0, a normwise error of
`u`. Thirty-eight squarings amplify that entry until it overflows. `expm` itself
chooses `s = 0` for every `b`. The test suite checks this case, and the
triangular case, to a few ulps.

### What MTL5 implements

`include/mtl/operation/expm.hpp` follows Al-Mohy and Higham's Algorithm 6.1:

1. **Powers.** Form `A^2`, `A^4` and `A^6` as needed.
2. **Degree and scaling.**
   - Try `m = 3`, `5`, `7`, `9` in turn. Each is accepted when the `α_p` bound
     is below `θ_m` and the ℓ correction is zero.
   - Otherwise use `m = 13`, with the smallest `s` the `α_p` bounds allow, plus
     the ℓ correction.
3. **Padé approximant.** Evaluate `p_m` and `q_m` in Higham's schemes: about
   six matrix products for `m = 13`.
4. **Solve.** LU-factorize `q_m` and solve for `r_m` one column at a time.
5. **Squaring.** Square `s` times. For an upper triangular `A`, each step also
   recomputes the diagonal and superdiagonal exactly.

It differs from Algorithm 6.1 in three places.

- **Exact norms instead of estimates.** Algorithm 6.1 estimates `||A^8||` and
  `||A^10||` with the block 1-norm estimator of Higham and Tisseur (2000). MTL5
  has no `normest1` yet. Instead it uses exact 1-norms of the powers it forms
  and bounds the others by submultiplicativity, for example
  `||A^8|| <= min(||A^4||^2, ||A^2|| ||A^6||)`. An estimate is a lower bound on
  the norm. An upper bound can only increase `s`, so the backward-error
  guarantee is preserved. In the rare case where an estimate would have allowed
  a smaller `s`, MTL5 squares more often than SciPy does.
  `||(|A|)^(2m+1)||_1` for the ℓ correction is also exact. It is `k`
  vector–matrix products, `O(k n^2)`, renormalized at every step so it cannot
  overflow.
- **Precision-aware θ_m.** Covered in the next section.
- **Pre-scaling for narrow exponent ranges.** The powers up to `A^8`, and
  products of their norms up to `||A||^10`, are formed before scaling. In float
  that overflows once `||A||` is about 7e3, so a large matrix is first scaled
  down to `||A|| <= θ_13 2^K`, with `K = max_exponent/10 - 4`.
  - This limits how far the `α_p` refinement can lower `s`: at most `K`
    squarings below the norm-based choice, where `K` is 8 for float and 98 for
    double.
  - A type without `numeric_limits` is not pre-scaled.

The function:

```cpp
#include <mtl/mtl.hpp>   // or <mtl/operation/expm.hpp>

mtl::mat::dense2D<double> A = ...;
auto E  = mtl::expm(A);           // dense2D<double>
auto Ed = mtl::expm<double>(Af);  // float matrix, products and LU accumulated in double
```

- **Accumulator.** The leading template parameter is passed to every matrix
  product, to the LU factorization and to the solves. The default `void` keeps
  their BLAS and LAPACK dispatch.
- **Errors.** A non-square `A` throws `std::invalid_argument`. A NaN or infinite
  entry gives a matrix of NaN. A singular Padé denominator throws
  `std::runtime_error`; it cannot happen for finite `A` within the `θ_m` bounds.

### Precision: θ_m follows the element type

Higham's θ_m values are for `u = 2^-53`. MTL5 runs on float, double, `long
double`, double-double, posits and other types. The degree and scaling must
follow each type's own unit roundoff:

- **Too fine a table wastes work.** A double table used for float squares more
  often than float needs.
- **Too coarse a table loses accuracy.** A double table used for double-double
  stops at an error of `2^-53`.

`tools/expm_theta.py` solves Higham's equation with mpmath for any `u`. It
starts by checking itself: it must reproduce Higham's double values to the 16
digits he prints, or it stops. It then generates the tables in the header, for
bfloat16, binary16, float, double, x87 extended, double-double, binary128,
quad-double and binary256:

| u | θ_3 | θ_5 | θ_7 | θ_9 | θ_13 |
|---|---|---|---|---|---|
| 2^-24 (float) | 0.426 | 1.88 | 3.93 | 4.23* | 4.25* |
| 2^-53 (double) | 0.0150 | 0.254 | 0.950 | 2.10 | 4.25* |
| 2^-64 (x87) | 0.00420 | 0.118 | 0.552 | 1.38 | 4.02 |
| 2^-106 (double-double) | 3.28e-5 | 0.00645 | 0.0690 | 0.273 | 1.32 |
| 2^-113 (binary128) | 1.46e-5 | 0.00397 | 0.0488 | 0.209 | 1.10 |

\* capped, as follows. For a large `u` the equation allows a large `θ_m`, and
then the denominator `q_m(A)` becomes ill-conditioned. At `u = 2^-24` the
script's bound on `κ(q_13(A))` is 2.7e7, and solving with `q_m` amplifies
rounding errors by that much.

Al-Mohy and Higham had already lowered double's θ_13 from 5.37 to 4.25 to
condition `q_13` better. The script makes that a rule for every `u`: each `θ_m`
is capped where its condition bound reaches the bound at θ_13 = 4.25. For
double this changes θ_13 alone. It also binds for float's θ_9 and θ_13 and for
the 8- and 16-bit formats.

At run time:

- **Unit roundoff.** `detail::unit_roundoff<T>()` is half of the machine
  epsilon. For a type without a `numeric_limits` specialization,
  `numeric_limits<T>::epsilon()` is zero, so the epsilon is measured instead by
  halving until `1 + ε/2 == 1`.
- **Table choice.** `expm` uses the table with the largest `u` that does not
  exceed the type's own. That is conservative for a type between two tables.
- **Below the last table.** A type with `u` below `2^-237` uses the last table,
  and its backward error is then bounded by `2^-237`.

The degree is capped at 13 at every precision. In high precision this means
small θ values and many squarings. Fasi and Higham (2019) show that higher
degrees pay off there, and that is a possible extension.

### Accuracy and cost

- **What is guaranteed.** The guarantee is on the backward error. The forward
  error of `expm(A)` also depends on the conditioning of the exponential at `A`,
  which can be large. For the rotation generator `t [0 -1; 1 0]` it grows with
  `t`.
- **What the tests check:**
  - closed forms: zero, diagonal, nilpotent, rotation and the nonnormal example
    above;
  - the identities `exp(A) exp(-A) = I` and `exp(2A) = exp(A)^2`;
  - a 4×4 nonnormal matrix and a complex variant against a 60-digit mpmath
    reference, for double, float, `long double`, `std::complex<double>` and a
    scalar without `numeric_limits`.
- **Cost.**
  - Up to about 6 matrix products for the powers and the Padé polynomials.
  - One LU factorization and `n` triangular solve pairs.
  - `s` squarings.
  - For `m = 13` that is roughly `(6 + s + 4/3) n^3` multiply-adds, plus
    `O(n^2)` for the norms.

## History

| Year | Development |
|---|---|
| 1850 | Sylvester coins the word "matrix". |
| 1858 | Cayley's *Memoir on the theory of matrices* founds matrix algebra, and already considers square roots of matrices. |
| 1867, 1888 | Laguerre and Peano define the matrix exponential by its power series. |
| 1883 | Sylvester gives a definition of `f(A)` for general `f`, through the eigenvalues. |
| 1870s–1890s | Lie develops continuous transformation groups. The exponential map from a Lie algebra to its group becomes the link between the infinitesimal and the global. |
| 1881, 1892 | Frobenius arranges rational approximants of a power series into a table. Padé's thesis is the first systematic study of them, with particular attention to the exponential. They now carry his name. |
| 1977 | Ward makes scaling and squaring with Padé approximants a practical algorithm, with an accuracy estimate. |
| 1978 | Moler and Van Loan's "Nineteen dubious ways" surveys the methods and their failures. It names no best method, but cautiously suggests that scaling and squaring might be acceptable. The 2003 revision still finds none completely satisfactory. |
| 2005 | Higham's backward error analysis fixes the degree (13 for double) and the θ_m thresholds. MATLAB's `expm` adopted the method. |
| 2008 | Higham's *Functions of Matrices* collects the theory and algorithms. Section 1.10 is a history of the subject. |
| 2009 | Al-Mohy and Higham remove overscaling with the `α_p` bounds and exact triangular entries. Their Algorithm 6.1 is what SciPy's `scipy.linalg.expm` implements, and MATLAB's `expm` documentation cites it alongside the 2005 paper. |
| 2019 | Fasi and Higham give a scaling and squaring algorithm for arbitrary precision, taking the unit roundoff as an input. |

The thread through this history is the same one MTL5 is built around. The
mathematics of `exp(A)` was settled by 1900. What took the next century was
computing it reliably in finite precision. The answer is not a fixed formula
but a choice of degree and scaling made against the unit roundoff. That choice
must move with the arithmetic: MTL5's custom number types are exactly the case
where double's θ_m table is the wrong one.

## What comes next

Issue #650 has four phases. `expm` is phase 1.

- **Phase 2:** `schur(A)`, which returns `Q` and the quasi-triangular `T`, and a
  triangular Sylvester solver.
- **Phase 3:** `sqrtm`, by the Schur method (Björck & Hammarling 1983; Higham
  1987 for real matrices), and `logm`, by inverse scaling and squaring (Al-Mohy &
  Higham 2012).
- **Phase 4:** Fréchet derivatives and condition estimates. Also
  `expm_multiply(A, b)`, which computes the action of the exponential without
  forming it and matters for sparse `A` (Al-Mohy & Higham 2011).
- **A `normest1`** (Higham & Tisseur 2000) would let `expm` match Algorithm
  6.1's norm estimates. Phase 4 needs it as well.

## References

- N. J. Higham, *Functions of Matrices: Theory and Computation*, SIAM, 2008.
  doi:10.1137/1.9780898717778.
- N. J. Higham, "The scaling and squaring method for the matrix exponential
  revisited", SIAM J. Matrix Anal. Appl. 26(4), 1179–1193, 2005.
- A. H. Al-Mohy and N. J. Higham, "A new scaling and squaring algorithm for the
  matrix exponential", SIAM J. Matrix Anal. Appl. 31(3), 970–989, 2009.
- C. Moler and C. Van Loan, "Nineteen dubious ways to compute the exponential of
  a matrix", SIAM Review 20(4), 801–836, 1978; "… twenty-five years later",
  SIAM Review 45(1), 3–49, 2003.
- R. C. Ward, "Numerical computation of the matrix exponential with accuracy
  estimate", SIAM J. Numer. Anal. 14(4), 600–610, 1977.
- M. Fasi and N. J. Higham, "An arbitrary precision scaling and squaring
  algorithm for the matrix exponential", SIAM J. Matrix Anal. Appl. 40(4),
  1233–1256, 2019.
- N. J. Higham and F. Tisseur, "A block algorithm for matrix 1-norm estimation,
  with an application to 1-norm pseudospectra", SIAM J. Matrix Anal. Appl.
  21(4), 1185–1201, 2000.
- A. H. Al-Mohy and N. J. Higham, "Computing the action of the matrix
  exponential, with an application to exponential integrators", SIAM J. Sci.
  Comput. 33(2), 488–511, 2011.
- A. H. Al-Mohy and N. J. Higham, "Improved inverse scaling and squaring
  algorithms for the matrix logarithm", SIAM J. Sci. Comput. 34(4), C152–C169,
  2012.
- Å. Björck and S. Hammarling, "A Schur method for the square root of a
  matrix", Linear Algebra Appl. 52/53, 127–140, 1983.
- N. J. Higham, "Computing real square roots of a real matrix", Linear Algebra
  Appl. 88/89, 405–430, 1987.
