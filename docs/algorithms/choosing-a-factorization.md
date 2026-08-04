# Choosing a symmetric or Hermitian factorization

MTL5 offers four entry points for factoring a symmetric or Hermitian matrix:
`cholesky_factor`, `cholesky_h_factor`, `ldlt_factor`, `ldlt_h_factor`. This page
says which one to call, and why the choice is smaller than it looks.

## The rule

**Call the `_h` form unconditionally.**

```cpp
mtl::cholesky_h_factor(A);   mtl::cholesky_h_solve(A, x, b);   // positive definite
mtl::ldlt_h_factor(A);       mtl::ldlt_h_solve(A, x, b);       // possibly indefinite
```

The `_h` routines are **not complex-only**. For a real element type conjugation is
the identity, so `cholesky_h_factor` *is* `cholesky_factor` — not "equivalent to",
not "agrees within tolerance", but the same arithmetic producing the same bits:

```
real SPD    cholesky_factor=0  cholesky_h_factor=0   bitdiff=0
real sym    ldlt_factor=0      ldlt_h_factor=0       bitdiff=0
```

That invariant is asserted in `tests/unit/operation/test_cholesky.cpp` and
`test_ldlt.cpp` with `==`, deliberately, rather than an epsilon comparison. It is
not an accident of the current implementation — it is a property the tests exist
to keep true.

So generic code templated on the scalar type needs **no `is_complex_v` branch**.
Writing one is the common mistake: it doubles the API surface a caller has to
cover and buys nothing.

## The full picture

What each routine does with each kind of input, measured rather than asserted:

| input | `cholesky_factor` | `cholesky_h_factor` | `ldlt_factor` | `ldlt_h_factor` |
|---|---|---|---|---|
| real symmetric positive definite | `0` | `0` — bit-identical | `0` | `0` — bit-identical |
| real symmetric, indefinite | `k+1` (not PD) | `k+1` (not PD) | `0` | `0` — bit-identical |
| **Hermitian** (`A == Aᴴ`) | compile error | `0` | **`LDLT_NOT_SYMMETRIC`** | `0` |
| **complex symmetric** (`A == Aᵀ`) | compile error | **`CHOLESKY_NOT_HERMITIAN`** | `0` | **`LDLT_NOT_HERMITIAN`** |

Two things follow from the table.

**Every wrong choice is caught.** Feeding a routine the structure it does not
handle produces a negative return code or a compile error, never a plausible
wrong answer. That was not always so: until [#352](https://github.com/stillwater-sc/mtl5/issues/352),
`ldlt_factor` ran a Hermitian matrix to completion and returned a wrong solution
under `info == 0`. The guards exist because that failure mode is invisible.

**The guards are scale-relative, not exact.** A matrix assembled in floating
point — from an FEM pass, or a blocked `Bᴴ·B` whose `(i,j)` and `(j,i)` sums
accumulate in different orders — is Hermitian only to rounding. An exact test
passes on matrices written as literals and fails on matrices that are computed;
during #352 a **one-ULP** perturbation defeated an exact guard and produced an
answer wrong in the first significant digit. The threshold is `n · eps · scale`
(`detail/structure_tol.hpp`), degrading to an exact test for element types
without a `std::numeric_limits` specialization.

## When you want the plain forms

**`ldlt_factor` — a complex symmetric matrix.** `A == Aᵀ` with a non-real
diagonal is a real case (moment matrices, some frequency-domain formulations) and
`L·D·Lᵀ` is its correct factorization. This is the one situation where the `_h`
form is the wrong answer, and it is why `ldlt_factor` still accepts complex.

**`cholesky_factor` — nothing, beyond real code that already calls it.** It is
restricted to real element types, and that is not a gap waiting to be filled:
"positive definite" is a statement about an **ordering**, and a complex symmetric
matrix has no real diagonal to order. There is no complex-symmetric Cholesky to
offer. For complex input, `L·Lᴴ` is the only meaningful factorization, which is
exactly what `cholesky_h_factor` computes.

The asymmetry between the two is therefore principled, not historical: `ldlt` has
two genuinely different complex factorizations, and `cholesky` has one.

### Design note: why `cholesky_factor` rejects rather than dispatches

Since `L·Lᴴ` is the *only* meaningful Cholesky for a complex matrix, there is no
ambiguity for a restriction to protect against — so `cholesky_factor` could
simply dispatch to the Hermitian algorithm for complex input and "just work" for
generic callers. That was considered and **deliberately rejected**
([#366](https://github.com/stillwater-sc/mtl5/issues/366)). The reasons, recorded
here so the question does not get re-opened by each new consumer:

- **The caller should name the factorization they are getting.** `A = L·Lᵀ` and
  `A = L·Lᴴ` are different objects; having one spelling silently mean either
  depending on the element type makes generic code harder to reason about, not
  easier.
- **Consistency with `ldlt`.** There, dispatch is not available even in
  principle — both complex forms are meaningful — so `_h` must be explicit. One
  naming convention across both families beats a special case in one.
- **It follows the rule that produced these routines.**
  [#352](https://github.com/stillwater-sc/mtl5/issues/352) and
  [#353](https://github.com/stillwater-sc/mtl5/issues/353) were both cases of a
  routine quietly doing something adjacent to what was asked. Restricting and
  naming the alternative is the habit that fixed them.

The cost is one line at each call site, and the `static_assert` names the
replacement, so the compiler tells you what to write.

## Which factorization, not which spelling

Orthogonal to the `_h` question:

- **`cholesky_h`** needs positive definiteness and takes a square root per pivot.
  Returns `k+1` if pivot `k` is not positive, which doubles as a definiteness test.
- **`ldlt_h`** is square-root-free and handles **indefinite** matrices, since `D`
  may carry negative entries. Same `O(n³/3)` cost, and no precision loss from
  `sqrt` on an ill-conditioned pivot.

If you do not know that the matrix is positive definite, use `ldlt_h`. If you are
using the factorization *as* a definiteness test, use `cholesky_h` and read the
return code.

## Not yet available

**Complex eigenvalues.** `hessenberg_factor` and `eigen_symmetric` reject complex
element types: they apply `H·A·H`, which is a similarity transform only because a
*real* Householder reflector is Hermitian. The unitary reduction `A → H·A·Hᴴ`
with a real subdiagonal and phase accumulation is tracked in
[#362](https://github.com/stillwater-sc/mtl5/issues/362).

Complex `qr`, `lq` and the Householder reflectors themselves are supported
([#361](https://github.com/stillwater-sc/mtl5/pull/361)), including a real `R`
diagonal matching LAPACK's guarantee.

## Summary

| you have | call |
|---|---|
| anything symmetric or Hermitian, positive definite | `cholesky_h_factor` |
| anything symmetric or Hermitian, possibly indefinite | `ldlt_h_factor` |
| complex **symmetric** (`A == Aᵀ`), indefinite | `ldlt_factor` |
| complex symmetric, positive definite | does not exist — see above |
