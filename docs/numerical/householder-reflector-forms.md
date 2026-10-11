# Householder reflector forms

`householder`, `apply_householder_left`, `apply_householder_right`,
`qr_factor` and `lq_factor` can build and apply a Householder reflector in
either of two forms. Both produce a valid factorization, and they produce
different bits, and for some inputs different signs. The default is unchanged.
The second form is opt-in, for callers that need to reproduce a hand-written
textbook QR or LQ exactly.

```cpp
#include <mtl/operation/qr.hpp>

mtl::qr_factor(A, tau);                                  // default: LAPACK form
mtl::qr_factor(A, tau, mtl::tag::textbook_reflector{});  // opt-in: textbook form
```

## Why

A reflector H = I − τ·v·vᵀ maps a column x onto a multiple of e₁. The algebra
fixes H up to a sign, and leaves open how v is scaled and how the sums are
formed. MTL5's default follows LAPACK's `dlarfg`, and the textbook (Golub &
Van Loan §5.1) makes different choices:

| | default (LAPACK form) | textbook form |
|---|---|---|
| ‖x‖ | x scaled by max\|xᵢ\| before squaring | Σxᵢ², unscaled, from zero, ascending i |
| v | normalized, v(0) = 1 | not normalized: v = x, v(0) = x₀ − α |
| reflected onto | +‖x‖ when x₀ > 0 (Parlett's v₀ = −σ/(x₀ + ‖x‖)) | α = −sign(x₀)·‖x‖ |
| scalar | τ = 2v₀²/(σ + v₀²) | vᵀv, summed from zero |
| column update | `A(i,j) -= τ·v(i)·w` | `f = (2·s)/vᵀv`, then `A(i,j) -= f·v(i)` |
| R's diagonal | non-negative for x₀ > 0 | −sign(x₀)·‖x‖ |

Each column is rounded through a different sequence of operations, so R differs
in the last bits. Where x₀ > 0 the two forms also reflect onto opposite signs, so
that row of R, and the matching column of Q, change sign. Both are correct:
Q·R = A and Q is orthogonal either way.

That doesn't matter to most callers. It matters to a caller whose results are
pinned to the textbook form. branes-ai/cortex is porting a multi-state-constraint
Kalman filter (MSCKF) to MTL5 and requires every stage to reproduce the pre-port
output (#559). It uses the textbook form twice: a QR that keeps only R, for the
square-root covariance, and a projection onto the left null space of the feature
Jacobian, built from three reflectors applied across `[H_f | H_x | r]`. A
prototype on the default form measured:

- in the stage benchmarks, 10 of 117 outputs moved. The null-space projection
  changed at O(1), because a different reflector sign gives a different, equally
  valid basis;
- on EuRoC V2_03, 1 of 16,265 update decisions flipped, and the trajectory
  diverged from there (ATE 0.269 → 0.289 m).

This is the same situation as [summation order in
Cholesky](summation-order-in-cholesky.md), and it has the same kind of remedy: a
tag that selects the other form.

## What

```cpp
namespace mtl::tag { struct textbook_reflector {}; }   // <mtl/tag/reflector_form.hpp>
```

| call | form | LAPACK |
|---|---|---|
| `householder(x)` | default; returns `{v, τ}` with v(0) = 1 | — |
| `householder(x, tag::textbook_reflector{})` | textbook; returns `{v, vtv, alpha}` | — |
| `apply_householder_left(A, v, τ, row, col)` | default | — |
| `apply_householder_left(A, v, vtv, row, col, tag::textbook_reflector{})` | textbook | — |
| `qr_factor(A, tau)` | default | `geqrf` when `MTL5_HAS_LAPACK` and A is a `float`/`double` dense matrix |
| `qr_factor(A, tau, tag::textbook_reflector{})` | textbook | **never** |
| `apply_householder_right(A, v, τ, row, col)` | default | — |
| `apply_householder_right(A, v, vtv, row, col, tag::textbook_reflector{})` | textbook | — |
| `lq_factor(A, tau)` | default | — |
| `lq_factor(A, tau, tag::textbook_reflector{})` | textbook | — |

The tagged `householder` returns a `textbook_reflection<T>`, which supports
structured bindings:

```cpp
auto [v, vtv, alpha] = mtl::householder(x, mtl::tag::textbook_reflector{});
```

### Skipped columns

A column is skipped when `!(norm > 0)`: it is zero, or it contains a NaN. The
tagged `householder` then returns `vtv = 0`, and the tagged
`apply_householder_left` and `apply_householder_right` do nothing when
`!(vtv > 0)`, which also covers a vᵀv
that underflows. These are the tests a hand-written loop uses, so NaN handling
matches too: a NaN column is left as it is, not propagated through a reflector.

A column that is zero *below* the pivot is not skipped. If x = (x₀, 0, …, 0) with
x₀ > 0, the textbook reflector is H = I − 2e₁e₁ᵀ, which maps x₀ to −x₀. The
default returns τ = 0 there and leaves the column alone.

### The tagged `qr_factor`

For each column k it computes the textbook reflector of `A(k:m−1, k)`, skips the
column if the reflector says so, applies it to columns k+1 onward, and writes
R(k,k) = α. So R, on and above the diagonal, is bit-identical to a hand-written
textbook QR, including its signs.

Below the diagonal and in `tau` it stores the same reflector in MTL5's packed
form: v/v(0), with τ = 2·v(0)²/vᵀv. v(0) = x₀ − α can't be zero, because α has
the opposite sign to x₀, so |v(0)| = |x₀| + ‖x‖. Because of this,
`qr_extract_Q` and `qr_solve` work on the result unchanged. The Q they produce
agrees with the textbook Q to rounding, not to the bit, since the packed form is
a rescaling of v. A caller that needs Q, or Qᵀb, bit for bit should apply the
reflectors with the tagged `apply_householder_left` as the original code does.

A skipped column gets τ = 0 and keeps its entries, like the default's identity
reflection.

### The tagged right update and `lq_factor`

`apply_householder_right(A, v, vtv, row, col, tag)` is the row mirror of the
left update. It computes A := A·H for rows `row` onward and columns
`col .. col+v.size()−1`. Each row i is updated as

```text
s = Σ_j A(i, col+j)·v(j)      from zero, ascending j
f = (2·s)/vᵀv
A(i, col+j) -= f·v(j)
```

For a real reflector H = Hᵀ, so there is no conjugate to place, unlike the
default's complex path.

`lq_factor(A, tau, tag)` is the row mirror of the tagged `qr_factor`. For each row
k it computes the textbook reflector of `A(k, k:n−1)`, skips the row if the
reflector says so, applies it on the right to rows k+1 onward, and writes
L(k,k) = α. So L, on and below the diagonal, is bit-identical to a hand-written
textbook LQ. Above the diagonal and in `tau` the reflector is stored in the
packed form, as in QR, so `lq_extract_Q` works on the result unchanged, with Q
matching the textbook Q to rounding. `lq_factor` has no LAPACK path in either
form.

### Limits

- **Real element types only.** The textbook form has no single complex
  counterpart, so the tagged overloads `static_assert` on a complex type.
- **No scaling.** The squares are formed as a hand-written loop forms them, so
  in `double` entries above about 1e154 overflow and entries below about 1e−154
  underflow. The default's scaling avoids both. That's the cost of reproducing
  the loop.
- **QR and LQ only.** The Hessenberg reduction, the eigensolvers and the SVD use
  the default form only.
- **No accumulator.** The point of the tag is a fixed order of operations, so the
  tagged overloads don't take an `Accumulator`.

## How

### Reproducing a hand-written QR

If the code being ported looks like this:

```cpp
for (k = 0; k < std::min(m, n); ++k) {
    double norm = 0;
    for (i = k; i < m; ++i) norm += A[i][k] * A[i][k];
    norm = std::sqrt(norm);
    if (!(norm > 0)) continue;
    double alpha = A[k][k] >= 0 ? -norm : norm;
    for (i = k; i < m; ++i) v[i] = A[i][k];
    v[k] -= alpha;
    double vtv = 0;
    for (i = k; i < m; ++i) vtv += v[i] * v[i];
    if (!(vtv > 0)) continue;
    for (j = k + 1; j < n; ++j) {
        double s = 0;
        for (i = k; i < m; ++i) s += v[i] * A[i][j];
        double f = (2 * s) / vtv;
        for (i = k; i < m; ++i) A[i][j] -= f * v[i];
    }
    A[k][k] = alpha;
    for (i = k + 1; i < m; ++i) A[i][k] = 0;
}
```

then this is its MTL5 equivalent, bit for bit on and above the diagonal:

```cpp
mtl::vec::dense_vector<double> tau;
mtl::qr_factor(A, tau, mtl::tag::textbook_reflector{});
auto R = mtl::qr_extract_R(A);   // exact zeros below the diagonal
```

A null-space projection, or anything else that applies reflectors to a matrix
of its own, uses the two lower-level calls:

```cpp
using mtl::tag::textbook_reflector;
auto [v, vtv, alpha] = mtl::householder(col, textbook_reflector{});
mtl::apply_householder_left(M, v, vtv, row, 0, textbook_reflector{});
```

As with Cholesky, the bits match only if **FP contraction** is the same in both
codebases. With FMA available a compiler may fuse `s += v*a` or `a -= f*v`, and
it decides per loop. MTL5's tests compile with `-ffp-contract=off` when they
compare bits; see [condition 2 in the Cholesky
page](summation-order-in-cholesky.md#reproducing-a-hand-written-factorization).

### Choosing between the two

| situation | use |
|---|---|
| Results must match an existing textbook-form implementation | `tag::textbook_reflector` |
| Results must match MTL5 before #559, or LAPACK's `geqrf` | the default |
| Columns may be very large or very small in magnitude | the default, which scales |
| Complex matrices | the default |
| You need the same bits in LAPACK and non-LAPACK builds | `tag::textbook_reflector`, because it never dispatches |
| Nothing is pinned | the default |

## Examples

### The sign of R

```cpp
mtl::vec::dense_vector<double> x{3.0, 4.0};

auto [v, tau] = mtl::householder(x);                         // H*x = +5 e1
auto [w, vtv, alpha] = mtl::householder(x, mtl::tag::textbook_reflector{});
// alpha == -5, w == {8, 4}, vtv == 80:                     H*x = -5 e1
```

Both reflections are exact here. With x₀ = 3 > 0 the default reflects onto +5
and the textbook form onto −5, so a QR built from either has the opposite sign
in that row of R. A caller that uses only R·Rᵀ, or only the span of Q's columns,
won't notice. A caller that compares bits will.

### A column that is zero below the pivot

```text
A = | 0  1  1 |       textbook R = | 0   1   1 |     tau = (0, 2, 2)
    | 0  2  3 |                    | 0  −2  −3 |
    | 0  0  4 |                    | 0   0  −4 |
```

Column 0 is zero, so it is skipped. Column 1 is zero below its pivot, and the
reflector I − 2e₁e₁ᵀ flips row 1. Column 2's remaining entry 4 is reflected onto
−4. The default leaves this matrix as it is, with τ = (0, 0, 0).

## Testing

`tests/unit/operation/test_householder_textbook.cpp` compares bits (`==`)
against serial reference loops written as above:

- the tagged `householder` equals the reference on `double`, `float` and a
  class-typed scalar, for x₀ positive, negative and zero, and for a tail of zeros;
- zero and NaN columns are skipped, and applying a skipped reflector leaves A
  untouched;
- the tagged `apply_householder_left` equals the reference update with row and
  column offsets, on all three types, for a 40 × 85 `[H_f | H_x | r]` shape, and
  for a 300 × 2000 matrix that splits across a pool of 4 threads;
- the tagged `apply_householder_right` equals the reference update with row and
  column offsets, on all three types, and for a 2000 × 300 matrix that splits
  across the pool;
- the tagged `qr_factor`'s R equals the reference R for shapes from 1 × 1 to
  81 × 81, plus 256 × 200, on all three types, and the tagged `lq_factor`'s L
  equals the reference L for the transposed shapes, plus 200 × 256;
- the matrix above gives exactly the R and τ shown, and its transpose gives the
  transposed L under `lq_factor`;
- `qr_extract_Q` and `qr_solve` work on the tagged result: Q·R reproduces A and Q
  is orthogonal to 1e−13, and a square system solves to 1e−11. `lq_extract_Q`
  works on the tagged LQ the same way: L·Q reproduces A to 1e−13;
- `qr_solve` skips a column that the tagged `qr_factor` skipped for a NaN below
  the diagonal, and returns a finite solution of the system that was factored.
  Before #560's review fix, `qr_solve` applied the τ = 0 reflector anyway and
  0·NaN made the whole solution NaN;
- the default and the textbook form produce different R, and different L, for
  the same matrix.

The target is compiled with `-ffp-contract=off`. As a teeth check, five
patched copies of the headers each fail the tests that should notice:

| mutation | fails |
|---|---|
| the update sums s in descending i | the `apply_householder_left` and `qr_factor` comparisons |
| the norm sums x² in descending i | the `householder` and `qr_factor` comparisons |
| `qr_factor` keeps the column's own diagonal instead of writing α | the `qr_factor` comparison, the exact 3 × 3 case, and Q·R = A |
| the right update sums s in descending j | the `apply_householder_right` and `lq_factor` comparisons |
| `lq_factor` keeps the row's own diagonal instead of writing α | the `lq_factor` comparison, and L·Q = A |

(`2·(s/vᵀv)` in place of `(2·s)/vᵀv` is *not* a useful mutation: multiplying by
2 is exact, so the two round identically.)
