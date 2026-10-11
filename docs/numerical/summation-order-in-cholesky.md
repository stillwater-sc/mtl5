# Summation order in Cholesky

`cholesky_factor` can form each entry of L in either of two orders. Both are
backward stable, and they produce different bits. The default is unchanged. The
second order is opt-in, for callers that need to reproduce a hand-written
factorization exactly.

```cpp
#include <mtl/operation/cholesky.hpp>

mtl::cholesky_factor(P);                                    // default: sum, then subtract
mtl::cholesky_factor(P, mtl::tag::subtract_term_by_term{});  // opt-in: term by term
```

## Why

Every entry of a Cholesky factor is a matrix entry minus a sum of products of
entries already computed:

```text
L(j,j) = sqrt( A(j,j) - Σ_{k<j} L(j,k)² )
L(i,j) =     ( A(i,j) - Σ_{k<j} L(i,k)·L(j,k) ) / L(j,j)        i > j
```

The formula doesn't say *where* A(i,j) enters the sum, and the two common choices
round differently:

| ordering | loop | rounding sequence for 3 products |
|---|---|---|
| **sum, then subtract** (MTL5's default) | `s = 0; s += p_k …; r = a - s` | `a − ((p₀ + p₁) + p₂)` |
| **term by term** | `s = a; s -= p_k …; r = s` | `((a − p₀) − p₁) − p₂` |

Real arithmetic gives the same answer either way. Floating point doesn't, because
each `+` and `−` rounds, and the intermediate values the two forms round at are
different numbers.

That has two consequences:

1. **Neither form is the correct one.** Both are backward stable: the computed L
   is the exact factor of a matrix within a small multiple of the unit roundoff of
   A. They differ in the last few bits of L, in a pattern that depends on the
   matrix.
2. **A port can't change the form without changing results.** If code being moved
   onto MTL5 used the term-by-term form, then switching to MTL5's default changes
   bits, even though nothing about the algorithm changed. That's harmless for most
   callers. It isn't harmless for a caller that holds the port to the pre-port
   results: in a filter, a residual that moves in the last bit can flip an
   accept/reject decision, and from that point the trajectory follows a different
   path.

That second case is what prompted the opt-in (#556). branes-ai/cortex is porting a
multi-state-constraint Kalman filter (MSCKF) to MTL5 and requires every stage to
reproduce the pre-port output. Its triangular solves already ported bit for bit,
because `trsm`'s generic path is term by term. Its covariance factorization didn't:

- in the stage benchmarks, 37 of 117 outputs moved;
- on the EuRoC V2_03 sequence, 2 of 16,265 update decisions flipped, and the
  trajectory diverged from there (ATE 0.269 → 0.261 m).

## What

### The two orderings

```cpp
namespace mtl::tag {
struct subtract_sum {};           // the default
struct subtract_term_by_term {};  // the opt-in
}
```

| call | ordering | LAPACK |
|---|---|---|
| `cholesky_factor(A)` | sum, then subtract | `potrf` when `MTL5_HAS_LAPACK` and A is a `float`/`double` dense matrix |
| `cholesky_factor(A, tag::subtract_sum{})` | sum, then subtract | same as above: this is the same call |
| `cholesky_factor(A, tag::subtract_term_by_term{})` | term by term | **never** |

The term-by-term overload never dispatches to LAPACK, because `potrf`'s summation
order is its own and can't be the term-by-term one. In a LAPACK build the two
calls therefore differ in more than ordering: the default may be a blocked LAPACK
factorization, while the opt-in is always MTL5's own loop.

Everything else is the same in both orderings:

- the products and the order of k;
- the return codes: 0 on success, k+1 when the k-th pivot is not positive;
- the in-place storage of L in the lower triangle;
- the threading. The column update is split over rows, each `L(i,j)` is written by
  exactly one thread, and the result doesn't depend on the thread count.

### Accuracy

Neither ordering is more accurate across the board. The tables give the backward
error ‖A − LLᵀ‖_F / ‖A‖_F, computed in `long double`. Each row is the mean over 40
random SPD matrices with log-spaced spectra, factored in `double`.

As measured in #556 (gcc `-O2`):

| n | cond | term by term | sum, then subtract |
|---|---|---|---|
| 81 | 1e2 | 2.0e-16 | 1.4e-16 |
| 81 | 1e6 | 1.2e-16 | 2.2e-16 |
| 81 | 1e10 | 0.9e-16 | 2.3e-16 |
| 141 | 1e10 | 1.2e-16 | 3.0e-16 |

Reproduced for this page with `cholesky_factor` itself (gcc 15 `-O2
-ffp-contract=off`). The eigenvectors are a product of 20 random Householder
reflections:

| n | cond | term by term | sum, then subtract |
|---|---|---|---|
| 81 | 1e2 | 1.8e-16 | 1.2e-16 |
| 81 | 1e6 | 1.0e-16 | 1.7e-16 |
| 81 | 1e10 | 0.8e-16 | 1.7e-16 |
| 141 | 1e10 | 1.0e-16 | 1.8e-16 |

Both measurements show the same pattern. Term by term is ahead on ill-conditioned
spectra, which is a filter covariance's usual case. Sum-then-subtract is ahead on
well-conditioned ones. The size of the gap depends on the test matrices: with only
3 reflections per matrix (eigenvectors much closer to the coordinate axes), the
two orderings came out within 10 % of each other at cond 1e6 and above. Every
number in both tables is a small multiple of the unit roundoff (1.1e-16).

**So choose the ordering for reproducibility, not for accuracy.** If nothing
pins your results, the default is fine.

The two orderings also take the same time: the same products, in the same loops.

### NaN input

A NaN anywhere in the part of A that the factorization reads makes some pivot NaN.
Both orderings test the pivot with `!(diag > 0)`, so a NaN pivot is reported as
not positive definite: the call returns k+1. Before #556, the default tested
`diag <= 0`, which is false for NaN, so it returned 0 with NaNs in L.
`cholesky_h_factor` already used `!(diag > 0)`.

## How

### Reproducing a hand-written factorization

If the code being ported looks like this:

```cpp
for (j = 0; j < n; ++j) {
    double d = P[j][j];
    for (k = 0; k < j; ++k) d -= L[j][k] * L[j][k];
    L[j][j] = std::sqrt(d);
    for (i = j + 1; i < n; ++i) {
        double s = P[i][j];
        for (k = 0; k < j; ++k) s -= L[i][k] * L[j][k];
        L[i][j] = s / L[j][j];
    }
}
```

then this is its MTL5 equivalent, bit for bit:

```cpp
mtl::mat::dense2D<double> L = P;            // only the lower triangle is read
int info = mtl::cholesky_factor(L, mtl::tag::subtract_term_by_term{});
if (info != 0) { /* pivot info-1 was not positive */ }
```

Three conditions for the bits to match:

1. **The loop really is term by term.** Check where the matrix entry enters. A
   loop that starts `double s = 0;` and ends `L[i][j] = (P[i][j] - s) / …` is the
   *default* form; use `cholesky_factor(L)` for that, in a build without LAPACK, or
   with an element type LAPACK doesn't take.
2. **FP contraction is the same.** With FMA available, a compiler may fuse
   `s - L[i][k]*L[j][k]` into one fused multiply-add, and it decides per loop.
   GCC's default is `-ffp-contract=fast`, Clang's is `on`, and both fuse.
   MTL5's tests compile with `-ffp-contract=off` when they compare bits. If you
   compare bits across two codebases, build both with `-ffp-contract=off`, or build
   for a target without FMA. See [the contraction section of the mult
   analysis](../performance/ikj-gemm-and-crs-dense.md#bit-identity-and-fp-contraction)
   for a measured case.
3. **The solves match too.** `cholesky_solve` forms each substitution step as
   `(b(i) - Σ) / L(i,i)`, i.e. sum then subtract. If the original code solved term
   by term, solve with `trsm`, whose default generic path is
   `s = alpha*B(i,c); s -= A(i,j)*B(j,c) …`. Exceptions: `trsm` is sum-then-subtract
   when given a custom `Accumulator`, and it dispatches to BLAS for column-major
   `float`/`double` when BLAS is linked.

### Choosing between the two

| situation | use |
|---|---|
| Results must match an existing term-by-term implementation | `tag::subtract_term_by_term` |
| Results must match MTL5 before #556, or LAPACK's `potrf` | the default |
| Nothing is pinned | the default |
| You need the same bits in LAPACK and non-LAPACK builds | `tag::subtract_term_by_term`, because it never dispatches |

## Examples

### A 3×3 case where the orderings differ

The matrix below is `float` and SPD: `B Bᵀ / 7` for a small integer matrix B.

```text
A = (1/7) · | 134  122  105 |
            | 122  114   91 |
            | 105   91   94 |
```

The two orderings agree on the first five entries of L. They differ on the last
pivot, the only entry whose sum has two products to round:

| entry | sum, then subtract | term by term | exact (long double) |
|---|---|---|---|
| L(0,0) | 4.37525511 | 4.37525511 | 4.37525503233 |
| L(1,0) | 3.98344135 | 3.98344135 | 3.98344132451 |
| L(1,1) | 0.646459281 | 0.646459281 | 0.646459834266 |
| L(2,0) | 3.42837143 | 3.42837143 | 3.42837157815 |
| L(2,1) | −1.01586652 | −1.01586652 | −1.0158667026 |
| **L(2,2)** | **0.801783383** | **0.801783085** | 0.801782305688 |

L(2,2) = sqrt(A(2,2) − L(2,0)² − L(2,1)²). Sum-then-subtract rounds
L(2,0)² + L(2,1)² first and subtracts the result from A(2,2). Term by term
subtracts L(2,0)² from A(2,2), rounds, and then subtracts L(2,1)². Here A(2,2) is
13.43 and the two squares are 11.75 and 1.03, so the pivot that remains is 0.64.
That is a cancellation: the leading bits cancel, so a rounding difference
at the scale of 13.43 is large relative to 0.64. After the square root, L(2,2)
differs by 5 `float` ulps between the two orderings, in the seventh significant
digit. In this example term by term lands closer to the
exact value. On another matrix it can be the other way round, as the accuracy
tables show.

### Making the choice explicit in code that cares

```cpp
// The filter's covariance factor is pinned to the pre-port term-by-term
// results (see docs/numerical/summation-order-in-cholesky.md).
constexpr mtl::tag::subtract_term_by_term covariance_order{};

int info = mtl::cholesky_factor(S, covariance_order);
```

Naming the tag at the call site documents the dependency where a reader will see
it. Code that doesn't depend on the ordering shouldn't pass a tag at all.

## Testing

`tests/unit/operation/test_cholesky_ordering.cpp` compares bits (`==`), not values
within a tolerance, since a tolerance would pass either ordering:

- the opt-in equals a reference term-by-term loop, on `double`, `float` and a
  class-typed scalar, for n from 1 to 81, and at n = 768 with the pool at 4
  threads;
- the default still equals the reference sum-then-subtract loop;
- `tag::subtract_sum` is the same call as no tag;
- the two orderings differ on an 81×81 matrix, so the tag selects different
  arithmetic;
- NaN and non-SPD pivots are reported by both.

The target is compiled with `-ffp-contract=off`, for the reason in condition 2
above. As a teeth check, a copy of the header patched to ignore the tag fails
exactly the two tests that should notice: the bit-for-bit comparison and "the
two orderings round differently".
