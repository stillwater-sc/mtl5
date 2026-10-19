# Dense direct solvers: factor once, solve many, refine in mixed precision

How MTL5 factors a dense matrix and solves `A·x = b` directly, and *why* the layer
is a **family** of factorizations rather than one general method. This is the dense
counterpart to the [sparse direct solvers](../sparse-direct-solvers-design.md): the
LAPACK-family workhorses — LU, Cholesky, LDLᵀ, QR, LQ, SVD — that decompose a matrix
into triangular/orthogonal factors and then solve by substitution. Each exploits the
structure of its matrix class, all share a small kernel of elementary transforms and
a common triangular-solve back-end, and all dispatch to LAPACK or a native reference
through the [operation layer](operation-dispatch-architecture.md).

The subsystem is ~14 free-function operations in
[`operation/`](https://github.com/stillwater-sc/mtl5/tree/main/include/mtl/operation):
`lu`, `cholesky`, `ldlt`, `ldlt_bk`, `qr`, `lq`, `svd`, `inv`, plus the shared
primitives `householder`, `givens`, `hessenberg`, and `lu_iterative_refine`.

---

## Decision 1 — the factor / solve split

Every factorization is exposed as two operations plus a convenience wrapper, using
LU as the archetype:

```cpp
int  lu_factor(M& A, std::vector<size_type>& pivot);          // decompose (O(n³))
void lu_solve (const M& LU, const pivot&, VecX& x, VecB b);   // apply     (O(n²))
int  lu       (M& A, VecX& x, VecB b);                        // factor then solve
```

The split exists because the two halves have different cost and reuse. Factorization
is O(n³) and depends only on `A`; a solve is O(n²) and depends on the right-hand
side. Separating them lets a caller **factor once and solve many** — the common case
of applying `A⁻¹` to a sequence of vectors (time-stepping, multiple loads, an inner
solve in an outer iteration) pays the cubic cost a single time. `cholesky`, `qr`,
`ldlt`, and the rest follow the same `_factor` / `_solve` / convenience shape.

---

## Decision 2 — in-place overwrite: the LAPACK storage convention

The factors are written **into the input matrix**, packed by triangle:

| Factorization | What overwrites `A` |
|---|---|
| LU | `L` (unit lower) below the diagonal, `U` on and above |
| Cholesky | `L` in the lower triangle; upper triangle untouched |
| QR | `R` on and above the diagonal; Householder vectors below (+ a `tau` array) |
| LDLᵀ (Bunch-Kaufman) | `L` below the diagonal, block-diagonal `D` on it (+ a pivot array) |

There is no separate n² allocation for the result — the decomposition reuses `A`'s
storage, and only small side arrays (`pivot`, `tau`) are extra. This is the LAPACK
economy, and adopting it is what lets the LAPACK path dispatch **directly**: the
buffers already have the layout `getrf`/`potrf`/`geqrf` expect, so no repacking
stands between MTL5's call and the vendor routine. The cost is that `A` is destroyed;
a caller who needs the original copies it first.

---

## Decision 3 — a factorization per structure, not one method

The design decision that defines the layer: rather than route everything through
general LU, it provides the *specialized* factorization for each matrix class,
because exploiting structure buys a 2× work saving and better stability.

| Factorization | Matrix class | Why |
|---|---|---|
| **LU**, partial pivoting | general square | the baseline; pivoting for stability |
| **Cholesky** `A = L·Lᵀ` | symmetric **positive definite** | half the work, no pivoting; *succeeds iff SPD* |
| **LDLᵀ**, Bunch-Kaufman | symmetric **indefinite** | 1×1/2×2 pivot blocks handle indefiniteness without breakdown; bounded growth (≤ 2.57) |
| **QR** (Householder) | general / rectangular `m ≥ n` | orthogonal, robust; least squares |
| **LQ** | rectangular `m ≤ n` | the row-space analog |
| **SVD** `A = U·S·Vᵀ` | any | the most general — rank, pseudo-inverse, least squares |

The split is principled, not incidental. Cholesky is real-only (`A = L·Lᵀ`) with a
separate `cholesky_h_factor` for the Hermitian `L·Lᴴ` case — a `static_assert`
enforces the distinction, because the two are genuinely different for complex data.
And Cholesky's failure return (`k+1` when `A(k,k) ≤ 0`) makes the factorization
double as a **definiteness test**: it succeeds exactly when `A` is SPD, so "is this
matrix SPD?" and "factor it cheaply" are one call. Choosing the right member of the
family is a real decision, which is why there is a companion
[user guide on choosing a factorization](../algorithms/choosing-a-factorization.md);
this page is the design of the family it chooses from.

---

## Decision 4 — LAPACK-or-native dispatch, per the operation layer

Each `_factor` is guarded exactly like `mult` and the eigensolvers:

```cpp
template <Matrix M>
int lu_factor(M& A, std::vector<size_type>& pivot) {
#ifdef MTL5_HAS_LAPACK
    if constexpr (/* column-major dense float/double */) {
        int info = interface::lapack::getrf(...);      // vendor path
        /* convert 1-based Fortran pivots → 0-based */
        return info;
    }
#endif
    /* native: partial-pivoted elimination (the reference floor) */
}
```

Same [build-is-the-backend](operation-dispatch-architecture.md#decision-4--the-build-is-the-backend)
model: with LAPACK linked and a column-major float/double operand, the call routes
to `getrf`/`potrf`/`geqrf`/`gesvd`; otherwise the native reference algorithm runs.
LAPACK is column-major, so a `col_major` `dense2D` dispatches directly while other
layouts take the native path. The native factorizations are correctness references,
not performance-tuned — consistent with the
[benchmark finding](blas-kernel-architecture.md) that native LAPACK trails the
vendors by an order of magnitude; the value of the native path is universality
(any type, any layout, no dependency), and LAPACK is the speed.

---

## Decision 5 — shared elementary transforms across the family (and the eigensolvers)

QR, LQ, Hessenberg reduction, SVD, and the symmetric eigensolver do not each
re-derive their orthogonal machinery. They build on two shared operations:

- **`householder(x) → (v, tau)`** — the reflector `H = I − tau·v·vᴴ` with
  `H·x = beta·e₁` and `v(0) = 1` implicit, the workhorse of every orthogonal
  factorization.
- **Givens rotations** — the plane rotations used to introduce or chase zeros.

Factoring these out means QR and the
[eigensolvers'](eigensolvers-architecture.md) tridiagonalization use the *same*
`householder`, tested to agree bit-for-bit with LAPACK's reference. The
dense-factorization and eigenvalue subsystems are two clients of one small,
well-tested set of orthogonal primitives — a shared kernel rather than duplicated
reflector code.

---

## Decision 6 — triangular substitution as the common solve back-end

However `A` is factored, the *solve* reduces to triangular substitution, and all the
`_solve` operations funnel through the same two kernels:

```cpp
void lu_solve(const M& LU, const pivot&, VecX& x, VecB b) {
    /* apply the pivot permutation to b → x */
    lower_trisolve(LU, x, /*unit_diag=*/true);    // L y = Pb
    upper_trisolve(LU, x, /*unit_diag=*/false);   // U x = y
}
```

Cholesky solves with `L` then `Lᵀ`; QR applies `Qᵀ` then back-substitutes `R`. The
factorization determines *which* triangular factors exist; `lower_trisolve` /
`upper_trisolve` (also public operations) are the shared O(n²) substitution kernels
every solve calls. One back-end, many front-ends — the substitution code is written
once and reused across the whole family.

---

## Decision 7 — mixed-precision iterative refinement: factor cheap, refine accurate

This is where the factorization layer meets MTL5's reason for existing.
`lu_iterative_refine` solves `A·x = b` by factoring `A` **once in a low working
precision** and then correcting the answer with a residual formed in a **higher
precision**:

```text
factor  A ≈ P·L·U            (Working precision — the expensive O(n³) step, cheap)
solve   x = U\(L\Pb)          (Working precision)
repeat  r = b − A·x           (Residual precision — the accurate part)
        d = U\(L\Pr)          (Working precision — reuse the factors)
        x += d                (Residual precision)
```

The costly factorization runs in the cheap precision, while an accurately-computed
residual recovers accuracy *far beyond the working precision's own solve* (Wilkinson;
Higham, ch. 12). This is the classical technique modern hardware revives — factor in
fp16/tf32, refine to fp64 — and it is the dense twin of `sparse/iterative_refine`.
`Working` and `Residual` are **any** arithmetic types (`float`/`double`/`long
double`, or a custom type a caller composes in); the core stays
[dependency-free](../position-mixed-precision-acceleration.md) — "MTL5 never depends
on an external number library." The direct-solver family is thus not just a port of
LAPACK's factorizations; it is the place classical numerical linear algebra becomes a
[mixed-precision](mixed-precision-custom-types-SIMD.md) laboratory.

---

## Error codes double as structure tests

The `_factor` functions return an `int`: `0` on success, `k+1` when step `k` fails —
a zero pivot `U(k,k)` (LU singular), a non-positive `A(k,k)` (Cholesky: not SPD), a
non-real Hermitian diagonal (a distinct negative code so it cannot collide). So a
factorization is also a diagnosis: the return value tells you *why* a matrix could
not be factored, and for Cholesky/LDLᵀ that is precisely a definiteness/structure
verdict. Companion operations `backward_error` and `factorization_properties` provide
the accuracy and quality checks on top.

---

## What the design deliberately does not do

- **The native paths are references, not tuned kernels.** They exist for
  universality — any element type, any layout, zero dependencies — and defer to
  LAPACK for speed. They are not blocked/vectorized to vendor performance.
- **In-place, destructive factorization.** `A` is overwritten; preserving it is the
  caller's copy. The trade is LAPACK-compatible storage and no n² scratch.
- **Column-major is the fast lane.** Direct LAPACK dispatch wants column-major
  float/double; other layouts and types are correct via the native path but do not
  hit the vendor routine.

---

## Where it sits in the stack

```text
containers   (dense2D operand; the factors overwrite it in place)
   │
operations   (householder / givens primitives; lower/upper_trisolve back-end;
   │          the LAPACK-vs-native dispatch)
   │
▶ DENSE DIRECT SOLVERS   (operation/{lu,cholesky,ldlt,ldlt_bk,qr,lq,svd,inv}
        factor (structure-specific) → triangular solve → optional mixed-precision refine
```

It is the dense sibling of the sparse direct solvers and the direct counterpart to
the iterative and Krylov-eigenvalue families: where those *apply* `A`, these *factor*
it. It adds the factorization algorithms and the factor/solve split on top of the
shared orthogonal primitives, triangular kernels, and dispatch the rest of the
library already provides.

---

## File map

| File | Role |
|---|---|
| [`operation/lu.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/operation/lu.hpp) | LU with partial pivoting (`lu_factor`/`lu_solve`/`lu`); LAPACK `getrf` |
| [`operation/cholesky.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/operation/cholesky.hpp) | `L·Lᵀ` (real SPD) and `L·Lᴴ` (Hermitian); LAPACK `potrf` |
| [`operation/ldlt.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/operation/ldlt.hpp) / [`ldlt_bk.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/operation/ldlt_bk.hpp) | symmetric-indefinite LDLᵀ, Bunch-Kaufman pivoting |
| [`operation/qr.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/operation/qr.hpp) / [`lq.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/operation/lq.hpp) | Householder QR / LQ; LAPACK `geqrf` |
| [`operation/svd.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/operation/svd.hpp) | `A = U·S·Vᵀ` via Jacobi / iterative QR or LAPACK |
| [`operation/householder.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/operation/householder.hpp), [`givens.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/operation/givens.hpp), [`hessenberg.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/operation/hessenberg.hpp) | shared elementary orthogonal transforms |
| [`operation/lower_trisolve.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/operation/lower_trisolve.hpp), [`upper_trisolve.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/operation/upper_trisolve.hpp) | the shared triangular-substitution back-end |
| [`operation/lu_iterative_refine.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/operation/lu_iterative_refine.hpp) | mixed-precision iterative refinement |
| [`operation/inv.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/operation/inv.hpp), [`backward_error.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/operation/backward_error.hpp), [`factorization_properties.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/operation/factorization_properties.hpp) | explicit inverse; accuracy / quality checks |

For the sparse counterpart, see the
[sparse direct solvers](../sparse-direct-solvers-design.md) design doc; for the
LAPACK-vs-native dispatch, the [operation dispatch layer](operation-dispatch-architecture.md);
for the orthogonal primitives it shares, the
[eigensolvers](eigensolvers-architecture.md) doc. The accumulator paths of these
factorizations are left-looking forms of their own; [The Accuracy
Architecture](../architecture/accuracy-architecture.md) explains why, and
[Accumulation Configurations](../algorithms/accumulation-configurations.md) gives
what they measure.
