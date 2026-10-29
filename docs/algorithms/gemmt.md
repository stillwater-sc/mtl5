# Triangular-output products and congruences

```cpp
#include <mtl/operation/gemmt.hpp>

// One triangle of C = alpha A B + beta C, mirrored to the other
mtl::gemmt(alpha, A, B, beta, C);                          // lower, mirrored
mtl::gemmt(alpha, A, B, beta, C, mtl::uplo::upper);
mtl::gemmt(alpha, A, B, beta, C, mtl::uplo::lower, mtl::gemmt_fill::keep);

// C = A P A^T, P symmetric; A dense or compressed2D
mtl::congruence(A, P, C);
mtl::congruence(alpha, A, P, beta, C);                     // C = alpha A P A^T + beta C

// A diagonal middle term: C = alpha K diag(r) K^T + beta C
mtl::congruence(alpha, K, r, beta, C);
```

Where these operations come from, the Joseph covariance update and the
`GEMMT` BLAS extension, is told in [gemmt-origins.md](gemmt-origins.md).

## gemmt

`gemmt` computes `C = α A B + β C` for an `n × k` matrix `A` and a `k × n`
matrix `B` whose product the caller knows to be symmetric. Only one triangle
of `C`, the diagonal included, is formed: `uplo::lower` (the default) or
`uplo::upper`. That is half the multiply–adds of `mult`.

The other triangle is then mirrored from the formed one, so `C` is exactly
symmetric: `C(i,j)` and `C(j,i)` hold the same bits. This is `syrk`'s
full-result convention. `gemmt_fill::keep` leaves the other triangle as it
was, as the BLAS routine does.

With `β ≠ 0` and the default mirror, the input `C` must be symmetric. Only the
formed triangle of the old `C` is read, so an asymmetric `C` loses the other
triangle's `β C` terms and the result is not `α A B + β C`.

`B` can be a dense matrix, `trans()` of one, or `trans()` of a
`compressed2D`. For a transposed CRS matrix `R`, entry `(i,j)` sums over the
stored entries of row `j` of `R`, so its cost is that row's number of
non-zeros rather than `k`.

`α` and `β` follow the BLAS convention. They are converted to `C`'s element
type first, as `?gemmt` takes them, so the generic and BLAS paths combine with
the same coefficients. With `β = 0`, `C` is not read, so a NaN left in it does
not propagate; with `α = 0`, `A` and `B` are not read.

### Rounding

On the generic path every entry of the formed triangle is summed exactly as
`mult`'s generic kernel sums it: from zero, adding `A(i,0) B(0,j)`,
`A(i,1) B(1,j)`, … in ascending `k`. The triangle therefore agrees with
`mult` bit for bit, and the test suite checks this for a class-type scalar
that rounds at every step. With an explicit `Accumulator`, each entry is
summed in that type and rounded once to `C`'s element type before the
`α`/`β` combine, as `mult<Acc>` and `syrk<Acc>` do.

For a transposed CRS `B`, the sum leaves out the terms whose `B` entry is not
stored. Adding an exact zero product does not change a finite sum, so the
result agrees with the dense one bit for bit whenever `A` is finite.

### BLAS dispatch

`?gemmt` is a BLAS extension. MKL, BLIS and recent OpenBLAS export it;
reference BLAS added it in 3.12 under the name `?gemmtr`, and older OpenBLAS
releases lack it. CMake links and runs a small `dgemmt_` call and defines
`MTL5_BLAS_HAS_GEMMT` only if it returns the right answer. Linking alone is
not enough: OpenBLAS 0.3.26, the version in Ubuntu 24.04, exports `dgemmt_`
but checks `lda` and `ldb` against each other's dimensions, so it rejects a
valid call with `k > n` and leaves `C` unwritten (fixed in 0.3.27). When
cross-compiling, the probe cannot run and `gemmt` keeps its generic path. A
build that links BLAS without CMake defines `MTL5_BLAS_HAS_GEMMT` by hand,
and only for a library that has a working `?gemmt`.

With `MTL5_BLAS_HAS_GEMMT`, `gemmt` dispatches when the accumulator allows
BLAS, `C` is a dense `float` or `double` matrix, and `A` and `B` are dense
matrices of the same type or `trans()` of one. The operands' orientations
need not agree: each maps onto `transa` or `transb`, and a row-major `C` is
computed as `Cᵀ = Bᵀ Aᵀ` with the other triangle. Everything else takes the
generic path. The vendor's summation order is its own, so a BLAS result is
not bit-identical to `mult`.

## congruence

`congruence(A, P, C)` computes the symmetric `m × m` matrix `C = A P Aᵀ` for an
`m × n` matrix `A` and a symmetric `n × n` matrix `P`. It forms `T = A P` with
`mult` and then the lower triangle of `T Aᵀ` with `gemmt`, mirrored. On the
generic path the lower triangle is therefore bit-identical to
`mult(mult(A, P), trans(A))`. `P`'s symmetry is the caller's promise; it is
neither checked nor used to save work. With `β ≠ 0`, `C` must be symmetric
too: only its lower triangle is read, and the upper triangle of the result is
the mirror of the lower one, so an asymmetric `C` loses its upper `β C` terms.

This is the shape of every covariance update in a Kalman filter:

| Step | Product |
|---|---|
| Prediction | `P ← F P Fᵀ + Q` |
| Clone augmentation | `P ← G P Gᵀ` |
| Innovation covariance | `S = H P Hᵀ` |
| Joseph update | `(I − K H) P (I − K H)ᵀ + K R Kᵀ` |

`A` may be a `compressed2D`. The Jacobians `F`, `G` and `H` are mostly
identity blocks and strips, and both products then walk only the stored
entries: the first through `mult`'s CRS × dense kernel (#550), the second
through `gemmt` with a transposed CRS `B`. A dense and a CRS `A` with the same
entries give the same bits.

The vector overload, `congruence(α, K, r, β, C)`, computes `α K diag(r) Kᵀ + β C`
for the `K R Kᵀ` term of the Joseph form when `R` is diagonal. It forms
`T(i,k) = K(i,k) r(k)` directly, one rounding per entry, and then calls `gemmt`.
With `β = 1` it adds the term to a `C` already holding the first one.

### For adopters

Moving from `mult` followed by a symmetrize, `(C + Cᵀ)/2`, to `congruence` is
not bit-identical. The average rounds the two triangles' sum, while
`congruence` keeps one triangle as `mult` computed it and copies it across.
Each is a correctly symmetric result of the same accuracy; they differ in the
last bit of some entries.

## Performance

`benchmarks/bench_congruence.cpp` times `C = A P Aᵀ` for a transition-shaped `A`,
`F = Φ ⊕ I`: a dense 15 × 15 block and the identity elsewhere. `P` is a
symmetric `d × d` covariance and everything is row-major `double`. The arms are:

- **two_step:** `mult(A, P)`, then `mult(T, trans(A))` and `(C + Cᵀ)/2`. This is
  the form branes-ai/cortex uses today.
- **copy:** the same, with `trans(A)` first copied into a dense matrix, so that
  the second product is a plain `mult` that BLAS can take.
- **congruence:** `congruence(A, P, C)` with a dense `A`.
- **CRS:** `congruence(A, P, C)` with `A` a `compressed2D`.

The times are in microseconds per call, the minimum over 15 rounds. They were
measured on one core of an i7-12700K, pinned, with `MTL5_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1` and the `powersave` governor. The generic build has
neither BLAS nor `MTL5_NATIVE_FAST_GEMM`; the BLAS build links OpenBLAS with
`MTL5_BLAS_HAS_GEMMT`.

| d | nnz(A) | two_step | copy | congruence | CRS | two_step, BLAS | copy, BLAS | congruence, BLAS | CRS, BLAS |
|---|---|---|---|---|---|---|---|---|---|
| 15 | 225 | 1.44 | 1.44 | 1.15 | 1.56 | 1.02 | 0.64 | 0.78 | 1.47 |
| 30 | 240 | 9.05 | 8.39 | 6.26 | 4.06 | 6.42 | 2.84 | 2.80 | 4.00 |
| 60 | 270 | 72.9 | 56.2 | 42.1 | 10.0 | 54.4 | 16.8 | 14.3 | 10.2 |
| 100 | 310 | 388 | 263 | 204 | 20.6 | 293 | 66.7 | 61.5 | 21.1 |
| 150 | 360 | 1440 | 1091 | 803 | 40.7 | 995 | 217 | 224 | 41.6 |
| 200 | 410 | 3542 | 2499 | 1819 | 60.1 | 2545 | 490 | 446 | 61.5 |

- **Generic, dense `A`:** `congruence` takes 0.73–0.78× the time of the copy arm
  from `d = 60` up and half the time of two_step at `d = 200`. The second
  product does half the work, and the dense `A` is copied out as `Aᵀ` so that
  it runs in the vectorizable i-k-j form, as `mult` does for row-major
  operands. The first product is a full `mult` in both arms, which is why the
  saving is about a quarter and not a half.
- **BLAS, dense `A`:** `congruence` takes 0.85–1.03× the time of the copy arm. OpenBLAS's
  `dgemmt` is barely faster than its `dgemm`, and at `d = 15` it is slower.
  The gain over two_step comes from avoiding `trans(A)`, which `mult` cannot
  hand to BLAS.
- **CRS `A`:** both products walk only the stored entries, so the time follows
  `nnz(A)` rather than `d²`. At `d = 200` it is 30 times faster than the
  generic dense arm and 7 times faster than the BLAS one. When `A` is fully
  dense (`d = 15`), the indirection makes it the slowest arm.

The CSVs are not committed; rerun the benchmark to reproduce the table.

## Limits

- Only `float` and `double` dispatch to BLAS; `?gemmt` for complex types is
  not wrapped.
- `A` as a `compressed2D` is read through `A(i,k)` on the generic path of
  `gemmt` itself. The CRS-aware path is for the transposed `B`, which is where
  `congruence` puts it.
- Every entry of `T = A P` is formed, as `mult` forms it. A congruence that
  also exploited `P`'s symmetry would be `symm` followed by `gemmt`, which
  saves nothing on the generic path, since `symm` reads both triangles.
