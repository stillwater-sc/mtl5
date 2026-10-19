# Accumulation Configurations: What Each One Buys

Every reduction in MTL5, whether a dot product, a matrix product, a substitution or an entry of a factor, can be computed in a chosen **accumulator**. This page reports what each accumulation configuration does to accuracy across the library, measured on the same routines and inputs. It shows where a better accumulator buys accuracy, where it buys nothing, and why.

The measurements come from the `[586:*]` test cases (#586). Each section names the tag that reproduces it. The reasons behind the results — which loop orders let an accumulator see a whole reduction, and which don't — are in [The Accuracy Architecture](../architecture/accuracy-architecture.md).

## The configurations

| | configuration | what it rounds | how to select it |
|---|---|---|---|
| 1 | **plain** | each product, then each partial sum | `Accumulator = T` (or a wider built-in type, such as `double` for `float` data) |
| 2 | **FMA** | each partial sum, once per term; the product is not rounded | `math::fma_accumulator<T>` |
| 2c | **compensated** (Ogita–Rump–Oishi Dot2) | as if in twice the working precision, rounded once | `math::compensated_accumulator<T>` |
| 3 | **exact** (super-accumulator, quire) | nothing until the end; one rounding | `math::exact_accumulator<T>` for `float` and `double`; a Universal quire for posits and other types |

All four are used the same way. The operation takes the accumulator as its first template argument, and an optional result type after it:

```cpp
using exact = mtl::math::exact_accumulator<float>;
float d = mtl::dot<exact, float>(x, y);          // one rounding
mtl::lu_factor<exact>(A, pivot);                 // each entry of L and U rounded once
mtl::itl::pc::ilu_0<float, mtl::mat::parameters<>, exact> M(A);
```

- **Default.** With the default `Accumulator = void`, nothing changes: the operation takes its BLAS, LAPACK, SIMD or native path.
- **Any other accumulator** takes the native, accumulator-aware path, never BLAS or LAPACK. That path forms each output entry as one reduction (`interface::accumulator_allows_blas_v`, in `include/mtl/interface/dispatch_traits.hpp`).
- **Adding your own.** A new accumulator plugs in through an ADL hook: `accumulator_clear`, `accumulator_assign`, `accumulator_add_product`, `accumulator_value` and, optionally, `accumulator_merge`. The protocol is in `include/mtl/math/accumulator_traits.hpp`; [Mixed-Precision Kernels](mixed-precision-kernels.md) describes the element / accumulate / result precisions.
- **References.** The exact accumulator is described in [Exact Accumulator](../architecture/exact-accumulator.md), and the compensated one in [Error-Free Transformations](../architecture/error-free-transformations.md).

## How the errors are measured

- **Exact errors.** The reference is the exact accumulator itself. An error `|exact − computed|` is formed by adding `−computed` into a copy of the exact sum and rounding once (`exact_test::err`, `tests/unit/operation/exact_accumulator.hpp`). So errors are exact, not estimates.
- **Solver residuals.** The direct-solver residuals `‖b − A·x‖ / ‖b‖` are formed the same way, entry by entry.
- **Exceptions.** The preconditioner and three-factor tests use an infinity-norm residual computed in double, and the refinement row uses the solver's own reported residual. Their sections say so.
- **No FMA contraction.** Every test that compares configurations is built with `-ffp-contract=off` (`tests/unit/CMakeLists.txt`). With contraction on, the compiler may turn `acc += a*b` into a fused multiply-add, which silently makes configuration 1 into configuration 2.
- **Measured numbers.** They are written in each test file's header comment. The assertions encode the ordering they show, with a margin. They do not encode the numbers themselves.

To run a row:

```bash
ctest --test-dir build -R op_test_accumulation_configs_mt     # inner products, norms, determinism
ctest --test-dir build -R op_test_accumulation_solvers_mt     # direct, refinement, Krylov, eigen
```

or run a test binary with a tag, such as `"[586:dot]"`.

## Inner products: `dot` and `gemm`

`[586:dot]`, `[586:gemm]`, `tests/unit/operation/test_accumulation_configs_mt.cpp`

The battery is GenDot, the ill-conditioned dot product generator of Ogita, Rump and Oishi (`generators::gen_dot`; see [Ill-Conditioned Sums and Dot Products](../generators/ill-conditioned.md)). It uses 40 instances of length 100 per anticipated condition number. The achieved condition number is typically about 10 times the anticipated one. The table gives the geometric mean of the relative error:

| type | cond | plain | fma | **compensated** | exact |
|---|---:|---:|---:|---:|---:|
| float | 1e4 | 2.8e-3 | 2.9e-3 | **1.3e-8** | 1.3e-8 |
| float | 1e6 | 0.20 | 0.19 | **4.4e-8** | 1.5e-8 |
| float | 1e8 | 9.6 | 11 | **1.7e-6** | 1.5e-8 |
| float | 1e12 | 1.0e5 | 1.1e5 | **1.5e-2** | 1.6e-8 |
| double | 1e8 | 2.8e-8 | 3.8e-8 | **2.9e-17** | 2.9e-17 |
| double | 1e12 | 4.1e-4 | 3.3e-4 | **3.2e-17** | 3.2e-17 |
| double | 1e16 | 4.8 | 4.3 | **9.5e-16** | 3.5e-17 |
| double | 1e24 | 1.1e8 | 1.3e8 | **2.6e-8** | 2.8e-17 |

**What is asserted, on every instance:**
- **Exact.** The exact accumulator returns the correctly rounded result, and its error is at most that of plain, FMA and compensated. This is provable: the correctly rounded value is the nearest representable number, so no configuration can be closer.
- **Plain and FMA** stay within the classical bound `γ_n · Σ|xᵢyᵢ|`, with `γ_n = n·u / (1 − n·u)`.
- **Compensated** stays within the Dot2 bound `u·|x·y| + γ_n² · Σ|xᵢyᵢ|`.

On the means, compensated is below plain and FMA at every condition number, and within 1.5× of exact while `cond·u ≤ 0.01`.

**Reading.** Plain and FMA lose accuracy in proportion to `u·cond` from the start. Compensated stays at the exact level until the condition number nears `1/u`, about 1.7e7 in float and 9e15 in double, and then grows as `u²·cond`. Exact is correctly rounded throughout. [Accurate Sum and Dot Product](../examples/accurate-sum-dot.md) extends this picture to K-fold precision and condition numbers up to 1e120.

**`gemm`.** The test multiplies 24×64 by 64×24 in float. Each diagonal entry of C is a GenDot product with condition number 1e6. Every entry of the exact result is correctly rounded. Every entry of plain, FMA and compensated stays within its bound.

### Finding: FMA is not more accurate than plain on a whole sum

`err(fma) ≤ err(plain)` is a **per-term** guarantee: FMA removes the rounding of each product. It is not an ordering of whole sums. On an ill-conditioned dot product the error comes from cancellation in the running sum, and FMA does not touch that. In the table above:
- FMA is ahead of plain at float 1e6 and double 1e12;
- FMA is behind plain at float 1e8 and double 1e8.

Which of the two wins depends on the instance, and even the mean flips. Only the exact accumulator dominates. So the tests never assert `err(fma) ≤ err(plain)` on the battery, and neither should a test of your own. Assert `err(exact) ≤ min(err(plain), err(fma))` and the bounds.

FMA wins strictly where the product rounding *is* the whole error. Take `dot([−fl(a·b), a], [1, b])`: plain returns exactly 0, while FMA and exact return `a·b − fl(a·b)` exactly. The test checks this on 2000 float samples.

## Plain sums: `sum` and `trace`

`[586:sum]`, `tests/unit/operation/test_sum_trace_accumulator.cpp`

- **How a sum enters.** Each element enters as `add_product(acc, x, 1)`, and `x·1` is exact. FMA then rounds the same `acc + x` once that the plain sum does, so **plain and FMA are bit-identical for a plain sum**.
- **Result.** On 20 cancelling float sums of 1000 terms each, over 2^-12 .. 2^12, exact is correctly rounded on all 20. Plain, and so FMA, is correctly rounded on none.
- **Constructed case.** For the sum `2²⁴ + 1 − 2²⁴`, plain gives 0 and exact gives 1. The same holds for `trace` of the matching diagonal.

## Sparse products: `spgemm`

`[586:spgemm]`, `tests/unit/operation/test_spgemm_accumulator.cpp`

This is a random cancelling product of two 40×60 and 60×40 float matrices, with magnitudes 2^-12 .. 2^12, 1589 entries in all. The reference for each entry is an independent binary64 Kulisch sum of the exact products, rounded once to float.

| | plain | fma | exact |
|---|---:|---:|---:|
| entries not correctly rounded | 584 | 446 | **0** |

## Norms: `two_norm` and `frobenius_norm`

`[586:norms]`, `tests/unit/operation/test_accumulation_configs_mt.cpp`

The input is float, and the result is double. The reference is the square root of the exact sum of squares.

| input | plain | fma | exact |
|---|---|---|---|
| 1000 entries near 1e30 | `inf` | `inf` | correct |
| 1000 entries near 1e-30 | 0 | 0 | correct |
| a 40×30 matrix mixing 1e25, 1 and 1e-28 (`frobenius_norm`) | `inf` | `inf` | correct |

- **Why it works.** The accumulator path does no scaling. The exact accumulator's range covers every square of a float, and that alone avoids overflow and underflow.
- **A wide plain accumulator also survives.** `frobenius_norm<double, double>` of the same float matrix lands within 1e-12 of the exact value. For float data, accumulating in double is the cheap fix.

## Direct solvers and refinement

`[586:direct]`, `[586:refine]`, `tests/unit/operation/test_accumulation_solvers_mt.cpp`

The test factors and solves Hilbert(n) in float, `x(i) = (1 + (7919·i) mod 13) / 13`, `b = A·x`. Each entry is the relative residual, computed exactly:

| | n | plain | fma | exact |
|---|---|---:|---:|---:|
| sparse LU | 5 | 2.4e-8 | 1.5e-8 | **1.1e-9** |
| | 8 | 1.5e-8 | 3.5e-8 | **2.9e-9** |
| sparse Cholesky | 5 | 3.0e-8 | 2.1e-8 | **7.2e-9** |
| | 8 | 2.6e-8 | 1.3e-8 | **1.1e-9** |
| dense LU | 5 | 1.8e-8 | 8.5e-9 | **1.1e-9** |
| | 8 | 2.8e-8 | 1.3e-8 | **2.9e-9** |
| dense Cholesky | 5 | 3.0e-8 | 2.1e-8 | **7.2e-9** |
| | 8 | 2.6e-8 | 1.3e-8 | **1.1e-9** |
| dense `inv`, `x = inv(A)·b` | 5 | 1.4e-3 | 3.1e-4 | **2.5e-5** |
| | 8 | 1.7e-1 | 6.9e-1 | **3.8e-3** |

- **What is asserted.** For n = 5..8, exact is at least 2× below both plain and FMA for LU and Cholesky, and at least 4× for `inv`. The measured margins are larger: 4.3× for dense LU, 2.9× for dense Cholesky, 10× for `inv`, and 2.8× for the sparse factorizations. FMA is worse than plain in two cells (sparse LU at n = 8, `inv` at n = 8), so the finding above holds here as well.
- **Hilbert(9).** Its smallest eigenvalue is about 5e-13. Under plain and FMA, sparse and dense Cholesky both stop with *not positive definite*. Under exact, both factor it, with a residual below 1e-7.
- **Refinement.** This is the strongest result. The factor is in float, and the residual is in double (`supernodal_lu_solve_refined<float, Acc>`, Hilbert(n), n = 9..11). Plain and FMA stall at relative residuals between 1.2e-4 and 5.3e-4. Exact converges to 1.1e-12 or better. Only the accumulator differs between the three runs.

### LDLᵀ: exact is lowest, since the three-factor split

The terms of LDLᵀ have three factors, `L(i,k)·d_k·L(j,k)`. The product `L(i,k)·d_k` used to be rounded before it reached the accumulator, and LDLᵀ showed no ordering. Since #602, an exact, compensated or wider accumulator receives that product as two exact pieces (`math::two_product`), so the whole term reaches it. Dense and sparse LDLᵀ give the same numbers. Hilbert(n), n = 5..10: exact is lowest at every n, by 1.06× at n = 6 (2.0e-8 / 1.3e-8 / 1.2e-8) up to 28× at n = 9 (3.4e-7 / 1.3e-7 / 1.2e-8). The ordering is asserted for n = 5..8.

The split's own gain is measured over an ensemble in `tests/unit/operation/test_three_factor_terms.cpp`. There are 300 perturbed Hilbert matrices per n, in float, using ILDL; dense LDLᵀ agrees. The values are the geometric mean of `‖A·x − 1‖∞`, computed in double:

| n | plain | fma | exact, pre-rounded terms | exact, split terms |
|---|---:|---:|---:|---:|
| 6 | 5.4e-7 | 4.2e-7 | 3.2e-7 | 2.8e-7 |
| 8 | 7.1e-7 | 5.8e-7 | 4.1e-7 | 3.5e-7 |
| 12 | 1.4e-6 | 1.1e-6 | 7.0e-7 | 5.4e-7 |
| 20 | 2.9e-6 | 2.3e-6 | 1.4e-6 | 9.1e-7 |

The split adds 1.1–1.6× on top of exact accumulation, growing with n. The compensated accumulator takes the split too, and matches the exact split's mean to two digits: 5.3e-7 at n = 12 and 8.6e-7 at n = 20, over 100 draws. That is asserted, within 5%.

### No ordering: sparse QR and Bunch–Kaufman

- **Sparse QR.** The three are within a small factor and swap places. At n = 8 they are 1.9e-7 / 1.1e-7 / 5.0e-7. The test asserts only that each one solves.
- **Dense Bunch–Kaufman LDLᵀ** (`ldlt_bk`), on Hilbert(n) and on the indefinite saddle `[[H, 1], [1ᵀ, 0]]`, n = 4..9. FMA is sometimes below exact: Hilbert(6) gives 4.9e-8 / 1.1e-8 / 1.5e-8, and the saddle at n = 7 gives 9.4e-8 / 2.3e-6 / 8.0e-8. The cause is not the three-factor rounding: its terms are `L(i,p)·W(j,p)`, with W the stored Schur column, so no product is rounded early. The test asserts only that each one solves, to below 1e-5.

## Preconditioners

`[586:pc]`, `tests/unit/itl/test_pc_accumulator.cpp`

**The ordering follows the length of the reductions, not which preconditioner it is.** Residuals are `‖A·M⁻¹·b − b‖∞` in float, computed in double, and given as plain / fma / exact.

- **Stencil patterns: no ordering.** A 5-point stencil gives reductions of at most four terms, and a tridiagonal gives one. A one-term update `A(i,j) − L·U` rounds twice in plain arithmetic but once with FMA, so FMA and exact agree there, and there is little for an accumulator to keep.
  - ILU(0) on a tridiagonal, where it is exact LU, n = 1024: 3.5e-7 / 3.0e-7 / 3.0e-7.
  - ILU(0) and IC(0) on a 5-point Laplacian, n = 256: 4.0e-1 / 4.0e-1 / 4.0e-1. The dropped fill is the whole error.
- **Dense pattern: the direct-solver ordering returns.** On a full pattern, ILU(0) and IC(0) are the complete factorizations. Hilbert(n), b = 1:

  | n | IC(0) | ILU(0) |
  |---|---|---|
  | 5 | 1.4e-5 / 4.3e-6 / **1.0e-6** | 9.7e-6 / 3.7e-6 / **3.1e-7** |
  | 8 | 2.0e-4 / 2.9e-4 / **3.9e-5** | 2.0e-4 / 2.5e-4 / **5.4e-6** |
  | 9 | NaN / NaN / **7.0e-5** | 2.0e-4 / 2.6e-4 / **2.1e-6** |

  The test asserts, for n = 5..8, that exact is at least 2× below plain and FMA for IC(0) and at least 4× for ILU(0).

  At n = 9, IC(0) breaks down under plain and FMA: a pivot cancels below zero, and its square root is NaN. The exact accumulator still factors it, as dense Cholesky does.
- **ILUT** with no dropping on a dense pattern is the same elimination as ILU(0) and gives the same numbers, bit for bit. Exact is asserted at least 4× below both.
- **ILDL** has three-factor terms and takes the #602 split. Exact is lowest; its last column below is the exact accumulator fed pre-rounded terms:

  | n | plain | fma | exact | exact, pre-rounded |
  |---|---:|---:|---:|---:|
  | 5 | 2.2e-6 | 9.2e-6 | **1.7e-6** | 2.7e-6 |
  | 6 | 9.2e-5 | 2.4e-5 | **9.2e-6** | 1.7e-5 |
  | 7 | 7.8e-5 | 9.0e-5 | **2.0e-5** | 8.9e-5 |
  | 8 | 2.9e-4 | 2.0e-4 | **5.0e-6** | 4.3e-5 |

  Single matrices are noisy; the ensemble in the LDLᵀ section above is the steadier measure.

## Krylov solvers

`[586:krylov]`, `tests/unit/operation/test_accumulation_solvers_mt.cpp`

**No ordering in float.** The test runs CG on a 400-unknown 2-D Laplacian, symmetrically scaled by a factor of 1 and of 10, and GMRES(20) on a convection–diffusion problem. Iteration counts and true residuals agree to a few percent across plain, FMA and exact. The test asserts only that each one converges, to a true residual below 4e-6 in under 1000 iterations.

**Why.** The accumulator covers each dot product and each sparse matrix-vector product. But the recurrences update x, r and p in float, and each iteration rounds them again. A better reduction inside an iteration cannot undo the rounding of the iteration itself. The [architecture page](../architecture/accuracy-architecture.md) discusses this.

The original result that a quire helps CG (#237, #238) was found with posits, whose tapered precision may be what made the difference there. Its retest is a peer-repo row (see below).

## Eigensolvers and the SVD

`[586:eigen]`, `[586:svd]`

**`power_iteration` on a cancelling operator: exact converges, plain and FMA do not.**
- **The operator.** `A = P⁻¹·D·P` is exactly representable in float, with largest eigenvalue `2n`. Its matrix-vector product cancels by about four orders of magnitude.
- **The run.** 5000 iterations, tolerance 1e-6. The values are the relative error of the eigenvalue.

| | plain | fma | exact |
|---|---|---|---|
| n = 16 | 5.4e-3, not converged | 2.0e-2, not converged | **8.0e-6, converged in 14 iterations** |
| n = 32 | 8.9e-2 | 8.2e-1 | **6.9e-5** |

This needed the accumulator to reach the eigensolvers' matrix-vector product (#591, #593). Before that, all three gave 5.4e-3 and none converged. FMA is worse than plain here too.

**No ordering**, asserted only to work:
- **`eigenvalue_symmetric`.** The accumulator reaches only the Householder tridiagonalization. The implicit QR iteration that follows works on two- and three-term expressions, not sums. λ_max of Hilbert(4..7) is accurate in all three configurations, and that is asserted. λ_min is not improved. For Hilbert(7) the relative error is 7.6e-3 plain against 0.43 exact; that number is recorded in the test header, not asserted.
- **`power_iteration` on the 1-D Laplacian.** It is identical in all three, because each row of its matrix-vector product has three well-scaled terms.
- **`arnoldi`** on the cancelling operator: no clean ordering. At n = 32, k = 16 it gives 6.4e-3 plain, 9.7e-2 FMA and 1.3e-2 exact. Its Gram–Schmidt updates of w run in float and cancel as heavily as the matrix-vector product does. This is recorded in the test header; no test asserts it.
- **`svd`** (one-sided Jacobi), `tests/unit/operation/test_svd_accumulator.cpp`. Below is the largest relative error over the singular values of a float matrix, against an exact-accumulated double SVD, given as plain / fma / exact / double accumulator:

  | input | errors |
  |---|---|
  | Hilbert(4) | 1.4e-6 / 1.4e-6 / 9.2e-5 / 9.2e-5 |
  | Hilbert(6) | 5.6e-2 / 5.7e-2 / 3.1e-2 / 3.1e-2 |
  | Hilbert(8) | 3.8 / 0.35 / 3.0 / 3.0 |

  Jacobi's error comes from *applying* the plane rotations, rounded in float at every sweep. The inner products only choose the rotation angle, and an exact angle applied in float is no better than a rounded one. These numbers are recorded in the test header. The test asserts that every configuration gives a valid SVD.

## Determinism

`[586:determinism]`, `tests/unit/operation/test_accumulation_configs_mt.cpp` and `test_accumulation_solvers_mt.cpp`

- **Exact is independent of summation order.**
  - Shuffling a 2000-term GenDot input ten times leaves the exact result bit-identical. Plain and FMA move.
  - A 300 000-term `dot` with an exact accumulator gives the same bits at any `MTL5_NUM_THREADS`. Plain and FMA do not.
  - Ten iterations of CG on a 2-D Laplacian with 160 000 unknowns give the same bits threaded and serial under exact. Under plain float they do not.
- **Compensated** merges its per-thread partials with a `two_sum`. The result stays within its bound, but it is not bit-identical across thread counts.
- **`Accumulator = void` is configuration 1 at the same type** for the native GEMM and GEMV only. It does not hold under BLAS or `MTL5_NATIVE_FAST_GEMM`, where `void` picks kernels that block the sum or split it across SIMD lanes. `dot` and `two_norm` with `void` always take a SIMD reduction, so it does not hold for them either.

Which kernels thread, and how, is in [On-Node Threading](on-node-threading.md).

## Configuration 3 with real quires

MTL5 stays independent of the Universal number library, so its configuration-3 rows use the exact accumulator for `float` and `double`. The rows with Universal's quires belong in the peer repositories that depend on both libraries:
- the BLAS level in mp-blas;
- the Krylov solvers in mp-iterative, including the posit + quire CG retest (#237, #238);
- direct solvers and refinement in mp-ir or mp-numerics.

Each should carry the `[586:<row>]` tag it mirrors. What to assert depends on the row:
- **Where the exact result is correctly rounded** (`dot`, `gemm`, `sum`, `spgemm`, norms), `err(3) ≤ min(err(1), err(2))` is guaranteed: no representable value is closer. Assert it with the bounds, and never assert a strict chain such as `err(3) ≤ err(2) ≤ err(1)`.
- **For solvers**, assert what was measured, with a margin, as the MTL5 rows above do. A solver's output is not one correctly rounded reduction, and there is no guaranteed ordering: Bunch–Kaufman has a case where FMA's residual is below exact's.

As of this writing those rows are not yet written; they are tracked in #586.

## Which accumulator when

| situation | use | why |
|---|---|---|
| a long, cancelling reduction in a dot, `gemm`, `spgemm` or sum | exact (or compensated) | the error is cancellation; only keeping the low bits helps |
| a moderately conditioned dot or sum, cost matters | compensated | exact-level accuracy up to `cond ≈ 1/u` at a small constant cost (see [Accurate Sum and Dot Product](../examples/accurate-sum-dot.md)) |
| norms of float data with a wide range | a `double` accumulator, or exact | avoids overflow and underflow without scaling |
| dense or sparse LU, Cholesky, LDLᵀ, `inv`, and their solves | exact | each entry of the factor is one long reduction; residuals drop by 3–10× and near-singular SPD matrices still factor |
| mixed-precision iterative refinement with a low-precision factor | exact, in the factor | the one case where it decides whether refinement converges |
| full-pattern ILU(0), IC(0), ILUT, ILDL | exact | as the direct solvers |
| stencil-pattern preconditioners, Krylov solvers in float | anything; plain is enough | reductions are short, or the recurrence rounds every iteration |
| QR, Bunch–Kaufman, `eigenvalue_symmetric`, SVD | anything; no ordering | the error is in rotations, reflector application or pivoting, not in a reduction |
| reproducible results across thread counts | exact | associative, so the result is the same for every partition |
| a product's rounding is the whole error | FMA | removes the product rounding per term; it is no help against cancellation |

## Related

- [The Accuracy Architecture](../architecture/accuracy-architecture.md): why each algorithm behaves as it does
- [Mixed-Precision Kernels](mixed-precision-kernels.md): element, accumulate and result precision
- [Measuring Solver Accuracy](measuring-solver-accuracy.md): residuals, norms and forward error
- [On-Node Threading](on-node-threading.md): which kernels thread and which stay bit-identical
- [Exact Accumulator](../architecture/exact-accumulator.md), [Error-Free Transformations](../architecture/error-free-transformations.md)
- [#586](https://github.com/stillwater-sc/mtl5/issues/586): the configuration matrix; [#592](https://github.com/stillwater-sc/mtl5/issues/592): this page
