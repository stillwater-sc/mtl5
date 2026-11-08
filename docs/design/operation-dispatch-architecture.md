# The operation dispatch layer: one API, many kernels, chosen at compile time

How MTL5 turns a single call — `mult(A, x, y)` — into whichever kernel the operand
types and the build configuration allow: an external BLAS `gemv`, MTL5's own SIMD
GEMV, a sparse CRS traversal, or the portable generic loop — and *why* every one
of those choices is made at **compile time**, with no runtime branch, no virtual
call, and no policy object.

This is the layer the container docs keep gesturing at when they say an operation
is "selected by the category tag." The containers
([`dense2D`](../architecture/containers/dense2d-architecture.md),
[`compressed2D`](../architecture/containers/compressed2d-architecture.md), …) carry
type-level identity — `value_type`, `orientation`, `category`, `ashape`,
contiguity. The [expression layer](expression-template-architecture.md) produces
typed operands. This layer *reads* that identity, together with the build's
feature macros, and routes to a kernel. It is the traffic controller between typed
operands and the code that actually computes.

It lives in
[`operation/`](https://github.com/stillwater-sc/mtl5/tree/main/include/mtl/operation)
(the public free functions and generic kernels) and
[`interface/`](https://github.com/stillwater-sc/mtl5/tree/main/include/mtl/interface)
(the guarded bindings to external libraries). The archetype is
[`operation/mult.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/operation/mult.hpp);
this page reads it closely.

---

## Decision 1 — operations are free functions, separate from containers

`mult`, `dot`, `lu_factor`, `two_norm`, `eigenvalue_symmetric` are **free
functions** in `mtl::`/`mtl::operation`, not member functions of the matrix and
vector classes. The containers hold data and expose their identity through traits;
the algorithms live outside and dispatch on that identity.

This is the classic generic-programming split — data structures and algorithms are
orthogonal — and it is load-bearing here. A free function can be overloaded and
concept-constrained on operand type, and can select an *external* implementation,
without any container knowing those implementations exist. `dense2D` has no idea
BLAS is a thing; `mult` does. That separation is what lets the same container flow
to a native kernel in one build and an MKL kernel in another with zero change to
the container.

---

## Decision 2 — dispatch is a compile-time `if constexpr` ladder

The public `mult(M, x, y)` is a single function template whose body is a chain of
`if constexpr` branches, each guarded by a compile-time predicate:

```cpp
template <typename Accumulator = void, Matrix M, Vector VIn, Vector VOut>
void mult(const M& A, const VIn& x, VOut& y) {
    if constexpr (interface::is_compressed2D_v<M>) {
        detail::mult_sparse_crs<Accumulator>(A, x, y);            // sparse: CRS traversal
    } else if constexpr (!interface::accumulator_allows_blas_v<Accumulator>) {
        detail::mult_generic<Accumulator>(A, x, y);              // custom accumulator: generic
    } else {
#ifdef MTL5_HAS_BLAS
        if constexpr (interface::BlasDenseMatrix<M> && ...) { interface::blas::gemv(...); return; }
#endif
#ifdef MTL5_NATIVE_FAST_GEMM
        if constexpr (interface::BlasDenseMatrix<M> && ...) { /* SIMD GEMV */ return; }
#endif
        detail::mult_generic(A, x, y);                           // portable floor
    }
}
```

Because every branch is `if constexpr`, the compiler instantiates **only** the
selected path for a given set of template arguments. There is no runtime type
test, no dispatch table, no branch in the emitted code — the dead arms are
discarded during instantiation. The public API is one symbol; the machine code is
whatever the operands and build permit. That is the entire design goal: **a
uniform interface with hand-kernel performance, resolved statically.**

---

## Decision 3 — a small predicate vocabulary factors out eligibility

The branch conditions are not re-derived in every operation. They are named,
reusable predicates in
[`interface/dispatch_traits.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/interface/dispatch_traits.hpp):

| Predicate | Asks |
|---|---|
| `is_blas_scalar_v<T>` | is `T` a `float` or `double`? (what standard BLAS/LAPACK support) |
| `BlasDenseMatrix<M>` / `BlasDenseVector<V>` | float/double element **and** a contiguous `data()` **and** shape? |
| `is_row_major_v<M>` | row-major layout? (drives the orientation bridge, Decision 8) |
| `is_compressed2D_v<M>` | is this the CRS sparse matrix? (route to the sparse kernel) |
| `accumulator_allows_blas_v<Acc>` | is the accumulator the default `void`? (Decision 6) |
| `is_suitesparse_eligible_v<M>` | CRS + float/double? (route to an external sparse solver) |

Every operation gates on this shared vocabulary rather than open-coding "is it
float, contiguous, right orientation." Eligibility logic is written once and read
everywhere, so `mult`, `dot`, `lu_factor`, and the rest agree on what "can be
accelerated" means. A concept like `BlasDenseMatrix` is the precise contract an
operand must meet to be handed to a Fortran routine — float/double, a real
pointer, and known dimensions — and nothing weaker slips through.

---

## Decision 4 — the build *is* the backend

The acceleration branches are wrapped in `#ifdef MTL5_HAS_BLAS` /
`#ifdef MTL5_NATIVE_FAST_GEMM`, preprocessor macros that CMake sets as compile
definitions when the corresponding option is on:

```cmake
if(MTL5_WITH_BLAS)  target_compile_definitions(mtl5 INTERFACE MTL5_HAS_BLAS)  endif()
```

This is deliberate and consequential. In a build **without** BLAS, the BLAS branch
does not merely go unused — it is not compiled at all, so no external symbol is
referenced and the binary carries **no link dependency** on a BLAS library. In a
build **with** BLAS, the branch compiles and the eligible types route to it. There
is no runtime flag to flip and no in-process policy switch: **the build selects the
backend, once, for the whole program.** This is the exact model the
[benchmark methodology](../benchmarks/systems.md) rests on — one binary per
backend, the build *is* the backend — and the operation layer is where that
compile-time choice becomes a kernel call.

---

## Decision 5 — an ordered preference, with a generic floor that is never optional

When more than one path is eligible, the ladder expresses a fixed preference:

```text
external BLAS/LAPACK   (if linked and the operands qualify)
      ↓ else
native-fast SIMD       (MTL5's own blocked GEMM / SIMD GEMV, if built)
      ↓ else
generic C++            (always present)
```

Each accelerated path is a *fallback guard* for the previous one, and the generic
kernel at the bottom has **no** `#ifdef` and **no** eligibility constraint. That is
intentional: the generic loop is the floor that guarantees the operation works for
*any* operand — a custom number type (posit, interval), a non-contiguous view, a
sparse structure, an exotic accumulator — even when nothing above it applies.
Acceleration is opportunistic; **correctness is universal.** The generic kernels
are themselves identity- and accumulator-correct (they use `math::zero<Result>()`
and `accumulator_traits`, not literal `0` and `+=`), so the floor is not a
degraded path — it is the fully-general one that the fast paths are specializations
of.

---

## Decision 6 — correctness constrains the order: the accumulator override

The preference in Decision 5 is *not* simply "fastest first." It is
"correctness-constrained, then fastest," and the accumulator is where that shows.

`mult` takes a template parameter `Accumulator` selecting the precision in which
inner products are summed — the heart of mixed-precision work. External BLAS and
the SIMD kernel accumulate in a **hardware-fixed** precision; they cannot honor a
caller-chosen accumulator or result type. So the predicate
`accumulator_allows_blas_v<Accumulator>` (true only for the default `void`) sits
*above* the acceleration branches:

```cpp
} else if constexpr (!interface::accumulator_allows_blas_v<Accumulator>) {
    detail::mult_generic<Accumulator>(A, x, y);   // any non-default accumulator -> generic
    return;
}
```

A call like `mult<double>(A_float, B_float, C_double)` — fp32 operands summed in
fp64 — therefore **bypasses BLAS entirely**, even though the operands are float and
contiguous and BLAS is linked, because BLAS would silently sum in fp32 and give the
wrong answer for the requested semantics. (For the specific float→double case a
*widening* blocked GEMM is used — accumulate-in-fp64 through the SIMD micro-kernel
— rather than the scalar generic loop, so the mixed path is still fast where a
kernel exists.) The lesson the code encodes: **a request for a particular precision
overrides the speed preference**, because delivering the wrong number quickly is
not an option. This is the library's mixed-precision purpose expressed as a
dispatch rule.

---

## Decision 7 — the interface layer: thin C++ wrappers over the Fortran ABI, guarded

The operation layer never calls a raw Fortran symbol. Everything external is
confined to `interface/`, where each library gets a thin C++ wrapper behind its
guard macro. `blas.hpp` declares the Fortran ABI under `extern "C"` and wraps it:

```cpp
#ifdef MTL5_HAS_BLAS
extern "C" { void dgemv_(const char*, const int*, const int*, const double*, ...); /* … */ }

namespace mtl::interface::blas {
    inline float dot(int n, const float* x, int incx, const float* y, int incy) {
        return sdot_(&n, x, &incx, y, &incy);      // pointers, trailing underscore, hidden here
    }
    // gemv, gemm, trsv, … likewise
}
#endif
```

`mult` calls `interface::blas::gemv('T', …)`, expressing intent; the wrapper deals
with the ABI grit — everything passed by pointer, the trailing-underscore Fortran
names, `int` sizes, `char` flags. The messy, error-prone boundary lives in exactly
one place, per library: `blas.hpp`, `lapack.hpp`, and the sparse-solver bindings
`umfpack.hpp` / `superlu.hpp` / `klu.hpp` / `cholmod.hpp` / `spqr.hpp`. Each is
`#ifdef`-guarded so it compiles only when that library is linked. The operations
stay readable; the ABI stays quarantined.

---

## Decision 8 — the orientation bridge to column-major BLAS, without copies

BLAS is column-major (Fortran); MTL5's `dense2D` may be row- **or** column-major.
Rather than physically transpose a row-major matrix before handing it to BLAS —
an O(n²) copy — the dispatch rewrites the *call* using layout algebra. A row-major
`C = A*B` is the same bytes as a column-major `Cᵀ = Bᵀ·Aᵀ`, so:

```cpp
if constexpr (interface::is_row_major_v<MC>) {
    // C row-major: call gemm with A and B pointers SWAPPED, leading dims re-read
    interface::blas::gemm('N','N', n, m, k, alpha, B.data(), n, A.data(), k, beta, C.data(), n);
} else {
    interface::blas::gemm('N','N', m, n, k, alpha, A.data(), m, B.data(), k, beta, C.data(), m);
}
```

The same trick appears in the GEMV path as a choice of the `'T'`/`'N'` transpose
flag. For a triangular or symmetric operand it also flips which triangle is
stored: a row-major lower `A` is, read column-major, the upper triangle of `Aᵀ`.
So `trmv` and `trsv` on it call `?trmv`/`?trsv('U', 'T', …)` (#747), and `trsm`,
`trmm`, `symm`, `syrk`, `syr2k` and `symv` use the same mapping (#724). The orientation identity the container carries (`is_row_major_v`) is thus
consumed *here*, in the dispatch, to bridge to a fixed-layout library at zero data
movement — the layout algebra lives in the traffic controller, not in the kernels
and not in the container. (It is also the subtlest code in the layer, which is why
it is centralized and covered by tests rather than re-derived per call site.)

---

## Decision 9 — sparse and dense are different kernels under one name

`is_compressed2D_v<M>` routes a CRS matrix-vector product to `mult_sparse_crs`,
which walks `starts`/`indices`/`data` and touches only stored nonzeros (O(nnz)) —
never a dense BLAS `gemv`, which would be O(n²) over mostly zeros. A transposed
sparse view routes to a scatter kernel (`mult_sparse_crs_transposed`). Dense
operands, meanwhile, take the BLAS/native-fast/generic ladder. The *same* public
`mult` name covers all of them; the `category`/type identity decides which body
runs. This is the payoff of the whole tag system the container docs describe: a
`tag::sparse` matrix and a `tag::dense` matrix are the same call and utterly
different code, chosen with `if constexpr`.

---

## What the design deliberately does not do

- **No runtime dispatch or auto-tuning.** There is no policy object, no runtime
  "pick the fastest kernel for this size" heuristic. The backend is the build; a
  given binary always takes the same path for a given type. This is simpler and
  link-lean (no unused backend pulled in), at the cost of compiling per backend —
  the explicit trade the benchmark model makes.
- **Conservative eligibility.** Only float/double, contiguous, correctly-oriented
  operands reach BLAS; everything else falls to generic. Acceleration is offered
  only where it is provably safe, so the default is always a correct answer, never
  a fast-but-wrong one.
- **Some dispatch duplication.** The `if constexpr` ladders recur across
  operations; the `dispatch_traits` vocabulary shrinks but does not erase the
  repetition. The trade is explicit, greppable, zero-cost dispatch per operation
  over a centralized runtime mechanism that would cost a branch and hide the logic.

---

## Where it sits in the stack

```text
containers  (dense2D / compressed2D / dense_vector / … — carry value_type, orientation,
   │         category, ashape, contiguity)
   │
expressions (build typed operands; mat_mat_times_expr defers a product to here)
   │
▶ OPERATION DISPATCH  (operation/ + interface/dispatch_traits.hpp)
   │   reads operand type + build macros, selects a kernel via if constexpr
   ├─▶ generic C++ kernels            (operation/*.hpp — the universal floor)
   ├─▶ native-fast SIMD kernels       (detail/gemm_blocked.hpp, detail/gemv.hpp — see
   │                                   the BLAS kernel architecture)
   └─▶ external libraries             (interface/blas.hpp, lapack.hpp, superlu/klu/…)
```

It is the hinge between "what the operand *is*" (the container/expression type
identity) and "what code runs" (a specific kernel). Everything above it is about
representing values; everything below it is about computing fast; this layer maps
one to the other, statically.

---

## File map

| File | Role |
|---|---|
| [`operation/mult.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/operation/mult.hpp) | the archetype: mat·vec / mat·mat dispatch + generic and sparse kernels |
| [`operation/`](https://github.com/stillwater-sc/mtl5/tree/main/include/mtl/operation) | ~80 free-function operations (`dot`, `norms`, `lu`, `qr`, `cholesky`, `svd`, transcendentals, …) following the same pattern |
| [`interface/dispatch_traits.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/interface/dispatch_traits.hpp) | the compile-time eligibility predicate vocabulary |
| [`interface/blas.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/interface/blas.hpp) / [`lapack.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/interface/lapack.hpp) | guarded C++ wrappers over the Fortran BLAS/LAPACK ABI |
| [`interface/`](https://github.com/stillwater-sc/mtl5/tree/main/include/mtl/interface) sparse bindings | `umfpack` / `superlu` / `klu` / `cholmod` / `spqr` external solver wrappers |
| [`math/accumulator_traits.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/math/accumulator_traits.hpp) | the Element→Accumulate→Result model the mixed-precision path uses |

For the SIMD kernels this layer dispatches to, see the
[BLAS kernel architecture](blas-kernel-architecture.md); for the expression whose
matrix product defers here, the
[expression-template layer](expression-template-architecture.md); for why the build
selects the backend, the [benchmark systems](../benchmarks/systems.md) page.
