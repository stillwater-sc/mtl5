# The ITL iterative solvers: Krylov methods as four composable roles

How MTL5 solves `A·x = b` iteratively, and *why* the layer is built as four
independent, swappable roles rather than a family of monolithic solver classes. A
Krylov method — Conjugate Gradient, BiCGSTAB, GMRES, MINRES, … — is expressed once,
generically, and combined at the call site with *any* linear operator, *any*
preconditioner, and *any* convergence policy. The Cartesian product of those
choices is available with no combinatorial code, because each is a template
parameter with a small behavioral contract.

This is the layer that sits **atop** the [operation dispatch
layer](operation-dispatch-architecture.md): a Krylov solver is an *algorithm built
from* matrix-vector products and dot products, composing `mtl::mult` and `mtl::dot`
into a higher-level recurrence. Where the direct solvers
([sparse direct solvers](../sparse-direct-solvers-design.md)) factor `A`, the
iterative solvers only ever *apply* it.

The subsystem lives in
[`itl/`](https://github.com/stillwater-sc/mtl5/tree/main/include/mtl/itl)
(namespaces `mtl::itl`, `mtl::itl::pc`, `mtl::itl::smoother`). The archetype is
[`itl/krylov/cg.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/itl/krylov/cg.hpp);
this page reads it closely.

---

## The canonical call: `solve(A, x, b, M, iter)`

Every Krylov solve assembles the same four roles:

```cpp
compressed2D<double>       A   = /* the operator (typically sparse) */;
dense_vector<double>       x(n), b = /* right-hand side */;
itl::pc::ilu_0<...>        M(A);                       // preconditioner, built from A
itl::basic_iteration<double> iter(b, /*max*/1000, /*rtol*/1e-10);  // stopping policy

itl::cg(A, x, b, M, iter);      // run the algorithm
int err = iter;                 // 0 = converged, 1 = max_iter exceeded
```

Four collaborators — **operator** `A`, **solution/rhs** vectors, **preconditioner**
`M`, **iteration** `iter` — each varying independently. Swap `cg`→`gmres` for a
different Krylov method, `A` sparse→dense→matrix-free, `M` `identity`→`ilu_0` for a
different preconditioner, `basic_iteration`→`cyclic_iteration` for different
reporting. Nothing else changes.

---

## Decision 1 — solvers are free functions over four independent roles

`cg` is a free-function template, not a class:

```cpp
template <typename LinearOp, typename VecX, typename VecB,
          typename PC, typename Iter, typename Accumulator = void>
int cg(const LinearOp& A, VecX& x, const VecB& b, const PC& M, Iter& iter);
```

The five behavioral collaborators (`LinearOp`, the two vector types, `PC`, `Iter`)
are template parameters, each with a minimal contract the body exercises. There is
no `Solver` base class, no shared inheritance, no runtime dispatch — you select a
method by *calling its function*. This mirrors the
[operation layer](operation-dispatch-architecture.md)'s free-function philosophy
(algorithms are orthogonal to the data they act on) and is what lets the same CG
recurrence run against a `compressed2D`, a `dense2D`, or a matrix-free operator
with no change and no virtual-call overhead.

---

## Decision 2 — the operator is duck-typed: it only needs `A * p`

CG never names `A`'s concrete type. It applies `A` through one helper:

```cpp
apply_operator<Accumulator>(A, p, q);   // q = A * p  -- the ONLY thing CG asks of A
```

Krylov methods, mathematically, need nothing from `A` but the ability to compute
`A·v` for a vector `v`. The code honors that exactly: `LinearOp` is an unconstrained
template parameter, and the *only* operation performed on it is `apply_operator`
(`include/mtl/itl/krylov/apply_operator.hpp`). A `Matrix` goes to
`mtl::mult<Accumulator>`; anything else must support `A * x`. The consequences are
the payoff of the whole abstraction:

- **Sparsity is transparent.** A `compressed2D` operator routes `mult` to the CRS
  matvec (O(nnz)); a `dense2D` routes it to BLAS / native-fast. CG is identical for
  both — the [dispatch layer](operation-dispatch-architecture.md) picks the kernel,
  the solver is oblivious.
- **Matrix-free by construction.** `A` need not be a stored matrix at all. Any type
  with `A * x` — a stencil applied on the fly, an operator that never materializes
  its entries — is a valid `LinearOp`. This is the headline capability of a Krylov
  library, and here it costs nothing: it is simply the absence of any requirement
  beyond `A * x`. BiCG and QMR also apply `Aᵀ`; a matrix-free operator provides it
  as `trans(A)`, found by argument-dependent lookup, through `apply_transpose`.
  (Between the accumulator rollout, #254, and #278 the solvers called
  `mtl::mult` directly, which is constrained to `Matrix`, so a matrix-free
  operator did not compile. `tests/unit/itl/test_krylov_matrix_free.cpp` now
  runs all ten solvers on one.)

The contract is *structural* (a `Matrix`, or `A * x`), not a hard concept bound on
the signature — deliberately, so the matrix-free door stays open.

---

## Decision 3 — the preconditioner is a concept: `solve` / `adjoint_solve`

A preconditioner `M` approximates `A⁻¹`, and the solver applies it with
`M.solve(z, r)` (compute `z ≈ A⁻¹ r`). The contract is a two-line concept:

```cpp
template <typename P, typename X>
concept Preconditioner = requires(const P& p, X& x, const X& b) {
    { p.solve(x, b) };
    { p.adjoint_solve(x, b) };
};
```

Anything satisfying it plugs in, and the shipped preconditioners span the full
range of statefulness behind that one uniform face:

| Preconditioner | State | `solve(x, b)` |
|---|---|---|
| `identity` | none | `x = b` (no-op) |
| `diagonal` (Jacobi) | stores `inv(diag A)` | `x(i) = inv_diag(i) · b(i)` |
| `ilu_0` / `ilut` / `ic_0` / `ildl` | an incomplete factorization | a forward/back substitution |
| `ssor`, `block_diagonal` | sweeps / blocks | the corresponding approximate solve |

`identity` stores nothing and copies; `diagonal` precomputes the inverse diagonal
in its constructor from `A`; the incomplete factorizations precompute a sparse
`L`/`U`. All of them present `solve`, so the solver treats "no preconditioner" and
"a full ILU" identically — `M.solve` is one call either way. `adjoint_solve`
(apply `Mᵀ`, or `conj` of the diagonal) is required uniformly so the non-symmetric
methods (BiCG, QMR) that need `Mᵀ` compose the same way; for symmetric
preconditioners it equals `solve`.

---

## Decision 4 — the iteration object owns the stopping policy

The single most important separation in the layer: **the solver expresses the
recurrence; the iteration object decides when to stop.** CG's loop is

```cpp
while (!iter.finished(r)) {        // <-- convergence test lives in iter, not cg
    ++iter;
    /* ... the CG recurrence: p, q = A·p, alpha, x += alpha p, r -= alpha q, z = M⁻¹r ... */
}
return iter;                       // convertible to int error code
```

Everything about *when to stop* is in `basic_iteration`, not in `cg`:

```cpp
bool converged(Real r) const { return r <= rtol_ * norm_r0_ || r <= atol_; }
bool finished(Real r) {
    resid_ = r;
    if (converged(r))     { error_ = 0; return true; }   // relative or absolute tolerance met
    if (i_ >= max_iter_)  { error_ = 1; return true; }   // budget exhausted
    return false;
}
```

It tracks the iteration count, the current and initial residual, the relative
(`rtol`) and absolute (`atol`) tolerances, and a max-iteration budget; it exposes
`iterations()`, `resid()`, `relresid()`, and a `fail()` hook for solver breakdown.
Because it is a separate object:

- **Convergence policy changes without touching any solver.** Tighter tolerance,
  more iterations, a different stopping rule — all live in the iteration object.
- **Result and status ride back together.** The solver *returns* `iter`, which
  converts to an `int` error code, while the caller keeps the same object to read
  `iterations()`/`resid()`. One object is both the loop state and the report.

The solver owns numerics; the iteration owns control. Neither leaks into the other.

---

## Decision 5 — reporting variants by (compile-time) inheritance of the controller

The iteration controllers are the one place the layer uses a small inheritance
hierarchy, and it is for implementation reuse, not runtime polymorphism:

```text
basic_iteration<Real>            convergence test + state (the base)
   ├── cyclic_iteration<Real>    prints the residual every `cycle` iterations
   └── noisy_iteration<Real>     prints every iteration
```

`cyclic_iteration::finished` calls `base::finished` and then prints — it *refines*
the base behavior:

```cpp
bool finished(Real r) {
    bool done = base::finished(r);
    if (this->i_ % cycle_ == 0) out_ << "iteration " << this->i_ << " resid " << r << ...;
    return done;
}
```

Crucially, the solver is templated on the concrete `Iter` type and calls
`iter.finished(r)` on it, so the right `finished` is chosen **at compile time** by
the actual type passed — static dispatch, zero overhead, no `virtual`. The
inheritance exists only so `cyclic`/`noisy` reuse the base's residual/tolerance
machinery instead of copying it. It is the exception that proves the rule: MTL5
reaches for inheritance precisely when the variation is a behavioral refinement of
a shared stateful contract, and even then binds it statically.

---

## Decision 6 — the Cartesian product for free

The four roles are orthogonal, so their combinations come without combinatorial
code. Conceptually:

```text
{ cg, bicg, bicgstab, cgs, gmres, minres, qmr, tfqmr, idr_s }   solvers
        ×
{ dense2D, compressed2D, transposed view, matrix-free, … }      operators
        ×
{ identity, diagonal, ilu_0, ilut, ic_0, ildl, ssor, … }        preconditioners
        ×
{ basic, cyclic, noisy }                                        iteration policies
```

Every cell of that product is reachable by writing one call, because none of the
four axes knows about the others — the solver is generic over the operator and
preconditioner types, the preconditioner is generic over the matrix, the iteration
is generic over the vector. Ten solvers do not need forty preconditioner-aware
variants; each solver is written once against the contracts. That non-multiplication
is the entire reason the "template" is in "Iterative Template Library."

---

## Decision 7 — mixed precision threads through, from the operation layer

Every solver carries an optional `Accumulator` template parameter, forwarded to the
operations it composes:

```cpp
apply_operator<Accumulator>(A, p, q);   // mtl::mult<Accumulator> for a Matrix
value_type rho = mtl::dot<Accumulator, value_type>(r, z);
```

So the matrix-vector products and the inner products of the Krylov recurrence can
accumulate in a precision distinct from the operand type — a fp32 operator with
fp64 accumulation, say — reusing the exact
[dispatch-layer mechanism](operation-dispatch-architecture.md#decision-6--correctness-constrains-the-order-the-accumulator-override).
The solver does not implement mixed precision itself; it *passes the request
through* to `mult`/`dot`, so the library's mixed-precision story extends into the
iterative methods without the solver knowing how it is honored. `Accumulator = void`
(the default) leaves the ordinary dispatch untouched.

A matrix-free operator does its own accumulation inside `A * x`, so a non-void
`Accumulator` cannot reach its matvec. The solvers accept the pair anyway, and the
accumulator then governs only their dot products. This is option (b) of #278, chosen
to match the eigensolvers' `ev_matvec` (#591): rejecting the combination at compile
time would forbid a reasonable request — accurate inner products around an operator
the caller controls — for the sake of a contract the caller can already read.

---

## The wider `itl/` family

The Krylov + preconditioner + iteration core is the heart of the subsystem, but the
`itl/` umbrella gathers three adjacent iterative-method families that share its
composable spirit:

- **Krylov eigensolvers** (`itl/eigen/`: Arnoldi, Lanczos, power iteration) — the
  same subspace idea aimed at eigenpairs instead of a linear solve.
- **Smoothers** (`itl/smoother/`: Jacobi, Gauss-Seidel, SOR) — stationary iterations,
  used primarily as multigrid components.
- **Multigrid** (`itl/mg/`: restriction, prolongation, multigrid) — composes
  smoothers with grid-transfer operators.

They round out the iterative-methods toolbox; the design principles above (generic
over operator and vector, policy objects for control) recur throughout.

---

## What the design deliberately does not do

- **No solver base class or runtime solver selection.** You choose a method by
  calling its function; there is no `Solver` object to hold or switch at runtime.
  Simpler and overhead-free, at the cost of no polymorphic solver handle — the same
  trade the operation layer makes.
- **No hard operator concept at the signature.** `LinearOp` is unconstrained; the
  "a `Matrix`, or `A * x`" contract is enforced by use, not by a `requires` clause.
  This is deliberate — a formal `Matrix` bound would slam the matrix-free door that
  Decision 2 exists to keep open. (A `LinearOperator` concept exists in the library
  for callers who want to assert the contract; the solvers do not force it.)
- **A uniform preconditioner interface, even where a method ignores part of it.**
  CG never calls `adjoint_solve`, yet the concept requires it. The small tax buys a
  single preconditioner contract that every method — symmetric or not — composes
  against identically.

---

## Where it sits in the stack

```text
containers   (compressed2D operator, dense_vector solution/workspace)
   │
operations   (mult, dot, two_norm — dispatched to sparse/BLAS/native kernels)
   │
▶ ITL SOLVERS   (itl/krylov + itl/pc + itl/iteration)
      solver (recurrence)  ×  operator (apply)  ×  preconditioner (approx inverse)
                                                 ×  iteration (stop policy)
```

An iterative solver is the composition layer: it takes the primitive operations and
assembles them into a convergent algorithm, delegating *what to compute* to the
dispatch layer, *how to approximate `A⁻¹`* to the preconditioner, and *when to stop*
to the iteration object. It adds no kernels of its own — only the recurrence that
ties them together.

---

## File map

| File | Role |
|---|---|
| [`itl/krylov/cg.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/itl/krylov/cg.hpp) | the archetype solver (CG); the pattern all Krylov methods follow |
| [`itl/krylov/`](https://github.com/stillwater-sc/mtl5/tree/main/include/mtl/itl/krylov) | `bicg`, `bicgstab`, `cgs`, `gmres`, `minres`, `qmr`, `tfqmr`, `idr_s`, … |
| [`itl/iteration/basic_iteration.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/itl/iteration/basic_iteration.hpp) | the convergence controller (tolerances, budget, residual, error code) |
| [`itl/iteration/`](https://github.com/stillwater-sc/mtl5/tree/main/include/mtl/itl/iteration) | `cyclic_iteration` / `noisy_iteration` reporting refinements |
| [`itl/pc/`](https://github.com/stillwater-sc/mtl5/tree/main/include/mtl/itl/pc) | preconditioners: `identity`, `diagonal`, `ilu_0`, `ilut`, `ic_0`, `ildl`, `ssor`, `block_diagonal` |
| [`concepts/preconditioner.hpp`](https://github.com/stillwater-sc/mtl5/blob/main/include/mtl/concepts/preconditioner.hpp) | the `solve`/`adjoint_solve` concept |
| [`itl/eigen/`](https://github.com/stillwater-sc/mtl5/tree/main/include/mtl/itl/eigen), [`itl/smoother/`](https://github.com/stillwater-sc/mtl5/tree/main/include/mtl/itl/smoother), [`itl/mg/`](https://github.com/stillwater-sc/mtl5/tree/main/include/mtl/itl/mg) | Krylov eigensolvers, stationary smoothers, multigrid |

For the operations a solver composes, see the
[operation dispatch layer](operation-dispatch-architecture.md); for the direct
counterpart that factors `A` instead of applying it, the
[sparse direct solvers](../sparse-direct-solvers-design.md) design doc; for the
typical operator and workspace types, the
[compressed2D](../architecture/containers/compressed2d-architecture.md) and
[dense_vector](../architecture/containers/dense-vector-architecture.md) docs.
