# Measuring Solver Accuracy: Residuals, Norms, and Error

When a linear solver returns an approximate solution `x̂` to `A x = b`, how do you
know it worked? This page defines the metrics MTL5 uses to answer that — the
**residual** and its **norms** — and the often-misunderstood relationship between
the residual (which is cheap to measure) and the actual **solution error** (which
is what you usually care about). These metrics are reported by the solver tests
and the benchmark suite, and referenced by every algorithm page.

## The residual

A direct solver computes `x̂`, an approximation to the true `x`. Substituting it
back generally does not reproduce `b` exactly, because of finite-precision
arithmetic in the factorization and solve. The leftover mismatch is the
**residual vector**:

```text
r  =  A x̂ − b
```

If `x̂` were exact, `r` would be all zeros. It rarely is, so we summarize the
vector `r` with a single number — a **norm**.

## Norms

A norm `‖·‖` collapses a vector to a non-negative scalar "size." Three are common:

| Norm | Definition | Reading |
|------|------------|---------|
| `‖r‖₁` | `Σ_i \|r_i\|` | total absolute mismatch |
| `‖r‖₂` | `sqrt(Σ_i r_i²)` | Euclidean / root-sum-of-squares |
| `‖r‖∞` | `max_i \|r_i\|` | **worst single-equation mismatch** |

MTL5 reports `‖A x̂ − b‖∞` by default. The infinity norm is the strictest
per-equation statement ("*every* equation is satisfied to within this") and is
the cheapest to compute (one pass, a running max).

## Absolute vs relative residual

The plain residual is **scale-dependent**: multiply the whole system by 1000 and
`‖r‖∞` grows 1000× even though the solution is identical. To get a scale-free
number, normalize by the right-hand side:

```text
relative residual  =  ‖A x̂ − b‖∞ / ‖b‖∞
```

This matters in practice. For example MTL5's `generators::poisson2d_dirichlet`
carries an `h²` discretization scaling that makes the matrix entries large
(diagonal on the order of `1e5`), so its *absolute* residual is correspondingly
large while the solution is fine. The benchmark and example code therefore use
the **relative** residual so a fixed tolerance (say `< 1e-8`) means the same
thing regardless of how the matrix is scaled.

## Residual (backward error) vs forward error

The residual measures **backward error** — "for what nearby system did I find the
exact solution?" A small residual means `x̂` exactly solves `(A) x = (b − r)`, a
problem close to the one you posed. That is the right notion of "the solver did
its job."

It is *not* the same as the **forward error**, the distance to the true solution:

```text
forward error  =  ‖x̂ − x‖
```

The two are linked by the **condition number** `κ(A) = ‖A‖·‖A⁻¹‖`:

```text
   ‖x̂ − x‖        ‖A x̂ − b‖
  ─────────  ≲  κ(A) · ──────────
    ‖x‖              ‖b‖
```

So a tiny relative residual can still hide a large solution error when `A` is
**ill-conditioned** (large `κ`). Conversely, a backward-stable solver always
delivers a tiny residual — that is what it guarantees — but accuracy of `x̂`
itself is capped by the conditioning of the problem, not the solver.

Where the exact solution is known (e.g. a test builds `b = A·1` so `x = 1`), MTL5
also reports the **forward error** directly (`‖x̂ − 1‖∞`) alongside the residual.
Reporting both separates *solver quality* (residual) from *problem conditioning*
(the gap between residual and forward error).

## What "good" looks like

For a well-conditioned system solved in `double`, a backward-stable direct solver
(LU, Cholesky, QR, KLU) gives a relative residual around `1e-12`–`1e-15` — i.e.
`x̂` satisfies every equation to ~15 significant digits. Values far above that
signal either an ill-conditioned matrix (look at the forward error / `κ`) or a
numerically unstable solve.

In reduced precision the floor rises with the unit roundoff: expect ~`1e-6`–`1e-7`
for `float`, and larger still for half-precision types — which is exactly why
mixed-precision work (factor low, refine high) pairs a low-precision factorization
with a higher-precision residual.

## Reading the benchmark / test columns

- The solver unit tests assert a small residual (absolute or relative) as the
  correctness gate.
- `benchmarks/bench_klu` reports the residual for both native and external KLU,
  so its time/fill comparison is between two *correct* solvers — not a
  fast-but-wrong vs slow-but-right mismatch.

## References

- Higham, *Accuracy and Stability of Numerical Algorithms*, SIAM, 2002
  (backward error, conditioning, the residual–forward-error bound).
- Demmel, *Applied Numerical Linear Algebra*, SIAM, 1997.
