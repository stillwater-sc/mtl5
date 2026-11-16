# IDR(s): Induced Dimension Reduction

`mtl::itl::idr_s` solves a nonsymmetric system `A x = b` with short recurrences, so its memory does not grow with the number of iterations, as GMRES's does. It belongs to the same family as BiCGSTAB: IDR(1) is close to BiCGSTAB, and a larger shadow space `s` buys faster convergence for `s + 1` products with `A` per cycle and `3s + 4` vectors of memory.

This page covers the algorithm MTL5 implements, its options, and measurements that show where it stands against the other Krylov solvers in MTL5.

## The algorithm

MTL5 implements the biorthogonal IDR(s) of van Gijzen and Sonneveld ("Algorithm 913: An elegant IDR(s) variant that efficiently exploits biorthogonality properties", ACM TOMS 38(1), 2011).

Every residual lies in a sequence of nested spaces `G_0 ⊃ G_1 ⊃ ...`, each orthogonal to more of a fixed shadow space `P` (`s` vectors). Each cycle takes `s + 1` steps:

- **The first `s` steps** each make the new direction `G_k = A U_k` orthogonal to `P_0 ... P_{k-1}`. The small system `P^H G` is therefore lower triangular, and each step solves it by forward substitution in `O(s²)`.
- **The last step** moves into the next space with `r ← r − ω A M⁻¹ r`, picking `ω` for the cycle.

Each step costs one product with `A` and one preconditioner solve, and counts as one iteration. Preconditioning is on the right, so the residual the solver tests is the residual of the original system.

Until issue 874, MTL5 used the 2008 prototype (Sonneveld and van Gijzen, SIAM J. Sci. Comput. 31, Algorithm 3.1). Its `s × s` system was dense, solved by elimination at every step, and it became erratic at tight tolerances. On a 2D convection-diffusion system with ILU(0) at 1e-13, `s = 2` took 729 iterations and `s = 8` took 1612, while `s = 4` took 71. On a 200² grid, `s = 2`, 4 and 8 ran to 20000 iterations.

## Accuracy: the true residual decides

The solver updates its residual by recurrence, and in floating point that drifts away from `b − A x`. MTL5 handles the drift in two ways:

- **Residual replacement** (`idrs_options::residual_replacement`, on by default). At the start of a cycle, once the updated residual has fallen 100× below its largest norm since the last replacement, it is replaced by `b − A x`, at the cost of one product with `A`. Without it, IDR(16) on a nonnormal system stopped at an updated 9.1e-11 whose true value was 2.5e-9.
- **The final check.** When the updated residual meets the tolerance, or the iteration limit is reached, `b − A x` is computed. If that has not converged, the method restarts from the current `x`. This is the same rule as `cg`, `bicgstab`, `tfqmr` and the other MTL5 Krylov solvers. The returned iteration object reports the true residual.

A tolerance below what double precision can reach on a system is not met: the solver runs to `max_iter` and returns 1 with the true residual it reached. On the unpreconditioned 100² Laplacian, for example, that floor is about 1.1e-13.

## Breakdowns

A diagonal entry of `P^H G` that is negligible against `‖G_k‖` is a near-breakdown of the recurrence, not of the problem. IDR(s) then restarts from the current `x`.

On `σ I + S`, with `S` the 2D central-difference convection operator, IDR(1) met `P^H G = 2·10⁻¹⁷` (against `‖G‖ = 1`) after 102 iterations. BiCGSTAB breaks down on the same system. With restarts, IDR(1) converges: in 1399 iterations on x86-64 without FMA, and in 2149 with FMA contraction. A system this close to breakdown is sensitive to rounding, and the iteration counts vary between platforms.

A breakdown at the first step after a restart means the shadow space is degenerate for the current residual. An example is `P = e₁` for a diagonal `A` whose residual has a zero first entry: `P^H r` then stays 0. IDR(s) draws a new shadow space from `seed + 1`, `+ 2` and `+ 3`, even when `P` was supplied, and returns 2 only if that does not help.

| Return | Meaning |
|---|---|
| 0 | converged (true residual) |
| 1 | iteration limit |
| 2 | `P^H G` singular after three redraws of the shadow space |
| 3 | `t = A M⁻¹ r` is zero, or `ω = 0` |

## Options

```cpp
itl::idrs_options opt;
opt.s = 8;                        // shadow space dimension (clamped to n)
opt.kappa = 0;                    // omega strategy, see below
opt.smoothing = false;            // minimal-residual smoothing
opt.residual_replacement = true;  // see above
opt.seed = 42;                    // the shadow space's random draw
itl::idr_s(A, x, b, M, iter, opt);

itl::idr_s(A, x, b, M, iter, 4);  // s alone, default options
itl::idr_s(A, x, b, M, iter, opt, P);  // a shadow space of your own
```

- **`s`.** A larger `s` takes fewer products with `A`, but costs more vector work and memory per product. On the systems below, `s = 4` to 8 is the best trade-off. `s = 16` rarely beats 8 in time.
- **`kappa`.** The cycle's `ω` minimizes `‖r − ω t‖`. With `kappa > 0` it is enlarged when the angle between `r` and `t` is wide, scaled by `kappa / |cos(r, t)|` whenever `|cos(r, t)| < kappa`. This is Sleijpen and van Gijzen's "maintaining the convergence" strategy, and their default is 0.7. MTL5 defaults to 0, from the measurements below:
  - **No clear gain:** on convection-diffusion, 0.7 was never clearly better.
  - **Slower on Poisson:** up to 25% slower at `s ≤ 2`.
  - **Faster on `σ I + S` at σ = 0.5:** 25% fewer iterations at `s ≥ 4`.
  - **Unbounded at σ = 0.1:** the enlarged `ω` made the residual grow without bound (to 1e62 at `s = 1`), where `kappa = 0` stays bounded.
- **`smoothing`.** Replaces `x` and `r` by a smoothed pair whose residual norm decreases monotonically (Algorithm 913's smoothing). It costs two vectors and two reductions per product, and in the measurements below changed iteration counts by at most a few percent.
- **`seed` and `P`.** The shadow space is `s` random vectors, drawn from `std::mt19937_64` in the solver's own value type. The entries are 24-bit integers scaled by 2⁻²³, exact in any type with a 24-bit significand, with no detour through `double`. The draw is complex for a complex value type, which is the recommended shadow space for a complex system. The vectors are then orthonormalized.
  - **Reproducible:** a run is bit-reproducible for a given seed.
  - **Seed variance:** different seeds change iteration counts by a few percent on easy systems. On hard nonnormal ones the spread is wider: IDR(16) on convection-diffusion with eps = 0.9 took 217 to 356 iterations over eight seeds.
  - **Your own `P`:** a supplied `P` is orthonormalized too. That changes nothing in the method, which depends only on the span of `P`. A rank-deficient `P` throws `std::invalid_argument`.

## Measurements

The systems are 2D convection-diffusion (the 5-point operator with convection `eps` in both directions) and the 2D Laplacian. The runs start from `x = 0` with `b = ones`. Times are on an i7-12700K, single-threaded, with the default options.

Iterations, with the time of the solve:

| system | rtol | IDR(1) | IDR(2) | IDR(4) | IDR(8) | IDR(16) |
|---|---|---|---|---|---|---|
| convdiff 200², eps 0.3, ILU(0) | 1e-8 | 137, 77 ms | 115, 68 ms | 99, 68 ms | 91, 84 ms | 88, 133 ms |
| same, before issue 874 | 1e-8 | 142, 87 ms | 112, 72 ms | 100, 74 ms | 93, 88 ms | 89, 126 ms |
| same | 1e-13 | 160, 89 ms | 133, 79 ms | 133, 93 ms | 135, 123 ms | 151, 219 ms |
| same, before issue 874 | 1e-13 | 161 | 20000 (limit) | 20000 (limit) | 20000 (limit) | 236 |
| convdiff 100², eps 0.9, none | 1e-10 | 389, 22 ms | 299, 20 ms | 256, 22 ms | 233, 30 ms | 380, 81 ms |
| Laplacian 100², none | 1e-10 | 294, 17 ms | 253, 17 ms | 235, 20 ms | 230, 30 ms | 226, 49 ms |

At ordinary tolerances the biorthogonal variant takes about as many iterations as the prototype, and is 5-20% faster per solve, because its small system is triangular. At tight tolerances the prototype failed outright and the new one converges.

On the eps = 0.9 system without a preconditioner:

| solver | iterations | products with A | true residual |
|---|---|---|---|
| IDR(4) | 256 | 256 | 6e-11 |
| IDR(8) | 233 | 233 | 5e-11 |
| BiCGSTAB | 423 | 846 | 4e-11 |
| BiCGSTAB(2) | 238 | 952 | 8e-11 |
| GMRES(30) | 422 | 422 | 9e-11 |
| CGS, BiCG, QMR | do not converge in 20000 | | |

## When to choose IDR(s)

- **Over BiCGSTAB:** on nonsymmetric systems, IDR(4) to IDR(8) usually needs fewer products with `A` than BiCGSTAB. Here it was about a third as many, on a system where BiCGSTAB had falsely claimed convergence before issue 873.
- **Over BiCGSTAB(ℓ):** BiCGSTAB(ℓ) is the alternative when the spectrum has large imaginary parts. On `σ I + S` at σ = 0.1, BiCGSTAB(2) converged in 182 iterations, while no IDR(s) variant converged in 5000.
- **Over GMRES(m):** GMRES(m) minimizes the residual and never breaks down, but it costs `m` vectors and an orthogonalization that grows with `m`. IDR(s) has a fixed cost per product.
