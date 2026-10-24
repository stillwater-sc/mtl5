# Build Options

MTL5 provides several CMake options to control the build and enable optional features.

## Core Options

| Option | Default | Description |
|---|---|---|
| `MTL5_BUILD_TESTS` | ON | Build the Catch2 test suite |
| `MTL5_BUILD_EXAMPLES` | ON | Build example programs |
| `MTL5_BUILD_REGRESSION_TESTS` | OFF | Also build the large-scale solver regression tests in `tests/regression/` (slow; see [Regression tests](#regression-tests)) |

## External Library Bindings

These options link optional external libraries for hardware-accelerated or specialized solvers. MTL5 has native C++ implementations of all algorithms — these bindings are for users who want to delegate to optimized external libraries.

| Option | Default | Description |
|---|---|---|
| `MTL5_WITH_BLAS` | OFF | Link BLAS library for dense acceleration |
| `MTL5_WITH_LAPACK` | OFF | Link LAPACK library for factorizations |
| `MTL5_WITH_ZLIB` | OFF | Link zlib for transparent gzip (`.mtx.gz`) Matrix Market reading |
| `MTL5_WITH_UMFPACK` | OFF | Link UMFPACK library (SuiteSparse) |
| `MTL5_WITH_SUPERLU` | OFF | Link SuperLU library |
| `MTL5_WITH_SUITESPARSE_KLU` | OFF | Link KLU sparse solver (SuiteSparse) |
| `MTL5_WITH_SUITESPARSE_CHOLMOD` | OFF | Link CHOLMOD Cholesky (SuiteSparse) |
| `MTL5_WITH_SUITESPARSE_SPQR` | OFF | Link SuiteSparseQR (SuiteSparse) |

### Vendor threads inside MTL5 regions

A vendor BLAS/LAPACK, CHOLMOD and SPQR run thread teams of their own. With
`MTL5_NUM_THREADS > 1`, a vendor call made from inside an MTL5 parallel region
runs on one thread where the vendor allows a per-thread setting: MKL, and BLIS
0.9 or later. A call made outside a region keeps the vendor's threads. The
configure log names the vendor it found (`BLAS/LAPACK vendor: ...`); select one
with `BLA_VENDOR`:

```bash
cmake -B build -DMTL5_WITH_BLAS=ON -DBLA_VENDOR=FLAME          # BLIS
cmake -B build -DMTL5_WITH_BLAS=ON -DBLA_VENDOR=Intel10_64lp   # threaded MKL
cmake -B build -DMTL5_WITH_BLAS=ON -DBLA_VENDOR=OpenBLAS
```

OpenBLAS's thread count can only be set for the whole process, so MTL5 leaves
it alone. With a threaded MTL5, set `OPENBLAS_NUM_THREADS=1`; with a threaded
OpenBLAS, leave `MTL5_NUM_THREADS` at 1. The other variables are
`BLIS_NUM_THREADS`, `MKL_NUM_THREADS` and `OMP_NUM_THREADS`. See
[On-node threading](../algorithms/on-node-threading.md#vendor-libraries-inside-a-region).

### Matrix Market I/O: gzip and large files

`mtl::io::mm_read` / `mm_read_dense` read SuiteSparse `.mtx` files. Two features
support circuit5M-scale workloads:

**Transparent gzip.** Built with `-DMTL5_WITH_ZLIB=ON` (defines `MTL5_HAS_ZLIB`),
`mm_read("matrix.mtx.gz")` decompresses on the fly — no need to pre-extract a
downloaded `.tar.gz`. Without the flag, passing a `.gz` path throws a clear error
pointing at the option. Plain `.mtx` reading is unaffected and needs no zlib.

```bash
cmake -B build -DCMAKE_CXX_STANDARD=20 -DMTL5_WITH_ZLIB=ON
```

**Large-file load path.** The sparse coordinate reader assembles CRS directly from
a single triplet buffer sized from the header `nnz` and sorts it in place,
avoiding the extra full-size copy that the generic `coordinate2D::compress()`
makes. This roughly halves transient peak memory on very large matrices while
producing identical output (same `(row, col)` order, same duplicate
accumulation).

**circuit5M (`Freescale/circuit5M`, ~5.56M × 5.56M, ~59.5M nnz).** The loader is
built to handle this scale, but it is an opt-in large run (multi-GB file and RAM):
the matrix is not downloaded or factored in CI. To validate locally, fetch the
matrix, then load and factor it with native KLU in `double`, recording wall-clock
load/factor time and peak RSS. Expect the load to be dominated by I/O and the
single triplet buffer (~`nnz × 24` bytes) plus the CRS output; factor cost is
governed by the per-block ordering (see the native-KLU notes).

## Native Performance & Development

These tune or instrument **in-tree** builds (tests, benchmarks). They are
`BUILD_INTERFACE`-only and never leak into the installed/exported package.

| Option | Default | Description |
|---|---|---|
| `MTL5_WITH_HIGHWAY` | OFF | Use Google Highway for SIMD-accelerated native kernels (found or fetched at a pinned tag); scalar fallback otherwise |
| `MTL5_NATIVE_ARCH` | OFF | Tune for the host CPU (`-march=native`); lets the SIMD layer pick the widest ISA (non-portable binaries) |
| `MTL5_NATIVE_FAST_GEMM` | OFF | Route `mtl::mult` through the native blocked GEMM / SIMD GEMV path instead of the generic loop |
| `MTL5_SANITIZE` | *(empty)* | Comma-separated sanitizers for in-tree builds, e.g. `-DMTL5_SANITIZE=address,undefined` (GCC/Clang) |

```bash
# Native fast dense path, host-tuned (what benchmarks/run_sweeps.sh builds):
cmake -B build -DMTL5_NATIVE_FAST_GEMM=ON -DMTL5_WITH_HIGHWAY=ON -DMTL5_NATIVE_ARCH=ON

# ASan + UBSan over the kernels:
cmake -B build-asan -DMTL5_SANITIZE=address,undefined -DMTL5_WITH_HIGHWAY=ON
```

### Multithreaded GEMM

The native blocked GEMM parallelizes its row (`ic`) loop with the C++ standard
concurrency runtime (`std::thread`) — **no OpenMP dependency**. It is controlled
at runtime by the `MTL5_NUM_THREADS` environment variable (clamped to the
hardware concurrency); unset or `1` keeps the single-thread path unchanged.

```bash
MTL5_NUM_THREADS=8 ./your_program   # use up to 8 threads for native GEMM
```

## Build Presets

The project includes CMake presets for common configurations:

```bash
# Development (Debug, tests ON)
cmake --preset dev

# Or configure manually
cmake -B build -DCMAKE_CXX_STANDARD=20 -DMTL5_WITH_BLAS=ON
cmake --build build -j$(nproc)
```

## Running Tests

```bash
# Run all tests
ctest --test-dir build

# Run a specific test
ctest --test-dir build -R test_concepts

# Run with verbose output
ctest --test-dir build --output-on-failure
```

## Regression tests

The unit tests (`tests/unit/`) check each operation on small inputs, in seconds.
The regression tests (`tests/regression/`) check the solvers on problems large
enough to expose what small ones hide: error growth with n, ill-conditioning,
fill-in, and iteration counts. Every test case runs at several sizes, from 100
to 50,000 unknowns, and asserts a residual or backward-error bound that scales
with n.

They are off by default. `MTL5_BUILD_REGRESSION_TESTS=ON` builds them, and the
`ci-regression` preset turns it on together with a Release build:

```bash
cmake --preset ci-regression
cmake --build build-ci-regression --parallel
ctest --test-dir build-ci-regression -L regression --output-on-failure
```

Each regression test carries the CTest label `regression`, so one build directory
serves both tiers:

```bash
ctest --test-dir build-ci-regression -L regression    # regression tests only
ctest --test-dir build-ci-regression -LE regression   # unit tests only
ctest --test-dir build-ci-regression                  # both
```

A test that fails prints its residual, its bound, and the problem size, so the
failing configuration is visible without rerunning it.

### What they cover

There are 14 test programs, one per solver. Sizes in **n** are unknowns; **k × k**
is a 2-D grid, so n = k².

**Dense factorizations** (`dense/`). Each solves Ax = b and asserts the backward
error ‖Ax − b‖ / (‖A‖_F ‖x‖) against a bound that scales with n: n·ε for the
well-conditioned matrices, rising to 1000·n·ε for Moler, 10⁶·n·ε for
Vandermonde, 10⁸·n·ε for Pascal and 10¹⁰·n·ε for Hilbert.

| Test | Matrices | n |
|---|---|---|
| `test_lu` | Frank, Moler, Lehmer; Pascal and Hilbert (ill-conditioned) | 50 to 1000 |
| `test_cholesky` | random SPD (κ = 100), Lehmer, Moler, Minij | 100 to 1000 |
| `test_ldlt` | random SPD (κ = 100), Lehmer, Moler, Minij | 100 to 1000 |
| `test_ldlt_bk` | Bunch–Kaufman: random SPD, Lehmer, Moler, an indefinite tridiagonal | 100 to 1000 |
| `test_qr` | Frank, Kahan, Vandermonde, and a QR solve. Checks ‖QR − A‖/‖A‖ and ‖QᵀQ − I‖ | 100 to 1000 |
| `test_eigenvalue` | Symmetric eigenvalues of a random symmetric and a Wilkinson matrix. Checks that the eigenvalues sum to tr(A) | 100 to 501 |

There is no SVD regression test. The native SVD's accuracy at these sizes
doesn't yet meet a bound worth asserting (see the note in `test_eigenvalue.cpp`).

**Sparse direct solvers** (`sparse/`). Each asserts the relative residual
‖Ax − b‖ / ‖b‖ against n·ε, or n²·ε for the 1-D problems.

| Test | Problems | Size |
|---|---|---|
| `test_sparse_lu` | 1-D convection–diffusion (unsymmetric), 2-D Laplacian, 2-D Poisson | n up to 10,000; k up to 100 |
| `test_sparse_cholesky` | 1-D Laplacian, 2-D Laplacian and Poisson with AMD ordering | n up to 50,000; k up to 100 |
| `test_sparse_ldlt` | the same problems as Cholesky | n up to 50,000; k up to 100 |
| `test_sparse_qr` | 1-D and 2-D Laplacian, and a tall-skinny least-squares problem (two stacked 1-D Laplacians, 2n × n) | n up to 5000; k up to 50 |
| `test_klu_scaling` | native KLU on 2-D Poisson at 64², 128², 256² | see below |

`test_klu_scaling` is a timing test, not an accuracy test. Each refinement
quadruples n. An O(n²) factorization would take about 16 times as long per step,
an O(n^1.5) one about 8 times, and native KLU measures 5 to 7 times. The test
fails if any step's time ratio exceeds 13, between the two hypotheses, so it
catches an asymptotic regression rather than a slow machine. It takes the best
of 3 runs at each size.

**Iterative solvers** (`itl/`). Each asserts a relative residual below 1e−8.

| Test | Problems and preconditioners | Size |
|---|---|---|
| `test_cg` | 2-D Laplacian with identity and Jacobi; 2-D Poisson with IC(0); 1-D Laplacian with Jacobi | k up to 100; n = 1000 |
| `test_bicgstab` | 2-D Laplacian with identity and Jacobi; 1-D convection–diffusion with ILU(0) | k up to 70; n up to 20,000 |
| `test_gmres` | 2-D Laplacian with identity and Jacobi; 1-D convection–diffusion with ILU(0) | k up to 158; n up to 50,000 |

### Time limits

Each regression test has a 300 s limit. `test_cg`, `test_bicgstab` and
`test_gmres` have 600 s, set in `tests/regression/CMakeLists.txt`. GMRES needs
it: it took 486 s on a Xeon E5-2420 v2, so 300 s made the tier unrunnable on
older machines. The whole tier takes about 7 minutes on a GitHub runner, and
longer locally on slower hardware. Build in Release, since a Debug build is many
times slower.

### In CI

CI runs the tests in two tiers, both in `.github/workflows/ci.yml`:

| Tier | Job | Runs on | What it runs |
|---|---|---|---|
| 1 | `build` | every PR, draft or not; every push to `main`; release tags | the unit tests on 8 platforms: Linux GCC and Clang on x64 and ARM64, macOS (two Apple Clang versions), Windows MSVC and Clang-CL |
| 1 | `lapack`, `threaded`, `fp-contract`, `tsan`, `highway`, `benchmarks` | the same events, alongside `build` | unit tests in special configurations: with LAPACK, multithreaded, `-march=x86-64-v3`, ThreadSanitizer, Highway SIMD, and a benchmark build |
| 2 | `regression` | **non-draft PRs only**, once `build` has passed | the regression tests, on Linux x64 with GCC 13 |

The `regression` job:

1. configures with `cmake --preset ci-regression` and GCC 13, with a compiler
   cache (sccache) except on pull requests from forks, so a fork can't write to
   the shared cache;
2. builds with `--parallel 2`;
3. runs `ctest -L regression --output-on-failure --timeout 600`. The per-test
   limits above take precedence over `--timeout`.

Some consequences of when it runs:

- **Tier 2 waits for `build` only**, not for the other Tier 1 jobs, so it can
  start while those are still running.
- **A draft PR never runs Tier 2.** The job shows as skipped. Marking the PR
  ready (`gh pr ready <n>`) starts a new run that includes it.
- **Pushes to `main` and release tags run Tier 1 only.** Code reaches `main`
  through a PR that has already passed Tier 2.
- **A docs-only change runs neither tier.** `ci.yml` ignores commits that touch
  only `docs/`, `docs-site/`, Markdown, images, `LICENSE`, `.coderabbit.yaml` or
  `.nojekyll`.
- **A new commit on a PR cancels the PR's run in progress**, Tier 2 included, so a
  cancelled `regression` job on an older commit is expected.

