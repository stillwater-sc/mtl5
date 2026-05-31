# MTL5 Benchmark Harness

## Motivation

MTL5 operations like `mult()`, `lu_factor()`, and `two_norm()` can dispatch to
optimized libraries (BLAS, LAPACK, UMFPACK) at compile time via `#ifdef` guards
and `if constexpr`. This creates a measurement problem: how do you compare the
native C++ implementation against the accelerated path when the dispatch decision
is baked into the binary?

Three architectures were considered:

| Approach | Mechanism | Tradeoff |
|----------|-----------|----------|
| **Multi-binary** | Build N executables with different `-D` flags | Zero library changes, but results require post-hoc joining |
| **Dual-path** | Expose `::generic` namespace aliases | Single binary, but requires modifying every operation header |
| **Policy tags** | Template parameter selects implementation | Single binary, self-contained, extensible to future backends |

We chose **policy tags** because MTL5 will grow backends (CUDA, MKL, oneDNN)
and this architecture absorbs new backends without touching the harness core
or the library headers.

## Architecture

```
benchmarks/
  bench_all.cpp              CLI driver
  harness/
    backend.hpp              Backend tags + availability traits
    timer.hpp                High-resolution timing + statistics
    reporter.hpp             Console table + CSV output
    generators.hpp           Deterministic matrix/vector generators
    op_blas.hpp              Policy wrappers for BLAS-level ops
    op_lapack.hpp            Policy wrappers for LAPACK-level ops
    runner.hpp               Suite runners with fold-expression expansion
```

### Backend tags

Each backend is a simple struct with a name:

```cpp
struct Native { static constexpr std::string_view name = "native"; };
struct Blas    { static constexpr std::string_view name = "blas";   };
struct Lapack  { static constexpr std::string_view name = "lapack"; };
```

Availability is a compile-time trait gated by the same `#ifdef` macros
that the library already uses:

```cpp
template <> inline constexpr bool is_available_v<Native> = true;  // always

#ifdef MTL5_HAS_BLAS
template <> inline constexpr bool is_available_v<Blas> = true;
#endif
```

### Operation wrappers

Each benchmarked operation is a struct template specialized per backend.
The `Native` specialization calls the generic C++ code directly (bypassing
any dispatch logic), while accelerated specializations call the library API:

```cpp
template <typename Backend> struct gemm;

template <>
struct gemm<Native> {
    template <Matrix MA, Matrix MB, Matrix MC>
    static void run(const MA& A, const MB& B, MC& C) {
        detail::mult_generic(A, B, C);   // always the C++ loops
    }
};

template <>
struct gemm<Blas> {
    template <Matrix MA, Matrix MB, Matrix MC>
    static void run(const MA& A, const MB& B, MC& C) {
        // call BLAS dgemm_ directly
    }
};
```

### Fold-expression expansion

A `backend_list<Backends...>` type plus a `for_each_backend()` helper
expands benchmarks across all compiled-in backends in a single binary:

```cpp
using dense_backends = backend_list<
    Native
#ifdef MTL5_HAS_BLAS
    , Blas
#endif
>;

// At each matrix size, this expands to one measurement per backend
for_each_backend(dense_backends{}, [&]<typename Backend>() {
    auto t = measure([&]{ op::gemm<Backend>::run(A, B, C); }, ...);
    reporter.add(t);
});
```

## Building

```bash
# Native-only (no external libraries needed)
cmake -B build -DMTL5_BUILD_BENCHMARKS=ON
cmake --build build --target bench_all

# With BLAS + LAPACK comparison (uses the system default BLAS, e.g. OpenBLAS)
cmake -B build -DMTL5_BUILD_BENCHMARKS=ON \
      -DMTL5_WITH_BLAS=ON -DMTL5_WITH_LAPACK=ON
cmake --build build --target bench_all

# Against Intel MKL: select it through CMake's FindBLAS vendor, with the
# oneAPI environment sourced so the libraries are found at configure and run time.
source /opt/intel/oneapi/setvars.sh
cmake -B build-mkl -DMTL5_BUILD_BENCHMARKS=ON \
      -DMTL5_WITH_BLAS=ON -DMTL5_WITH_LAPACK=ON \
      -DBLA_VENDOR=Intel10_64lp
cmake --build build-mkl --target bench_all
```

> The CMake options are `MTL5_WITH_BLAS` / `MTL5_WITH_LAPACK`. The benchmark
> output labels the accelerated backend `blas` / `lapack` regardless of which
> library is linked, so "native vs MKL" is the `blas` / `lapack` rows of a
> binary built against MKL. Use separate build directories to keep OpenBLAS and
> MKL results apart.

## Running

```bash
# Full suite with default sizes
./build/benchmarks/bench_all

# A whole BLAS level: l1 (dot+nrm2), l2 (gemv), l3 (gemm)
./build/benchmarks/bench_all --suite l3

# Custom explicit sizes
./build/benchmarks/bench_all --suite gemm --sizes 32,64,128,256,512,1024

# Separate BLAS and LAPACK size ranges
./build/benchmarks/bench_all --blas-sizes 64,256,1024 --lapack-sizes 64,128,256

# Export to CSV for plotting
./build/benchmarks/bench_all --csv results.csv
```

### Suites

`all`, `blas` (= l1 + l2 + l3), `lapack`, the level groups `l1` (dot + nrm2),
`l2` (gemv), `l3` (gemm), and the individual ops `dot`, `nrm2`, `gemv`, `gemm`,
`lu`, `qr`, `cholesky`, `eig`.

### Sweeping size N (padding / odd-size overhead)

Instead of listing sizes, generate them with `--sweep` (or the per-tier
`--blas-sweep` / `--lapack-sweep`):

```bash
# Linear, inclusive: START:STOP:STEP
./build/benchmarks/bench_all --suite l3 --sweep 16:1024:16

# Geometric, inclusive: START:STOP:xFACTOR
./build/benchmarks/bench_all --suite blas --sweep 16:1024:x2

# Odd sweep -- a non-power-of-2 step yields only odd, non-aligned sizes
./build/benchmarks/bench_all --suite l1 --sweep 33:1024:97

# Dense sweep bracketing a power-of-2 cliff to measure padding overhead
./build/benchmarks/bench_all --suite l3 --sweep 250:262:1 --csv around_256.csv
```

The **default** size set is intentionally *not* all powers of two -- it brackets
each power of two with its `+/-1` neighbours and 1.5x midpoints
(`48, 64, 65, 96, 128, 129, 192, 255, 256, 257, 384, 512, 513, 768, 1024`), so a
plain run already surfaces odd-size / padding effects. Use a dense `--sweep`
around a boundary (e.g. `250:262:1`) to zoom in on a specific cliff, and `--csv`
to capture the curve for plotting. Pin threads (`OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1` / `OPENBLAS_NUM_THREADS=1`) for stable per-size numbers.

## Plotting

`plot_results.py` turns one or more bench_all CSVs into GFLOP/s-vs-N curves
(matplotlib; standard library otherwise). Pass several CSVs to overlay backends
from different builds -- e.g. a native/OpenBLAS/MKL comparison in one figure:

```bash
# One figure per operation from a single run
./benchmarks/plot_results.py results.csv

# Overlay OpenBLAS vs MKL for gemm (committed example data)
./benchmarks/plot_results.py \
    benchmarks/data/blas_sweep_openblas.csv \
    benchmarks/data/blas_sweep_mkl.csv \
    --labels openblas,mkl --op gemm --out gemm_gflops.png

# Wall-clock time, log-log
./benchmarks/plot_results.py results.csv --op gemm --metric median_ns --logx --logy
```

`benchmarks/data/` holds small committed example sweeps (the odd-size BLAS sweep
`65:1025:80`, single-threaded, native + OpenBLAS / MKL) so the script and curves
are reproducible without re-running the suite. (This is plotting *tooling* --
the NumPy/SciPy bindings live in the separate `mtl5-python` repo.)

## Example output

Native vs OpenBLAS on an AMD Ryzen 9 (single-threaded):

```
Operation            Backend        Size     Median(us)        Min(us)        Speedup    GFLOP/s
--------------------------------------------------------------------------------------------
gemm                 native           64        5014.50        4881.83         (base)       0.10
gemm                 blas             64          12.21           9.03        410.78x      42.94

gemm                 native          256      271065.37      266160.23         (base)       0.12
gemm                 blas            256         179.00         174.51       1514.34x     187.46

gemm                 native          512     2327056.64     2275839.99         (base)       0.12
gemm                 blas            512         928.69         906.14       2505.75x     289.05

lu_factor            native          256      130478.75      126964.53         (base)       0.09
lu_factor            lapack          256         408.50         397.86        319.41x      27.38

cholesky             native          256       42711.39       42133.33         (base)       0.13
cholesky             lapack          256         266.00         263.44        160.57x      21.02
```

## Adding a new backend

To add a CUDA backend, for example:

1. **Define the tag** in `harness/backend.hpp`:
   ```cpp
   struct Cuda { static constexpr std::string_view name = "cuda"; };

   #ifdef MTL5_HAS_CUDA
   template <> inline constexpr bool is_available_v<Cuda> = true;
   #endif
   ```

2. **Add to backend lists** in `backend.hpp`:
   ```cpp
   using dense_backends = backend_list<
       Native
   #ifdef MTL5_HAS_BLAS
       , Blas
   #endif
   #ifdef MTL5_HAS_CUDA
       , Cuda
   #endif
   >;
   ```

3. **Specialize operations** in a new `harness/op_cuda.hpp`:
   ```cpp
   template <>
   struct gemm<Cuda> {
       template <Matrix MA, Matrix MB, Matrix MC>
       static void run(const MA& A, const MB& B, MC& C) {
           // Copy to device, call cublasDgemm, copy back
       }
   };
   ```

4. **Include** the new header in `runner.hpp`.

No changes to `bench_all.cpp`, `timer.hpp`, `reporter.hpp`, or the runner
functions are needed. The fold expression automatically picks up the new
backend.
