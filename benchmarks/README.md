# MTL5 Benchmark Harness

## Methodology: one executable per backend

MTL5's public operations (`mult`, `dot`, `two_norm`, `lu_factor`,
`eigenvalue_symmetric`, ...) dispatch to BLAS/LAPACK **at compile time** when
`MTL5_HAS_BLAS` / `MTL5_HAS_LAPACK` are defined, and otherwise run the generic
C++ path. A dependent application therefore picks its backend **once, with build
flags, for the whole program**.

The benchmark mirrors exactly that. `bench_all` calls **only the public `mtl::`
API** — it does not select an implementation at runtime. The *build* is the
backend:

| Build | CMake flags | What every op runs |
|-------|-------------|--------------------|
| `native`   | *(none)* | generic C++ |
| `openblas` | `-DMTL5_WITH_BLAS=ON -DMTL5_WITH_LAPACK=ON` | system BLAS/LAPACK (OpenBLAS) |
| `mkl`      | `… -DBLA_VENDOR=Intel10_64lp` (oneAPI sourced) | Intel MKL |

So we build the same `bench_all.cpp` once per backend and run each, producing
**one CSV per backend** with a single `--label`. There is no `native` curve
hiding inside the OpenBLAS or MKL binary, and no in-process policy switching —
the numbers are what a real app compiled that way would get.

> Earlier revisions used per-op *policy tags* (Native/Blas/Lapack) measured
> together in a single binary. That produced a misleading split (e.g. two
> different `native` curves, one per linked library) because identical code was
> timed in two different process environments. The one-binary-per-backend model
> above replaces it.

## Layout

```
benchmarks/
  bench_all.cpp          CLI driver (sizes/sweeps/suites + --label)
  run_sweeps.sh          builds native/openblas/mkl variants and runs the sweeps
  plot_results.py        GFLOP/s-vs-N plots from the CSVs
  harness/
    timer.hpp            high-resolution timing + statistics
    reporter.hpp         console table + CSV output
    generators.hpp       deterministic matrix/vector generators
    runner.hpp           per-suite runners (call the public mtl:: API)
  data/                  committed example CSVs + rendered plots (see data/README.md)
```

## Build & run (reproducible, all variants)

```bash
# Builds native / openblas / mkl, pins to a P-core, writes one CSV per backend.
BENCH_CPU=4 benchmarks/run_sweeps.sh            # 4 = a P-core; see note below
# custom sweep:        BENCH_CPU=4 benchmarks/run_sweeps.sh 16:2048:32
```

`run_sweeps.sh` configures each variant in its own (clean) build dir, builds
`bench_all`, and runs the BLAS and LAPACK sweeps single-threaded. The MKL
variant is skipped automatically if `/opt/intel/oneapi/setvars.sh` is absent
(override with `MKL_SETVARS=...`).

**CPU pinning matters.** On hybrid CPUs (e.g. Intel P/E-core parts) an unpinned
single-threaded run lets short L1 kernels land on slow E-cores, skewing results.
Set `BENCH_CPU` to a performance core. The script also pins threads to 1
(`OMP_NUM_THREADS=1` and the vendor equivalents).

### Building a single variant by hand

```bash
cmake -B build-openblas -DMTL5_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
      -DMTL5_WITH_BLAS=ON -DMTL5_WITH_LAPACK=ON
cmake --build build-openblas --target bench_all
taskset -c 4 ./build-openblas/benchmarks/bench_all --suite blas --sweep 65:1025:80 \
      --label openblas --csv benchmarks/data/blas_sweep_openblas.csv
```

(The CMake options are `MTL5_WITH_BLAS` / `MTL5_WITH_LAPACK`. For MKL, `source
/opt/intel/oneapi/setvars.sh` first and add `-DBLA_VENDOR=Intel10_64lp`.)

## Running `bench_all`

```bash
./build-openblas/benchmarks/bench_all                       # default sizes, all suites
./build-openblas/benchmarks/bench_all --suite l3            # one BLAS level
./build-openblas/benchmarks/bench_all --suite gemm --sizes 64,128,256,512,1024
./build-openblas/benchmarks/bench_all --csv out.csv --label openblas
```

`--label NAME` sets the backend name recorded in the output (the build defaults
to `native` or `blas`; `run_sweeps.sh` passes `native`/`openblas`/`mkl`).

### Suites

`all`, `blas` (= l1 + l2 + l3), `lapack`, the level groups `l1` (dot + nrm2),
`l2` (gemv), `l3` (gemm), and the individual ops `dot`, `nrm2`, `gemv`, `gemm`,
`lu`, `qr`, `cholesky`, `eig`.

### Sweeping size N (padding / odd-size overhead)

Generate sizes with `--sweep` (or per-tier `--blas-sweep` / `--lapack-sweep`):

```bash
./build-openblas/benchmarks/bench_all --suite l3 --sweep 16:1024:16   # linear
./build-openblas/benchmarks/bench_all --suite blas --sweep 16:1024:x2 # geometric
./build-openblas/benchmarks/bench_all --suite l1 --sweep 33:1024:97   # all odd / non-pow2
./build-openblas/benchmarks/bench_all --suite l3 --sweep 250:262:1    # dense bracket of 256
```

The **default** size set is intentionally not all powers of two — it brackets
each power of two with `±1` neighbours and 1.5x midpoints
(`48, 64, 65, 96, 128, 129, 192, 255, 256, 257, 384, 512, 513, 768, 1024`), so a
plain run already surfaces odd-size / padding effects.

## Plotting

`plot_results.py` turns the per-backend CSVs into GFLOP/s-vs-N curves
(matplotlib; standard library otherwise). Pass the native/openblas/mkl CSVs to
overlay them — one clean curve per backend:

```bash
./benchmarks/plot_results.py benchmarks/data/blas_sweep_*.csv \
    --out benchmarks/data/blas_sweep_gflops.png
./benchmarks/plot_results.py benchmarks/data/lapack_sweep_*.csv \
    --out benchmarks/data/lapack_sweep_gflops.png
# single op / wall-clock / log-log:
./benchmarks/plot_results.py benchmarks/data/lapack_sweep_*.csv --op gemm --metric median_ns --logx --logy
```

Cross-backend speedups are computed at plot/analysis time across the CSVs (each
binary measures only its own backend, so there is no in-run baseline).

See `data/README.md` for the committed example sweeps, the platform they were
run on, and the rendered curves.

> The plotting script is benchmark *tooling*; the NumPy/SciPy bindings live in
> the separate `mtl5-python` repo.

## Adding a new backend (e.g. CUDA)

Because the benchmark uses the public API, a new backend is added in the
**library** (give the relevant `mtl::` ops a compile-time dispatch path guarded
by e.g. `MTL5_HAS_CUDA`), then add a build variant + `--label cuda` to
`run_sweeps.sh`. No harness changes are required.
