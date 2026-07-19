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
| `native-fast` | `-DMTL5_NATIVE_FAST_GEMM=ON -DMTL5_WITH_HIGHWAY=ON -DMTL5_NATIVE_ARCH=ON` | MTL5's own SIMD GEMM/GEMV (no external BLAS) |
| `openblas` | `-DMTL5_WITH_BLAS=ON -DMTL5_WITH_LAPACK=ON` | system BLAS/LAPACK (OpenBLAS) |
| `blis`     | `-DMTL5_WITH_BLAS=ON -DBLA_VENDOR=FLAME` | BLIS (BLAS-only; `libblis`) |
| `mkl`      | `… -DBLA_VENDOR=Intel10_64lp` (oneAPI sourced) | Intel MKL |

BLIS is selected through CMake's `FindBLAS` `BLA_VENDOR=FLAME` and requires a
BLIS BLAS on the system (e.g. `apt install libblis-dev`, or build BLIS and point
CMake at it). It is a **BLAS-only** library, so the `blis` variant runs the BLAS
L1/L2/L3 suites (LAPACK factorizations would need libFLAME, not wired here). Set
thread count with `BLIS_NUM_THREADS`. Both `run_sweeps.sh` and `run_scaling.sh`
skip the `blis` variant automatically if a BLIS BLAS cannot be located.

`native-fast` is the epic #82 path: `mult()` routes through the blocked GEMM
(`detail/gemm_blocked.hpp`) and SIMD GEMV (`detail/gemv.hpp`), built over Google
Highway with `-march=native`. It links **no external BLAS** — it is MTL5
competing with OpenBLAS/MKL on its own kernels.

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
  run_sweeps.sh          builds native/native-fast/openblas/blis/mkl variants and runs the sweeps
  plot_results.py        GFLOP/s-vs-N plots from the CSVs
  analyze_gate.py        % of a reference backend (--reference, default openblas) / % of FMA peak + the pass/fail perf gate
  harness/
    timer.hpp            high-resolution timing + statistics
    reporter.hpp         console table + CSV output
    generators.hpp       deterministic matrix/vector generators
    runner.hpp           per-suite runners (call the public mtl:: API)
  data/                  committed example CSVs + rendered plots (see data/README.md)
```

## Build & run (reproducible, all variants)

```bash
# Builds native / native-fast / openblas / mkl, pins to a P-core, one CSV each.
BENCH_CPU=4 benchmarks/run_sweeps.sh            # 4 = a P-core; see note below
# custom sweep:        BENCH_CPU=4 benchmarks/run_sweeps.sh 16:2048:32
# BLAS-only (the #82 gate; native LAPACK at large N is slow):
BENCH_CPU=4 BENCH_SUITES=blas benchmarks/run_sweeps.sh
```

`run_sweeps.sh` configures each variant in its own (clean) build dir, builds
`bench_all`, and runs the suites named in `BENCH_SUITES` (default `blas lapack`)
single-threaded. The MKL variant is skipped automatically if
`/opt/intel/oneapi/setvars.sh` is absent (override with `MKL_SETVARS=...`).

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

`all`, `blas` (= l1 + l2 + l3), `lapack`, the level groups `l1` (dot + nrm2 +
axpy + scal), `l2` (gemv), `l3` (gemm), and the individual ops `dot`, `nrm2`,
`axpy`, `scal`, `gemv`, `gemm`,
`lu`, `qr`, `cholesky`, `eig`.

### BLAS routine coverage

Benchmarked (the core BLAS routines MTL5 implements): **L1** `dot`, `nrm2`,
`axpy`, `scal`; **L2** `gemv`; **L3** `gemm`. Standard BLAS routines MTL5 does
**not** implement yet — and therefore cannot benchmark — are, for reference:
L1 `asum`, `iamax`, `copy`, `swap`, `rot`; L2 `ger`, `symv`, `trmv`, `trsv`;
L3 `symm`, `syrk`, `syr2k`, `trmm`, `trsm` (tracked in #227).

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

## The native-fast acceptance gate (epic #82)

The epic's goal is for MTL5's **own** dense kernels (no external BLAS) to land
**within 10–20% of OpenBLAS** for GEMM at practical sizes, and at the memory
ceiling for GEMV/L1. `analyze_gate.py` measures this from the CSVs:

```bash
# Build the gate variants (BLAS suite only -- native LAPACK at large N is slow):
BENCH_CPU=4 BENCH_SUITES=blas benchmarks/run_sweeps.sh 65:1025:80

# % of OpenBLAS and % of FMA peak, per op and size (peak = 1 P-core fp64):
benchmarks/analyze_gate.py benchmarks/data/blas_sweep_native-fast.csv \
    benchmarks/data/blas_sweep_openblas.csv --peak-gflops 78

# Pass/fail gate: GEMM native-fast >= 80% of OpenBLAS for N >= 256
benchmarks/analyze_gate.py benchmarks/data/blas_sweep_native-fast.csv \
    benchmarks/data/blas_sweep_openblas.csv --gate --op gemm --threshold 0.80 --min-size 256
```

The gate (`--gate`) exits non-zero if native-fast drops below the threshold, so
it can run in an opt-in workflow. It is **not** wired into the per-push CI:
shared CI runners have unstable clocks and no P-core pinning, which makes
absolute perf gates flaky. Run it on dedicated hardware.

**Measured result** (i7-12700K, 1 P-core, fp64, single-thread — see
`data/README.md`): GEMM native-fast is **80–84% of OpenBLAS for all N ≥ 256**
(~76–78% of FMA peak); GEMV is **~100–116% of OpenBLAS** and `dot`/`nrm2` are at
or above it (both bandwidth-bound). The GEMM gate **passes** at the 80% / N≥256
threshold. This is single-threaded; multithreading is #92.

## Multi-core GEMM scaling (#108)

The native blocked GEMM parallelizes its `ic` (row) loop with the C++ standard
concurrency runtime (set `MTL5_NUM_THREADS`). `run_scaling.sh` sweeps GEMM over
thread counts for native-fast **and** threaded OpenBLAS/MKL (their own
`*_NUM_THREADS`), pinning to physical performance cores; `analyze_scaling.py`
reports speedup + parallel efficiency and draws the scaling plot.

```bash
# native-fast / openblas / mkl, T in {1,2,4,8}, pinned to P-cores:
BENCH_PCPUS=0,2,4,6,8,10,12,14 benchmarks/run_scaling.sh
benchmarks/analyze_scaling.py benchmarks/data/gemm_scaling_*.csv \
    --plot benchmarks/data/gemm_scaling.png
```

> **Set `BENCH_PCPUS` to your topology** — one logical id per physical core
> (`lscpu -e=CPU,CORE,MAXMHZ`). The default is an i7-12700K's 8 P-cores.

**Measured (i7-12700K, fp64, N=2048):** native-fast scales **5.8× on 8 P-cores**
(56.8 → 330.8 GFLOP/s), vs OpenBLAS **7.15×** (528) and MKL **7.30×** (547). So
native-fast scales well to ~4 cores (3.7×, ~92% efficiency) but its efficiency
trails the tuned libraries at high thread counts — the single-thread ~80%-of-
OpenBLAS gap widens to ~62% at 8 threads. The simple per-`(jc,pc)` thread-team
spawn and `ic`-only partition leave room for a persistent thread pool and
multi-loop (BLIS-style) parallelization — a future optimization, tracked
separately from this measurement.

## Adding a new backend (e.g. CUDA)

Because the benchmark uses the public API, a new backend is added in the
**library** (give the relevant `mtl::` ops a compile-time dispatch path guarded
by e.g. `MTL5_HAS_CUDA`), then add a build variant + `--label cuda` to
`run_sweeps.sh`. No harness changes are required.
