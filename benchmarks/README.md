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

```text
benchmarks/
  bench_all.cpp          dense BLAS/LAPACK driver (sizes/sweeps/suites + --label)
  bench_klu.cpp          sparse scoreboard: native KLU vs SuiteSparse KLU (#138)
  bench_superlu.cpp      sparse scoreboard: native LU vs SuperLU (#186)
  bench_sparse.cpp       level-scheduled sparse triangular-solve scaling (#297)
  machines/              per-machine run profiles: the pin list, thread counts and ISA
                         flag for each benchmark host, committed instead of retyped
  run_sweeps.sh          builds native/native-fast/openblas/blis/mkl variants and runs the sweeps
  run_scaling.sh         multi-core GEMM scaling across backends and thread counts (#108)
  run_scaling_297.sh     native 1->N scaling of every threaded kernel family (#297)
  fetch_klu_matrices.sh      downloads the SuiteSparse circuit matrices for bench_klu
  fetch_superlu_matrices.sh  downloads the SuiteSparse unsymmetric matrices for bench_superlu
  plot_results.py        GFLOP/s-vs-N plots from the CSVs
  analyze_gate.py        % of a reference backend (--reference, default openblas) / % of FMA peak + the pass/fail perf gate
  analyze_scaling.py     speedup + parallel efficiency from the run_scaling*.sh CSVs
  harness/
    timer.hpp            high-resolution timing + statistics
    reporter.hpp         console table + CSV output
    generators.hpp       deterministic matrix/vector generators
    runner.hpp           per-suite runners (call the public mtl:: API)
  data/                  committed example CSVs + rendered plots (see data/README.md)
```

`bench_all` is the dense driver documented in the next few sections; the three
sparse binaries are covered under [Sparse direct-solver
scoreboards](#sparse-direct-solver-scoreboards).

All four are built only when `MTL5_BUILD_BENCHMARKS=ON`, which the `release`
preset now sets:

```bash
cmake --preset release && cmake --build build-release -j$(nproc)
```

That preset links no external library, so it builds every benchmark in its
native-only configuration — enough to run them, but the vendor comparison
columns stay empty. See the per-backend builds below.

## Build & run (reproducible, all variants)

```bash
# Builds native / native-fast / openblas / mkl, pins to a P-core, one CSV each.
# OUTDIR is REQUIRED and must be this machine's own directory: the CSVs are named
# by backend, not by machine, so a shared one silently overwrites another
# machine's committed results (#439).
OUTDIR=benchmarks/data/i7-12700k BENCH_CPU=4 benchmarks/run_sweeps.sh
# custom sweep:
OUTDIR=benchmarks/data/i7-12700k BENCH_CPU=4 benchmarks/run_sweeps.sh 16:2048:32
# BLAS-only (the #82 gate; native LAPACK at large N is slow):
OUTDIR=benchmarks/data/i7-12700k BENCH_CPU=4 BENCH_SUITES=blas benchmarks/run_sweeps.sh
```

`run_sweeps.sh` runs `benchmarks/preflight.sh` first and refuses to measure on a
dirty tree, against a competing build, or on a machine without thermal headroom
(see the run contract in `docs/benchmarks/systems.md`); the machine state it
captures is appended to every `.sysinfo` sidecar. It then configures each variant
in its own (clean) build dir and builds `bench_all` for **all** variants before
measuring any of them, so no backend is timed while the machine is still hot from
compiling its own binary. The suites named in `BENCH_SUITES` (default
`blas lapack`) then run single-threaded. The MKL variant is skipped automatically if
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

| Suite | Runs |
|-------|------|
| `all` | every suite below except `gemm-rect` / `ewise` |
| `blas` | `l1` + `l2` + `l3` |
| `l1` | `dot`, `nrm2`, `axpy`, `scal` |
| `l2` | `gemv`, `ger`, `symv`, `trmv`, `trsv` |
| `l3` | `gemm`, `trmm`, `trsm`, `symm`, `syrk`, `syr2k` |
| `lapack` | `lu`, `qr`, `cholesky`, `eig` |
| `gemm-rect` | rectangular GEMM shapes: the BLIS multi-loop 2D grid (#297) |
| `ewise` | element-wise vector/matrix expression sweeps (#297) |
| `int` | `int-dot` + `int-gemv` — the integer arms (#451 phase 4) |
| `int-dot` | fp64/fp32 baselines vs int32, int16→int32, int8→int32, uint8×int8→int32 |
| `int-gemv` | fp64/fp32 baselines vs int32 |

Any individual op above is also a suite name of its own (`--suite trsm`). The
authoritative list is always `bench_all --help`.

`gemm-rect` and `ewise` carry their own built-in shape sets and ignore
`--sizes` / `--sweep`.

`int` is deliberately **not** part of `all`: adding arms to the default run
would change every machine's committed CSV and break comparison with the
existing baselines. Ask for it by name.

## Integer arms (#451 phase 4)

```bash
benchmarks/run_int_bench.sh --outdir benchmarks/data/<machine> [--arch <flag>] [--pin 0,2,4,6]
benchmarks/machines/ryzen-9-8945hs-int.sh          # Zen 4, under WSL
```

### Read the curve, not a ratio

A dot product is a streaming reduction: two operands in, two ops each, no reuse.
At large `n` it is **bandwidth-bound**, and an int8 dot moves one byte per
element where fp64 moves eight. A large-`n` int8 speedup of ~8× is therefore
expected **whether or not the machine has VNNI at all** — that is bytes, not
arithmetic. The instruction only shows where the kernel is compute- or
latency-bound, which for a reduction means the L1-resident sizes.

So the suite sweeps by *footprint*, 1K → 4M elements, and the shape across that
range is the result. Measured on a Xeon E5-2420 v2 (SSE4, **no** VNNI), int8
against fp64:

| n | 1024 | 4 096 | 16 384 | 65 536 | 262 144 | 1 048 576 | 4 194 304 |
|---|---|---|---|---|---|---|---|
| `int8/fp64` | 0.96× | 2.25× | 1.87× | 2.49× | 2.95× | 3.69× | **7.34×** |

Flat at the small end, ~7× at the large end — the bandwidth effect with no
instruction behind it. That machine is the **control**: a Zen 4 advantage at the
*small* end that this table lacks is the instruction; the large end is bytes on
both. (Shape check only — not a contract-compliant run.)

### The guard

`run_int_bench.sh` refuses to time a build without the native quad
multiply-accumulate, because nothing in a timing distinguishes `vpdpbusd` from
the `vpmaddwd` decomposition, and having the hardware is not sufficient:

- Highway selects its VNNI target (`HWY_AVX3_DL`) only on the **full**
  conjunction `__AVX512VNNI__ ∧ __VAES__ ∧ __VPCLMULQDQ__ ∧ __AVX512VBMI__ ∧
  __AVX512VBMI2__ ∧ __AVX512VPOPCNTDQ__ ∧ __AVX512BITALG__`.
- **MSVC cannot define these.** `/arch:AVX512` covers F/CD/BW/DQ/VL only and
  there is no `/arch` for VNNI — so the MSVC profile
  `machines/ryzen-9-8945hs.ps1` cannot produce a VNNI build. Use WSL/gcc
  (`-march=znver4`) or clang-cl (`/clang:-march=znver4`).

`bench_all` prints the decision, and `build_isa` now records `AVX512VNNI`,
`AVX3_DL` and `DOTPROD`, so a sidecar says which path was measured:

```
SIMD backend:    SSE4   int8 quad dot: decomposed
```

Pass `--allow-decomposed` to measure the decomposition on purpose; the label
becomes `native-int-decomposed` so the CSV cannot be mistaken for the other.

### No `gemm` arm

The epic asks for int arms for dot, gemv and gemm. There is **no integer GEMM**
to benchmark — phases 0–3 delivered dot and gemv, and `mult(dense2D<int32_t>,…)`
runs the generic triple loop. An arm named `gemm_i32` would time the fallback
while implying the kernel, so there is none.

This is also where VNNI would pay most: a GEMM reuses each operand O(n) times
and is compute-bound, so the arithmetic density would show instead of being
masked by memory traffic. The tile is already settled (it is the float tile,
#464) and `kc` is already a multiple of 4; what remains is the quad-interleaved
pack layout.

### BLAS routine coverage

Benchmarked, and therefore implemented in MTL5: **L1** `dot`, `nrm2`, `axpy`,
`scal`; **L2** `gemv`, `ger`, `symv`, `trmv`, `trsv`; **L3** `gemm`, `trmm`,
`trsm`, `symm`, `syrk`, `syr2k`. That is the full L2/L3 core — #227 closed the
gap that earlier revisions of this file described as outstanding.

Standard BLAS routines with no public `mtl::` operation, and hence no benchmark:
L1 `asum`, `iamax`, `rot`, `copy`, `swap`. (`copy` has a raw binding in
`mtl/interface/blas.hpp` for internal use, but no public op; `copy`/`swap` are
normally expressed through assignment and `std::swap`.)

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
OUTDIR=benchmarks/data/i7-12700k BENCH_CPU=4 BENCH_SUITES=blas \
    benchmarks/run_sweeps.sh 65:1025:80

# % of OpenBLAS and % of FMA peak, per op and size (peak = 1 P-core fp64):
benchmarks/analyze_gate.py benchmarks/data/i7-12700k/blas_sweep_native-fast.csv \
    benchmarks/data/i7-12700k/blas_sweep_openblas.csv --peak-gflops 78

# Pass/fail gate: median GEMM ratio >= 80% of OpenBLAS for N >= 256,
# and no individual size below the 70% floor
benchmarks/analyze_gate.py benchmarks/data/blas_sweep_native-fast.csv \
    benchmarks/data/blas_sweep_openblas.csv --gate --op gemm \
    --threshold 0.80 --floor 0.70 --min-size 256
```

### What the gate asserts, and why

`--gate` exits non-zero if **either**:

- the **median** of the per-size ratios falls below `--threshold` (default 0.80), or
- **any single size** falls below `--floor` (default 0.70).

The median is the primary assertion because the epic's target — "within 10–20%
of OpenBLAS" — is an aggregate claim, and because it is the only statistic here
that reproduces. Two runs of the identical protocol on the same idle machine
gave — **2026-08-01 measurements, before #382 raised native-fast to ~97% of
OpenBLAS; the figures are retained because they are the evidence for the rule,
not a current result**:

| statistic | run A | run B | swing |
|---|---:|---:|---:|
| median of ratios | 82.3% | 82.5% | **0.2 pt** |
| mean | 82.1% | 82.2% | 0.1 pt |
| min (the pre-#327 rule) | 76.3% | 78.7% | 2.4 pt |
| worst single size | — | — | **6.2 pt** |

The old rule failed if *any* size was below threshold — gating on the minimum,
the noisiest statistic available — and the two runs failed at disjoint sets of
sizes. The median is ~30× more stable.

The floor keeps the rule honest: a genuine cliff at one size must still fail, so
no size may drop below `--floor`, set well under the threshold so ordinary
±6-point noise cannot trip it.

> **Raising the iteration count does not help.** Within-run stddev is already
> only 0.3–2.2% of the median at 10 iterations — the variance is *between* runs
> (turbo/thermal state, allocation, page layout), not within them.

It is **not** wired into the per-push CI: shared runners have unstable clocks
and no P-core pinning, which makes absolute perf gates flaky. Run it on
dedicated hardware.

**Measured results live on the per-system result pages, not here** — see
[Intel i7-12700K](../docs/benchmarks/i7-12700k.md), indexed from
[Benchmark systems](../docs/benchmarks/systems.md). Keeping numbers in one place
means a re-run updates them once; this file documents how to *produce* them.

## Multi-core GEMM scaling (#108)

The native blocked GEMM parallelizes its `ic` (row) loop with the C++ standard
concurrency runtime (set `MTL5_NUM_THREADS`). `run_scaling.sh` sweeps GEMM over
thread counts for native-fast **and** threaded OpenBLAS/MKL (their own
`*_NUM_THREADS`), pinning to physical performance cores; `analyze_scaling.py`
reports speedup + parallel efficiency and draws the scaling plot.

```bash
# native-fast / openblas / mkl, T in {1,2,4,8}, pinned to P-cores.
# OUTDIR is REQUIRED -- one directory per machine (#439).
OUTDIR=benchmarks/data/i7-12700k BENCH_PCPUS=0,2,4,6,8,10,12,14 \
    benchmarks/run_scaling.sh
benchmarks/analyze_scaling.py benchmarks/data/i7-12700k/gemm_scaling_*.csv \
    --plot benchmarks/data/i7-12700k/gemm_scaling.png
```

> **Set `BENCH_PCPUS` to your topology** — one logical id per physical core
> (`lscpu -e=CPU,CORE,MAXMHZ`). The default is an i7-12700K's 8 P-cores.

Measured scaling numbers are on the
[i7-12700K result page](../docs/benchmarks/i7-12700k.md). The structural finding
is stable across sessions: native-fast tracks the tuned libraries closely to 2
threads and loses ground as the count grows, because the per-`(jc,pc)` thread-team
spawn and `ic`-only partition leave room for a persistent thread pool and
multi-loop (BLIS-style) parallelization — a future optimization, tracked
separately from this measurement.

### Which variable sets the thread count

Concurrency is a **runtime** axis of an already-built binary — you do not
rebuild to change it. Which variable applies depends on what the binary was
built against:

| Build | Variable |
|-------|----------|
| `native-fast` (`-DMTL5_NATIVE_FAST_GEMM=ON -DMTL5_WITH_HIGHWAY=ON`) | `MTL5_NUM_THREADS` |
| `openblas` (`-DMTL5_WITH_BLAS=ON`) | `OPENBLAS_NUM_THREADS` |
| `blis` (`-DBLA_VENDOR=FLAME`) | `BLIS_NUM_THREADS` |
| `mkl` (`-DBLA_VENDOR=Intel10_64lp`) | `MKL_NUM_THREADS` |

`MTL5_NUM_THREADS` is read **once**, on first use of the pool, and clamped to
the hardware concurrency; unset or invalid means 1 (fully serial). See
`mtl/detail/thread_pool.hpp` and `docs/algorithms/on-node-threading.md`.

> **A plain build will not show GEMM scaling.** The threaded GEMM/GEMV paths in
> `mtl/operation/mult.hpp` sit inside `#ifdef MTL5_NATIVE_FAST_GEMM`, so in a
> build without it (the `release` preset, for instance) `MTL5_NUM_THREADS` leaves
> `--suite gemm` / `l3` timings unchanged. The L1/L2 reductions (`dot`, `nrm2`,
> `axpy`, `scal`) use the pool unconditionally and do scale. Build the
> `native-fast` variant for meaningful dense scaling numbers.

Labelling matters when sweeping by hand: `analyze_scaling.py` recovers the
thread count by parsing the `backend` CSV column, so pass
`--label <backend>-t<T>` exactly as the scripts do.

```bash
for T in 1 2 4 8; do
  MTL5_NUM_THREADS=$T OMP_NUM_THREADS=$T \
    taskset -c $(seq -s, 0 2 $((2*T-2))) \
    ./build-native-fast/benchmarks/bench_all --suite gemm --sizes 1024,2048 \
      --label native-fast-t$T --csv t$T.csv
done
```

## Threaded-kernel scaling across families (#297)

`run_scaling.sh` covers GEMM across backends. `run_scaling_297.sh` is the
native-only counterpart that sweeps **every** threaded kernel family 1→N,
writing one CSV per family with the `backend` column labelled `native-t<T>`:

| Suite | Family | Issues |
|-------|--------|--------|
| `gemm_rect` | rectangular GEMM, BLIS multi-loop 2D grid | #311 |
| `lu` / `qr` / `chol` | dense factorizations | #298, #300 |
| `ewise` | element-wise vector/matrix expression sweeps | #312 |
| `sparse` | level-scheduled sparse triangular solves | #301–#309 |

```bash
BENCH_PCPUS=0,2,4,6,8,10,12,14 THREADS="1 2 4 8" benchmarks/run_scaling_297.sh
benchmarks/analyze_scaling.py benchmarks/data/scaling_*.csv --plot out.png
benchmarks/analyze_scaling.py benchmarks/data/scaling_ewise.csv --op ewise-vec
```

Environment: `BENCH_PCPUS`, `THREADS`, `LAPACK_SIZES` (default
`1024,2048,4096`), `SPARSE_SIZES` (2-D grid sides, default `100,160`), `BUILD`
(default `build-scaling-297`), `JOBS`. `analyze_scaling.py` keys its series on
`(operation, size)`, so families that share a size no longer collide.

#### Sparse sizes grow expensive fast

The sparse family's cost is dominated by the **untimed factorization** each case
performs before its (millisecond) solves are measured, and it scales far worse
than the grid side suggests. Whole-suite wall clock at `T=1` on an i7-12700K:

| grid side | 2-D case | wall clock |
|---|---|---|
| 100 | n=10,000 | 11 s |
| 160 | n=25,600 | 3 m 38 s |
| 200 | n=40,000 | 8 m 46 s |
| 320 | n=102,400 | not measured |

That is **per thread count** — the factorization is serial, so every value in
`THREADS` pays it again. The default is therefore ~15 minutes for
`THREADS="1 2 4 8"`, in proportion with the rest of the driver.

The default was `200,320` until #321; a `T=1` run was killed after **3 h 28 min**
without finishing its first size. #322 clamped the 3-D grid, which is what
brought side 200 down to the 8 m 46 s above, but `200,320` remains impractical
for a four-thread-count sweep. Raise `SPARSE_SIZES` deliberately, with the table
above in mind.

Results are written up in `docs/design/issue-297-threading-results.md`; the plan
is `docs/design/issue-297-threading-benchmark-plan.md`.

## Sparse direct-solver scoreboards

Three binaries measure the sparse side. Each runs a built-in synthetic suite
with no arguments, so they are useful immediately after a plain build:

```bash
./build-release/benchmarks/bench_klu          # 2D Poisson, 32^2 .. 256^2
./build-release/benchmarks/bench_superlu      # 2D convection-diffusion (unsymmetric)
./build-release/benchmarks/bench_sparse       # synthetic sparse triangular solves
```

**`bench_klu`** (#138) — native KLU vs SuiteSparse KLU: factor + solve time,
fill, block structure, residual.

```bash
./bench_klu A.mtx B.mtx        # those matrices instead of the built-in suite
./bench_klu ext:Big.mtx        # external-only row: skip the native run (too slow)
./bench_klu --csv out.csv      # also write a CSV scoreboard
```

**`bench_superlu`** (#186) — the same shape for native LU vs SuperLU. SuperLU is
supernodal (BLAS-3) while native LU is scalar non-supernodal, so the ratio is
expected to grow with dense fill — quantifying that gap is the point.

The vendor columns require the corresponding build flag; without it each binary
says so in its header and reports native-only:

```bash
cmake -B build-klu -DMTL5_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
      -DMTL5_WITH_SUITESPARSE_KLU=ON          # or -DMTL5_WITH_SUPERLU=ON
```

Real matrices come from `fetch_klu_matrices.sh` / `fetch_superlu_matrices.sh`,
which download the SuiteSparse sets these scoreboards are tuned around.

**`bench_sparse`** (#297) — the level-scheduled solve phase of the native sparse
Cholesky / LDLT / LU and supernodal solvers. Native-only, no external library.
The solves are bit-identical across thread counts by construction (proven in
CI); this measures how much of that level structure becomes wall-clock speedup.

```bash
MTL5_NUM_THREADS=8 ./bench_sparse --csv t8.csv --label native-t8
./bench_sparse --file A.mtx B.mtx     # add SPD/general matrices from disk
./bench_sparse --sizes 100,150        # 2-D grid side lengths (default 100,160)
```

`run_scaling_297.sh` drives it as its `sparse` family — one process per thread
count, pinned to physical cores, into `benchmarks/data/scaling_sparse.csv`.

## Adding a new backend (e.g. CUDA)

Because the benchmark uses the public API, a new backend is added in the
**library** (give the relevant `mtl::` ops a compile-time dispatch path guarded
by e.g. `MTL5_HAS_CUDA`), then add a build variant + `--label cuda` to
`run_sweeps.sh`. No harness changes are required.
