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
```

One committed profile per machine, so the arch flag, pin list, output directory
and the `--allow-decomposed` decision live in the repository rather than in a
shell history:

```bash
bash benchmarks/machines/ryzen-9-8945hs-int.sh      # Zen 4, under WSL -- the only NATIVE x86 part
bash benchmarks/machines/jetson-orin-nano-int.sh    # A78AE, native SDOT; OUTDIR follows nvpmodel
bash benchmarks/machines/i7-12700k-int.sh           # Alder Lake: has AVX-VNNI, cannot reach it (§7)
bash benchmarks/machines/xeon-e5-2420-int.sh        # SSE4: no VNNI at all -- the control
```

| machine | arch flag | pin | quad dot | why it is in the set |
|---|---|---|---|---|
| Ryzen 9 8945HS | `-march=znver4` | `0,2,…,14` | native `vpdpbusd` | the only native x86 datapoint |
| Jetson Orin Nano | `-mcpu=native` | `0,…,5` | native `SDOT`/`UDOT` | native, and **not x86** |
| i7-12700K | `-march=alderlake` | `0,2,…,14` | decomposed | modern memory system, instruction absent |
| Xeon E5-2420 v2 | `-march=native` | `0,…,5` | decomposed | no VNNI silicon exists — the control |

The two decomposed profiles pass `--allow-decomposed` on the operator's behalf,
because on those parts the decomposition is the measurement rather than a
misconfiguration. The two native profiles deliberately do **not**: if the guard
trips there, the build did not get the ISA it was supposed to, and that is worth
stopping for.

The Xeon pins `0,1,2,3,4,5` where the others pin `0,2,4,…` — its SMT siblings are
**blocked** (`0,6`), (`1,7`) … not adjacent pairs, so the interleaved list would
put two threads on three cores and idle the rest.

⚠️ **The two native machines are native for *opposite* pairings**, and a
cross-machine comparison that pairs arms by name will get the sign of the effect
wrong:

| pairing | x86 AVX3_DL | x86 AVX10.2 | NEON + DotProd | NEON + I8MM |
|---|---|---|---|---|
| `u8 × i8` | **native** `vpdpbusd` | native | emulated | **native** `USDOT` |
| `i8 × i8` | emulated | native `vpdpbssd` | **native** `SDOT` | native |
| `u8 × u8` | emulated | native `vpdpbuud` | **native** `UDOT` | native |

So `gemm_u8i8_i32_quad` is the fast arm on x86 and the *slow* one on the Jetson.
x86 gets exactly one pairing before AVX10.2 and ARM exactly two before I8MM, so
neither is a superset of the other and **neither machine is a baseline for the
other's arms**.

`bench_all` therefore reports support **per pairing**, and
`mtl::simd::has_native_quad_dot_v<NA, NB>` is the compile-time query:

```text
SIMD backend:    AVX3_DL   int8 quad dot: PARTIAL
                 u8*i8 native   i8*i8 emulated   u8*u8 emulated
```

`PARTIAL` is the common case — every machine measured so far that has the
instruction at all has it for some pairings and not others; only AVX10.2 and
NEON+I8MM report `NATIVE`. A partial build runs without `--allow-decomposed`
(it is a legitimate measurement) but is labelled `native-int-partial`, because
half its arms are decomposed and nothing in a timing says which half.

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

### Troubleshooting: `/usr/bin/env: 'bash\r': No such file or directory`

The script was checked out with **CRLF** line endings, so the shebang names an
interpreter called `bash\r`. This happens on Windows, where Git's default
`core.autocrlf=true` converts on checkout — and it affects **every** `.sh` in
this repository, not just this one, so it will bite the moment any part of the
benchmark contract is run under WSL.

The repository now carries a `.gitattributes` pinning `*.sh` to `eol=lf`, which
prevents it. An **already-checked-out** tree still has the old endings, though;
attributes only apply at checkout.

Both repairs below **discard uncommitted changes to the files they touch** —
commit or stash first.

```bash
# just the two scripts
rm benchmarks/run_int_bench.sh benchmarks/machines/ryzen-9-8945hs-int.sh
git checkout -- benchmarks/run_int_bench.sh benchmarks/machines/ryzen-9-8945hs-int.sh

# or the whole working tree
git rm --cached -r . && git reset --hard
```

Do **not** reach for `sed -i 's/\r$//' …` instead. It fixes the line endings and
leaves the tree looking modified, so `preflight` refuses the run. That is worth
stating precisely, because `git diff` will tell you the file is unchanged:

```console
$ sed -i 's/\r$//' s.sh
$ git diff --numstat s.sh          # empty -- no content change
$ git status --porcelain s.sh
 M s.sh                            # ... but modified, and it stays that way
warning: in the working copy of 's.sh', LF will be replaced by CRLF the next time Git touches it
```

Under `core.autocrlf=true` — the setting that produced the CRLF in the first
place — Git reports the file modified because the working tree no longer matches
what a checkout *would* write, not because the content differs. `preflight` tests
exactly `git status --porcelain`, so it rejects, and correctly: a benchmark run
on a tree Git considers modified cannot be attributed to a commit. Re-checkout
instead, which leaves the tree genuinely clean.

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

`bench_all` prints the decision **per operand pairing**, and `build_isa` records
`AVX512VNNI`, `AVX3_DL` and `DOTPROD`, so a sidecar says which path was measured:

```text
SIMD backend:    SSE4   int8 quad dot: DECOMPOSED
                 u8*i8 emulated   i8*i8 emulated   u8*u8 emulated
```

Three states, and **`PARTIAL` is what every machine measured so far reports** —
each has the instruction for some pairings and not others, because x86 and ARM
implement opposite ones (see [the quad arms](#the-quad-arms-and-what-they-revise)
below). Only AVX10.2 and NEON+I8MM report `NATIVE`, and neither has been
measured.

| state | guard | label |
|---|---|---|
| `NATIVE` | runs | `native-int` |
| `PARTIAL` | runs — it is a legitimate measurement, it just has to be labelled | `native-int-partial` |
| `DECOMPOSED` | **refuses** without `--allow-decomposed` | `native-int-decomposed` |

Pass `--allow-decomposed` to measure the decomposition on purpose; the label
records it so the CSV cannot be mistaken for the other.

### The `gemm` arm, and why it inverts the dot result

A dot has no operand reuse, so it is bandwidth-bound and the operand *width*
dominates. A GEMM reuses each operand O(n) times: the panels are packed once and
read from cache and registers thereafter, so how many bytes they occupied in
memory stops mattering. Measured on a Xeon E5-2420 v2 (SSE4, **no** VNNI), n=512:

| arm | GOP/s |
|---|---|
| `gemm_f32` | **18.8** |
| `gemm_i32` | 13.5 |
| `gemm_i16_i32` | 12.8 |
| `gemm_i8_i32` | 13.3 |

**Narrowing the operand buys nothing here** — `gemm_i8_i32` is no faster than
`gemm_i32`, and every integer arm is *slower* than fp32. That is not a defect:
the widening path promotes int8 into int32 lanes and then does the same int32
multiply-add, so it inherits int32's arithmetic cost plus a promotion, and the
memory saving it was built for is irrelevant once operands are reused. Integer
multiply is also simply slower than float FMA on this ISA — there is no integer
FMA, so it is a separate multiply and add.

`gemm_i8_i32` is the **widen-on-load** path: narrow operands promoted to int32
lanes, then an ordinary multiply-add. It is *not* `vpdpbusd`, which consumes
four k-values per instruction and needs a quad-interleaved pack layout and a
different micro-kernel.

### The quad arms, and what they revise

That micro-kernel now exists, and it changes the conclusion above. The two new
arms run the **quad multiply-accumulate** itself — four k-values per
instruction, from quad-interleaved panels — against the same operands:

| arm | GOP/s | vs `gemm_i8_i32` |
|---|---|---|
| `gemm_f32` | 18.4 | fp32 baseline |
| `gemm_i32` | 13.2 | i32 × i32, same-type |
| `gemm_i16_i32` | 12.1 | i16 × i16, widen |
| `gemm_i8_i32` | 13.2 | i8 × i8, widen |
| `gemm_i8_i32_quad` | **16.5** | i8 × i8, quad |
| `gemm_u8i8_i32_quad` | **21.8** | u8 × i8, quad |

Xeon E5-2420 v2, SSE4, n=512, `int8 quad dot: decomposed` — **this machine still
has no VNNI.** From the committed run in
`benchmarks/data/xeon-e5-2420/int_arms.csv`, produced by `run_int_bench.sh` with
its preflight and its native-quad guard (see [The guard](#the-guard) above);
provenance in the `.sysinfo` sidecar. Figures are best-of-iteration; on this run
the median and the best agree to within 5.1% for every arm at every size, so the
choice of statistic does not carry the result.

### Read the ratios one variable at a time

The four int8 numbers differ in **two** things — kernel and operand signedness —
so only same-row-different-one-thing pairs are controlled comparisons. On the
Xeon, across the four sizes:

| comparison | n=128 | n=256 | n=512 | n=1024 | what varies |
|---|---|---|---|---|---|
| `gemm_i8_i32_quad` ÷ `gemm_i8_i32` | 1.18× | 1.23× | **1.25×** | **1.25×** | the kernel, operands fixed |
| `gemm_u8i8_i32_quad` ÷ `gemm_i8_i32_quad` | 1.27× | 1.32× | **1.32×** | **1.32×** | operand signedness, kernel fixed |
| `gemm_u8i8_i32_quad` ÷ `gemm_i8_i32` | 1.50× | 1.62× | 1.65× | 1.66× | **both — not a controlled result** |

The last row is the product of two effects and must not be quoted as what the
quad kernel buys. That distinction is not pedantry, and the four-machine data
below shows how badly it misleads: the same naive ratio reads **4.78×** on the
Ryzen and **1.52×** on the Jetson, while the actual kernel effect on those
machines is 2.15× and 3.64× — overstating on one, understating on the other.

### Four machines, and what the Xeon alone could not say

**The Xeon is the low outlier, by about a factor of three.** The kernel effect is
real everywhere and *grows with n* everywhere, but its magnitude is a property of
the machine, not a constant. GEMM at n=1024, GOP/s, best-of-iteration,
single-threaded:

| machine | native pairing | fp32 | `i8` widen | `i8_quad` | `u8i8_quad` | kernel | native ÷ emulated (raw) | best int8 ÷ fp32 |
|---|---|---|---|---|---|---|---|---|
| Xeon E5-2420 v2 (SSE4) | none | 19.7 | 13.4 | 16.8 | **22.2** | 1.25× | — | 1.12× |
| i7-12700K (AVX2) | none | 146.3 | 49.2 | 134.4 | **145.9** | **2.73×** | — | 1.00× |
| Ryzen 9 8945HS (AVX3_DL) | `u8×i8` | 141.4 | 96.5 | 207.9 | **461.0** | 2.15× | **2.22×** | **3.26×** |
| Jetson Orin Nano (NEON) | `i8×i8` | 14.3 | 8.6 | **31.5** | 13.1 | **3.64×** | **2.40×** | **2.20×** |

- **kernel** = `gemm_i8_i32_quad ÷ gemm_i8_i32` — same operands, quad against
  widen-on-load. Ranges **1.25× to 3.64×**. It rises with n — i7 2.19→2.73,
  Ryzen 1.69→2.15, Jetson 3.21→3.64 — which is what a register-blocking change
  should do: the small sizes never leave cache, so the operand-traffic reduction
  has nothing to pay for yet. (The Xeon rises 1.18→1.25 and then goes flat: its
  last two points differ by 0.1%, against a 9.6% run-to-run spread on that
  machine. Three of four rise monotonically and the fourth does not contradict
  them.)
- **native ÷ emulated (raw)** = the machine's native pairing ÷ its emulated one,
  *same kernel, same machine, same run*. Only the two `PARTIAL` machines can
  supply it. **This is a RAW ratio, not the instruction's contribution** — the two
  pairings also differ in signedness and in decomposition path, so it moves two
  variables. Netted below.
- Note the two decomposed machines reach **parity with fp32 and no more**
  (1.12×, 1.00×; the Xeon's 12% sits against a 9.6% spread on that machine),
  while both native machines clear it by 2.2–3.3×.

**Which arm is fastest inverts between the two native machines**, because they
implement opposite pairings: `u8i8_quad` is 2.2× the `i8_quad` on the Ryzen, and
the `i8_quad` is 2.4× the `u8i8_quad` on the Jetson. Comparing those two machines
by arm *name* gives 461.0 against 13.1 — a 35× "machine gap" that is almost
entirely a naming artifact.

### What the instruction is worth, and why the dot understated it

The raw native-over-emulated ratios above move **two** variables — nativeness and
the pairing's decomposition shape — which is the error this page warns about two
sections earlier. The shape term is measurable: it is the same ratio on the
machines where *both* pairings are emulated.

| | shape control (both emulated) |
|---|---|
| Xeon E5-2420 v2 | 1.32× |
| i7-12700K | 1.09× |

favouring `u8 × i8`. Dividing that out — and noting it acts in *opposite
directions* on the two native machines, because they implement opposite pairings:

| | raw | shape acts | **instruction, net** |
|---|---|---|---|
| Ryzen 9 8945HS (native `u8×i8`) | 2.22× | *with* the native arm | **1.68–2.04×** |
| Jetson Orin Nano (native `i8×i8`) | 2.40× | *against* the native arm | **2.60–3.18×** |

**The Jetson figure carries a caveat the Ryzen one does not.** No ARM machine
here emulates both pairings, so its shape control is borrowed from x86, where the
decomposition is structurally similar (two native calls plus a shift and a
subtract) but not identical. Treat 2.60–3.18× as indicative and the 2.40× raw
ratio as the measurement.

The same ratios on the **dot**, for contrast:

| | dot (L1-resident, n=1024) | GEMM (n=1024) |
|---|---|---|
| Ryzen 9 8945HS | 1.50× raw → **1.13–1.35×** net | 2.22× raw → **1.68–2.04×** net |
| Jetson Orin Nano | **2.75×** raw (no same-ISA control) | 2.40× raw |

The Ryzen dot figure reproduces §6's ~1.2×, which is the cross-check that both
suites measure the same thing. **The GEMM figure is roughly 1.5× that**, and that
is what the hardware plan predicted in direction: a dot is bandwidth-bound, so
instruction efficiency barely shows, while a GEMM is compute-bound throughout and
shows it fully. The magnitude is now measured rather than assumed.

Physically coherent, which is worth checking on a 3× result: one quad instruction
does four times the multiply-accumulates of an fp32 FMA of the same width, so 4×
is the ceiling. Measured against each machine's own fp32 GEMM: Ryzen 3.26×,
Jetson 2.20×, decomposed i7 1.00×.

**The best-int8-over-fp32 column is not affected by any of this.** It compares
each machine's fastest int8 arm against its *own* fp32 GEMM, so no pairing
confound enters — which is why the conclusion that rests on it (only the machines
with the instruction beat fp32 by a margin worth having) stands on the raw data.
The decomposed parts are not *below* fp32; they are beside it, at 1.00× and
1.12×, the first exactly at parity and the second against a 9.6% spread.

### What this replaced

The original claim was that on a machine without VNNI there is no int8 GEMM win
at all. That was true of the *widen-on-load* kernel and does not survive changing
the kernel — the i7, with no usable instruction anywhere in it, gets **2.73×**
from the kernel shape alone.

The reason is visible in the disassembly rather than inferred. Highway's
*decomposition* of the quad accumulate is a pair of `vpmaddwd` plus
sign-extension shifts, which still folds four products per accumulator lane in a
handful of instructions — where widen-on-load runs four independent
promote-multiply-add chains.

So the operand width still contributes nothing in a GEMM. What has changed is the
size of what remains: *the instruction*, read as the four-products-per-lane
**kernel shape**, is worth 1.25–3.64× and needs no VNNI silicon at all; *the
silicon on top of that* is worth a further **1.7–2.0× on Zen 4** net of the shape
control (2.22× raw), and more than that on the Jetson though without a same-ISA
control to pin it. Those are separate quantities and this is the first time the
programme could measure both.

Both int8 arms are compiled into the same binary on purpose. They compute
bit-identical results — integer addition is associative — so nothing in an
*answer* distinguishes them, and comparing a quad number on one machine against
a widen number on another is precisely the missing-control error that cost this
programme 20% on the dot headline once already.
