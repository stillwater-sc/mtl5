# Benchmark Systems

Every performance number published for MTL5 is tied to a specific machine, a
specific set of vendor libraries, and a specific pinning policy. This page is
the index of those machines. Each one links to its full result page.

Performance claims do not travel between machines. A GEMM efficiency measured
on a desktop hybrid CPU says little about a server part with different cache
sizes, memory channels, and turbo behaviour — so read the results page together
with the system description here.

## Systems

| System | CPU | Cores used | Results |
|--------|-----|-----------|---------|
| `i7-12700K` | 12th Gen Intel Core i7-12700K | 8 P-cores (E-cores excluded) | [Results](i7-12700k.md) |
| `ryzen-9-8945hs` | AMD Ryzen 9 8945HS (Zen 4) | 8 cores (SMT siblings excluded) | [Results](ryzen-9-8945hs.md) |
| `xeon-e5-2420v2` | Intel Xeon E5-2420 v2 (Ivy Bridge-EP) | 6 cores (SMT siblings excluded) | pending — #430 |
| `jetson-orin` | NVIDIA Jetson Orin, ARM Cortex-A78AE | 6 cores | pending — #430 |
| `arduino-ventuno-q` | Qualcomm "Monaco", 4× Cortex-A78C + 4× Cortex-A55 | 4 A78C cores (A55 excluded) | null run (no cache sizes exposed); see its section |

A machine is registered here as soon as it is part of an experiment, which can
precede its first result page: the bottom two rows exist because #430 needs the
cache hierarchies and pinning policies of all four machines settled *before* it
starts measuring. A row without a results link means the hardware is
characterized and the suites have not been run on it yet.

## `i7-12700K` — desktop hybrid (Alder Lake)

The primary MTL5 development machine.

| | |
|---|---|
| CPU | 12th Gen Intel Core i7-12700K |
| Topology | 8 P-cores (2 threads each, logical 0–15) + 4 E-cores (logical 16–19) |
| Clocks | P-cores 4.9–5.0 GHz max, E-cores 3.8 GHz max, 800 MHz min |
| Cache | L1d 512 KiB (12 instances), L2 12 MiB (9 instances), L3 25 MiB (shared) |
| Memory | 31 GiB |
| OS | Ubuntu 24.04.4 LTS, kernel 6.8.0-136 |
| Compiler | GCC 13.3.0, `-O3 -DNDEBUG` (CMake `Release`) |
| CPU governor | `powersave` (`intel_pstate`; still boosts, but clocks are not pinned) |

### Cache hierarchy — two hierarchies, not one

This machine has **two different cache configurations in one socket**, and which
one MTL5 detects depends on where the detecting thread is scheduled (#432). Both
readings are recorded because neither is "the" answer for this CPU:

| Pinned to | L1d | L2 (raw / cores sharing) | L3 | fp64 `kc` `mc` | fp32 `kc` `mc` |
|---|---|---|---|---|---|
| P-core, `taskset -c 0` (Golden Cove) | 48 KiB, 12-way | 1.25 MiB / **1** | 25 MiB / 12 | `768` `106` | `768` `213` |
| E-core, `taskset -c 16` (Gracemont) | 32 KiB, 8-way | 2 MiB / **4** | 25 MiB / 12 | `512` `64` | `512` `128` |
| — Haswell defaults, for comparison | 32 KiB | 256 KiB | 8 MiB | `512` `32` | `512` `64` |

Read with `util_test_cache_info -s` from a `ci`-preset build. Blocking figures are
the 128-bit (SSE) build; an AVX2 build divides them by the wider SIMD width.

Three things follow, and all matter when reading any result from this machine:

1. **Detection is per-core-class and must be pinned.** The two rows are what the
   detector reports under `taskset -c 0` and `taskset -c 16`; both are stable
   across repeats since #432, but they are *different*, so every measurement must
   record which class it pinned to. Unpinned, the detector reports the smallest
   per-core budget across the cores the process may use — deterministic, and
   chosen so blocks fit whichever kind the work lands on.
2. **The E-core L2 is one 2 MiB instance shared by four cores**, so a core's share
   is 512 KiB. The `mc` column reflects that discount: `64` rather than the `256`
   the raw 2 MiB would give. Before #432 the raw figure was used, sizing the
   packed A block for roughly 4x the L2 a core actually has.
3. **Sharing is counted in physical cores, not logical CPUs.** The P-core L1d/L2
   are shared by an SMT pair yet report `1`, because the pinning policy runs one
   thread per physical core and leaves the sibling idle — counting CPUs would
   halve both. The `25 MiB / 12` L3 is the same rule read the other way: 20
   logical CPUs collapse to the 12 physical cores (8 P + 4 E) that share it.

Under this page's pinning policy (P-cores, E-cores excluded) the relevant row is
the first: against the Haswell defaults, detection blocks with a larger `kc` and
`mc`. **It is slower.**

### Cache hierarchy: asymmetric clusters, and a level MTL5 cannot see

From the module datasheet, with what detection reports beside it:

| | 4-core cluster | 2-core cluster |
|---|---|---|
| L1 per core | 128 KiB (64 KiB d + 64 KiB i) | same |
| L2 per core | 256 KiB private | same |
| L3 | 2 MiB shared by **4** → **512 KiB/core** | 2 MiB shared by **2** → **1 MiB/core** |
| System Cache | 4 MiB shared across **both** clusters | — |

Two things follow that matter when reading any result from this machine.

**The cores are identical but their cache *sharing* is not.** All six are A78AE
with the same private L1/L2, yet a core in the small cluster has twice the L3
share of one in the large cluster. That is a third topology class beyond the
homogeneous-private parts (`xeon-e5-2420v2`, `ryzen-9-8945hs`) and the hybrid
P/E one (`i7-12700K`): *symmetric cores, asymmetric sharing*. A thread grid
spanning both clusters therefore has non-uniform cache behaviour even though
every thread runs the same core.

**Detection handles it correctly, and the datasheet is why we know.** It reports
`l3 = 2 MiB / 4 cores`, i.e. it selected the **smaller** per-core share (512 KiB)
rather than the larger. That is #432's rule — take the minimum per-core budget
across the CPUs the process may run on — doing exactly what it was written for,
so blocks fit whichever cluster the work lands on.

**The 4 MiB System Cache is not modelled at all.** It does not appear in sysfs, so
`cache_info` cannot see it and `derive_blocking` never considers it. On this part
there is a real level between the 2 MiB cluster L3 and DRAM that the blocking
model is blind to. Worth remembering before concluding anything about `nc` here.

### Cache-blocking A/B result — detection loses here

`benchmarks/run_blocking_ab.sh`, AVX2 build (`MTL5_NATIVE_ARCH=ON`), fp64,
P-cores pinned, 5 interleaved rounds, min of 5. Ratio is detected / default
throughput; the analyzer does not call anything within **2%** of parity, so
`0.985` counts as a tie and everything below `0.98` is a loss:

| shape | T=1 | T=8 |
|---|---|---|
| 1024³ | 0.965 | **0.551** |
| 2048³ | 0.920 | 0.929 |
| 4096³ | 0.901 | 0.985 |
| 852 × 8192 × 1024 | 0.914 | 0.806 |
| 852 × 12288 × 1024 | 0.906 | 0.675 |

`detected kc=384 mc=213` against `default kc=256 mc=64`. Nine losses, one tie,
no wins. Three separate causes, and only the first is about cache sizing:

1. **At T=1 the larger blocks are slower at every size** — by 3.5% at 1024³ and
   8–10% at the other four. No threading is involved: the analytical "half of L1,
   half of L2" sizing is beaten by the hand-tuned constants on Golden Cove.
2. **A larger `mc` starves the thread partition.** `gemm_blocked` fixes
   `ic_nt = min(budget, nib)` from the *unbalanced* block count, so `mc` 64 → 213
   takes `nib` 16 → 5 at 1024³ and five threads of eight do the work.
3. **A larger `kc` inflates the packed B panel**, and once the jc loop splits
   into teams each holds one: 2 × 12.6 MB against a 25 MiB L3.

Causes 2 and 3 are the #408 / #429 family — a cache-derived parameter moving a
*block count* the thread partition is sensitive to — and are defects independent
of detection, which merely exposed them. Because of this result, cache detection
ships **opt-in** (`MTL5_ENABLE_CACHE_DETECTION`); MTL5's shipped blocking is the
compile-time defaults. Re-run the harness before assuming this verdict holds on
different hardware.

### Vendor libraries

| Backend | Version | Selected by |
|---------|---------|-------------|
| OpenBLAS | 0.3.26 (pthread build) | `-DMTL5_WITH_BLAS=ON -DMTL5_WITH_LAPACK=ON` |
| BLIS | 0.9.0 | `-DMTL5_WITH_BLAS=ON -DBLA_VENDOR=FLAME` |
| Intel MKL | oneAPI 2026.1 | `-DBLA_VENDOR=Intel10_64lp` (after `setvars.sh`) |
| SuiteSparse KLU | 7.6.1 | `-DMTL5_WITH_SUITESPARSE_KLU=ON` |
| SuperLU | 6.0.1 | `-DMTL5_WITH_SUPERLU=ON` |
| Google Highway | 1.0.7 | `-DMTL5_WITH_HIGHWAY=ON` (native-fast SIMD) |

OpenBLAS is the active `libblas.so.3` alternative on this host, so a build
configured with `MTL5_WITH_BLAS=ON` and no `BLA_VENDOR` genuinely links
OpenBLAS rather than the reference netlib BLAS — the `openblas` label means what
it says.

### A vendor BLAS inside an MTL5 region (#640)

The test is a batch of 64 independent n×n dgemms, split across the 8 tids of one
MTL5 region. The vendor's own team is also set to 8 threads. It runs pinned to
one logical CPU on each P-core (0,2,…,14). Each arm runs in its own process, 3
rounds, with the median of 15 reps after a 300 ms warm-up. The command is
`OUTDIR=benchmarks/data/i7-12700k-vendor-nested BENCH_PCPUS=0,2,4,6,8,10,12,14
benchmarks/run_vendor_nested.sh`; the data is in
`benchmarks/data/i7-12700k-vendor-nested/`. Vendors: OpenBLAS 0.3.32 (pthreads),
BLIS 1.1 (OpenMP) and MKL 2026.1 (`Intel10_64lp`, Intel OpenMP). The governor
is `powersave`, and each run waited for a 1-minute load average below 1.

The arms:
- `uncapped`: each tid calls `dgemm_` directly, which is what MTL5 did before
  #640.
- `capped`: each tid calls through `mtl::interface::blas::gemm`.
- `direct`: the whole batch is called from one thread outside a region, so the
  vendor's team is the only parallelism.
- `process1`: OpenBLAS set to one thread with `openblas_set_num_threads(1)`,
  the process-wide configuration the threading docs give it.

Median GFLOP/s over the rounds, with the range in brackets where it is wider
than 10%:

| Vendor | Arm | n = 128 | n = 256 | n = 512 |
|---|---|---|---|---|
| BLIS | uncapped | 0.1 | 0.6 | 3.0 |
| BLIS | **capped** | **405** | **457** | **507** |
| BLIS | direct | 190 | 331 | 442 |
| MKL | uncapped | 328 | 338 [305–369] | 381 [308–393] |
| MKL | **capped** | **499** | **468** | **513** |
| MKL | direct | 380 | 391 | 473 |
| OpenBLAS | uncapped | 244 | 163 [108–259] | 289 [284–407] |
| OpenBLAS | capped (no per-thread control: same code as uncapped) | 249 | 105 [102–253] | 391 [289–401] |
| OpenBLAS | **process1** | **462** | **430** | **486** |
| OpenBLAS | direct | 257 | 338 | 429 |

What it shows:

- **BLIS without the cap collapses**, by ×170 to ×3800. Each of the 8 tids starts
  an 8-thread OpenMP team, so 64 spinning threads share 8 cores. With the cap it
  runs at 405–507 GFLOP/s, which is also faster than `direct` (×1.15 to ×2.1):
  8 independent single-thread GEMMs keep every core busier than one 8-thread
  GEMM of the same size.
- **MKL degrades without the cap, but less**, and the cap gains ×1.35 to ×1.52.
  The uncapped arm also varies more between rounds.
- **OpenBLAS cannot be capped per thread, and with 8 concurrent callers it is
  erratic.** One arm configuration measured 102 GFLOP/s in one round and 259 in
  another. `uncapped` and `capped` run the same code, so their difference is
  that spread, not a result. The documented configuration, OpenBLAS at one
  thread for the whole process, is both the fastest and the steadiest arm:
  ×1.7 to ×2.6 over `uncapped`, within 0.5% between rounds.
- The checksums of every arm agree at each size.

A small vendor call does not measure the cap's per-call cost. Once a region's
first vendor call has set the vendor to one thread, every later call checks a
thread-local flag. A 256-element `ddot` timed through the wrapper and directly
moved by up to 2× between processes in either direction. That included the
OpenBLAS build, where the cap compiles to nothing at all, so that arm was taken
out of the benchmark.


### Pool dispatch and join latency (#636)

Many near-empty regions run on a pool of T threads, and each one is timed with
`steady_clock` (the invariant TSC here). Three latencies are recorded:
- **dispatch**: from the caller's last timestamp before `run()` to the latest
  worker's entry into the task;
- **join**: from the latest tid's exit from the task to the caller's return;
- **region**: the whole `run()`.

The arms are:
- `thread`: `thread_pool`.
- `spin`: `spin_pool` with its default spin of 4096 iterations, about 140 µs on
  a P-core.
- `sleep`: `spin_pool` with spin 0, so every wait sleeps at once.
- `hold`: `spin_pool` with a spin of 40000 iterations, about 1.4 ms, which
  outlasts the gap.

The gap between regions is either 0 (back to back) or 1000 µs (a 1 kHz loop).
Runs are pinned to one logical CPU per P-core (0,2,…,14). Each arm runs in its
own process, for 4 rounds with the arm order rotated each round, so every arm
runs in every position once: 20000 regions at gap 0 and 2000 at gap 1000. The
command is `OUTDIR=benchmarks/data/i7-12700k-pool-latency
BENCH_PCPUS=0,2,4,6,8,10,12,14 benchmarks/run_pool_latency.sh`, and the data is
in `benchmarks/data/i7-12700k-pool-latency/`. The governor is `powersave`, and
the 1-minute load average was 1.4 at the start.

Region latency in µs, as p50 / p99 / max, each the median over the rounds:

| Gap | Arm | T = 2 | T = 4 | T = 8 |
|---|---|---|---|---|
| 0 | thread | 3.03 / 3.57 / 39.6 | 3.32 / 5.23 / 73.5 | 6.08 / 7.99 / 39.6 |
| 0 | **spin** | **0.23 / 0.25 / 4.6** | **0.31 / 0.38 / 6.1** | **0.48 / 0.57 / 7.4** |
| 0 | sleep | 0.27 / 0.44 / 5.9 | 0.55 / 0.70 / 12.9 | 0.79 / 2.41 / 12.4 |
| 0 | hold | 0.24 / 0.25 / 3.3 | 0.34 / 0.39 / 6.4 | 0.51 / 0.60 / 7.9 |
| 1000 | thread | 20.4 / 102 / 573 | 24.9 / 118 / 223 | 41.4 / 58.4 / 1478 |
| 1000 | spin | 10.6 / 98.0 / 172 | 12.4 / 110 / 129 | 16.3 / 61.1 / 1368 |
| 1000 | sleep | 15.5 / 98.6 / 137 | 13.0 / 109 / 196 | 16.7 / 72.5 / 579 |
| 1000 | **hold** | **0.68 / 0.98 / 5.4** | **0.83 / 1.18 / 6.0** | **1.06 / 1.37 / 608** |

What it shows:

- **Back to back, `spin_pool` is about 10× faster than `thread_pool`** at
  every percentile up to p99: 0.23–0.48 µs per region at the median against
  3.0–6.1 µs. Most of `thread_pool`'s time is the condition-variable wake on
  dispatch (1.4–4.1 µs p50) and the matching wake of the caller on join
  (1.5–1.6 µs).
- **`spin_pool` with spin 0 is still faster than `thread_pool`**, 0.27–0.79 µs at
  the median. libstdc++'s `std::atomic::wait` itself spins 16 times (12 pauses,
  then 4 yields) before it calls the futex. Back to back, the next region often
  arrives within that window, and no mutex is taken.
- **At 1 kHz with the default spin, every pool pays the kernel's wake-up.** The
  gap outlasts the spin, so the workers are asleep when the region arrives:
  10–17 µs at the median, about 100 µs at p99, and maxima up to 1.5 ms. These
  are wake-ups from idle cores under `powersave`, not costs of the pool.
  `spin_pool`'s join stays fast even then (0.5 µs p50 against 9–11 µs), because
  the caller is awake and spinning.
- **`hold`, a spin longer than the gap, keeps a 1 kHz loop at about 1 µs**:
  p99 1.0–1.4 µs, max 6 µs at T ≤ 4. At T = 8 the pool fills all 8 pinned
  cores, so anything else the OS runs there preempts a spinner, and the max
  rises to 608 µs. A real-time deployment gives the pool cores of its own, as
  the threading docs say. The cost is 8 cores spinning continuously.
- **A yield in the spin loop matters when the pool fills its cores.** Before
  every 64th spin iteration became a yield, `spin` at T = 8 and gap 1000 had a
  dispatch median of 155 µs in all three rounds of that run, while `sleep` stayed at
  20 µs. The scheduler had placed two of the pool's threads on one core, and
  the spinner held it. With the yield the median is 16 µs.
- **Widening the dispatch word to 64 bits cost nothing measurable.** The word
  was 32 bits, with a generation that could wrap, and libstdc++ waits on a
  4-byte atomic with a futex directly but on an 8-byte one through a shared
  table. With the workers asleep at 1 kHz, the dispatch median was
  10.1–15.9 µs with the 32-bit word (3ace550) and 9.6–15.6 µs with the 64-bit
  one (1e32af6).

### Pinning policy

This is a **hybrid P/E-core CPU**, which makes pinning mandatory rather than
optional. An unpinned single-threaded run lets short L1 kernels land on a 3.8 GHz
E-core and reports a number that has nothing to do with the 5.0 GHz P-core the
same code would reach in production. The failure mode is documented in
[the multi-core scaling investigation](../performance/multicore-scaling-investigation.md).

| Run type | Pinning | Rationale |
|----------|---------|-----------|
| Single-thread sweeps | `taskset -c 4` | One P-core (5.0 GHz bin) |
| Scaling runs, `T` threads | first `T` of `0,2,4,6,8,10,12,14` | One logical id per **physical** P-core — SMT siblings excluded, so scaling reflects cores, not hyperthreads |

Single-threaded runs additionally pin every vendor's threading to one thread
(`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`BLIS_NUM_THREADS=1`, `MTL5_NUM_THREADS=1`) so no backend quietly uses more of
the machine than another.

E-cores are never used for measurement. Nothing here characterises them.

## `ryzen-9-8945hs` — mobile Zen 4 (Hawk Point)

A second development machine, and the first Windows/MSVC entry. Its results page
is a **native-Windows** reproduction of the same suites, which is why its backend
set differs from the Linux machine (see the two notes below).

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M (Zen 4, "Hawk Point") |
| Topology | 8 physical cores, 16 logical (2-way SMT; siblings adjacent, logical 0–1, 2–3, …) |
| Clocks | 4.0 GHz base, up to ~5.2 GHz boost (not pinned; Windows balanced power plan) |
| Cache | L1 32 KiB d + 32 KiB i per core, L2 1 MiB per core (8 MiB total), L3 16 MiB (shared) |
| Memory | 27.8 GiB |
| OS | Windows 11 Pro 10.0.26200 (25H2) |
| Compiler | MSVC 19.51.36247 (Visual Studio 18 2026), `/O2 /DNDEBUG` (CMake `Release`) |

These figures are self-reported by each binary via `mtl::util::identify()`
(`include/mtl/util/system_info.hpp`), and a `<csv>.sysinfo` tag is written next
to every result CSV — so the machine identity is recorded by the run, not typed
in by hand.

### Vendor libraries

| Backend | Version | Selected by |
|---------|---------|-------------|
| OpenBLAS | 0.3.34 (official Windows binary, `DYNAMIC_ARCH`) | explicit `-DBLAS_LIBRARIES=…\libopenblas.lib` |
| Intel MKL | 2026.1.0 (pip `mkl-devel`) | explicit `-DBLAS_LIBRARIES=…\mkl_rt.lib` |
| Google Highway | 1.4.0 (FetchContent) | `-DMTL5_WITH_HIGHWAY=ON` (native-fast SIMD) |
| SuperLU | 7.0.1 (vcpkg) | `-DMTL5_WITH_SUPERLU=ON` + vcpkg toolchain |
| SuiteSparse KLU | 7.12.3 (vcpkg) | `-DMTL5_WITH_SUITESPARSE_KLU=ON` + vcpkg toolchain |

**BLIS is absent by construction**: it has no native MSVC build (it configures
under Clang/MSYS only), so there is no `blis` curve here.

**OpenBLAS is the official prebuilt Windows binary, not vcpkg's.** vcpkg's
OpenBLAS on MSVC builds an untuned generic kernel (measured ~9 GFLOP/s GEMM, ~8×
below the tuned build) and ships BLAS only — its `DYNAMIC_ARCH` and LAPACK both
require a Fortran/mingw toolchain that vcpkg does not use on MSVC. The official
mingw binary has the runtime-dispatched CPU kernels and a bundled LAPACK, so it
serves as both the BLAS reference and the OpenBLAS LAPACK curve.

MKL is installed from the `mkl-devel` wheel rather than the oneAPI installer
(this machine's build account is not elevated). It is the same library at the
same version the Linux page used (2026.1); only the packaging differs. It is
linked through the single-dynamic-library dispatcher `mkl_rt.lib`.

### Register-tile sweep (#643): no tile clears the noise, so AVX-512 keeps 6×16

`bash benchmarks/machines/ryzen-9-8945hs-tile.sh` under WSL, fp64, 5 rotated
rounds, data in `benchmarks/data/ryzen-9-8945hs-tile-double/`. Only `mr` changes;
nr = 16, kc = 128, mc = 128 and nc = 8192 are identical across the arms.
Geometric-mean ratio of the median against `default` (mr = 6):

| Tile | Accumulators (of 32) | T=1 | T=8 |
|---|---|---|---|
| 8×16 | 16 | 0.995 | 1.046 |
| 10×16 | 20 | 0.996 | 1.036 |
| 12×16 | 24 | 1.032 | 1.021 |
| 14×16 | 28 | 1.002 | 0.994 |

The default arm's round-to-round spread was 5–41%, as expected inside a VM
(Windows places the virtual CPUs). 12×16 is nominally ahead at both thread counts
(×1.032 and ×1.021), but neither gain, nor any other tile's, clears that spread,
so AVX-512 builds keep the AVX2 description. That is a
measured reason to keep it, not a measured loss for a larger tile. A bare-metal
Linux run would be the way to settle it.

### Pinning policy

This is a **homogeneous** 8-core part (no P/E split), but SMT still makes pinning
mandatory: an unpinned single-thread run can share a physical core with its SMT
sibling and report a contended number. Pinning uses a Windows processor-affinity
**bitmask** (there is no `taskset`); `benchmarks/run_sweeps.ps1` and
`run_scaling.ps1` set it via `System.Diagnostics.Process.ProcessorAffinity`.

| Run type | Affinity mask | Rationale |
|----------|---------------|-----------|
| Single-thread sweeps | `0x1` | logical 0 = one thread of physical core 0 (sibling idle) |
| Scaling runs, `T` threads | OR of `1 << (2·i)`, i < T | one logical id per physical core 0,2,…,2(T−1) — SMT siblings excluded |

The logical→physical map was confirmed with `GetLogicalProcessorInformationEx`
(a `topology_probe`), not assumed. Single-threaded runs additionally force every
vendor's threading to one thread (`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `MTL5_NUM_THREADS=1`).

## `xeon-e5-2420v2` — server Ivy Bridge-EP

The oldest machine in the set, and kept deliberately. It has **no AVX2 and no
FMA**, so Highway selects its 128-bit target and the GEMM micro-kernel runs 2
doubles wide at close to this core's arithmetic ceiling. That makes it the
**negative control** for cache-blocking experiments (#430): a blocking change
that appears to help here is almost certainly measuring something else.

| | |
|---|---|
| CPU | Intel Xeon E5-2420 v2 @ 2.20 GHz |
| Topology | 6 physical cores / 12 threads (SMT), 1 socket, 1 NUMA node |
| Logical→physical | logical 0–5 = physical cores 0–5, logical 6–11 = their SMT siblings (confirmed with `lscpu -e`, not assumed) |
| Clocks | 2.2 GHz base, 2.7 GHz max turbo, 1.2 GHz min |
| Cache | L1d 32 KiB/core 8-way (192 KiB total), L2 256 KiB/core (1.5 MiB total), L3 15 MiB shared, 64 B line |
| ISA | SSE4.2, AVX, AES, F16C — **no AVX2, no FMA** |
| Memory | 15 GiB |
| OS | Ubuntu 24.04.4 LTS, kernel 6.8.0-137 |
| Compilers | GCC 13.3.0, Clang 18.1.3, `-O3 -DNDEBUG` (CMake `Release`) |
| CPU governor | `ondemand` (`intel_cpufreq`; clocks are not pinned) |

### Vendor libraries

| Backend | Version | Notes |
|---------|---------|-------|
| Reference BLAS | 3.12.0 (`libblas3`) | netlib reference, **not** OpenBLAS |
| Reference LAPACK | 3.12.0 (`liblapack3`) | netlib reference |
| Google Highway | 1.4.0 | fetched by CMake (`-DMTL5_WITH_HIGHWAY=ON`) |

OpenBLAS, BLIS and MKL are **not installed**. The `libblas.so.3` alternative
resolves to the netlib reference implementation, so a `MTL5_WITH_BLAS=ON` build
on this host links reference BLAS. Vendor-comparison numbers are therefore not
meaningful here and should not be produced from this machine — it is a machine
for measuring MTL5's own kernels against each other.

### Blocking parameters

Recorded because this machine is the reason #426's central claim went untested:
its L1d and L2 happen to **equal** the hardcoded Haswell defaults, so runtime
cache detection changes nothing here. Only its L3 differs (15 MiB against the
8 MiB default), and L3 is not applied to `nc` (#429).

| Backend | width\<double\> | fp64 `mr nr kc mc nc` | detected == default? |
|---------|---------------|------------------------|----------------------|
| Highway 1.4.0 (128-bit) | 2 | `6 4 512 32 2048` | yes, every field |
| Scalar fallback | 1 | `6 2 1024 16 1024` | yes, every field |

Measured GEMM ceiling for context (fp64, Highway, `n = 3072`, min of several):
**9.2–9.4 GFLOP/s** single-threaded and **~46 GFLOP/s** on 6 threads. The
single-thread figure sits against a non-FMA SSE2 arithmetic ceiling of 8.8
GFLOP/s at the 2.2 GHz base clock and 10.8 at the 2.7 GHz single-core turbo —
i.e. the kernel is compute-bound, which is exactly why it is the control.

### Pinning policy

Homogeneous cores (no P/E split), but SMT makes pinning mandatory for the same
reason it does on the Zen 4: an unpinned single-thread run can share a physical
core with its sibling and report a contended number.

| Run type | Pinning | Rationale |
|----------|---------|-----------|
| Single-thread sweeps | `taskset -c 0` | one thread of physical core 0 (sibling 6 idle) |
| Scaling runs, `T` threads | `taskset -c 0-(T-1)`, `T <= 6` | logical 0–5 are distinct physical cores |
| SMT-inclusive runs | `taskset -c 0-11` | label explicitly; not the default |

Single-threaded runs additionally force `MTL5_NUM_THREADS=1` (and the vendor
equivalents where a vendor is linked).

## `jetson-orin` — Jetson Orin Nano (Tegra234, 6× Cortex-A78AE)

| | |
|---|---|
| CPU | NVIDIA Jetson **Orin Nano** Engineering Reference Developer Kit (Super), Tegra234 |
| Topology | **6× Cortex-A78AE** (ARMv8.2), no SMT, all six online. Six is the module's **physical** core count, not a power-mode reduction — `nvpmodel.conf` defines `CPU_A78_0..5` and nothing else. **Two asymmetric clusters**: one 4-core and one 2-core (datasheet) |
| Clocks | 729.6 MHz – **1497.6 MHz** (15 W mode cap). Governor **`schedutil`**, *not* pinned — see the caveat below |
| Cache | L1d 64 KiB 4-way **private**, L2 256 KiB **private**, L3 2 MiB **per cluster**, 64 B line. Plus a **4 MiB System Cache** shared across clusters that sysfs does not expose — see below |
| ISA | NEON, 128-bit (`nr=4` for fp64 ⇒ 2 doubles). **No SVE** — a data point for #427 |
| Memory | 7 GiB total, **shared with the GPU** |
| OS | Ubuntu 22.04.5 LTS, **L4T R36.4.7** (JetPack 6.x, GCID 42132812), kernel 5.15.148-tegra |
| Compilers | GCC 11.4.0, `-O3 -DNDEBUG` (CMake `Release`) |
| Power mode | **15 W (mode 0)**. GPU capped 306–612 MHz with 4 TPCs active, EMC 2133 MHz |
| Thermal at rest | cpu 47.4 °C, gpu 46.2 °C, soc ~47 °C — far below throttle |

> **Clocks were not pinned for the run on this page.** The governor is
> `schedutil` and `jetson_clocks` was not applied, so cores ramp between 729.6
> and 1497.6 MHz rather than sitting at the cap. Both A/B arms are affected
> equally and the interleaved min-of-5 protocol absorbs much of it, but these are
> **15 W-mode, unpinned** numbers and should not be compared against a `MAXN`
> or `jetson_clocks` run. Single-thread GEMM measured 6.7 GFLOP/s, roughly 56% of
> the ~12 GFLOP/s fp64 NEON peak this core reaches at the 15 W cap.

The 6-core figure matters beyond bookkeeping: the thread pool clamps
`MTL5_NUM_THREADS` to `hardware_concurrency`, so a run asking for 8 threads here
gets a budget of 6, and every thread-grid calculation is bounded by 6. The A/B
CSVs recorded only the *requested* count, which is why the first analysis of this
machine used the wrong budget; the benchmark now records the effective pool size
alongside it.

### Commands to fill this in

```bash
# identity, topology, clocks
lscpu; lscpu -e; cat /proc/cpuinfo | head -20
# cache hierarchy, exactly as MTL5 detects it (non-x86 -> sysfs path)
for d in /sys/devices/system/cpu/cpu0/cache/index*; do \
  echo "$(cat $d/level) $(cat $d/type) $(cat $d/size) $(cat $d/coherency_line_size)"; done
# and what MTL5 itself derives from that
ctest --test-dir build -R util_test_cache_info --output-on-failure -V | grep l1d=
# power/thermal state -- the numbers are meaningless without these
sudo nvpmodel -q; sudo jetson_clocks --show; cat /sys/devices/virtual/thermal/thermal_zone*/temp
```

### Cache-blocking A/B result — neutral, and the grid explains the one loss

`kc` doubles (64 KiB L1d) and `mc` halves — the opposite corner from the Zen 4.
Result: **0 faster, 1 slower, 9 indistinguishable**, at an effective budget of 6
(the run asked for 8; the pool clamps to `hardware_concurrency`), in **15 W mode
with unpinned clocks**.

| shape | T=1 | T=8 (pool 6) |
|---|---|---|
| 64 × 4096 × 1024 | 0.961 | 0.988 |
| 64 × 6144 × 1024 | 0.965 | **0.760** |
| 1024³ | 0.961 | 0.905 |
| 2048³ | 0.994 | 0.990 |
| 4096³ | 1.006 | 1.004 |

The single loss is the single shape where the two arms get different thread
grids: `mc = 16` gives `nib = 4`, so `ic_nt = min(6,4) = 4` and then
`jc_nt = 6/4 = 1` by integer division — a grid of 4 where the default's `mc = 32`
yields 2 × 3 = 6. Predicted 0.667, measured 0.760. `3 × 2` and `2 × 3` were both
legal, so full utilisation existed and the factorization missed it (#429).

Everywhere the grids match, the arms match. **`mc` shows no locality effect on
this machine at all** — its entire measured influence is through the grid.

### Pinning and thermal policy (applies regardless of the module)

This is the part that differs from every x86 machine here, and it is not
optional. A Jetson's clocks are governed by a **power mode** and are subject to
thermal throttling, so an unpinned, unconfigured run measures the cooling
solution rather than the code.

| Requirement | Why |
|---|---|
| Fix `nvpmodel` to one mode and record it | mode selects online cores *and* clock caps; the default differs per module and per JetPack image |
| Run `jetson_clocks` to disable DVFS ramping | otherwise early reps run at a lower clock than late ones, which reads as a warm-up effect |
| Record temperature before and after each run | a run that throttles mid-sweep must be discarded, not averaged |
| Pin with `taskset -c 0-(T-1)` | homogeneous cores, no SMT expected — confirm with `lscpu -e` before relying on it |
| Allow a fixed idle gap between reps | sustained load will throttle where interleaved short runs do not |

Interleave A/B variants **within one session** as on every other machine, and
treat any run whose end temperature crossed the throttle point as invalid data
rather than a slow result.

## `arduino-ventuno-q` — Arduino Ventuno Q (Qualcomm "Monaco", 4× Cortex-A78C + 4× Cortex-A55)

| | |
|---|---|
| Board | Arduino Ventuno Q, host `ventunoq-01`. The device tree names the SoC, not the board: `Qualcomm Technologies, Inc. Monaco Monza addons` |
| Topology | **big.LITTLE, 8 cores, no SMT.** 4× **Cortex-A78C** r0p2 (MIDR part `0xd4b`) on **CPUs 0–3**; 4× **Cortex-A55** r2p0 (`0xd05`) on CPUs 4–7. The big cores are the *low* ids here — the common "big cores are 4–7" layout would have pinned the run to the little ones |
| Clocks | A78C 902–**2112 MHz** as configured (`scaling_max_freq`), against a 2361.6 MHz hardware maximum; A55 844.8–1958.4 MHz. Governor **`schedutil`** on the A78C cluster, `ondemand` on the A55 cluster, *not* pinned |
| Cache | **Not exposed.** The kernel lists L1d, L1i, L2 and L3, but with no sizes (`lscpu -C` prints the levels and blank sizes), so MTL5's sysfs probe reads 0 for every level — see below |
| ISA | NEON, 128-bit, with dot product (`asimddp`). **No SVE** — like the Orin |
| OS | Ubuntu 24.04.5 LTS, kernel 6.8.0-1084-qcom |
| Compilers | GCC 13.3.0, `-O3 -DNDEBUG` (CMake `Release`), `MTL5_NATIVE_ARCH=ON` |
| Thermal | 35.5 °C before, 35.9 °C after the session; limit 126 °C. The board never came near throttling |

Profile: `benchmarks/machines/arduino-ventuno-q.sh`. It reads the A78C ids from
`/proc/cpuinfo` by part number rather than assuming them, and its output
directory follows the A78C cluster's governor, as the Jetson's follows its power
mode.

### Cache-blocking A/B result — a null run, because there was nothing to detect

All four arms ran with **identical** blocking (`mr,nr,kc,mc,nc = 6,4,512,32,2048`
and the same `mc` actually used at every point). The detecting arms found no
cache sizes (`cache_l1d_bytes=0`, `cache_l2_bytes=0` in their sidecars), so
`derive_blocking` fell back to the compile-time model. Unlike the Xeon's null
run, this is **not** agreement between the hardware and the model: it says
nothing about whether detection would help on an A78C. The analyzer now names
this case from the sidecar instead of reporting that the caches match. (The
kconly and mconly sidecars from this run say `detection_enabled=0`: the benchmark
then recorded only the both-levels macro. It now writes `detection_levels`, and
the analyzer infers the levels from the arm name for older sidecars.)

Taken under `schedutil` with clocks unpinned, commit `46ea958`, T ∈ {1, 4} on
CPUs 0–3, 8 interleaved rounds (5 requested, rounded up to a multiple of the 4
arms), min of 5. Ratio is detected / default:

| shape | T=1 | T=4 |
|---|---|---|
| 64 × 4096 × 1024 | 1.015 | 1.045 |
| 64 × 6144 × 1024 | 1.008 | 0.989 |
| 1024³ | 1.010 | 1.012 |
| 2048³ | 1.013 | 0.987 |
| 4096³ | 1.012 | 0.997 |

Two things it does measure:

* **The noise floor of this board under `schedutil`.** Every ratio compares two
  separately built binaries that ran identical blocking (the arms differ only in
  their detection macros, which here changed nothing). Nine of ten are within
  1.5% of 1.0; the 64 × 4096 × 1024 point at T=4 is off by 4.5%, more than the
  analyzer's 2% floor. Some of that may be build-to-build difference (code
  layout, alignment) rather than run-to-run noise; this run cannot separate them. That is the error the structural null-run check exists to catch, and a
  reason to repeat this run with the `performance` governor before trusting a
  small effect here.
* **Throughput.** The default arm reaches 9.1 GFLOP/s fp64 on one A78C, about
  54% of the 16.9 GFLOP/s NEON peak at 2112 MHz (two 128-bit FMA pipes, 8 flops a
  cycle), close to the Orin's 56%. Four cores reach 36.6 GFLOP/s at 2048³, a
  speedup of 4.0.

To make the A/B mean something here, the detecting arm needs cache sizes the
kernel does not provide. `MTL5_BLOCKING_KC` / `MTL5_BLOCKING_MC` can impose them
by hand, but only once the A78C configuration is known: its L1d and L2 sizes are
chosen by the SoC integrator, so the core name alone does not fix them.

### Register-tile sweep (#643): 8×4 wins, and 12×4 loses at four threads

`benchmarks/machines/arduino-ventuno-q-tile.sh`, `schedutil`, 5 rotated rounds,
data in `benchmarks/data/arduino-ventuno-q-schedutil-tile-{double,float}/`. Every
arm is the same source with only `mr` changed; nr, kc, mc and nc are identical
across the arms (fp64 nr = 4, kc = 512, mc = 32; fp32 nr = 8, kc = 512, mc = 64),
and the checksums agree across all arms. The ratios are median GFLOP/s against
`default` (mr = 6 in both runs, which predate the change), as a geometric mean
over the shapes:

| Tile (fp64 / fp32) | Accumulators (of 32) | fp64 T=1 | fp64 T=4 | fp32 T=1 | fp32 T=4 | Worst shape (fp64 / fp32) |
|---|---|---|---|---|---|---|
| 8×4 / 8×8 | 16 | **1.174** | **1.102** | **1.170** | **1.219** | 1.060 / 1.156 |
| 10×4 / 10×8 | 20 | 1.156 | 1.045 | 1.171 | 1.160 | 0.967 / 1.064 |
| 12×4 / 12×8 | 24 | 1.061 | 0.930 | 1.063 | 1.206 | 0.840 / 1.048 |
| 14×4 / 14×8 | 28 | 1.049 | 0.878 | 1.030 | 1.028 | 0.736 / 0.930 |

* **The peak is at half the register file**, not at the 3/4 that #351 measured on
  AVX2. 12×4 is what the 3/4 rule would give on 32 registers, and it loses.
* **fp32 agrees on 8.** 8×8 is best at T=4 on every shape and ties 10×8 at T=1.
  fp32 has no 12-row loss: its wide shapes are 128 rows (mc = 64), so each
  thread's 32-row block takes 12-row panels at 36 rows of work, not 24 for 16.
* **The T=4 losses are a divisibility effect, not register pressure.** They fall on
  the wide 64-row shapes, where each of the four threads gets a 16-row block. 8-row
  panels tile it exactly; 12-row panels cover it with 24 rows of work and 14-row
  panels with 28. It is the #408 family again: a register-tile change that moves
  how a block divides.
* **Shipped as a result:** aarch64 builds now derive 8×4 fp64 and 8×8 fp32
  (`target_hw_traits`, 32 registers with a measured budget of 16 accumulators).
  In these sessions that took one A78C from 8.3 to 9.7 GFLOP/s fp64, 57% of
  the 16.9 GFLOP/s peak at 2112 MHz, and from 17.1 to 19.9 GFLOP/s fp32, 59% of
  33.8. Four cores went from 35.5 to 37.6 GFLOP/s fp64 and from 61.9 to
  77.2 GFLOP/s fp32 at 2048³.
* **Noise.** The clocks were unpinned, so the default arm spread 12–14% between
  rounds at T=1 (2–3% at T=4 on the squares). The 8×4 gain is consistent across
  every shape at both thread counts, which noise of that size would not produce.
  The absolute numbers are lower than the A/B session above (8.3 against
  9.1 GFLOP/s for the same 6×4 tile, both under `schedutil`). This data does not
  say why; compare ratios within a session, not GFLOP/s across them.

**Not measured, and covered by the same change:** the Jetson's A78AE and every
other aarch64 core (Apple M-series, the Neoverse cores of the CI
runners). The tile is a compile-time choice per ISA, so all of them moved
together on one board's fp64 data.

### Pinning and clock policy

| Requirement | Why |
|---|---|
| Pin to the A78C cores only (CPUs 0–3), read from `/proc/cpuinfo` | the A55 cores are slower and have a different cache hierarchy; a team that includes one runs at its pace |
| Record the governor, and set `performance` on the A78C policy for a comparable session | there is no `nvpmodel`; the governor is what decides the clocks |
| Note the 2112 MHz cap | `performance` runs at `scaling_max_freq`, not the 2361.6 MHz hardware maximum |
| Keep `schedutil` and `performance` runs in separate directories | the profile does this; they must never be averaged together |

## Cache-blocking A/B — the four-machine result (#430)

All four machines re-run after the thread-grid fix (#441), 5 rounds x 5 reps per
point, every binary verified to contain the fix. **Detection is neutral at best
and harmful at worst: across the 30 informative points, 1 was faster, 8 were
slower, and 21 were indistinguishable.** The single win is 2.7%, barely over the
noise threshold. `MTL5_ENABLE_CACHE_DETECTION` stays opt-in.

| machine | ISA measured | detected kc, mc | verdicts (faster / slower / same) |
|---|---|---|---|
| i7-12700K | AVX2 | 384, 213 | 1 / 7 / 2 |
| Ryzen 9 8945HS | **AVX-512** | 256, 256 | 0 / 1 / 9 |
| Jetson Orin Nano 15 W | NEON | 1024, 16 | 0 / 0 / 10 |
| Xeon E5-2420 v2 | AVX | identical to defaults | null run |

### What the grid fix bought

| point | before #441 | after |
|---|---|---|
| i7 1024³, T=8 | 0.551 | 0.707 |
| Zen 4 1024³, T=8 | 0.590 | 0.928 |
| Jetson wide/short, T=6 | 0.760 | 0.938 |

The catastrophic multi-thread losses are gone. The Zen 4 row is **not** a
controlled comparison: that sidecar reads `build_isa=...AVX512F` where the
earlier run was AVX2, so the ISA changed with the fix. This is the first Zen 4
measurement taken with the intended ISA, and it supersedes rather than extends
the old one — which is exactly what `build_isa` was added to make visible.

### CORRECTED: it is mc, not kc — see the study page

The first round of this experiment concluded that "mc is harmless and kc is where
the loss lives", inferred across machines that each happened to vary something
different. The four-arm follow-up, which varies them **one at a time on a single
machine**, refuted it and reversed the direction:

| i7-12700K, single-threaded | kc | mc | median ratio |
|---|---|---|---|
| `kconly` | 256 → **384** | 64 → 42 | **0.999** (free) |
| `mconly` | 256 | 64 → **320** | **0.919** (8% loss, 5/5 shapes) |

Raising `mc` is the harm, and it shows up with one thread. Raising `kc` is free
serially — though *not* in parallel, where the i7 loses ~6% under `kconly` too.
Neither is ever a win.

The full design, the exact hypothesis tests, the identical-binary calibration and
the mechanism hypothesis are on
[Cache blocking A/B](../performance/cache-blocking-ab-study.md). The operational
conclusion is unchanged: **detection stays opt-in and off by default.**

### The anomaly worth chasing

i7 1024³ at T=8 is 0.707, and it is not noise: detected 312–326 GFLOP/s across
five rounds against 438–461, ranges nowhere near overlapping. What makes it
interesting is that the pre-#441 explanation no longer applies — at m=1024, T=8,
mc_cache=213, the planner caps mc to 128, giving nib=8 and a perfectly balanced
8x1 grid. **The worst remaining point is the balanced one**, so thread starvation
is ruled out and `kc` plus L2 residency is what is left.

The experiment that settles it is a **kc-only vs mc-only** A/B, and the harness
now runs it. Four arms, all the same source, differing only in which detected
level feeds the blocking:

| arm | detects | moves |
|---|---|---|
| `default` | nothing | — (what MTL5 ships) |
| `detected` | L1 + L2 | kc and mc |
| `kconly` | L1 | kc |
| `mconly` | L2 | mc |

Run all four in **one session**, which is what makes them comparable — the arms
rotate order round by round, so no arm systematically leads.

Each machine has a committed profile under `benchmarks/machines/`, so the pin
list, thread counts, ISA flag and output directory are **in the repository
rather than in someone's shell history** — and land in the sidecar as
`harness_profile`, `harness_pcpus`, `harness_arms`:

```bash
benchmarks/machines/i7-12700k.sh            # P-cores 0,2,...,14, T in {1,8}
benchmarks/machines/jetson-orin-nano.sh     # 6 cores, T in {1,6}, OUTDIR follows nvpmodel
benchmarks/machines/arduino-ventuno-q.sh    # A78C cores only (read from cpuinfo), T in {1,4}, OUTDIR follows the governor
```

The integer suite (#451) has its own profile per machine, named `*-int.sh`, since
it needs a different flag set — the ISA flag decides whether the quad
multiply-accumulate is native, which the blocking A/B does not care about:

```bash
bash benchmarks/machines/ryzen-9-8945hs-int.sh    # znver4, native vpdpbusd
bash benchmarks/machines/jetson-orin-nano-int.sh  # A78AE, native SDOT; OUTDIR follows nvpmodel
bash benchmarks/machines/i7-12700k-int.sh         # alderlake, decomposed (AVX-VNNI unreachable)
bash benchmarks/machines/xeon-e5-2420-int.sh      # SSE4, decomposed; pins 0-5, not 0,2,4,...
```

The nc-model sweep (#479) has its own profile per machine, named `*-nc-sweep.sh`.
It **measures nothing** — pure enumeration — so it needs no preflight, no pinning
and no quiet machine, and exists to tell you which shapes a later timing session
should bother with. See [the benchmarks README](../../benchmarks/README.md) for
what to read in the output:

```bash
bash benchmarks/machines/xeon-e5-2420-nc-sweep.sh
bash benchmarks/machines/i7-12700k-nc-sweep.sh
bash benchmarks/machines/ryzen-9-8945hs-nc-sweep.sh
bash benchmarks/machines/jetson-orin-nano-nc-sweep.sh
```

The register-tile sweep (#643) has its own profile per 32-register machine, named
`*-tile.sh`. Each runs `default` (mr = 6) against `tile8`, `tile10`, `tile12` and
`tile14` (on aarch64, where `default` is now mr = 8, the ARM profiles run `tile6`
in place of `tile8`): the same source with only `mr` changed (`MTL5_GEMM_MR_OVERRIDE`), so nr,
kc, mc and nc stay fixed. It runs once for fp64 and once for fp32, each into its own
`...-tile-<dtype>` directory. The ARM profiles reuse their base profile, so the
pinning and the governor or power-mode label are the same as the A/B's:

```bash
benchmarks/machines/arduino-ventuno-q-tile.sh       # NEON, A78C cores, T in {1,4}
benchmarks/machines/jetson-orin-nano-tile.sh        # NEON, 6 cores, T in {1,6}
bash benchmarks/machines/ryzen-9-8945hs-tile.sh     # AVX-512 under WSL, T in {1,8}
```

`isaregs` is the rejected candidate, kept so it stays measurable: the target
ISA's full register file at #351's 3/4 rule (`MTL5_GEMM_TARGET_REGISTER_FILE`). On
NEON and AVX-512 it derives the same tile as `tile12`, so the sweep leaves it out.
Each sidecar records `vec_registers`, `target_vec_registers` and `mr_override`, so
the arms can be told apart; sidecars written from this change on also record
`acc_vectors`. The committed sweeps were built before the field existed, when
every target derived its tile from the 3/4 rule; in them `mr_override` alone
tells the arms apart. The results are in the Ventuno Q and Ryzen sections above;
NEON now ships 8×nr.

```powershell
pwsh benchmarks/machines/ryzen-9-8945hs.ps1  # /arch:AVX512, 8 physical cores
```

Every profile refuses to run on a machine it does not recognise (`-Force` /
`FORCE=1` overrides). That is not fussiness: the CSVs are named by arm and the
runner clears them before writing, so a profile executed on the wrong host
replaces one machine's committed evidence with another's — the #439 failure
wearing a friendlier face. Anything can still be overridden for a one-off
(`ROUNDS=3 ARMS="detected default" benchmarks/machines/i7-12700k.sh`), and the
profile prints when a pin list came from the environment rather than from itself.

Then compare each arm against the baseline:

```bash
./benchmarks/analyze_blocking_ab.py \
    benchmarks/data/i7-12700k/blocking_ab_{kconly,default}.csv
```

If `kc` is the sole cause, the product of this whole line of work is not
"detection off" but **"detect L2, ignore L1"** — a *win* on every machine whose
mc is currently pinned at the default 64.

One thing not to over-read: mc is derived from L2 *given kc*, so `mconly` does
not reproduce the `detected` arm's mc. On the i7, `detected` gets mc=213 from
kc=384 while `mconly` gets mc=320 from the default kc=256. That is the honest
question — "what does this L2 imply for the blocking we ship?" — but it means the
four arms are not a clean 2x2 of the same numbers.

### What the older CSVs cannot tell you

The `mc` column in any CSV written before `mc_used` existed is the **configured**
bound, not the step any loop used. Three stages sit between them — the L2 bound,
the budget cap in `plan_gemm_grid`, and the even-partition round-off in
`balanced_mc` — so the i7 rows say `mc=213` for runs that stepped 210 serially
and 128 on eight threads, and even a *default* arm configured at 32 steps 30 and
29. Every harness now records `mc_used`, `nib`, `njb`, `ic_nt`, `jc_nt` per point,
and `analyze_blocking_ab.py` prints them beside each verdict; older CSVs are
labelled as not carrying them.

## The run contract

Every harness in `benchmarks/` **must** satisfy this. It exists because each rule
below was written **after** the corresponding mistake had already produced a
number someone believed: a Zen 4 run that deleted the i7's committed CSVs, an
analyzer that called a 4.8% "win" between two byte-identical arms, a Jetson that
asked for eight threads on a six-core part.

Where each harness stands, so no CSV is read as more gated than it was:

| Harness | Preflight | Pins | Per-machine `OUTDIR` | Sidecar state | Interleaves |
|---|---|---|---|---|---|
| `run_blocking_ab.sh` / `.ps1` | yes | yes | yes | yes | yes |
| `run_scaling.sh` / `.ps1` | yes | yes | yes | yes | **no** |
| `run_sweeps.sh` / `.ps1` | yes | yes | yes | yes | **no** |
| `run_scaling_297.sh` / `.ps1` | **no** | yes | no | no | no |

**Why `run_scaling` and `run_sweeps` do not interleave.** Not an oversight, a
dependency: `bench_all` truncates its `--csv` on every invocation, and
`analyze_scaling.py` keeps the **last** row for a `(backend, op, size, T)` key
rather than the best — so extra rounds would silently overwrite instead of
accumulating, which is worse than not interleaving at all. It needs append
support in `bench_all` and best-of-rounds in the analyzers first (#442).

What they do instead is **build every variant before measuring any of them**.
Previously each backend was built and then immediately measured, so every arm was
timed on a machine still hot from compiling its own binary while the next arm
compiled during the previous one's cooldown — a per-arm bias in a comparison
whose entire purpose is comparing arms. That is the largest order effect
available to remove without the round support above.

`run_scaling_297.sh` is the remaining gap.

| Rule | Why |
|---|---|
| **Preflight, and stop on failure** — `benchmarks/preflight.sh` / `.ps1` | A dirty tree, a competing build or a hot machine invalidates the session. Finding out afterwards means the CSVs are already written |
| **Pin** — one logical id per physical core | On a hybrid CPU pinning also decides *which* cache hierarchy is detected, so an unpinned run is ambiguous, not merely noisy |
| **Interleave, alternating order per round** | Warm-up and thermal drift otherwise accrue to whichever arm runs first, in a fixed direction |
| **Min of N**, not mean | The minimum is the least contaminated sample; a mean folds in every interruption |
| **`OUTDIR` per machine, required with no default** | CSVs are named by arm, not by machine, and the runners clear them before writing |
| **Sidecar** — every CSV carries a `.sysinfo` | A number without its machine and its build is not evidence |

### What every sidecar records

Provenance comes from three places, deliberately, because each can be wrong
independently of the others:

| Source | Keys | Answers |
|---|---|---|
| The **binary** (`mtl/build_info.hpp`, `mtl/util/system_info.hpp`) | `git_commit` `git_dirty` `cxx_flags` `cmake_type` `build_isa` `cpu_*` `os_*` `compiler*` | What was built, and for which ISA |
| **Preflight** | `tree_git_commit` `tree_git_dirty` `competing_load` `loadavg_1m` `cpu_online` `cpu_affinity` `governor` `turbo` `power_mode` `thermal_before_c` `thermal_limit_c` `thermal_headroom_c` | What the machine was doing |
| The **harness** | `binary_stale` `thermal_after_c` `harness` `harness_rounds` `harness_reps` | Whether the run was self-consistent |

Two pairs are worth understanding, since each pair looks redundant and is not:

- **`cxx_flags` vs `build_isa`.** Flags are the *intent*; `build_isa` is the
  *effect*, taken from the compiler's own predefined macros. `MTL5_NATIVE_ARCH`
  adds `-march=native` through `target_compile_options`, so it appears in no
  `CMAKE_CXX_FLAGS` variable at all — a build can read `cxx_flags=-O3 -DNDEBUG`
  and still be an AVX build. This is exactly the question the Zen 4 run could not
  answer: `/arch:AVX512` intended, AVX2 measured, nothing in the CSV to say so.
- **`git_commit` vs `tree_git_commit`.** The first is where the *binary* came
  from, the second where the *tree* is now. They diverge when someone edits and
  forgets to rebuild, and the harness records that as `binary_stale=1` and warns.

### Gates, and where they are enforced

Preflight **fails** — it does not warn — on conditions that make a result
meaningless. A warning scrolls past and the CSV gets committed anyway.

| Condition | Action | Override |
|---|---|---|
| Working tree dirty | fail | `--allow-dirty` / `-AllowDirty` (or `ALLOW_DIRTY=1`), and it is recorded as `tree_git_dirty=1` |
| A build or benchmark already running | fail | none — fix the machine |
| Threads requested exceed available cpus | fail | none |
| Thermal headroom below the margin | fail **only where a sensor and a limit are both readable** | `MIN_THERMAL_HEADROOM_C` (default 15) |
| Governor not `performance` | warn, and record the value | — |

**What "dirty" excludes.** The harness's own `OUTDIR` — the CSVs from a previous
run, and the cmake logs the PowerShell drivers write beside them (now gitignored
as well). None of that says anything about whether the source that built the
binary is reproducible, and counting it stamped `tree_git_dirty=1` on sidecars
whose code was pristine. It also made the gate unusable: the Windows driver
builds before it measures, so the run tripped over its own logs, and the escape
hatch the message suggested (`set ALLOW_DIRTY=1`) is cmd.exe syntax that does
nothing in PowerShell. A gate a normal run trips by existing is not a gate. Use
`-AllowDirty` there, or `$env:ALLOW_DIRTY = "1"`.

The thermal rule is asymmetric on purpose. Windows desktop firmware usually does
not implement `MSAcpi_ThermalZoneTemperature`, and failing there would make the
Zen 4 machine unbenchmarkable; passing silently would hide the gap. Unavailable
is written as `thermal_before_c=unavailable` — recorded, never assumed cool.

Governor is recorded rather than enforced because **every** machine on this page
currently runs a non-performance governor (`powersave` on the i7, `ondemand` on
the Xeon, `schedutil` on the Jetson). Failing on it would block all of them; what
makes the data defensible is that a `schedutil` run can never be silently
compared against a pinned one.

### Sidecars written before this contract

The CSVs committed before `preflight` existed carry the `.sysinfo` of their day:
CPU, OS, compiler and caches, and **no** provenance or machine state. They are
not retrofittable — the information was never captured — so read any sidecar
without a `preflight_version` key as pre-contract, and treat its build and
machine state as unknown rather than as default.

## Methodology

One executable per backend. MTL5 dispatches to BLAS/LAPACK **at compile time**,
so the build *is* the backend — there is no runtime policy switch, and each
binary measures only its own backend. This mirrors what a dependent application
actually gets. The reasoning is spelled out in the
[benchmark harness](../../benchmarks/README.md) page.

Each binary is verified to link what its label claims before any measurement —
`native` resolves no external BLAS, `openblas` resolves `libopenblas.so.0`,
`blis` resolves `libblis.so.4`, `mkl` resolves the `libmkl_*` trio — and each
binary self-reports its compiled configuration in its output header.

BLIS is a **BLAS-only** library. An MTL5 build against it has no LAPACK path, so
a "blis" factorization curve would silently be the generic native path wearing a
vendor label. BLIS therefore appears in BLAS results and is absent from LAPACK
results by construction, not by oversight.

## Running the cache-blocking A/B on a new machine

Cache detection is opt-in (`MTL5_ENABLE_CACHE_DETECTION`) because it measured
**slower** than the compile-time defaults on the i7-12700K. That verdict is one
machine's; this is how to establish it for another. See #430 for the wider
experiment this feeds.

### What the A/B actually varies

The `detected` arm applies **L1 → `kc`** and **L2 → `mc`**, and nothing else.
Detected **L3 is deliberately not applied**: `l3_bytes` feeds only `nc`, `nc` sets
the jc block count `njb = ceil(n/nc)`, and the nest hands jc blocks to teams
round-robin — so an `njb` that is not a multiple of `jc_nt` unbalances them.
Applying a detected L3 measured a 10–25% regression on a wide/short shape for
exactly that reason, and `runtime_blocking` pins `nc` to the compile-time value
so it cannot drift indirectly through `kc` either.

So this experiment answers *"do detected `kc`/`mc` beat the hand-tuned
constants?"* — **not** *"does cache detection help?"* in general. A machine whose
L3 differs wildly from the 8 MiB default (most do) still has that half of the
question untested here. Any conclusion about L3 or `nc` waits on the
`balanced_nc` work in #429; until then, treat detected-L3 results as
experimental and out of scope for this runbook.

**Linux (GCC/Clang):**

```bash
cmake --preset release -DMTL5_WITH_HIGHWAY=ON -DMTL5_NATIVE_ARCH=ON
cmake --build build-release --target bench_blocking_ab_detected bench_blocking_ab_default -j4

# BENCH_PCPUS is one logical id per PHYSICAL core -- REPLACE with this machine's
# map, from `lscpu -e=CPU,CORE`. The values below are the i7-12700K's P-cores;
# for the Xeon E5-2420 v2 it is 0,1,2,3,4,5. THREADS must not exceed that count
# (the runner rejects it rather than over-subscribing).
# OUTDIR is REQUIRED and must be this machine's own directory -- the CSVs are
# named by arm, not by machine, and the runner clears them before writing.
BENCH_PCPUS=0,2,4,6,8,10,12,14 THREADS="1 8" ROUNDS=5 \
    OUTDIR=benchmarks/data/i7-12700k ./benchmarks/run_blocking_ab.sh

./benchmarks/analyze_blocking_ab.py \
    benchmarks/data/i7-12700k/blocking_ab_{detected,default}.csv
```

**Windows (MSVC)** — there is no `taskset`, so pinning is an affinity bitmask and
the driver is PowerShell, following `run_scaling.ps1`:

```powershell
pwsh benchmarks/run_blocking_ab.ps1 `
     -PCores "0,2,4,6,8,10,12,14" -Threads "1,8" -Rounds 5 `
     -Arch "/arch:AVX512" -OutDir "benchmarks\data\ryzen-9-8945hs"

python benchmarks/analyze_blocking_ab.py `
     benchmarks\data\ryzen-9-8945hs\blocking_ab_detected.csv `
     benchmarks\data\ryzen-9-8945hs\blocking_ab_default.csv
```

One caveat that is specific to Windows and does not apply on Linux: cache
detection there falls back to CPUID, which describes whichever core the thread is
running on, and .NET can only set `ProcessorAffinity` *after* the process starts —
so detection may run before the pin takes effect. That is harmless on a
**homogeneous** part, where every core reports the same hierarchy (which covers
`ryzen-9-8945hs`), and it is why the Linux path reads sysfs under the affinity
mask instead. Do not use the PowerShell driver on a hybrid Windows machine
without closing that gap first — the detected arm's figures would not be
reproducible.

Four things decide whether the result means anything:

- **An explicit ISA flag is not optional.** Without one Highway compiles for the
  baseline 128-bit target rather than the machine's real ISA, and since `kc`, `mc`
  and `nc` all divide by the SIMD width, you would be measuring blocking for a
  vector length the production build never uses. On GCC/Clang that is
  `-DMTL5_NATIVE_ARCH=ON`. **On MSVC that option does nothing** — it is guarded
  `if(MTL5_NATIVE_ARCH AND NOT MSVC)`, and MSVC x64 defaults to SSE2 — so pass
  the ISA directly, e.g. `-DCMAKE_CXX_FLAGS="/arch:AVX512"`. This is easy to miss
  because the run completes and produces a clean table either way.
- **`BENCH_PCPUS` must list one logical id per *physical* core**, in the order to
  use — the example above is the i7's and must be replaced for any other machine.
  On a hybrid CPU it does more than control noise: it fixes *which cache hierarchy
  gets detected at all* (#432), so an unpinned run there is ambiguous rather than
  merely noisy. The script rejects a `THREADS` larger than the list rather than
  quietly over-subscribing.
- **The shape list is derived from the machine**, once, from the detected arm, and
  handed to both. Do not substitute a fixed list: the regime where the jc loop
  parallelizes needs `nib <= T/2` and `njb >= 2`, and both bounds contain that
  machine's own `mc` and `nc`. A square-only list measures the one regime where
  the thread-partition effects cannot appear.
- **Commit the CSVs and their `.sysinfo` sidecars** to **this machine's own
  directory** under `benchmarks/data/` — `i7-12700k/`, `ryzen-9-8945hs/`, and so
  on. The output files are named by *arm*, not by machine, and the runner clears
  them before writing, so a shared directory means the next machine's run
  destroys the previous one's committed results. Both drivers now require an
  explicit output directory for that reason. The sidecar records the hierarchy
  each arm was blocked for, which is what makes the rows interpretable later —
  and on a hybrid part it is the evidence that the run was pinned to the core
  class you intended.

The analyzer refuses to compare incomplete or unevenly-sampled runs, declines any
difference under 2%, and reports a **null run** when both arms compiled to
identical blocking — which is the expected outcome on a machine whose L1/L2
already match the defaults (`xeon-e5-2420v2`), and a useful self-test of the
harness.

## Adding a system

1. Add a row to the table at the top of this page and a section describing the
   hardware, libraries, and pinning policy. Read every hardware figure **from the
   machine** rather than from a datasheet — the shipped configuration (power mode,
   clock cap, which BLAS the alternatives system actually resolves) is what
   determines the numbers.
2. Run the suites documented in the [harness README](../../benchmarks/README.md) on the new machine.
3. Add a result page `docs/benchmarks/<system>.md` following the existing one.
4. Register the page in `FILE_MAP` in `docs-site/sync-content.mjs`.

Steps 1 and 2–4 can be separated: a machine may be registered (step 1) while its
results are still pending, which is how `xeon-e5-2420v2` and `jetson-orin` stand
today. A machine that is *part of an experiment* belongs on this page before the
experiment runs, so that its pinning policy is agreed in advance rather than
reconstructed from whatever the first run happened to do.

The per-system pages are self-contained, so a new machine never requires editing
an existing result page.
