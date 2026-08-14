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
| `jetson-orin` | NVIDIA Jetson, ARM Cortex-A78AE | 8 cores | pending — #430 |

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

### Pinning policy

This is a **hybrid P/E-core CPU**, which makes pinning mandatory rather than
optional. An unpinned single-threaded run lets short L1 kernels land on a 3.8 GHz
E-core and reports a number that has nothing to do with the 5.0 GHz P-core the
same code would reach in production. The failure mode is documented in
[the multi-core scaling investigation](../design/multicore-scaling-investigation.md).

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

## `jetson-orin` — ARM Cortex-A78AE (pending characterization)

> **This entry is incomplete on purpose.** The fields below have not been read
> from the machine, and guessing them is worse than leaving them blank — the whole
> point of this page is that performance claims do not travel between machines.
> Fill them by running the commands below **on the Jetson** and replacing each
> `TODO`; do not populate them from a datasheet, since the shipped configuration
> (power mode, clock cap, memory) is what determines the numbers.

| | |
|---|---|
| CPU | TODO — exact module (AGX Orin / Orin NX) and core count |
| Topology | TODO — cores, clusters, SMT (expected: 8 cores, no SMT) |
| Clocks | TODO — depends on the selected `nvpmodel` mode |
| Cache | TODO — L1d, L2 per core, L3/SLC, line size |
| ISA | TODO — NEON (expected); confirm whether SVE is exposed (relevant to #427) |
| Memory | TODO — and note it is **shared with the GPU** |
| OS | TODO — JetPack / L4T version and kernel |
| Compilers | TODO |
| Power mode | TODO — `nvpmodel -q` output |

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
