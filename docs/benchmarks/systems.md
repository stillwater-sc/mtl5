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
