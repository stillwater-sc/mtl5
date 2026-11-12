# Benchmark Systems

Every performance number published for MTL5 is tied to a specific machine, a
specific set of vendor libraries, and a specific pinning policy. This page is
the index of those machines. Each one links to its full result page.

Performance claims do not travel between machines. A GEMM efficiency measured
on a desktop hybrid CPU says little about a server part with different cache
sizes, memory channels, and turbo behaviour — so read the results page together
with the system description here.

## Systems

| System | CPU | Cores used | Results | mtl5_bench card |
|--------|-----|-----------|---------|-----------------|
| `i7-12700K` | 12th Gen Intel Core i7-12700K | 8 P-cores (E-cores excluded) | [Results](i7-12700k.md) | [Card](cards/i7-12700k.md) |
| `ryzen-7-2700x` | AMD Ryzen 7 2700X (Zen+) | 8 cores (SMT) | — | [Card](cards/ryzen-7-2700x.md) (preliminary) |
| `ryzen-7-8845hs` | AMD Ryzen 7 8845HS (Zen 4) | 8 cores (SMT siblings excluded) | — | [Card](cards/ryzen-7-8845hs.md) |
| `ryzen-9-8945hs` | AMD Ryzen 9 8945HS (Zen 4) | 8 cores (SMT siblings excluded) | [Results](ryzen-9-8945hs.md) | [Card](cards/ryzen-9-8945hs.md) |
| `xeon-e5-2420v2` | Intel Xeon E5-2420 v2 (Ivy Bridge-EP) | 6 cores (SMT siblings excluded) | — | [Card](cards/xeon-e5-2420v2.md) |
| `jetson-orin` | NVIDIA Jetson Orin, ARM Cortex-A78AE | 6 cores | [Results](jetson-orin.md) | [Card](cards/jetson-orin.md) |
| `arduino-ventuno-q` | Qualcomm "Monaco", 4× Cortex-A78C + 4× Cortex-A55 | 4 A78C cores (A55 excluded) | [Results](arduino-ventuno-q.md) | [Card](cards/arduino-ventuno-q.md) |

**Results** pages are written analyses of one machine. The i7-12700K and
Ryzen 9 8945HS pages cover the suites in `benchmarks/README.md` (GEMM against
the vendors, factorizations, threading, sparse direct) and the platform
benchmark's #737 sessions. The Jetson Orin and Ventuno Q pages read the
platform benchmark and the interleave sweeps.
**mtl5_bench cards** are the platform benchmark's generated capability cards
(#721): every operation family against this machine's measured FMA peak and
bandwidth. A card page is written by `benchmarks/publish_platform.cmake` from a
session committed under `benchmarks/data/<machine>/mtl5_bench/`, and shows the
latest published session; earlier sessions stay in the data directory. A card
marked *preliminary* was not measured under #737's methodology (pinned cores,
the machine's clock policy, two sessions).

The sessions' `bench_interleave_rows` sweeps are compared across the machines
on [Interleave width across the benchmark machines](interleave-width.md) (#780).

A machine is registered here as soon as it is part of an experiment, which can
precede its first result page: the Jetson and Ventuno rows first existed
because #430 needed the cache hierarchies and pinning policies of all four
machines settled *before* it started measuring. A row without a results link
has a card but no written analysis yet.

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

### Cache-blocking A/B result

Detection loses here: see the i7-12700K section of the [cache-blocking A/B](cache-blocking-ab.md#i7-12700k-detection-loses-here) page.

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

### Threading measurements

The vendor BLAS inside an MTL5 region (#640) and the pool dispatch and join latency (#636) are on the [i7-12700K results](i7-12700k.md#threading-measurements-636-640) page.

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

### Register-tile sweep

No tile clears the noise, so AVX-512 keeps 6×16: see the [register-tile sweep](register-tile-sweep.md#ryzen-9-8945hs-no-tile-clears-the-noise-so-avx-512-keeps-616) page.

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
>
> The #737 platform sessions use a different policy: 25 W (mode 1) and
> `jetson_clocks`, with all six cores at 1344 MHz. See the
> [Jetson Orin analysis](jetson-orin.md).

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

### Cache-blocking A/B result

Neutral, and the grid explains the one loss: see the Jetson Orin section of the [cache-blocking A/B](cache-blocking-ab.md#jetson-orin-neutral-and-the-grid-explains-the-one-loss) page.

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

### Cache-blocking A/B result and register-tile sweep

- Cache blocking: a null run, because there was nothing to detect ([cache-blocking A/B](cache-blocking-ab.md#arduino-ventuno-q-a-null-run-because-there-was-nothing-to-detect)).
- Register tile: 8×4 wins, and 12×4 loses at four threads ([register-tile sweep](register-tile-sweep.md#arduino-ventuno-q-84-wins-and-124-loses-at-four-threads)).

### Pinning and clock policy

| Requirement | Why |
|---|---|
| Pin to the A78C cores only (CPUs 0–3), read from `/proc/cpuinfo` | the A55 cores are slower and have a different cache hierarchy; a team that includes one runs at its pace |
| Record the governor, and set `performance` on the A78C policy for a comparable session | there is no `nvpmodel`; the governor is what decides the clocks |
| Note the 2112 MHz cap | `performance` runs at `scaling_max_freq`, not the 2361.6 MHz hardware maximum |
| Keep `schedutil` and `performance` runs in separate directories | the profile does this; they must never be averaged together |

## Cross-machine analyses and the run contract

These used to be on this page:

- [Cache-blocking A/B across the machines](cache-blocking-ab.md) (#430): the four-machine result, and how to run it on a new machine.
- [Register-tile sweep](register-tile-sweep.md) (#643).
- [Run contract and methodology](run-contract.md): what every harness must record and enforce.

## Adding a system

1. Add a row to the table at the top of this page and a section describing the
   hardware, libraries, and pinning policy. Read every hardware figure **from the
   machine** rather than from a datasheet — the shipped configuration (power mode,
   clock cap, which BLAS the alternatives system actually resolves) is what
   determines the numbers.
2. Run a platform session on the new machine, as the
   [harness README](../../benchmarks/README.md) describes, and publish it with
   `benchmarks/publish_platform.cmake`. That writes the machine's capability
   card, `docs/benchmarks/cards/<system>.md`, and links it from the table above.
3. Optionally, write an analysis page, `docs/benchmarks/<system>.md`. Register it
   in `FILE_MAP` in `docs-site/sync-content.mjs`, and add it to the
   "Per-machine analyses" group of the Benchmarking sidebar in
   `docs-site/astro.config.mjs`.

Steps 1 and 2–3 can be separated: a machine may be registered (step 1) while its
results are still pending. A machine that is *part of an experiment* belongs on
this page before the experiment runs, so that its pinning policy is agreed in
advance rather than reconstructed from whatever the first run happened to do.

The per-system pages are self-contained, so a new machine never requires editing
an existing result page.
