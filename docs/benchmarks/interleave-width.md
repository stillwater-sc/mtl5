# Interleave width across the benchmark machines

The generic row kernels (`gemv`, `trmv` and the forward substitution on a
row-major matrix) sum W rows per pass, one accumulator per row (#749, #772).
The first sweeps were built with the earlier default, W = 6 for float and
double, measured only on an i7-12700K (#775); the #885 sweeps with today's 8
on x86-64. The default only marks a row (`*`), so the times compare. This page measures W on every machine of
the #737 platform campaign (#780). The per-ISA defaults below come from it, and
so does the table [run-time selection](#run-time-selection-778) uses (#778).

## How it was measured

- **Tool:** `bench_interleave_rows --reps 21`, n = 4096. It's built in its own
  build directory, the same generic build as the platform session's `native`
  variant: no BLAS, no `MTL5_NATIVE_FAST_GEMM`, no `-march`. That means SSE2
  on x86-64, NEON on AArch64, and `/O2` under MSVC.
- **Where:** it runs on the session's first CPU, under the session's clock
  policy, after its capability card's runs (`INTERLEAVE=1` in the machine
  profiles), and only on a machine that passed the session's preflight.
- **Repeats:** three runs per compiler in each session, and two sessions per
  machine, the second after a cold boot. Those are the same sessions as the
  machines' capability cards ([benchmark systems](systems.md)). Each cell
  below is the median of all the machine's runs: 6, or 9 and 12 where #885
  added sessions at one commit (the Xeon and the Ryzen 7; the Ryzen 9). Each
  run's value is already a median of 21.
- **Compilers:** GCC and Clang on Linux, MSVC and clang-cl on Windows. The
  Jetson has no Clang.
- **Why the median over runs:** each run's fastest width is listed too, but
  the median is what counts, because the first run of a sweep often favours
  a narrower width than the runs after it.

## Findings

- **x86-64: W = 8 is the better default.** Over every x86-64 machine,
  compiler and kernel, W = 8 averages 1.4% slower than each kernel's fastest
  width, 10% at worst (Clang on the i7-12700K, f32 gemv). W = 6 averages 4.1%
  slower, and 33% at worst: clang-cl's f64 `trmv`, slower at W = 6 than at
  W = 4 on both Ryzens and in both sessions. On the two Zen 4 machines,
  under both compilers, W = 8 is the fastest width of every kernel or within
  1% of it. The Ivy Bridge
  Xeon is the exception: it prefers 4 for f64 and loses at most 7% at W = 8.
- **AArch64: W = 6 is not safe; W = 4 is the better default.** With Clang
  (on the Arduino Ventuno Q), W = 4 is the fastest width in every kernel, and
  W = 6 is 18-24% slower than it. With GCC, gemv agrees: f64 and f32 are
  fastest at 4 on both AArch64 machines. On the Jetson, GCC's f64 `trmv` and
  forward substitution are fastest at 2, and lose 12% at 4.
- **GCC on AArch64 had a separate problem, now fixed (#795).** In these
  sessions row-major f32 `trmv` and the forward substitution take 5.6-7.2x
  longer at W = 4 than at W = 2, and 1.4-4.5x longer at W = 6 to 12, on both
  machines (GCC 11 and 13). Clang on the same machine does not. The cause was
  the `keep_scalar` barrier from #776, which was on for GCC on AArch64 although
  it had been measured only on x86-64. An A/B on both boards (median of 3 runs,
  `--reps 21`):

  | GCC, f32 | W = 4, with the barrier | W = 4, without it | W = 2, before |
  |---|---:|---:|---:|
  | Jetson Orin, `trmv` | 50.04 ms | 6.31 ms | 7.29 ms |
  | Jetson Orin, forward substitution | 39.45 ms | 5.58 ms | 7.08 ms |
  | Ventuno Q, `trmv` | 31.66 ms | 3.12 ms | 4.75 ms |
  | Ventuno Q, forward substitution | 26.22 ms | 2.99 ms | 4.70 ms |
  | Jetson Orin, `gemv` | 8.23 ms | 8.34 ms | |
  | Ventuno Q, `gemv` | 5.51 ms | 9.14 ms | |

  So on AArch64 the triangular kernels no longer take the barrier, and W = 4
  is their fastest width, 13-36% faster than the 2 they were held at. `gemv`
  keeps it: it is neutral on the Jetson and 1.66x faster with it on the
  Ventuno. The tables below are from the 2026-11-11 sessions, with the
  barrier.

## Defaults per ISA class

These are the built defaults (`include/mtl/interface/interleave_rows.hpp`).
Every target caps the width at half its vector registers. Targets other than
x86-64 and AArch64 are assumed to have 8, so they get 4: the measured 6 that
was the default before #772 is capped there, as it is on 32-bit x86.

| ISA class | float | double | worst loss against the fastest width |
|---|---|---|---|
| x86-64 (SSE2, generic build) | 8 | 8 | 10% (i7-12700K, Clang, f32 gemv); 7% on the Xeon |
| AArch64 (NEON, generic build) | 4 | 4 | 0% under Clang; 12% under GCC (Jetson, f64 `trmv` and forward substitution) |

- **x86-64 with AVX-512:** none of these machines has it. The Zen 4 machines
  support AVX-512, but the generic build doesn't use it, so that class is
  still unmeasured.
- **Run-time selection (#778):** it recovers the rest, for the CPUs it knows;
  see the next section.

## Run-time selection (#778)

With `MTL5_INTERLEAVE_RUNTIME`, float and double compile each of the three
kernels at W = 2, 4, 6, 8 and 12. On first use they take the width a table
lists for the CPU and the compiler, and the compile-time default where it
lists nothing (`include/mtl/interface/interleave_runtime.hpp`). It is off by
default: a file that calls all three kernels for both types took 0.5-0.9 s
longer to compile on the i7 (1.2 s without), and their code grew from
24-29 KB to 62-78 KB.

The table is each kernel's fastest width in the sweeps above, for the cores
those machines have:

| core | identified by | GCC | Clang | MSVC, clang-cl |
|---|---|---|---|---|
| Alder Lake (i7-12700K) | Intel family 6, model 0x97, 0x9A | f64 12, 12, 12; f32 6, 8, 8 | f64 8; f32 6 | |
| Ivy Bridge (Xeon E5-2420 v2) | Intel family 6, model 0x3A, 0x3E | f64 4, 6, 4; f32 8, 4, 4 | f64 4; f32 6 | |
| Zen 4 (both Ryzens) | AMD family 0x19, model 0x60-0x7F | | | 8 |
| Cortex-A78AE (Jetson Orin) | Arm part 0xd42 | f64 4, 2, 2; f32 4 | | |
| Cortex-A78C (Ventuno Q) | Arm part 0xd4b | 4 | 4 | |

The three numbers are `gemv`, `trmv` and the forward substitution. On
AArch64 the part comes from `/proc/cpuinfo` (Linux only). On the Ventuno that
is the A78C, not the A55, which is where the sweeps ran. The Jetson's f32
`trmv` and forward substitution are 4 since #795. The sweeps were a generic
build, so an x86-64 build with AVX uses no entry.

How close it comes, measured with the table in place
(`bench_interleave_rows`, same method as above, one session, median of the
runs): the worst kernel, as time at the chosen width over time at the
fastest, against today's fixed default.

| machine, compiler | runs | run-time choice, worst | fixed default, worst |
|---|---:|---:|---:|
| i7-12700K, GCC | 6 | 1.007 (f64 gemv) | 1.044 (f64 trmv) |
| i7-12700K, Clang | 3 | 1.007 (f64 trmv) | 1.094 (f32 gemv) |
| Jetson Orin, GCC | 3 | 1.000 | 1.128 (f64 trmv) |
| Ventuno Q, GCC | 3 | 1.000 | 1.000 |
| Ventuno Q, Clang | 3 | 1.000 | 1.000 |
| Xeon E5-2420 v2, GCC | 3 | 1.019 (f64 trmv) | 1.051 (f64 gemv) |
| Xeon E5-2420 v2, Clang | 3 | 1.000 | 1.081 (f32 trisolve) |
| Ryzen 7 8845HS, MSVC | 3 | 1.000 | 1.000 |
| Ryzen 7 8845HS, clang-cl | 3 | 1.000 | 1.000 |
| Ryzen 9 8945HS, MSVC | 6 | 1.055 (f32 trmv) | 1.055 (f32 trmv) |
| Ryzen 9 8945HS, clang-cl | 6 | 1.136 (f32 trmv) | 1.136 (f32 trmv) |

The Xeon and Ryzen rows are from #885, at `b2d022726b3d`: one session on the
Xeon and the Ryzen 7, two on the Ryzen 9, each started after a cold boot. On
the Zen 4 machines the table's width is the fixed default, 8, so the two
columns agree.

**The Ryzen 9 misses 3% for f32, and no table entry can fix it.** In some
runs its f32 kernels at W = 8 slow down by 12-16% (`gemv` 2.46 ms in the
2026-11-12 sessions, 3.0 ms in these), while W = 6 stays where it was
(2.68 ms) and becomes the fastest. Other runs of the same binary in the same
session are at 1.000: clang-cl's f32 worst was 1.122, 1.159, 1.154, 1.138,
1.000 and 1.142 in its six runs, MSVC's 1.000 three times and then 1.110,
1.125 and 1.140. So it is a state of the machine from run to run, not a width
that suits it better. The Ryzen 7 8845HS shares the Zen 4 entry and was at
1.000 in every run under both compilers, where W = 6 would be about 10%
slower. The two parts can't be told apart by CPUID family and model, so the
entry stays 8. The sweep tables below take the median over all sessions, and
there W = 8 remains the Ryzen 9's fastest f32 width.

f64 `gemv` on the i7 under GCC is bandwidth-bound at n = 4096. Widths 6 to 16
are within about 5% of each other there, and run to run the fastest moves
among 8, 12 and 16.

**The dispatch** is a switch on a cached width per call.
`bench_interleave_dispatch` times the generic `gemv` built without and with
run-time selection, at the same width, out of line as a caller's call is.
At n = 64 the cost is under 1% on the i7 (GCC and Clang), the Jetson and the
Ventuno under Clang, and 1-3.4% on the Ventuno under GCC (f64 the most). At
n = 256 it is at most 1.6% (Clang on the i7). At n = 16 it is 1-2 ns on the
i7, and inside the run-to-run spread on the boards.

**The table's widths are for n = 4096.** At small n they need not be the
fastest: on the i7 under GCC, the chosen 12 (f64) and 6 (f32) are 3-4% slower
than 8 at n = 64 and 2-3% at n = 256. Under Clang the chosen f32 6 is 8%
faster than 8 at n = 64 and 256, and level with it at n = 16.

<!-- BEGIN generated by benchmarks/interleave_table.py table -->

### Summary

Per machine and compiler: the fastest width of each kernel, and the worst cost of a fixed width against the fastest (time at that width / time at the fastest; 1.000 is no loss).

| machine | ISA | compiler | fastest (f64 gemv, trmv, trisolve / f32 ...) | W=4 | W=6 | W=8 | W=6 / W=4 |
|---|---|---|---|---|---|---|---|
| arduino-ventuno-q | AArch64 | clang++ | 4, 4, 4 / 4, 4, 4 | 1.000 | 1.237 | 1.472 | 1.237 |
| arduino-ventuno-q | AArch64 | g++ | 4, 4, 4 / 4, 2, 2 | 6.654 | 4.408 | 3.164 | 1.252 |
| i7-12700k | x86-64 | clang++ | 8, 8, 8 / 6, 6, 6 | 1.124 | 1.052 | 1.100 | 0.957 |
| i7-12700k | x86-64 | g++ | 12, 12, 12 / 6, 8, 8 | 1.088 | 1.041 | 1.029 | 0.984 |
| jetson-orin | AArch64 | g++ | 4, 2, 2 / 4, 2, 2 | 7.186 | 3.976 | 4.532 | 1.582 |
| ryzen-7-8845hs | x86-64 | clang-cl | 8, 8, 8 / 8, 8, 8 | 1.193 | 1.331 | 1.000 | 1.143 |
| ryzen-7-8845hs | x86-64 | msvc | 8, 8, 8 / 8, 8, 8 | 1.320 | 1.081 | 1.000 | 0.981 |
| ryzen-9-8945hs | x86-64 | clang-cl | 8, 8, 6 / 8, 8, 8 | 1.208 | 1.297 | 1.009 | 1.146 |
| ryzen-9-8945hs | x86-64 | msvc | 8, 8, 8 / 8, 8, 8 | 1.327 | 1.071 | 1.000 | 0.973 |
| xeon-e5-2420v2 | x86-64 | clang++ | 4, 4, 4 / 6, 6, 6 | 1.093 | 1.037 | 1.072 | 1.037 |
| xeon-e5-2420v2 | x86-64 | g++ | 4, 4, 4 / 8, 4, 4 | 1.025 | 1.021 | 1.046 | 1.021 |

### A fixed default per ISA class

Time at a fixed width / time at the fastest width, over every machine, compiler, kernel and type of the class: geometric mean / worst case.

| ISA | W=2 | W=4 | W=6 | W=8 | W=12 |
|---|---|---|---|---|---|
| x86-64 | 1.625 / 2.683 | 1.100 / 1.327 | 1.041 / 1.331 | 1.014 / 1.100 | 1.518 / 3.094 |
| AArch64 | 1.242 / 1.708 | 1.523 / 7.186 | 1.500 / 4.408 | 1.640 / 4.532 | 2.053 / 3.799 |

### Per machine

### arduino-ventuno-q -- clang++

0xd4b; Ubuntu clang version 18.1.3 (1ubuntu1); vector registers 32; built default W = 6 (f64), 6 (f32); median of 6 runs (2 sessions), each the median of 21. Data: `benchmarks/data/arduino-ventuno-q/interleave_rows/`.

| kernel | W=1 | W=2 | W=4 | W=6 | W=8 | W=12 | W=16 | fastest | W=6 / fastest | W=6 / W=4 | per-run fastest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemv f64 | 17.749 | 10.629 | **8.254** | 9.840 | 10.425 | 12.238 | 14.228 | 4 | 1.192 | 1.192 | 4: 6 |
| trmv f64 | 10.225 | 5.800 | **4.523** | 5.381 | 5.768 | 6.620 | 10.535 | 4 | 1.190 | 1.190 | 4: 6 |
| trisolve f64 | 10.282 | 5.877 | **4.483** | 5.300 | 5.620 | 8.893 | 10.459 | 4 | 1.182 | 1.182 | 4: 6 |
| gemv f32 | 16.730 | 9.563 | **6.389** | 7.544 | 9.406 | 11.408 | 14.054 | 4 | 1.181 | 1.181 | 4: 6 |
| trmv f32 | 8.852 | 5.042 | **3.290** | 4.059 | 4.429 | 5.426 | 9.395 | 4 | 1.234 | 1.234 | 4: 6 |
| trisolve f32 | 8.729 | 4.962 | **3.179** | 3.934 | 4.313 | 8.601 | 9.318 | 4 | 1.237 | 1.237 | 4: 6 |

### arduino-ventuno-q -- g++

0xd4b; g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0; vector registers 32; built default W = 6 (f64), 6 (f32); median of 6 runs (2 sessions), each the median of 21. Data: `benchmarks/data/arduino-ventuno-q/interleave_rows/`.

| kernel | W=1 | W=2 | W=4 | W=6 | W=8 | W=12 | W=16 | fastest | W=6 / fastest | W=6 / W=4 | per-run fastest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemv f64 | 18.844 | 10.431 | **8.104** | 9.587 | 9.992 | 12.108 | 14.022 | 4 | 1.183 | 1.183 | 4: 6 |
| trmv f64 | 10.479 | 5.710 | **4.449** | 5.124 | 5.359 | 6.266 | 6.877 | 4 | 1.152 | 1.152 | 4: 6 |
| trisolve f64 | 10.442 | 5.777 | **4.393** | 5.118 | 5.292 | 6.106 | 6.708 | 4 | 1.165 | 1.165 | 4: 6 |
| gemv f32 | 16.659 | 9.005 | **5.556** | 6.958 | 7.777 | 9.213 | 11.011 | 4 | 1.252 | 1.252 | 4: 6 |
| trmv f32 | 8.976 | **4.761** | 31.675 | 6.787 | 7.572 | 7.675 | 8.175 | 2 | 1.426 | 0.214 | 2: 6 |
| trisolve f32 | 8.755 | **4.712** | 26.197 | 20.768 | 14.909 | 15.145 | 17.940 | 2 | 4.408 | 0.793 | 2: 6 |

### i7-12700k -- clang++

12th Gen Intel(R) Core(TM) i7-12700K; Ubuntu clang version 21.1.8 (6ubuntu1); vector registers 16; built default W = 6 (f64), 6 (f32); median of 6 runs (2 sessions), each the median of 21. Data: `benchmarks/data/i7-12700k/interleave_rows/`.

| kernel | W=1 | W=2 | W=4 | W=6 | W=8 | W=12 | W=16 | fastest | W=6 / fastest | W=6 / W=4 | per-run fastest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemv f64 | 8.338 | 5.115 | 4.002 | 3.817 | **3.705** | 3.751 | 3.845 | 8 | 1.030 | 0.954 | 8: 6 |
| trmv f64 | 4.346 | 2.725 | 2.125 | 1.993 | **1.921** | 1.951 | 2.694 | 8 | 1.037 | 0.938 | 8: 4, 12: 2 |
| trisolve f64 | 4.304 | 2.710 | 2.085 | 1.996 | **1.897** | 2.242 | 2.749 | 8 | 1.052 | 0.957 | 8: 6 |
| gemv f32 | 6.990 | 3.940 | 2.910 | **2.625** | 2.887 | 2.876 | 2.970 | 6 | 1.000 | 0.902 | 6: 6 |
| trmv f32 | 3.759 | 2.110 | 1.589 | **1.416** | 1.518 | 1.508 | 2.292 | 6 | 1.000 | 0.891 | 6: 6 |
| trisolve f32 | 3.728 | 2.059 | 1.542 | **1.372** | 1.474 | 1.906 | 2.359 | 6 | 1.000 | 0.889 | 6: 6 |

### i7-12700k -- g++

12th Gen Intel(R) Core(TM) i7-12700K; g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0; vector registers 16; built default W = 6 (f64), 6 (f32); median of 6 runs (2 sessions), each the median of 21. Data: `benchmarks/data/i7-12700k/interleave_rows/`.

| kernel | W=1 | W=2 | W=4 | W=6 | W=8 | W=12 | W=16 | fastest | W=6 / fastest | W=6 / W=4 | per-run fastest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemv f64 | 8.373 | 5.096 | 3.817 | 3.708 | 3.633 | **3.562** | 3.647 | 12 | 1.041 | 0.971 | 12: 5, 16: 1 |
| trmv f64 | 4.331 | 2.665 | 2.107 | 2.010 | 2.010 | **1.953** | 2.054 | 12 | 1.029 | 0.954 | 12: 6 |
| trisolve f64 | 4.309 | 2.645 | 2.079 | 1.976 | 1.933 | **1.910** | 2.045 | 12 | 1.034 | 0.950 | 12: 6 |
| gemv f32 | 6.968 | 7.333 | 2.779 | **2.734** | 2.745 | 2.769 | 3.510 | 6 | 1.000 | 0.984 | 6: 3, 8: 1, 12: 2 |
| trmv f32 | 3.734 | 2.052 | 1.544 | 1.472 | **1.441** | 1.458 | 1.840 | 8 | 1.022 | 0.953 | 8: 6 |
| trisolve f32 | 3.694 | 2.016 | 1.481 | 1.432 | **1.413** | 1.421 | 1.768 | 8 | 1.013 | 0.967 | 8: 5, 12: 1 |

### jetson-orin -- g++

ARMv8 Processor rev 1 (v8l); g++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0; vector registers 32; built default W = 6 (f64), 6 (f32); median of 6 runs (2 sessions), each the median of 21. Data: `benchmarks/data/jetson-orin/interleave_rows/`.

| kernel | W=1 | W=2 | W=4 | W=6 | W=8 | W=12 | W=16 | fastest | W=6 / fastest | W=6 / W=4 | per-run fastest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemv f64 | 25.134 | 13.599 | **12.888** | 19.942 | 24.796 | 41.006 | 40.270 | 4 | 1.547 | 1.547 | 4: 6 |
| trmv f64 | 13.853 | **7.976** | 8.976 | 12.552 | 14.895 | 21.994 | 21.157 | 2 | 1.574 | 1.398 | 2: 5, 4: 1 |
| trisolve f64 | 13.549 | **7.800** | 8.710 | 12.431 | 15.082 | 22.457 | 21.260 | 2 | 1.594 | 1.427 | 2: 5, 4: 1 |
| gemv f32 | 24.912 | 13.213 | **7.735** | 12.236 | 14.143 | 18.181 | 23.286 | 4 | 1.582 | 1.582 | 4: 6 |
| trmv f32 | 13.188 | **6.926** | 49.775 | 11.046 | 12.965 | 14.921 | 17.179 | 2 | 1.595 | 0.222 | 2: 6 |
| trisolve f32 | 13.027 | **6.800** | 39.427 | 27.035 | 30.816 | 25.833 | 28.998 | 2 | 3.976 | 0.686 | 2: 6 |

### ryzen-7-8845hs -- clang-cl

AMD Ryzen 7 8845HS w/ Radeon 780M Graphics; Clang 22.1.3 with MSVC-like command-line; vector registers 16; built default W = 8 (f64), 8 (f32); median of 9 runs (3 sessions), each the median of 21. Data: `benchmarks/data/ryzen-7-8845hs/interleave_rows/`.

| kernel | W=1 | W=2 | W=4 | W=6 | W=8 | W=12 | W=16 | fastest | W=6 / fastest | W=6 / W=4 | per-run fastest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemv f64 | 10.671 | 5.910 | 3.584 | 3.255 | **3.145** | 6.963 | 7.773 | 8 | 1.035 | 0.908 | 6: 2, 8: 7 |
| trmv f64 | 5.748 | 3.259 | 1.999 | 2.285 | **1.717** | 3.745 | 4.007 | 8 | 1.331 | 1.143 | 8: 9 |
| trisolve f64 | 5.754 | 3.263 | 1.949 | 1.768 | **1.740** | 3.694 | 3.975 | 8 | 1.016 | 0.907 | 6: 2, 8: 7 |
| gemv f32 | 10.102 | 5.455 | 3.006 | 2.755 | **2.532** | 7.833 | 9.240 | 8 | 1.088 | 0.917 | 6: 2, 8: 7 |
| trmv f32 | 5.430 | 2.901 | 1.617 | 1.492 | **1.443** | 3.969 | 4.834 | 8 | 1.034 | 0.923 | 6: 2, 8: 7 |
| trisolve f32 | 5.415 | 2.925 | 1.707 | 1.490 | **1.431** | 3.922 | 4.716 | 8 | 1.041 | 0.873 | 6: 2, 8: 7 |

### ryzen-7-8845hs -- msvc

AMD Ryzen 7 8845HS w/ Radeon 780M Graphics; MSVC 19.52.37029.0; vector registers 16; built default W = 8 (f64), 8 (f32); median of 9 runs (3 sessions), each the median of 21. Data: `benchmarks/data/ryzen-7-8845hs/interleave_rows/`.

| kernel | W=1 | W=2 | W=4 | W=6 | W=8 | W=12 | W=16 | fastest | W=6 / fastest | W=6 / W=4 | per-run fastest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemv f64 | 10.672 | 5.906 | 3.620 | 3.174 | **3.056** | 4.191 | 8.728 | 8 | 1.039 | 0.877 | 8: 9 |
| trmv f64 | 5.740 | 3.204 | 1.944 | 1.803 | **1.715** | 2.577 | 4.563 | 8 | 1.051 | 0.927 | 6: 1, 8: 8 |
| trisolve f64 | 5.815 | 3.226 | 1.948 | 1.790 | **1.701** | 2.554 | 4.501 | 8 | 1.052 | 0.919 | 8: 9 |
| gemv f32 | 10.087 | 5.393 | 3.056 | 2.503 | **2.315** | 4.357 | 9.037 | 8 | 1.081 | 0.819 | 6: 1, 8: 8 |
| trmv f32 | 5.436 | 2.843 | 1.566 | 1.537 | **1.498** | 2.492 | 4.737 | 8 | 1.026 | 0.981 | 6: 1, 8: 8 |
| trisolve f32 | 5.394 | 2.887 | 1.631 | 1.571 | **1.496** | 2.483 | 4.671 | 8 | 1.050 | 0.963 | 6: 1, 8: 8 |

### ryzen-9-8945hs -- clang-cl

AMD Ryzen 9 8945HS w/ Radeon 780M Graphics; Clang 22.1.3 with MSVC-like command-line; vector registers 16; built default W = 8 (f64), 8 (f32); median of 12 runs (4 sessions), each the median of 21. Data: `benchmarks/data/ryzen-9-8945hs/interleave_rows/`.

| kernel | W=1 | W=2 | W=4 | W=6 | W=8 | W=12 | W=16 | fastest | W=6 / fastest | W=6 / W=4 | per-run fastest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemv f64 | 10.586 | 5.922 | 3.559 | 3.228 | **3.162** | 6.971 | 7.717 | 8 | 1.021 | 0.907 | 6: 5, 8: 7 |
| trmv f64 | 5.655 | 3.208 | 1.978 | 2.266 | **1.746** | 3.766 | 3.994 | 8 | 1.297 | 1.146 | 8: 12 |
| trisolve f64 | 5.675 | 3.212 | 1.908 | **1.744** | 1.760 | 3.697 | 3.973 | 6 | 1.000 | 0.914 | 6: 5, 8: 7 |
| gemv f32 | 9.951 | 5.402 | 2.981 | 2.675 | **2.512** | 7.675 | 9.112 | 8 | 1.065 | 0.898 | 6: 5, 8: 7 |
| trmv f32 | 6.722 | 3.549 | 1.914 | 1.734 | **1.675** | 4.995 | 6.054 | 8 | 1.035 | 0.906 | 6: 5, 8: 7 |
| trisolve f32 | 6.686 | 3.556 | 2.006 | 1.776 | **1.660** | 4.953 | 5.927 | 8 | 1.070 | 0.885 | 6: 5, 8: 7 |

### ryzen-9-8945hs -- msvc

AMD Ryzen 9 8945HS w/ Radeon 780M Graphics; MSVC 19.52.37029.0; vector registers 16; built default W = 8 (f64), 8 (f32); median of 12 runs (4 sessions), each the median of 21. Data: `benchmarks/data/ryzen-9-8945hs/interleave_rows/`.

| kernel | W=1 | W=2 | W=4 | W=6 | W=8 | W=12 | W=16 | fastest | W=6 / fastest | W=6 / W=4 | per-run fastest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemv f64 | 10.596 | 5.838 | 3.603 | 3.179 | **3.062** | 4.136 | 8.631 | 8 | 1.038 | 0.882 | 8: 12 |
| trmv f64 | 5.704 | 3.192 | 1.963 | 1.789 | **1.704** | 2.575 | 4.563 | 8 | 1.050 | 0.912 | 8: 12 |
| trisolve f64 | 5.775 | 3.229 | 1.959 | 1.777 | **1.690** | 2.552 | 4.458 | 8 | 1.052 | 0.907 | 8: 12 |
| gemv f32 | 9.990 | 5.358 | 3.011 | 2.430 | **2.268** | 4.291 | 8.883 | 8 | 1.071 | 0.807 | 8: 12 |
| trmv f32 | 5.383 | 2.827 | 1.874 | 1.823 | **1.772** | 3.113 | 5.929 | 8 | 1.028 | 0.973 | 4: 4, 6: 1, 8: 7 |
| trisolve f32 | 6.682 | 3.526 | 1.942 | 1.875 | **1.796** | 3.088 | 5.860 | 8 | 1.044 | 0.966 | 6: 1, 8: 11 |

### xeon-e5-2420v2 -- clang++

Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz; Ubuntu clang version 21.1.8 (6ubuntu1); vector registers 16; built default W = 8 (f64), 8 (f32); median of 9 runs (3 sessions), each the median of 21. Data: `benchmarks/data/xeon-e5-2420v2/interleave_rows/`.

| kernel | W=1 | W=2 | W=4 | W=6 | W=8 | W=12 | W=16 | fastest | W=6 / fastest | W=6 / W=4 | per-run fastest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemv f64 | 19.778 | 13.911 | **13.241** | 13.698 | 13.724 | 15.183 | 17.592 | 4 | 1.035 | 1.035 | 4: 9 |
| trmv f64 | 10.533 | 7.559 | **7.237** | 7.307 | 7.385 | 8.398 | 15.583 | 4 | 1.010 | 1.010 | 2: 1, 4: 6, 6: 2 |
| trisolve f64 | 10.226 | 7.318 | **7.119** | 7.385 | 7.286 | 10.250 | 15.218 | 4 | 1.037 | 1.037 | 2: 2, 4: 7 |
| gemv f32 | 19.420 | 10.719 | 9.066 | **8.560** | 8.712 | 10.037 | 11.475 | 6 | 1.000 | 0.944 | 6: 7, 8: 2 |
| trmv f32 | 10.170 | 5.612 | 4.749 | **4.459** | 4.666 | 5.288 | 10.845 | 6 | 1.000 | 0.939 | 6: 8, 8: 1 |
| trisolve f32 | 9.947 | 5.436 | 4.617 | **4.223** | 4.528 | 8.622 | 11.090 | 6 | 1.000 | 0.915 | 6: 9 |

### xeon-e5-2420v2 -- g++

Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz; g++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0; vector registers 16; built default W = 8 (f64), 8 (f32); median of 9 runs (3 sessions), each the median of 21. Data: `benchmarks/data/xeon-e5-2420v2/interleave_rows/`.

| kernel | W=1 | W=2 | W=4 | W=6 | W=8 | W=12 | W=16 | fastest | W=6 / fastest | W=6 / W=4 | per-run fastest |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gemv f64 | 20.691 | 13.603 | **13.122** | 13.397 | 13.725 | 14.157 | 15.415 | 4 | 1.021 | 1.021 | 4: 8, 6: 1 |
| trmv f64 | 10.387 | 7.340 | **7.074** | 7.205 | 7.354 | 7.709 | 9.178 | 4 | 1.019 | 1.019 | 4: 6, 6: 3 |
| trisolve f64 | 10.131 | 7.213 | **6.911** | 7.037 | 7.203 | 7.494 | 9.010 | 4 | 1.018 | 1.018 | 4: 8, 6: 1 |
| gemv f32 | 19.098 | 19.796 | 8.789 | 8.707 | **8.574** | 10.703 | 14.971 | 8 | 1.016 | 0.991 | 4: 1, 8: 8 |
| trmv f32 | 10.025 | 5.460 | **4.513** | 4.516 | 4.620 | 5.517 | 7.666 | 4 | 1.001 | 1.001 | 4: 5, 6: 4 |
| trisolve f32 | 10.054 | 5.348 | **4.373** | 4.397 | 4.451 | 5.430 | 7.177 | 4 | 1.005 | 1.005 | 4: 8, 6: 1 |

<!-- END generated -->

## Reproducing

Run a platform session with `INTERLEAVE=1`: `--interleave` in
`benchmarks/remote_session.sh`, or `$env:INTERLEAVE = '1'` for a Windows
profile. Publish it with `benchmarks/publish_platform.cmake`. Then:

```
python3 benchmarks/interleave_table.py import <machine> <session stamp> <results dir>
python3 benchmarks/interleave_table.py table
```

The first command copies the sweeps to
`benchmarks/data/<machine>/interleave_rows/<stamp>/`: one CSV per compiler,
with its `.sysinfo` sidecar, and the raw outputs. The second rewrites the
generated part of this page.
