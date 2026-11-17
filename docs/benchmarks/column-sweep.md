# Column sweep against the row kernel

A contiguous column-major A takes the column sweep in `gemv`, `trmv` and the
forward substitution (#777). It replaced the row kernels there because they
read a column-major A at a stride of m elements, which on large matrices was
2.5-6 GB/s. #884 found the sweep slower than the row kernel on small and on
short, wide matrices: 4 x 1000 at half its speed. This page records what was
measured on the five benchmark machines and what the dispatch is now.

Whichever of these paths runs, each row sums the same terms in ascending
column order with the same expression as the one-row loop. With contraction
off the bits are the same: `tests/unit/operation/test_trmv_trsv_interleaved.cpp`
holds every path to it, built with `-ffp-contract=off`. A build that
contracts may fuse a multiply and an add in one path and not in another, as
for the row kernels (#749), so it can differ in the last bit. BLAS and the
native-fast blocked kernels, where a build has them, take these calls first
and sum in their own order.

## What changed

- **Eight columns per pass.** The sweep loaded and stored every accumulator
  of its strip for each column, so a short strip was bound by that traffic,
  not by A. `detail::sweep_columns<8>` keeps each accumulator in a register
  across eight terms (`column_sweep_columns`).
- **Short strips in registers.** A strip of at most 16 rows
  (`column_sweep_fixed_rows`) takes `detail::sweep_fixed_rows<R>`, with R
  known at compile time, so the accumulators stay in registers for the whole
  pass. GCC's loop vectorizer is off for that function: with AVX it
  vectorized the column loop and rebuilt the columns with shuffles, where the
  SLP form (R accumulators in one register, x broadcast per column) is the
  one this loop wants.
- **The triangular diagonal blocked.** Up to n = 2048 (f64) the whole matrix
  is one strip, so in `trmv` and the forward substitution all the work was
  in the strip's diagonal block, one column at a time. That block now goes
  eight columns at a time: the rows past the group take all eight columns
  unrolled, and the rows inside it their few terms one by one. The forward
  substitution solves a group's rows first, then updates the rows below.
- **Wide column strides stay as they were.** At 8 KB or more between columns
  (`column_sweep_unrolled_stride`; n >= 1024 for f64), eight columns per pass
  read eight streams that far apart. On the Jetson and the Ventuno that ran
  at 0.46-0.75 of one column per pass, as on the Xeon at 4096. The public
  calls then run #777's one-column instantiation itself.
- **The smallest shapes take the row kernels:** `gemv` with m <= 2, `trmv`
  with n <= 32, the forward substitution with n <= 64, as #884 set them.
  Each cutoff minimized the geometric mean of the time over the best of the
  two kernels across all the builds below. Larger cutoffs ran into GCC's
  code for the row kernels on a column-major A on x86, 1.3-3.3x slower with
  `-march=native` than generic. #898 fixed that and raised the `trmv`
  cutoff to n <= 64 (see the last section).
- **MSVC keeps #777's sweep.** Its code for the eight-column pass ran at
  0.23-0.89 of the one-column loop from 48 rows up, so under MSVC
  `column_sweep_columns` is 1. It still gets the row-kernel cutoffs.

## How it was measured

- **Tool:** `bench_column_sweep` (`benchmarks/bench_column_sweep.cpp`). For
  each shape it times #777's sweep (`sweep1_ns`), the new sweep kernel
  (`sweep_ns`), the row kernel at the build's default interleave width
  (`rows_ns`), and the public `mult`, `trmv` and `lower_trisolve`
  (`public_ns`). Each entry is the median of 11 samples of at least 2 ms,
  after every kernel has run once.
- **Shapes:** square 2-4096; short, wide m = 1-64 by n = 256, 1000, 4096;
  tall 1000 x 4, 4096 x 8, 256 x 16; float and double.
- **Builds:** `run_column_sweep.sh` compiles the bench with GCC and Clang at
  CMake Release's `-O3 -DNDEBUG`, generic and native (the Ventuno with its
  profile's flags, #806). `run_column_sweep.ps1` builds it on Windows with
  MSVC (generic and `/arch:AVX2`) and clang-cl (generic and
  `-march=native`). The Jetson has no Clang.
- **Machine state:** one thread (`MTL5_NUM_THREADS=1`), pinned to CPU 0,
  which is a P-core on the i7-12700K and an A78C on the Ventuno. On Linux
  the governor of that CPU's policy is set to `performance`, plus
  `jetson_clocks` at 25 W on the Jetson. The Ryzen ran on its Balanced
  plan, not changed. Each CSV's `.sysinfo` sidecar records all of it.
- **Data:** `benchmarks/data/<machine>/column_sweep/<session>/`, at commit
  c93ee00. The tables below come from `benchmarks/column_sweep_table.py`.

## Results

Each cell is the geometric mean over the shapes of the old sweep's time
divided by the public call's, its minimum in brackets, then the same against
the row kernel. Above 1, the public call is faster.

| machine | build | gemv: vs old (min) / vs rows | trmv (lower): vs old (min) / vs rows | forward substitution: vs old (min) / vs rows |
|---|---|---|---|---|
| arduino-ventuno-q | clang++-generic | 2.33 (0.52) / 1.67 | 1.39 (0.82) / 1.55 | 1.39 (0.79) / 1.55 |
| arduino-ventuno-q | clang++-native | 2.02 (0.53) / 1.58 | 1.38 (0.79) / 1.51 | 1.39 (0.79) / 1.52 |
| arduino-ventuno-q | g++-generic | 1.42 (0.55) / 1.67 | 1.34 (0.94) / 1.59 | 1.40 (0.98) / 1.63 |
| arduino-ventuno-q | g++-native | 1.43 (0.62) / 1.74 | 1.33 (1.00) / 1.61 | 1.39 (0.98) / 1.65 |
| i7-12700k | clang++-generic | 1.91 (0.63) / 1.08 | 1.52 (1.00) / 1.20 | 1.43 (0.83) / 1.11 |
| i7-12700k | clang++-native | 1.89 (0.62) / 1.32 | 1.71 (0.97) / 1.44 | 1.59 (1.00) / 1.24 |
| i7-12700k | g++-generic | 1.47 (0.74) / 2.35 | 1.55 (0.95) / 1.97 | 1.47 (0.90) / 1.18 |
| i7-12700k | g++-native | 1.44 (0.56) / 4.88 | 1.51 (0.86) / 2.80 | 1.55 (0.99) / 1.27 |
| jetson-orin | g++-generic | 1.41 (0.54) / 1.58 | 1.33 (0.98) / 1.41 | 1.48 (0.91) / 1.44 |
| jetson-orin | g++-native | 1.40 (0.60) / 1.59 | 1.35 (0.98) / 1.42 | 1.49 (0.92) / 1.47 |
| ryzen-7-8845hs | clangcl-generic | 1.70 (0.49) / 1.07 | 1.38 (0.92) / 1.20 | 1.11 (0.76) / 1.22 |
| ryzen-7-8845hs | clangcl-native | 2.06 (0.68) / 1.41 | 1.47 (0.94) / 1.76 | 1.14 (0.74) / 1.66 |
| ryzen-7-8845hs | msvc-avx2 | 1.15 (0.73) / 1.05 | 1.19 (0.94) / 1.22 | 1.20 (0.93) / 1.40 |
| ryzen-7-8845hs | msvc-generic | 1.15 (0.79) / 0.96 | 1.17 (0.99) / 1.18 | 1.18 (0.97) / 1.30 |
| xeon-e5-2420v2 | clang++-generic | 1.57 (0.50) / 1.06 | 1.29 (0.74) / 1.20 | 1.31 (0.79) / 1.18 |
| xeon-e5-2420v2 | clang++-native | 1.72 (0.55) / 1.10 | 1.38 (0.94) / 1.35 | 1.30 (0.76) / 1.17 |
| xeon-e5-2420v2 | g++-generic | 1.22 (0.50) / 2.23 | 1.31 (0.76) / 1.92 | 1.28 (0.83) / 1.25 |
| xeon-e5-2420v2 | g++-native | 1.20 (0.47) / 3.49 | 1.17 (0.61) / 2.33 | 1.28 (0.83) / 1.24 |

## Where it is slower

`column_sweep_table.py` also lists every shape where the public call takes
more than 1.15 times as long as #777's sweep kernel. Most are the smallest
shapes (2 x 2 to 16 x 16), where the public call's own checks add 5-90 ns to
a kernel called directly. The cases that cost more than 0.1 us:

- **The forward substitution under Clang at n = 96-256:** 0.74-0.87 of the
  old sweep (0.2-0.5 us) on the Xeon, the Ryzen (clang-cl) and the Ventuno.
  The blocked diagonal compiles less well there than the one-column loop.
- **`trmv` on the Xeon at n = 32-96:** 0.61-0.84 (0.2-0.3 us), under GCC
  with `-march=native` at the `n <= 32` cutoff, and under Clang above it.
- **`gemv` with 48-64 rows and 4096 columns on the Jetson and the Ventuno
  under GCC:** 0.83-0.87 (4-19 us), on the eight-column pass. Shorter rows
  of the same shapes gain on both.
- **clang-cl `gemv` at 12 x 1000, 12 x 4096 and 4096 x 8:** 0.82-0.85
  (0.6-2.2 us).
- **n = 1024-2048 on the Xeon and under clang-cl:** 0.83-0.87 (13-267 us).
  At these strides the public calls run #777's own one-column instantiation,
  so this is the run-to-run spread of a memory-bound loop, not a different
  kernel.

## The row kernels on a column-major A under GCC (#898)

The cutoffs above were held low by GCC on x86. For a column-major A it could
vectorize the row kernels' column loop as an in-order reduction: R x 4
columns loaded, then each row's chain rebuilt with shuffles and scalar
merges. Which code it emitted depended on the calling context, so the same
`mult_rows_interleaved<8>` ran 64 x 64 f64 in 0.35 us in one binary and
2.2 us in another (i7-12700K, `-O3 -march=native`).

Since #898, under GCC on x86-64, a row kernel on a contiguous column-major A
runs each block's column loop as `sweep_fixed_rows<W>` once it has at least
16 columns. That is the same terms in the same order, with GCC's loop
vectorizer off for that function (`row_kernel_fixed_rows_v`). Row-major
code is unchanged: the f64 W = 8 kernels compile to identical code before
and after. The other compilers and the ARM boards keep the row kernel's own
loop, which measured as fast or faster there. Through `sweep_fixed_rows`,
`trmv` ran at 0.8-0.94 on the Jetson and the Ventuno under GCC, and at 0.83
under MSVC.

The row-kernel time, before over after, as a geometric mean over the shapes
up to 256:

| machine | build | gemv | trmv |
|---|---|---|---|
| i7-12700K | g++-generic | 1.96 | 1.58 |
| i7-12700K | g++-native | 3.86 | 2.08 |
| Xeon E5-2420 v2 | g++-generic | 1.96 | 1.44 |
| Xeon E5-2420 v2 | g++-native | 2.99 | 1.78 |

Clang, whose code is unchanged, measured 0.97-1.00 on both machines.
Through the public calls, which use the row kernels only below the cutoffs,
`trmv` is 4-6% faster with the n <= 64 cutoff, and `gemv` and the forward
substitution are unchanged (0.98-1.04). The Xeon's load rose from 0.4 to 2.4
during its run (it is also a Kubernetes node), and its worst shapes moved as
much under the unchanged #777 sweep as under the public calls.

With those times, the cutoff simulation over every build moves `trmv` from
n <= 32 to n <= 64: the time over the best of the two kernels goes from
1.046 to 1.013 in geometric mean, and the worst case from 2.0 to 1.5.

It also suggested `gemv` m <= 4 (1.110 -> 1.098). Measured, m = 3 then ran
at half its speed in the sweep, on every build: the row kernel takes it as a
2-row and a 1-row block, where the sweep takes one 3-row strip. So `gemv`
keeps m <= 2. The forward substitution's row kernel was not affected and
keeps n <= 64.
The sessions are under `benchmarks/data/{i7-12700k,xeon-e5-2420v2}/column_sweep/`,
the later of the two in each.

## Reproducing

```bash
CPU=0 benchmarks/run_column_sweep.sh                 # Linux
powershell -File benchmarks\run_column_sweep.ps1    # Windows
benchmarks/column_sweep_table.py benchmarks/data
```
