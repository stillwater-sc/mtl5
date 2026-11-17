# Column sweep against the row kernel

A contiguous column-major A takes the column sweep in `gemv`, `trmv` and the
forward substitution (#777). It replaced the row kernels there because they
read a column-major A at a stride of m elements, which on large matrices was
2.5-6 GB/s. #884 found the sweep slower than the row kernel on small and on
short, wide matrices: 4 x 1000 at half its speed. This page records what was
measured on the five benchmark machines and what the dispatch is now.

The results are bit-identical whichever path runs: each row sums the same
terms in ascending column order with the same expression as the one-row loop.
`tests/unit/operation/test_trmv_trsv_interleaved.cpp` holds every path to it,
built with `-ffp-contract=off`.

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
  with n <= 32, the forward substitution with n <= 64. Each cutoff
  minimized the geometric mean of the time over the best of the two kernels
  across all the builds below. Larger cutoffs run into GCC's `-O3
  -march=native` code for the row kernels on a column-major A on x86, 2-5x
  slower than its generic code from about 32 rows.
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

`column_sweep_table.py` also lists every shape where the public call is more
than 15% slower than #777's sweep kernel. Most are the smallest shapes (2 x 2
to 16 x 16), where the public call's own checks add 5-90 ns to a kernel
called directly. The cases that cost more than 0.1 us:

- **The forward substitution under Clang at n = 96-192:** 0.74-0.84 of the
  old sweep (0.1-0.5 us) on the Xeon, the Ryzen (clang-cl) and the Ventuno.
  The blocked diagonal compiles less well there than the one-column loop.
- **`trmv` on the Xeon at n = 32-48:** 0.61-0.84 (0.1-0.3 us), under GCC
  with `-march=native` at the `n <= 32` cutoff, and under Clang above it.
- **Single shapes, 0.82-0.85:** clang-cl `gemv` at 12 x 1000 and 12 x 4096,
  and 4096 x 8; the Jetson's f32 64 x 4096 under GCC; clang-cl's f64 2048 x
  2048, which runs the same one-column loop as the old sweep.

## Reproducing

```bash
CPU=0 benchmarks/run_column_sweep.sh                 # Linux
powershell -File benchmarks\run_column_sweep.ps1    # Windows
benchmarks/column_sweep_table.py benchmarks/data
```
