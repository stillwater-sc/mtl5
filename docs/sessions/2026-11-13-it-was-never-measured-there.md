# Session: it was never measured there

**Date**: 2026-11-12 into 2026-11-13 (overnight, then the next day)
**Duration**: One long working session, with a context compaction partway through
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Work through the open items the #737 platform sessions had left behind:

- #799, the analysis programme;
- #810, a tuned reference for the direct solvers;
- #804, the column-major `gemv`;
- #805, the Ventuno's threaded LU;
- #806, the Ventuno's quad integer gemm;
- #795, GCC's f32 triangular kernels on AArch64.

Each was taken through a fix, a measurement on the machine that showed it, and
the pages that described it.

**Merged:** #811, #812, #813, #814, #815, #817, #818, #819, #820, #821.

**Closed:** #799, #804, #805, #806, #810, #795.

This log covers the work from #811 on. The earlier PRs of 2026-11-12 (#801 to
#809) have their own CHANGELOG entries.

## The through-line

Four of the six problems came from a setting that was correct on the machine
where it was measured, then applied to one where nobody had measured it.

- **The `keep_scalar` barrier (#795)** was measured on an i7-12700K and enabled
  for GCC on x86-64 *and* AArch64. On both ARM boards it was what made f32
  `trmv` and the forward substitution 5.6–7.2× slower at W = 4. #798 had worked
  around the symptom by holding those kernels at W = 2.
- **`-march=native` (#806)** picks the host's ISA on every x86 machine and on
  the Jetson. On the Ventuno, GCC 13 resolves it to nothing: the A78C + A55
  mix is not a pair it knows, so the driver passes no `-march` or `-mcpu` on.
  Every native-fast build there was baseline Armv8.0-a.
- **The thread pool's sleep (#805)** cost nothing on the x86 machines, where
  waking a core is cheap. The A78C's idle states take 1.5–1.8 ms to leave, and
  the unblocked LU opens about 150 parallel regions at n = 512.
- **`cholmod_common::nthreads_max` (#817)** exists from CHOLMOD 5 on. The
  benchmark set it unconditionally, so the `blas` build failed on the
  Jetson's Ubuntu 22.04, which ships CHOLMOD 3.

Two of them were also misread, and the record had the right answer all
along. The sidecars of the 2026-11-11 sessions said `build_isa=NEON` on the
Ventuno and `NEON DOTPROD` on the Jetson. Issue #806 and both analysis pages
had the two boards the other way round. They read the Jetson's 3.9× quad
gemm as an emulated path beating a native one, when it was the native one.

## Work completed

### Direct solvers against tuned references (#810)

- **#812:** the CHOLMOD binding.
  - The #774 bugs: a double free on move, the leaks when a constructor throws,
    and an indefinite matrix factored silently. CHOLMOD's simplicial default is
    L D Lᵀ, which reports nothing for an indefinite matrix, so the binding now
    asks for L Lᵀ on both paths.
  - Separate analyze, factorize and solve steps; its first unit test; a CI lane.
- **#814:** `mtl5_bench` entries per phase (analysis, factorization, solve),
  a 3D grid (`direct3d`, 20³ in `medium`), and CHOLMOD and MKL PARDISO
  reference entries in the `blas` variant. MKL is loaded at run time, so its
  BLAS can't replace OpenBLAS for the other entries. The direct solvers' check
  had passed an all-NaN solution; it now fails.
- **#815:** two i7-12700K sessions and
  `docs/benchmarks/direct-solver-references.md`. On the 3D grid CHOLMOD and
  PARDISO reach 35–41% of peak; MTL5's supernodal LDLᵀ takes 7.2× PARDISO's
  time, its supernodal LU 35×, the up-looking Cholesky 768×.
- **#817, #818:** CHOLMOD on both ARM boards. The supernodal LDLᵀ takes 3.6×
  CHOLMOD's time on the Jetson and 4.8× on the Ventuno, on the 3D grid, and
  matches it on the 2D one.

### #804, #805 and the ARM sessions

- **#813:** the four analysis pages mark the column-major `gemv` fixed (#809),
  with the Ryzen 9's MSVC A/B: f64 22.23 → 4.00 ms.
- **#811:** the thread pool's workers spin up to 1 ms
  (`MTL5_THREAD_POOL_SPIN_US`) before they sleep.
- **#818:** two cold-boot sessions on each ARM board.
  - On the Ventuno, `factor/lu` f64 512×512 takes 17.2 ms on four threads,
    where it took 109 ms before #811: 1.4× faster than one thread.
  - The Ventuno's first OpenBLAS column: `gemm` at 89% of peak on one core.

### The Ventuno's ISA (#806)

- **#819:**
  - `MTL5_NATIVE_ARCH_FLAGS` (default `-march=native`) replaces the flags
    `MTL5_NATIVE_ARCH` adds.
  - A configure-time guard on AArch64 Linux warns when the flags miss a
    feature `/proc/cpuinfo` lists.
  - `platform.sh --arch`.
  - The Ventuno's profiles pass `-march=armv8.2-a+dotprod+fp16+rcpc
    -mtune=cortex-a78c`. That is what every core lists. `-mcpu=cortex-a78c`
    also enables PAuth, which the kernel doesn't report.
- **#820:** two Ventuno sessions with the new flags. `gemm_quad` i8 takes
  3.68 ms (was 23.7), 4.5× faster than the plain i8 gemm. The 8-bit dots gain
  2.8–3.4×. native-fast's Gauss–Seidel smoother had taken 1.9× the generic
  build's time and now matches it.

### The barrier on AArch64 (#795)

- **#821:** `keep_scalar_triangular_v` is `keep_scalar_v` on x86-64 and false
  on AArch64; `trmv` and the forward substitution use it. The A/B, f32 at
  W = 4:

  | | with the barrier | without it |
  |---|---:|---:|
  | Jetson, `trmv` | 50.04 ms | 6.31 ms |
  | Jetson, forward substitution | 39.45 ms | 5.58 ms |
  | Ventuno, `trmv` | 31.66 ms | 3.12 ms |
  | Ventuno, forward substitution | 26.22 ms | 2.99 ms |

  `gemv` keeps the barrier: on the Ventuno it is 1.66× faster with it. The
  width-2 exception of #798 goes, so every AArch64 compiler gets W = 4.

## The parts that went wrong

- **A title closed an issue.** #813 was titled "... is fixed (#804)", and
  GitHub counted "fixed (#804)" as a closing keyword. #804 was meant to close,
  so no harm; the memory note on closing keywords now covers titles.
- **The 32³ grid.** The first 3D size was 32³. The up-looking
  `sparse_cholesky` took over ten minutes per factorization there, so `medium`
  became 20³.
- **The solve's byte model** counted the index arrays, which put CHOLMOD at
  164% of the bandwidth roof. It now counts the values only.
- **The first ARM session pair was discarded.** The Jetson's `blas` build
  failed on CHOLMOD 3, and the Ventuno's had been configured before OpenBLAS
  was installed. A test build on the Jetson during its session also perturbed
  one run. Both boards were rerun after #817.
- **A local build started while another was running.** `pgrep` showed a
  universal-repo `ninja`, and the command went on anyway. Nothing compiled,
  because the target name was wrong; after that every local build waited for
  an idle machine.
- **Claims corrected on self-review**, before CodeRabbit saw them:
  - the 1-thread times "unchanged" where the refinement's had fallen;
  - "19 of 44" outliers on 4 threads, where it was 26;
  - a guess that the #811 A/B had run without `jetson_clocks`;
  - an f64 spread of "1–6%" that was 6.4%.
- **CodeRabbit's rate limit** skipped #813, #814 and the last commits of #819
  and #820. Those were reviewed by hand.
- **Permission denials:**
  - The issue comments on #804, #805 and #810.
  - The first merge of #819, which `/resolve-issue` hadn't asked for.

  Each went ahead once it was asked for explicitly.

## Lessons

- **A setting measured on one ISA is a hypothesis on another.** The barrier,
  `-march=native` and the pool's sleep were all right where they were
  measured. A guard that compares what the build got with what the hardware
  has (#819) turns the next one into a warning instead of a slow card.
- **Read the sidecar before the summary.** `build_isa=` had the right answer
  for #806 from the first session. Every page that described it was written
  from the expectation instead.
- **Workarounds hide causes.** #798's W = 2 for the triangular kernels was a
  correct reading of the symptom, and it left the barrier in place, costing
  13–36% against W = 4 without it.
- **A diagnostic run separates suspects cheaply.** The Jetson's CHOLMOD was
  3.7–5.7× slower on six threads. Six short runs, varying the pool and the
  BLAS threads separately, put it on OpenBLAS 0.3.20 rather than on #811.

## Follow-ups

- **#778:** run-time selection of the interleave width. On the Jetson, GCC's
  f64 `trmv` and forward substitution still prefer W = 2 and lose 12% at 4.
- **The direct solvers (#195 and the analysis page):** a supernodal Cholesky
  with BLAS-3 updates, a cheaper supernodal LU, and threading over the
  elimination tree. MTL5 is 3.6–7.2× behind on the 3D grid.
- **The Jetson's OpenBLAS 0.3.20:** its threaded CHOLMOD is slower than one
  thread. A newer OpenBLAS, or a capped BLAS thread count in that entry,
  would make the 6-thread `ref_cholmod` figures meaningful.
- **#774:** SPQR's move and the binding's other open items.
- **The other sweep and A/B scripts** still pass `-march=native`; on the
  Ventuno the CMake guard now warns about it.
