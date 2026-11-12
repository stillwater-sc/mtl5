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
  mtl5_bench.cpp         the platform benchmark: probes + every registered operation, checked (#721)
  platform/              mtl5_bench's registry, probes, runner and entries (entries_*.hpp)
  bench_klu.cpp          sparse scoreboard: native KLU vs SuiteSparse KLU (#138)
  bench_superlu.cpp      sparse scoreboard: native LU vs SuperLU (#186)
  bench_sparse.cpp       level-scheduled sparse triangular-solve scaling (#297)
  bench_block_scaled_dot.cpp  block-scaled integer gemv: per-block vs dequantize vs fused (#628)
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

## Platform benchmark: `mtl5_bench` (#721)

`mtl5_bench` is the standard characterization of a machine. It runs the same way
on every machine, so results from a Jetson, a Ventuno Q and an x86 desktop can be
compared directly.

### On a new machine: one command

```bash
git clone https://github.com/stillwater-sc/mtl5 && cd mtl5
benchmarks/platform.sh                # Windows: .\benchmarks\platform.ps1
```

`platform.sh` does the following, in order:
1. Gates the session with `preflight.sh`.
2. Builds `mtl5_bench` once per variant, each in a clean directory under
   `build-platform/`, and builds every variant before measuring any of them:
   - `native`;
   - `native-fast` (Highway and the blocked GEMM, with `-march=native` except
     on SVE hardware (#720) and macOS);
   - `blas` (if CMake finds a system BLAS/LAPACK; skipped otherwise).
3. Gates again after the builds.
4. Runs each variant at 1 thread and at all CPUs. MTL5's pool and every
   vendor library's pool are pinned to the same count.

**Thread scaling.** The thread pool is sized once per process, so scaling is measured across runs. Every run after a variant's 1-thread run is compared with it (`mtl5_bench --baseline <variant>-t1.json`), and the JSON and card then report each entry's speedup and parallel efficiency, plus a per-area median. A baseline from another size class is refused.

Results go to `build-platform/results/<host>-<UTC time>/`:
- `<variant>-t<T>.json`, with its `.sysinfo` sidecar, to which `preflight`'s
  machine-state record is appended;
- `<variant>-t<T>.md`, the capability card;
- `summary.md`, all the cards in one file;
- `logs/`.

Options: `--class`, `--variants`, `--threads`, `--jobs` (default: half the
CPUs), `--out`, `--machine` (default: the ARM board's device-tree model, else
the CPU brand), and `--no-gate`, which records the machine state without
stopping on it.

CI runs the script end to end at `--class small` on Linux x64, Linux ARM64,
macOS and Windows.

### Machine profiles: pinned cores and a clock policy (#737)

Run without options, `platform.sh` is correct only on a homogeneous machine
without SMT. Elsewhere the scheduler is free to put threads on SMT siblings or
little cores, and the script warns. For a session to publish, use the
machine's profile:

```bash
benchmarks/machines/i7-12700k-platform.sh          # 8 P-cores, governor performance
benchmarks/machines/xeon-e5-2420v2-platform.sh     # 6 cores, SMT siblings excluded
benchmarks/machines/jetson-orin-platform.sh        # 6 A78AE, nvpmodel + jetson_clocks
benchmarks/machines/arduino-ventuno-q-platform.sh  # 4 A78C, A55s excluded
CLASS=large benchmarks/machines/i7-12700k-platform.sh
```

A profile does four things:
1. **Identifies the machine.** It refuses to run anywhere else; `FORCE=1`
   overrides.
2. **Derives the CPUs from the topology:** one CPU per physical core, of the
   core class to measure, never a hard-coded list.
3. **Applies the clock policy.** Every change goes through `sudo -n` and is put
   back when the profile exits.
4. **Calls `platform.sh`.** It passes `--cpus` (a run at T threads gets
   `taskset -c` on the first T CPUs) and `--clock-policy`, and every sidecar
   records them as `bench_cpus=` and `clock_policy=`. `publish_platform.cmake`
   puts both on the card page.

`NO_CLOCK_POLICY=1` measures without the policy, for a card published with
`PRELIMINARY=ON`. `INTERLEAVE=1` also runs `bench_interleave_rows` on the first
measured CPU, still under the policy, with GCC and with Clang where it's
installed (#780). After publishing the session, `interleave_table.py import`
copies its sweeps to `benchmarks/data/<machine>/interleave_rows/<session>/`, and
`interleave_table.py table` regenerates the cross-machine tables on
`docs/benchmarks/interleave-width.md`.

**Root for the clock policy.** A machine whose user lacks passwordless sudo
needs one rule, installed with `sudo visudo -f /etc/sudoers.d/mtl5-bench`. The
`reboot` entry is for `remote_session.sh --reboot` below:

```
<user> ALL=(root) NOPASSWD: /usr/bin/tee /sys/devices/system/cpu/cpufreq/policy[0-9]/scaling_governor, /usr/bin/tee /sys/devices/system/cpu/cpufreq/policy[0-9][0-9]/scaling_governor, /usr/sbin/reboot
# a Jetson also needs: , /usr/sbin/nvpmodel, /usr/bin/jetson_clocks
```

**A new machine** gets a `<machine>-platform.sh` that follows one of these
four, plus a line in `docs/benchmarks/systems.md` naming its pinning policy.

**From another machine.** `benchmarks/remote_session.sh HOST PROFILE`:
1. pushes this checkout's commit over ssh to `~/mtl5-campaign` on `HOST`;
2. runs the profile there, detached, so a dropped connection doesn't end it;
3. waits for it to finish;
4. copies the results to `build-platform/remote/<machine>/<session>/`, where
   `<machine>` is the profile's machine name (`xeon-e5-2420v2` for
   `xeon-e5-2420v2-platform.sh`), not the ssh host, so the local directories
   match `benchmarks/data/<machine>/` whatever the host is called.

`--reboot` cold-boots the host first, for the second session. `--interleave`
and `--class` pass through to the profile.

**Windows machines.** The Ryzen NUCs have PowerShell profiles, driven the same
way from a Linux machine:

```bash
benchmarks/remote_session.sh nuc-ryzen-7-8845hs ryzen-7-8845hs-platform.ps1 --interleave
benchmarks/remote_session.sh nuc-ryzen-9-8945hs ryzen-9-8945hs-platform.ps1 --reboot --interleave
```

Locally: `powershell -ExecutionPolicy Bypass -File benchmarks\machines\ryzen-7-8845hs-platform.ps1`,
with `$env:CLASS`, `$env:INTERLEAVE`, `$env:NO_CLOCK_POLICY` and `$env:FORCE` in
place of the shell variables. `_platform_profile.ps1` holds what they share:
- **CPUs:** one logical CPU per core (`0,2,…,14`), passed to `platform.ps1
  -Cpus`. A run at T threads gets the processor affinity of the first T, which
  the benchmark inherits from the script when it starts.
- **Clock policy:** a copy of the active power scheme with the processor state at
  100–100% and the energy-performance preference at 0, active for the session.
  The previous scheme comes back and the copy is deleted afterwards. Boost is
  left as the scheme has it and recorded.
- **BLAS:** the newest official OpenBLAS Windows release unpacked under
  `~\opt\OpenBLAS-<version>`, found through `CMAKE_PREFIX_PATH`, with its DLL on
  `PATH` (the `?gemmt` probe and the runs need it). Without one, the `blas`
  variant is skipped.
- **Interleave sweep:** MSVC, and clang-cl when Visual Studio has the "C++ Clang
  tools for Windows" component.

A Windows host needs, once:
1. **OpenSSH Server,** with this machine's key. For an administrator the key goes
   in `C:\ProgramData\ssh\administrators_authorized_keys`, readable only by
   Administrators and SYSTEM; Windows ignores the user's own file for them.
2. **PowerShell as the ssh shell:** `New-ItemProperty -Path HKLM:\SOFTWARE\OpenSSH
   -Name DefaultShell -Value C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe
   -PropertyType String -Force`. Under the default `cmd.exe`, git over ssh gets
   the repository path with its quotes and fails.
3. **Git for Windows, CMake and Visual Studio** with the C++ workload.
4. **An administrator account,** for the power scheme and the reboot.
5. **Optionally, a Defender exclusion** for `~\mtl5-campaign`: real-time
   scanning of the builds and runs is the largest source of timing noise there.
6. **No scan after each Defender signature update.** Defender updates its
   signatures every few hours and scans after each one; on the Ryzen 9 such a
   scan overlapped two runs and made the memory-bound entries 10-30% slower.
   The policy "Turn on scan after security intelligence update" (Computer
   Configuration > Administrative Templates > Windows Components > Microsoft
   Defender Antivirus > Scan) set to *Disabled* stops it; real-time protection
   and the scheduled scans stay on. Whatever Windows still runs is recorded:
   each session writes `background_activity.txt`, and every sidecar it wrote
   gets `background_activity=` (`none`, or the Defender and Windows Update
   events in the session's window, plus an `unavailable:` entry for a log that
   could not be read).

`remote_session.sh` takes a `.ps1` profile as the sign of a Windows host, and
checks that the host agrees. It starts the session with `Win32_Process.Create`,
outside the ssh connection's job object, which Windows ends with the
connection. The reboot is `Restart-Computer`, checked by the boot time, and
the results come back with `scp`.

**Repeatability.** Each machine is measured twice, the second time after a
cold boot. `benchmarks/compare_sessions.py SESSION1 SESSION2` gives each run's
median, min and max ratio of session-2 to session-1 time, and lists the entries
outside ±5% (`--band`). Its last line is the repeatability band for the card
page's `NOTE=`.

### Publishing a session

A results directory under `build-platform/` is not kept: it is ignored by git.
To publish a session, run `publish_platform.cmake` with the machine's name from
`docs/benchmarks/systems.md`, in lower case. Run it on the machine that produced
the session; CMake is the only thing it needs.

```bash
cmake -DRESULTS=build-platform/results/<host>-<UTC time> -DMACHINE=ryzen-9-8945hs \
      -P benchmarks/publish_platform.cmake
```

```powershell
cmake "-DRESULTS=build-platform\results\<host>-<UTC time>" -DMACHINE=ryzen-9-8945hs `
      -P benchmarks\publish_platform.cmake
```

The `-D` options go before `-P`. Anything after `-P <script>` is passed to the
script as an argument, not defined as a variable.

The script checks the session before it writes anything:
- every run has its JSON (schema `mtl5-bench/1`), its sidecar and its card;
- `summary.md` exists, so the session finished;
- every run is the same commit and the same class;
- no run has a failed check (a `FAIL` blocks a card: file it, fix it, re-run);
- no run was built from a dirty tree.

Then it does three things:
1. **Copies the session** to `benchmarks/data/<machine>/mtl5_bench/<session>/`.
   `<session>` is `YYYYMMDDTHHMMSSZ`: the UTC start time in the results
   directory's default name, or, if the directory was renamed (`--out`), the
   time `summary.md` was written at the end. The files are the JSON, sidecars,
   cards, `summary.md` and the two preflight records.
   - This directory is **append-only**. A new run is a new session directory,
     and earlier sessions are never replaced.
   - Publishing the same session twice is a no-op. Publishing different files
     under an existing session's name is refused.
   - `check_sidecars.sh` enforces the sidecars in CI.
   - These files are what stillwater-sc/mtl5-python#121 ingests for graphs and
     trends.
2. **Writes the machine's card page,** `docs/benchmarks/cards/<machine>.md`.
   - The page opens with a provenance table: session, commit, class, runs, failed
     checks, CPUs, clock policy, a link to the raw data, and the earlier sessions.
   - "Across the runs" compares the session's runs side by side: each area's
     median %roof, and each entry's median time with the fastest run in bold.
   - The session's cards follow, one section per run. Each run's full entry
     table is folded into a `<details>` block.
   - The page always shows the **latest** published session. Publishing an older
     one archives its data and leaves the page alone.
   - The page is generated, so don't edit it by hand. Publish a new session
     instead.
3. **Registers the page** in `docs-site/sync-content.mjs`'s `FILE_MAP`, if it
   isn't there yet.
4. **Links the page from `systems.md`.** It sets the last cell of the machine's
   Systems-table row, the "mtl5_bench card" column, to
   `[Card](cards/<machine>.md)`, followed by `(preliminary)` when the page
   is. That cell belongs to the script: it follows the page, so a later final
   session drops the mark. Publishing an older session leaves the cell alone,
   like the page. If the row or the column isn't there, it warns instead.

Then review and commit what the script lists:

```bash
git add benchmarks/data/<machine>/mtl5_bench/<session> \
        docs/benchmarks/cards/<machine>.md docs-site/sync-content.mjs \
        docs/benchmarks/systems.md
```

Options, each a `-D` before `-P`:

| option | effect |
|---|---|
| `DRY_RUN=ON` | validate and report what would be written; write nothing |
| `PRELIMINARY=ON` | mark the page preliminary: measured outside #737's methodology (cores not pinned, clock policy not applied, one session) |
| `NOTE="..."` | a sentence for the page, after the preliminary mark |
| `ALLOW_DIRTY=ON` | accept runs built from a dirty tree; the page says so |
| `ALLOW_FAIL=ON` | accept runs with failed checks; the page says so |
| `SESSION=YYYYMMDDTHHMMSSZ` | name the session, if neither the directory name nor `summary.md` carries a time |

### What one run does

1. **Measures the machine's ceilings:**
   - **FMA peak in f64 and f32,** measured twice:
     - with a kernel in the widest vector ISA the CPU and OS support, chosen at
       run time (AVX-512F, AVX2+FMA or SSE2 on x86, NEON on AArch64);
     - over `simd::batch`, the SIMD target this build compiled.

     The ceiling is the higher of the two. Every build on a machine is
     therefore judged against the same machine-level ceiling: a build without
     Highway has a scalar `batch`, and judging it by that put its GEMM at
     thousands of percent.
   - **Memory bandwidth:** three-stream read, update (`a += s*b`) and STREAM
     triad. The best of the three is the bandwidth ceiling.
2. **Runs every registered operation** at a fixed problem size for the chosen
   class. Each operation goes through the public `mtl::` API, so the build
   decides the backend, as described above.
3. **Checks every result** against a long-double reference, using the standard
   `gamma_k` error bound. A fast wrong answer prints `FAIL` and makes the exit
   status 1. Matrix results (level-3 BLAS, `inv`, Schur, SVD, `expm`,
   `sylvester`, mixed GEMM) are checked twice, and both must pass: on 64
   sampled entries at the entrywise bound, and by Freivalds' check, which
   compares C·x with A·(B·x) for two probe vectors and so covers every entry
   in O(n²). In the M7 mutation tests the samples missed a single wrong GEMM
   entry, a wrong mirrored entry in GEMMT, a partial SYRK row and one Kronecker
   entry; Freivalds caught all four.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DMTL5_BUILD_BENCHMARKS=ON
cmake --build build --target mtl5_bench
./build/benchmarks/mtl5_bench --class medium --json results.json   # + results.json.sysinfo
./build/benchmarks/mtl5_bench --list                               # registered entries
./build/benchmarks/mtl5_bench --class small --filter blas3         # a subset, smoke-sized
```

`%roof` is the achieved rate as a percentage of the operation's **roofline**:
the lesser of the FMA peak and bandwidth × arithmetic intensity (flops per
compulsory byte). Low-intensity operations are judged against bandwidth, and
high-intensity ones against the FMA peak.
- **The ceilings are measured sustained rates, not hard limits.** A tuned
  streaming kernel can beat a plain-loop probe by 10–20%.
- **`*` marks a bandwidth-bound result whose working set fits in the
  last-level cache.** Its rate is a cache rate, but its percentage is against
  DRAM bandwidth. The `small` class is cache-resident on purpose: it is a smoke
  run. `medium` and `large` are sized well past any LLC, so they measure DRAM.
- **`!` marks a bandwidth-bound result above 150% of the DRAM roofline.**
  Tuned kernels beat the measured ceilings by 10–20%, and the bandwidth probe
  varies by about 10% between runs, but not by that much, so the entry's
  byte model is the likely cause. It is a warning, not a failed check, and the JSON
  field is `suspect_model`. It is not raised for compute-bound results, whose
  FMA-peak probe may not cover the kernel's SIMD width, or when the
  last-level cache size is unknown. symv was the first case: it charged all of A where it
  reads one triangle, and showed 184% (#743).
- **The probes and the operations use the same thread count,** MTL5's pool
  size (`MTL5_NUM_THREADS`, default 1).

| class | vectors | matrix–vector | matrix–matrix, factorizations | eigen / SVD | sparse kernels, smoothers (grid) | sparse direct, orderings (grid) | Krylov, preconditioners (grid) | multigrid (grid) | sparse eigen (grid) | catalogue time (i7-12700K, 1 thread) |
|---|---|---|---|---|---|---|---|---|---|---|
| `small`  | 2^15 | 512 | 128 | 64 | 64² | 32² | 32² | 63² | 24×25 | ~12 s |
| `medium` | 2^23 | 4096 | 512 | 256 | 512² | 96² | 128² | 255² | 64×65 | ~80 s |
| `large`  | 2^25 | 8192 | 1024 | 512 | 1024² | 192² | 256² | 511² | 128×129 | ~8 min; peak memory ~1 GiB (`ger`'s two 8192² matrices) |

`sparse_qr` runs on half the direct grid's side (see its entry), because one QR solve at 96² took over 4 s.

The registry covers 107 entries:
- **BLAS levels 1–3** (#721 milestone M1), plus DotK (`dot_k<3>`, M7), checked on an ill-conditioned integer dot product with an exact int64 answer.
- **Dense factorizations:** LU, Cholesky, LDLᵀ, Bunch–Kaufman, QR, LQ, Hessenberg, the Householder step, `inv`, and f32→f64 LU refinement.
- **Eigensolvers and matrix functions:** symmetric (tridiagonal QR and Jacobi), general, Schur, SVD, `expm` and `sylvester`.
- **Element-wise functions:** `exp`, `log`, `sin`, `sqrt` and `erf` (M3).
- **Sparse kernels:** SpMV, transposed SpMV, CRS×dense, SpGEMM and the level-scheduled lower solve.
- **Sparse direct solvers:** Cholesky, LDLᵀ, supernodal LDLᵀ, LU, supernodal LU, native KLU and QR.
- **Orderings:** AMD, COLAMD, RCM and the block-triangular form (M4).
- **Krylov solvers:** CG, MINRES, BiCG, BiCGSTAB, BiCGSTAB(ℓ), CGS, GMRES(30), QMR, TFQMR and IDR(s) at s = 1, 2, 4 and 8. Each is checked by the true residual it reaches.
- **Preconditioner setups:** diagonal, ILU(0), ILUT, IC(0), ILDL, SSOR and block-diagonal. Each must cut the Krylov iteration count by at least a quarter.
- **Smoothers:** Jacobi, Gauss–Seidel and SOR. Each is checked against its defining splitting, row by row.
- **Multigrid:** a 2D Poisson V(1,1) cycle.
- **Sparse eigensolvers:** Lanczos, Arnoldi, shift-invert and power iteration. They're checked against the analytic spectrum of the generated matrices (M5).
- **Mixed precision:** f32→f64 dot and GEMM (via `mult<double>`), and the block-scaled int8 dot (the Q8_0 layout). Each is checked at the wide precision's bound, so a kernel that silently accumulates in f32 fails.
- **Integer:** dot and GEMM in int32, int16→int32, int8→int32 and uint8×int8→int32, plus the quad (VNNI/SDOT) GEMM. The checks are exact: GEMM by Freivalds' C·x = A·(B·x) in int64. These entries report GOP/s with no %roof, because the probes measure floating-point peaks (M6).

All sparse inputs are generated: 2D Poisson, unsymmetric convection–diffusion, scrambled copies of both, and a planted block-triangular matrix. Nothing is downloaded. The orderings have no work model; they're timed and checked, and reported as `n/a`.

### Coverage is enforced

`benchmarks/check_coverage.sh path/to/mtl5_bench` checks that the catalogue
covers the library, and CI runs it in the benchmarks lane (#721 M7).
`benchmarks/platform/coverage.txt` maps every header under `operation/`, the
ITL solver, preconditioner, smoother, multigrid and eigensolver directories,
the sparse factorization, ordering, analysis and eigen directories, and
`simd/algorithm.hpp` to one of:

| kind | meaning |
|---|---|
| `bench <entry>` | an entry calls the header's operation |
| `via <entry>` | the operation runs inside that entry (`lower_trisolve` inside `trsv`) |
| `class <entry>` | the same kernel class as that entry (`cos` against `sin`) |
| `exempt <reason>` | not measured: queries, predicates, I/O, umbrellas, plumbing |

The check fails on:
- a header with no line, which is the case it exists for;
- a line for a header that no longer exists;
- an entry name that matches nothing in `mtl5_bench --list`;
- an exemption with no reason.

So adding an operation, or renaming an entry, means saying how it is measured.

Factorization and eigen rates use the conventional LAPACK operation counts (2/3 n³ for LU, 22 n³ for a full SVD, …), so they compare implementations rather than count what a given algorithm executes. Each entry's check is described at the top of its `entries_*.hpp`. The tracking issue lists the remaining areas. To add an operation, write a `problem`
factory and register it in `platform/entries_*.hpp`. A factory supplies setup,
run, check, and the flop and byte counts. The runner does the timing, so every
entry is measured the same way.

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
(see the run contract in `docs/benchmarks/run-contract.md`); the machine state it
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
| `all` | every suite below except `gemm-rect` / `ewise` / `eig-small` / `null-vector` / `int` |
| `blas` | `l1` + `l2` + `l3` |
| `l1` | `dot`, `nrm2`, `axpy`, `scal` |
| `l2` | `gemv`, `ger`, `symv`, `trmv`, `trsv` |
| `l3` | `gemm`, `trmm`, `trsm`, `symm`, `syrk`, `syr2k` |
| `lapack` | `lu`, `qr`, `cholesky`, `eig` |
| `gemm-rect` | rectangular GEMM shapes: the BLIS multi-loop 2D grid (#297) |
| `ewise` | element-wise vector/matrix expression sweeps (#297) |
| `eig-small` | n = 3…16: `eigen_symmetric` vs `eigen_symmetric_jacobi` vs `svd`, with eigen/singular vectors (#564) |
| `null-vector` | 2k × 9 and 2k × 12, k = 8…200: `null_vector` vs full and thin `svd` vs `eigen_symmetric` / `eigen_symmetric_jacobi` on AᵀA (#566) |
| `int` | `int-dot` + `int-gemv` — the integer arms (#451 phase 4) |
| `int-dot` | fp64/fp32 baselines vs int32, int16→int32, int8→int32, uint8×int8→int32 |
| `int-gemv` | fp64/fp32 baselines vs int32 |

Any individual op above is also a suite name of its own (`--suite trsm`). The
authoritative list is always `bench_all --help`.

`gemm-rect`, `ewise`, `eig-small` and `null-vector` carry their own built-in shape sets and ignore
`--sizes` / `--sweep`.

`int`, `eig-small` and `null-vector` are deliberately **not** part of `all`: adding arms to the default run
would change every machine's committed CSV and break comparison with the
existing baselines. Ask for it by name.

## The nc-model sweep (#479)

**What it is for.** #429 wants to balance the jc partition (`balanced_nc`), and
#479 must choose between six ways of sizing `nc` before that can ship. Choosing
needs evidence from **at least three microarchitectures** — a model picked from
one machine is how the hardcoded Haswell defaults arrived in the first place.

Timing six models on four machines is a large session, and most of it would
measure nothing: for most shapes every model computes the *same* `nc`, so timing
them compares a binary against itself. This sweep finds, per machine, the shapes
where the models actually differ — so the expensive session only runs those.

**It measures nothing itself.** Pure enumeration over `nc_for_model` and
`plan_gemm_grid`. It takes seconds, needs no preflight, no pinning and no quiet
machine, and can be run while the box is busy.

### Running it

One profile per machine, carrying that host's ISA flag and thread count:

```bash
bash benchmarks/machines/xeon-e5-2420-nc-sweep.sh
bash benchmarks/machines/i7-12700k-nc-sweep.sh
bash benchmarks/machines/ryzen-9-8945hs-nc-sweep.sh     # WSL; needs GCC >= 12 for znver4
bash benchmarks/machines/jetson-orin-nano-nc-sweep.sh
```

Each refuses to run on a machine it does not recognise, and refuses to run on a
dirty tree — see the provenance note below. Commit **both** files per dtype, the
CSV and its `.sysinfo`; CI rejects a CSV without one.

### The one line to read

```text
m1_balanced     8 of 20   <- the pair #429 needs
```

That is how many shapes are worth *timing* on this machine. **`0 of 20` is a
result, not a failure** — it says this machine cannot separate M0 from M1 and
should not be booked for that question. The Xeon gives 8.

The summary's other line, "N of 20 shapes separate at least two models", is
**not** the number to read: it counts a shape if any of the six differ, and at
`--threads 1` it happily reports "5 of 5" while M1 equals M0 on every one.

### Why the runner is fussy about ordering

The sidecar is written by the **binary**, and reports what `build_info.hpp` said
when the binary was *compiled*. So the tree must be clean and the build must be a
full one **before** the run:

```text
clean tree  →  FULL build (not --target)  →  run  →  commit
```

Every way of getting this wrong is silent — the sidecar comes out well-formed,
internally consistent, and naming the wrong commit. It took four attempts by
hand, which is why `run_nc_sweep.sh` does the ordering for you and then
**verifies the stamp against `git rev-parse`** rather than trusting it. If it
reports a mismatch, do not commit the data.

## The nc-model timing session (#479)

The sweep above says **where** the models disagree. This says **what that is
worth**. Two scripts, and they are not interchangeable:

| | plans | measures | needs a quiet box | takes |
|---|---|---|---|---|
| `*-nc-sweep.sh` | ✅ | ❌ | no | seconds |
| `*-nc-bench.sh` | ❌ | ✅ | **yes** | tens of minutes |

**Run the sweep first.** If it says `m1_balanced 0 of 20`, the timing session
would spend an hour comparing byte-identical code. `run_nc_bench.sh` reads the
sweep's sidecar and refuses on `0` for that reason.

```bash
bash benchmarks/machines/xeon-e5-2420-nc-bench.sh
bash benchmarks/machines/i7-12700k-nc-bench.sh
bash benchmarks/machines/ryzen-9-8945hs-nc-bench.sh   # WSL; GCC >= 12 for znver4
bash benchmarks/machines/jetson-orin-nano-nc-bench.sh
```

All six arms run **in one process**, interleaved within each round, so they share
one thermal state and one warmed pool. Six processes would make the machine, not
the model, the thing under test. The nest takes the model as a per-call
parameter (`gemm_blocked(..., nc_model)`) precisely so this is possible;
`MTL5_NC_MODEL` sets the process-wide default for anything that does not pass one.

### The three lines to read

```text
Noise floor: 2.14%  (worst deviation among 24 arm(s) whose nc equalled M0's,
             i.e. byte-identical code)
Best M1-over-M0 gain where M1's nc actually differed: 8.90% (over 8 arm(s))
Worst first-round excess over the minimum: 11.2%
```

**The noise floor is measured, not assumed.** Those 24 arms chose the *same* `nc`
as the baseline, so they ran byte-identical code and every deviation from `1.0`
is this session's own noise. A gain that is not comfortably above the floor is a
measurement of the box. The binary says so itself rather than leaving it to be
noticed.

Three distinct verdicts, and the sidecar's `effect_above_noise` carries all
three — `1`, `0`, or `unmeasured`:

- **`noise_floor_arms=0`** — no arm matched M0's `nc`, so the floor is *unmeasured*,
  which is **not** the same as zero. Every ratio is an upper bound.
- **`m1_differing_arms=0`** — M1 never differed here; this machine cannot address
  #429 at this thread count.
- **gain ≤ floor** — the effect is below what this session can see.

The first of those is a real trap and cost a rewrite here: the natural definition
of a control — *a shape where every model agrees* — never fires on a machine
whose detected L3 differs from the compile-time figure, because M2 and M4 then
always move. The control count was zero, the spread printed `0.00%`, and "is the
effect above the noise" became a comparison against nothing that no run could
fail. It read like a clean result. The control is therefore **per arm**, not per
shape.

`Worst first-round excess` large (say >25%) means the minimum has not converged —
raise `--rounds`. Timings are the **min** across rounds, not the mean: every
perturbation available on a shared box adds time and none removes it, so the mean
estimates the interference as much as the kernel.

### What the harness checks for you

- **Every arm's result is compared bit-for-bit against M0.** `nc` regroups
  columns; it must not reorder any C element's FMA chain. A mismatch is a defect
  in the blocking, not a tolerance question — the binary refuses to write a CSV.
- **`--threads 1` is rejected by the runner** (`FORCE_T1=1` overrides); the
  binary accepts it and prints a warning, because a deliberate single-thread
  control run is a legitimate thing to ask a binary for. At `jc_nt == 1` every
  balancing model is a no-op *by construction*, so it cannot address #429.
- **The sweep is a prerequisite, checked per dtype.** `run_nc_bench.sh` refuses
  when a requested dtype has no sweep, or when its sweep reports `0 of 20`.
  float and double have different blocking, hence different discrimination.
- **`MTL5_NC_MODEL` is rejected by `bench_blocking_ab`**, which does not vary the
  model: it would move the jc partition under every arm's label, two variables
  under one name.

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

### Every committed CSV needs its sidecar

`benchmarks/check_sidecars.sh` fails if a CSV under `benchmarks/data/` has no
`.sysinfo` beside it, and CI runs it on every push. A result without provenance
cannot be traced to a commit, a build configuration or a machine state — which is
what makes it checkable a year later — and the sidecar is written by the
**binary**, so a producer that was never taught to write one keeps emitting
untraceable CSVs. Twenty-eight such files reached the repository before anyone
noticed (#477), and they were found by auditing an open issue rather than by
anything failing.

It is a **ratchet, not a gate on history**: those files are listed in
`benchmarks/data/.sidecar-exempt` and cost nothing, but adding another one fails.
The list only ever shrinks — re-run the producer under the contract, commit the
CSV together with its sidecar, and delete the line. The checker also fails when
an exempt file has *gained* a sidecar, so a stale exemption cannot sit there as a
permanent amnesty.

Run it locally before committing benchmark data:

```bash
benchmarks/check_sidecars.sh
```

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
