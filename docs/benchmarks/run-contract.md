# Run contract and methodology

What every benchmark harness in `benchmarks/` must record and enforce, and how
the backends are measured. The reasoning behind the rules, and the failures that
produced them, are in the [benchmarking methodology](../performance/benchmarking-methodology.md).

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
| **Preflight** | `tree_git_commit` `tree_git_dirty` `competing_load` `idle_build_processes` `loadavg_1m` `cpu_online` `cpu_affinity` `governor` `turbo` `power_mode` `thermal_before_c` `thermal_limit_c` `thermal_headroom_c` | What the machine was doing |
| The **harness** | `binary_stale` `thermal_after_c` `harness` `harness_rounds` `harness_reps` | Whether the run was self-consistent |

`competing_load` lists the build tools and benchmarks that were running, and
fails the gate. A build tool counts only if it used more than 50 ms of CPU in a
one-second sample (#756). An idle one, such as a reusable MSBuild node left by an
earlier build or an open Visual Studio, is listed in `idle_build_processes` and
doesn't fail it. On macOS, which has no `/proc` to sample, every match counts,
and `idle_build_processes` is `unavailable`.

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

Governor is recorded rather than enforced because **every** machine on the [systems](systems.md) page
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
