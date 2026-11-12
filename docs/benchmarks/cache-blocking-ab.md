# Cache-blocking A/B across the machines (#430)

Cache detection (`MTL5_ENABLE_CACHE_DETECTION`) derives the GEMM blocking
parameters from the cache sizes the machine reports, instead of the compile-time
defaults. This page holds the A/B experiment that decides whether it should: the
four-machine result, each machine's own session, and how to run it on a new
machine. The follow-up experiment that showed the effect is `mc`, not `kc`, is
the [cache-blocking study](../performance/cache-blocking-ab-study.md). Each
machine's hardware, caches and pinning policy are on the
[benchmark systems](systems.md) page.

## The four-machine result

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
tells the arms apart. The results are on the [register-tile sweep](register-tile-sweep.md) page;
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

## Per machine

### i7-12700K: detection loses here

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

### Jetson Orin: neutral, and the grid explains the one loss

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

### Arduino Ventuno Q: a null run, because there was nothing to detect

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
