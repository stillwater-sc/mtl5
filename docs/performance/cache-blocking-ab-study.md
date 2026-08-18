# Cache blocking A/B: does the detected hierarchy beat the compile-time model?

MTL5 derives its GEMM blocking parameters — `kc`, `mc`, `nc` — from an analytical
model of a Haswell-class core. Runtime cache detection (#222/#426) replaces those
constants with the machine's own L1 and L2. The obvious expectation is that real
cache sizes beat hardcoded ones.

They do not. This page is the experiment that established it, the hypothesis that
came out of it, and the follow-up experiment that **refuted that hypothesis and
inverted it**.

## What the parameters do

| parameter | derived from | governs |
|---|---|---|
| `kc` | L1d size | depth of the packed B micro-panel |
| `mc` | L2 size, *given* `kc` | rows of A in one packed block |
| `nc` | L3 size | width of the packed B panel |

`nc` is deliberately **not** detected: it sets the jc block count, and the thread
partition is sensitive to it (#429). So detection moves `kc` and `mc` only — and
because `mc ≈ L2 / (2·kc·sizeof)`, raising `kc` *lowers* `mc`. The two are not
independent, which is what made the first round of analysis go wrong.

## Design

Four arms, one source, differing only in which detected level feeds the model:

| arm | detects | kc | mc |
|---|---|---|---|
| `default` | nothing | model | model |
| `detected` | L1 + L2 | detected | detected |
| `kconly` | L1 | detected | model L2, but *given* the detected kc |
| `mconly` | L2 | model | detected |

Protocol (the [run contract](../benchmarks/systems.md)): pinned to one logical id
per physical core, all arms **interleaved within one session** with the order
rotated per round, min-of-N per point, per-machine output directory, and machine
state captured in every sidecar. Shapes are derived once from the first arm and
handed to all of them, so no two arms are ever compared on different shapes.

### The controls are the point

Two of the arms collapse onto others on particular hardware, and that is a
feature. On Zen 4 the L1d is 32 KB — identical to the model — so `kconly`
compiles to *exactly* `default`. On the Jetson the L2 matches, so `mconly`
compiles to `default`, and `detected` compiles to `kconly`.

Those pairs are **null comparisons between identical binaries**. Whatever they
measure is the session, not the code, and that number is the floor under
everything else the session can claim.

## Hypothesis

From the first (two-arm) round, across four machines:

> **H₁(kc):** the throughput loss under detection is caused by `kc`; changes to
> `mc` are harmless.

It came from a cross-machine inference: the Zen 4 run moved `mc` alone and cost
nothing single-threaded, while both machines whose `kc` moved lost throughput.
The `kconly` / `mconly` arms exist to test it on one machine at a time.

**H₀:** `kc` and `mc` are interchangeable — neither is the seat of the loss.

## Method

Each arm yields one throughput per (shape, thread count); the paired quantity is
the log-ratio against the baseline arm on the same point. With 5 shapes × 2
thread counts there are n = 10 paired observations per arm per machine.

Two exact tests, no distributional assumptions:

- **sign test** — exact binomial on the direction of each pair
- **sign-flip permutation test** — exact over all 2ⁿ sign assignments of the
  paired log-ratios, so it weighs magnitude as well as direction

The permutation test is the one to read. The sign test ignores magnitude, and on
this data it rejects H₀ for a comparison of two *identical binaries* — a 0.3%
systematic offset in a consistent direction. That is a real effect and a useless
one, and it is exactly why the null-run controls are run.

**Power.** With n = 5 (one thread count on one machine) the smallest attainable
two-sided p is 2/2⁵ = **0.0625**, reached when all five points agree in sign. Any
per-regime result quoted at 0.0625 below is at the floor of what the design can
resolve; pooling to n = 10 is what buys p < 0.05. This is a limitation of five
shapes, not a weak effect.

## Calibration: what the identical-binary pairs measure

| machine | pair | median | range | p (perm) |
|---|---|---|---|---|
| Zen 4 | `kconly` vs `default` | 0.997 | 0.984 – 1.018 | 0.41 |
| Jetson | `mconly` vs `default` | 1.003 | 0.986 – 1.015 | 0.63 |
| Zen 4 | `detected` vs `mconly` | 1.012 | 0.974 – 1.033 | 0.17 |
| Jetson | **`detected` vs `kconly`** | **0.965** | **0.904 – 1.006** | 0.0625 |

Three of the four are clean, and they put the practical resolution of a healthy
session at about **±2%** — which is where the analyzer's noise floor already sat.

The fourth is not clean, and it disqualifies its session. See *Data quality*.

## Results

Ratios are arm ÷ `default`; below 1.0 means detection lost. Median over 5 shapes,
with the same-sign count and the exact permutation p.

### Single-threaded — the clean regime, no thread partition involved

| machine | arm | kc, mc | median | range | same-sign | p |
|---|---|---|---|---|---|---|
| i7-12700K | `kconly` | 384, 42 | **0.999** | 0.966 – 1.029 | 3/5 | 0.69 |
| i7-12700K | `mconly` | 256, 320 | **0.919** | 0.862 – 0.970 | 5/5 | 0.0625 |
| Zen 4 | `kconly` | *(null)* | 0.996 | 0.984 – 0.998 | 5/5 | 0.0625 |
| Zen 4 | `mconly` | 256, 256 | 1.015 | 0.982 – 1.055 | 4/5 | 0.38 |

### Multi-threaded (T = 8)

| machine | arm | median | range | same-sign | p |
|---|---|---|---|---|---|
| i7-12700K | `kconly` | 0.935 | 0.919 – 0.973 | 5/5 | 0.0625 |
| i7-12700K | `mconly` | 0.901 | 0.799 – 1.001 | 4/5 | 0.13 |
| i7-12700K | `detected` | 0.934 | **0.715** – 0.998 | 5/5 | 0.0625 |
| Zen 4 | `kconly` | 0.997 *(null)* | 0.994 – 1.018 | 4/5 | 1.00 |
| Zen 4 | `mconly` | 0.939 | 0.858 – 0.987 | 5/5 | 0.0625 |

### The contrast that answers the question

Comparing the two arms **directly**, paired on the same points, removes the
baseline entirely:

| machine | regime | median `kconly` ÷ `mconly` | range | p (perm) |
|---|---|---|---|---|
| i7-12700K | pooled, n=10 | **1.074** | 0.932 – 1.187 | **0.027** |
| i7-12700K | T=1, n=5 | 1.076 | 1.061 – 1.162 | 0.0625 (5/5) |
| Zen 4 | T=8, n=5 | 1.058 | 1.010 – 1.163 | 0.0625 (5/5) |

## Verdict: H₁(kc) is refuted, and the direction reverses

Single-threaded on the i7 — where no thread partition can confound it — raising
`kc` by 50% is **free** (median 0.999), while raising `mc` costs **8%** (median
0.919, all five shapes in the same direction). The hypothesis had it backwards.

The reason the first round got it wrong is instructive: it inferred causation
from the `detected` arm, where *both* parameters move, and exonerated `mc` using
the Zen 4 machine — where `mc` genuinely is free single-threaded (1.015). One
machine's behaviour was generalised to a rule. The i7 contradicts it.

**But `mc` is not the whole story at T = 8.** There the i7 loses ~6% under
`kconly` as well (median 0.935, 5/5), even though `kconly` at that point runs a
*smaller* `mc` than default. So multi-threading adds a second effect that
single-thread measurements cannot see — the packed-B panel `kc × nc` is shared
across a jc team, and `kc` sizes it. The honest summary:

- **`mc` upward is the largest single harm, and it is visible with one thread.**
- **`kc` is free serially and not free in parallel.**
- **Neither is ever a win.** Across all machines and arms, no configuration beats
  `default` by more than the noise floor.

The operational conclusion for #426 is therefore unchanged, and now for a
measured reason rather than an inferred one: **cache detection stays opt-in and
off by default.**

## Evidence that the mechanism is `mc` and not the shape

The `mc_used` column (#448) records the block size the loops actually stepped
by, after the L2 bound, the thread-budget cap and the even-partition round-off.
Within the *single* i7 `mconly` arm — one binary, one session:

| shape, T | mc used, vs default | ratio |
|---|---|---|
| 256×8192, T=8 | **32** (½×) | 1.001 |
| 256×12288, T=8 | **32** (½×) | 0.987 |
| 1024³, T=8 | 128 (2×) | 0.799 |
| 2048², 4096², T=8 | 256 (4×) | 0.856, 0.901 |
| every shape, T=1 | 318 (5×) | 0.862 – 0.970 |

Every point where `mc` actually grew lost; the two where the partition clamped it
*below* the default lost nothing. Same binary, same arm, same session — the
effect tracks the parameter, not the shape.

## Data quality: the Jetson session is disqualified

Its `detected` and `kconly` arms are the **same configuration** (kc = 1024,
mc = 16 — on that machine `mc` moves only because `kc` does), yet they differ by
a median of 3.9%, reaching 10%. Per-round throughput at 2048³, T=1:

```
default   6.80  5.88  6.09  5.89  6.83
kconly    6.70  5.78  6.01  6.00  6.69
detected  6.10  6.07  6.00  6.08  5.91     <- never caught a fast round
```

Within-arm spread across rounds, median over all points:

| machine | median | p90 | worst |
|---|---|---|---|
| i7-12700K | 1.8% | 4.8% | 5.8% |
| Zen 4 | 2.5% | 7.6% | 32.6% |
| Jetson (15 W) | **12.8%** | 16.2% | 16.5% |

The machine is moving by more than the effect. That is the 15 W power mode with
`schedutil` and unpinned clocks; the run needs `nvpmodel -m 0 && jetson_clocks`
before its numbers mean anything. **No Jetson result is quoted in the verdict
above.**

## Next: the model ignores the C strip

Why should a *larger* `mc` hurt when the model sizes it to fill half of L2? The
model counts the packed A block, `mc × kc`, and nothing else. But the same cache
holds the C strip being accumulated, which is `mc × nc`:

| i7, per ic block | A block `mc·kc` | C strip `mc·nc` | L2 |
|---|---|---|---|
| `default`, mc=60 | 123 KB | 1.97 MB | 1.25 MB |
| `mconly`, mc=318 | 651 KB | **10.4 MB** | 1.25 MB |

C dominates and is absent from the derivation.

**The obvious repair does not survive its own calibration point.** Adding the C
term to the *compile-time* model forces the strip to be `mc × nc`, since `n` is
unknown there, and reproducing the shipped `mc = 64` then needs

```
mc * (kc + nc) * sizeof  <=  8.5 * L2
```

which is not a residency statement at all — and the `nr = 4` hardware family
needs 2.5 rather than 8.5, so no constant repairs it.

The reason is that the strip is `mc × min(n, nc)`, and **`n` is known at the
call** — which is already where `mc` is capped for the thread budget. So the
bound belongs in the runtime planner, not in the model:

```
mc  <=  L2 / ((kc + min(n, nc)) * sizeof)
```

On the i7 that yields **128** at `n = 1024` (the shipped `mc = 64` untouched),
**71** at `n = 2048`, and **37** at `n = 4096` — tightening exactly where the
measurements preferred a smaller `mc`, and leaving alone where they did not. It
is also nearly independent of `kc`, where the current model is inversely
proportional to it; that difference in *shape* is the thing to measure.

Implemented as `detail::c_strip_mc_cap` and the **`ccap` arm** (#453): L2
detection plus the cap, so `ccap` vs `mconly` isolates the bound and `ccap` vs
`default` answers the shipped question.

### First result: the effect is large, and A does not belong in the budget

Xeon E5-2420 v2, 6 physical cores, 3 arms × 6 rounds in one session. The control
holds — `mconly` is a null run here (L2 already matches the model), median 1.001,
range [0.992, 1.006] — so the machine resolved to about ±0.7%.

| shape | T | ratio `ccap`/`default` | mc default → ccap |
|---|---|---|---|
| 2048² | 6 | **1.151** | 32 → 12 |
| 4096² | 6 | **1.158** | 32 → 12 |
| 1024² | 6 | 0.854 | 29 → 19 |
| 96×4096 | 6 | 0.832 | 16 → 8 |
| 96×6144 | 6 | 0.831 | 16 → 8 |
| all shapes | 1 | 0.993 – 1.005 | 30 → 12 |

Two things fall out. **Single-threaded, mc barely matters on this machine** —
cutting it from 30 to 12 costs nothing — which is the mirror image of the i7,
where *raising* it cost 3–14%. Small mc is safe; large mc is not.

And the multi-threaded split has a clean explanation:

| shape | mc | A = mc·kc | C = mc·min(n,nc) | C fits 256 KB L2? | ratio |
|---|---|---|---|---|---|
| 2048² | 32 | 128 KB | 512 KB | **no** | 1.151 |
| 4096² | 32 | 128 KB | 512 KB | **no** | 1.158 |
| 1024² | 29 | 116 KB | 232 KB | yes | 0.854 |
| 96×4096 | 16 | 64 KB | 256 KB | yes | 0.832 |

The bound fires whenever **A + C** exceeds L2, but every win is a shape where
**C alone** exceeds it, and every loss is a shape where C already fitted and the
bound shrank mc for nothing. So A does not belong in the budget: it is streamed
into the packed buffer and consumed once per jc block, while C is
read-modify-written across the whole kc loop and is the operand that must stay
resident.

### The `ccap2` arm, and what it predicts

`ccap2` charges the C strip alone. On this machine it leaves mc at the default
exactly where `ccap` lost — 1024² keeps 29, 96×4096 keeps 16 — while still
cutting 32 → 16 where `ccap` won. So it predicts, falsifiably: **the three losses
disappear and both wins survive.** If they do not, the C-strip story is wrong and
the wins have another cause.

```bash
ARMS="default ccap ccap2" ROUNDS=6 benchmarks/machines/i7-12700k.sh
```

### The prediction was half right, which killed the hypothesis

Xeon, `default` / `ccap` / `ccap2`, 6 rounds. `ccap` reproduced its earlier
session to within 0.5% — an independent replication of both the wins and the
losses.

| shape | T | `ccap` | `ccap2` | predicted | mc: def / ccap / ccap2 |
|---|---|---|---|---|---|
| 96×4096 | 6 | 0.832 | **0.993** | loss gone ✓ | 16 / 8 / 16 |
| 96×6144 | 6 | 0.836 | **0.999** | loss gone ✓ | 16 / 8 / 16 |
| 1024² | 6 | 0.856 | **1.001** | loss gone ✓ | 29 / 19 / 29 |
| 2048² | 6 | 1.152 | **1.012** | win survives ✗ | 32 / 12 / 16 |
| 4096² | 6 | 1.159 | **1.025** | win survives ✗ | 32 / 12 / 16 |

Charging A was indeed what caused the losses — all three vanish. But charging C
alone keeps only 1.2% and 2.5% of a 15% win. So the win is **not** "C fits in
L2": at mc=16 the strip is exactly 256 KB (100% of L2) and captures almost
nothing, while mc=12 (77%) captures all of it. And at 1024² the strip at the
untouched mc=29 is 237 KB — 93% of L2 — and leaving it alone is best. No
residency threshold produces both.

## What the reference implementations do

Haswell, double precision:

| | mr×nr | mc | kc | nc | A block | vs 256 KB L2 |
|---|---|---|---|---|---|---|
| BLIS | 6×8 | 72 | 256 | 4080 | 147 KB | 57% |
| OpenBLAS | 4×8 | **512** | 256 | **13824** | **1024 KB** | **400%** |
| MTL5 (AVX2) | 6×8 | 64 | 256 | 4096 | 131 KB | 51% |
| MTL5 (Xeon, nr=4) | 6×4 | 32 | **512** | 2048 | 131 KB | 51% |

**OpenBLAS does not target L2 for the A block at all** — 4× over on Haswell,
Sandy Bridge and Nehalem, 2× on Zen. Only SkylakeX, whose L2 is 1 MB, has A
inside L2 (589 KB, 58%). Our model and BLIS's assume L2 residency unconditionally;
the most-used BLAS in the world does not.

Both references use **kc = 256** for double, where our Xeon build derives 512 —
because `kc = (L1/2)/(nr·sizeof)` charges the **B micro-panel only**, so a
narrower `nr` inflates it. That was worth measuring rather than "fixing".

## The response curve, measured

2048³, best of 3 rounds, order rotated per round, `mc` overridden directly
(`bench_blocking_sweep`, an opt-in diagnostic build):

| mc | kc=256, T=1 | kc=512, T=1 | kc=256, T=6 | kc=512, T=6 |
|---|---|---|---|---|
| 12 | 9.61 | **9.74** | 50.27 | **51.24** |
| 24 | 9.58 | 9.65 | 48.91 | 50.25 |
| **32** *(shipped)* | 9.50 | 9.54 | 45.11 | **45.35** |
| 48 | 9.22 | 9.60 | 44.11 | 45.44 |
| 64 | 9.32 | 9.62 | 46.19 | 47.30 |
| 96 | 9.29 | 9.59 | 46.03 | 48.84 |
| 144 | 9.14 | 9.41 | 48.87 | 50.50 |
| 256 | 9.12 | 9.40 | 46.45 | 47.88 |

Three results, none of which either candidate model predicted:

1. **The curve is W-shaped, and the shipped `mc = 32` sits in the trough.** It
   reaches 88.5% of the best at T=6; both 12 and 144 beat it by ~12%. The two
   `kc` columns show the same shape independently, which is a replication.
2. **`mc` is a threading effect, not a residency one.** The spread across mc is
   **3.6% at T=1** and **13.0% at T=6**. A cache-residency story cannot be four
   times stronger with six threads on the same footprints.
3. **Our kc = 512 beats the reference kc = 256** on this machine at nearly every
   mc — so the "our kc is twice everyone else's" observation, which looked like a
   defect, is not one here.

So #453's premise does not survive: the L2 model is not what makes `mc` matter.
The `ccap` win was real and reproducible, but it came from moving off a bad
operating point, not from making C fit. The next question belongs to the
partition (#429, #412), not to the cache model.

## What this experiment changed in the harness

Each of these came from a defect this study found in its own data:

| change | why |
|---|---|
| `mc_used`, `nib`, `njb`, `ic_nt`, `jc_nt` per point (#448) | the CSV recorded the *configured* `mc`, which no loop ever used |
| four selectable arms, rotated per round (#449) | one session, so arms are comparable; rotation cancels position effects |
| `ROUNDS` rounded up to a multiple of the arm count | 5 rounds over 4 arms leaves one arm leading twice — the bias the rotation exists to remove |
| analyzer fails identical-parameter arms that disagree | the check that disqualified the Jetson, now automatic |
| `build_isa` in the sidecar | the first Zen 4 run was AVX2 when AVX-512 was intended, and nothing recorded it |

## Reproducing

```bash
benchmarks/machines/i7-12700k.sh            # or jetson-orin-nano.sh
```
```powershell
pwsh benchmarks/machines/ryzen-9-8945hs.ps1
```

The profiles carry each machine's pin list, thread counts and ISA flag, and
refuse to run on a machine they do not recognise. Then compare each arm against
the baseline, and run the consistency checks the runner prints — those are the
ones that decide whether the session is worth reading at all.
