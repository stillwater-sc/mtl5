# Benchmarking methodology, and the hypothesis register

This is the single statement of **how MTL5 decides that a performance change is
real**, and a register of every hypothesis this project has tested, with its
verdict.

It exists because the per-study documents
([cache-blocking A/B](cache-blocking-ab-study.md),
[multicore scaling](multicore-scaling-investigation.md),
[blocking and scheduling](blocking-and-scheduling-assessment.md)) each restate
the method in their own terms, and because the same three mistakes kept
recurring across them. The method below is mostly a list of those mistakes with
the countermeasure attached.

> **Scope.** This covers the *kernel* performance work — GEMM blocking, thread
> grids, SIMD micro-kernels. The sparse solver scoreboards
> (`bench_klu`, `bench_superlu`) compare against vendor libraries and use a
> different, simpler standard: wall-clock against a reference implementation on
> committed matrices.

---

## 1. What counts as evidence

### There is no significance test here, and pretending otherwise would be worse

MTL5 does **not** compute p-values, confidence intervals, or t-tests on
benchmark results. Two reasons, and they are honest limitations rather than
preferences:

1. **The samples are not IID.** Timings on a shared machine are contaminated by
   DVFS, scheduler migration, thermal drift and neighbours. Those perturbations
   are serially correlated, so the independence assumption behind a t-test does
   not hold and its p-value would be a fabricated precision.
2. **The distribution is one-sided.** Every perturbation available *adds* time;
   none removes it. The quantity of interest is the machine's capability at a
   given blocking, which is the **left edge** of the distribution, not its
   centre.

So the standard is:

| | |
|---|---|
| **Statistic** | minimum across rounds, per arm |
| **Comparison** | ratio to the baseline arm, same shape, same process |
| **Threshold** | the session's own **measured noise floor** |
| **Mechanism** | a recorded mediator, correlated with the effect |

A result is reported as real when the ratio clears the noise floor **and** a
recorded mediator moves with it. A ratio that clears the floor with no mechanism
is reported as an observation, not a finding.

### The noise floor is measured, not assumed

Every session contains **control arms**: arms whose blocking parameters came out
identical to the baseline's, so they are running **byte-identical code**. Their
deviation from 1.0 is, by construction, pure measurement noise.

```
Noise floor: 1.48%  (worst deviation among 24 arm(s) whose nc equalled M0's)
```

This is the single most important number in a session. Observed floors so far:

| session | floor | control arms |
|---|---|---|
| Xeon E5-2420 v2, `ondemand` | **1.48%** | 24 |
| i7-12700K, `powersave` | 4.31–7.40% | 36 |
| Ryzen 9 8945HS, WSL, unpinned | **17.06%** | 34 |
| Ryzen 9 8945HS, WSL, pinned governor | **7.49%** | 34 |
| Jetson Orin Nano, `schedutil` | 3.42–5.57% | 12 |

The Ryzen line is the argument for this whole section: at a 17% floor an
0.85 ratio is *unresolvable*, and the same arm at a 7.5% floor is a **confirmed
18% regression**. The conclusion changed because the floor changed, not because
the code did.

**A floor of `0.00%` from zero control arms is not a clean result** — see
§4, "controls that cannot fail".

### Bit-identity is a precondition, not a result

Blocking parameters change *how* work is grouped, never *what* is computed: the
FMA order for a given C element is fixed by the `pc` loop. So every arm must
produce a **byte-identical** result to the baseline before any of its timings are
reported.

Compared **element-wise**, not by checksum. A 64-bit fold can collide, and a
collision would admit a wrong arm's timings. `bench_nc_models` refuses to write a
CSV if any arm disagrees.

---

## 2. The run contract

Every measurement is subject to the provenance contract established in #442 and
enforced by `benchmarks/check_sidecars.sh` in CI.

**Every committed CSV has a `.sysinfo` sidecar** recording:

- *Build half* (written by the binary): `git_commit`, `git_dirty`, `cxx_flags`,
  `cmake_build_type`, the derived `mr/nr/kc/mc/nc`, detected cache sizes
- *Machine half* (written by the runner via `preflight.sh`): governor, turbo,
  thermal headroom before the run, competing load, CPU affinity

The runner enforces the ordering — `clean tree → preflight → FULL build → run →
verify the stamp against git rev-parse` — because every way of getting it wrong
produces a sidecar that is well-formed, internally consistent, and **wrong**.
A `--target` build silently leaves the previous commit's stamp in place.

**Known gap:** on WSL the governor reads as `unavailable`, so a pinned-governor
run there records its *effect* (a lower noise floor) but not the change itself.

See [systems.md](../benchmarks/systems.md) for per-machine topology and pinning
policy, and `benchmarks/README.md` for the commands.

---

## 3. Design rules

### One variable per ratio

Twice in this project a ratio was labelled as measuring one thing while two
things moved. Both times the reported figure was wrong by roughly 2×.

If two arms differ in more than one parameter, the ratio measures the pair, and
the write-up must say so or net the confound out against a control.

### Shapes are searched, not listed — and not derived from algebra either

A fixed shape list silently tests nothing: every measurement in #426 was square,
which is the one regime where `jc_nt` is structurally 1 and the effect under
study cannot appear.

But a list *computed* from a design note's algebra fails the same way. #430's
note said jc parallelism needs `m ≤ mc·T/2`; deriving shapes that way produced
`jc_nt == 1` on every one, because `plan_gemm_grid` caps `mc` at `ceil(m/budget)`
(#441) and prefers larger `ic_nt` on ties. Actual jc parallelism appears near
`m ≈ mr·T` — **m = 6, not 96**.

So the shape list is produced by **asking `plan_gemm_grid`** which shapes yield
the regime under study (`benchmarks/nc_shapes.hpp`), and the same header is
shared by the planner and the timing harness so they cannot drift.

### Plan offline, measure only what discriminates

Six models × 20 shapes × 4 machines is mostly wasted machine time: on most
shapes every model computes the same parameter, so timing them compares a binary
against itself. `sweep_nc_models` enumerates the disagreement from **pure
functions** in milliseconds, and the timing session runs only the shapes that
discriminate.

`m1_balanced  0 of 20` is a **result**: that machine cannot answer the question
and should not be booked for it.

### Arms interleave in one process

All arms of a comparison run back-to-back in a single process, **counterbalanced
by round** (arm order rotates), sharing one thermal state and one warmed pool.
Separate processes would put arms on different machine states and make the
machine the thing under test.

Order matters because the statistic is a per-arm minimum: a fixed order gives
every arm the same position every round, so a position effect never averages out
— it becomes an arm effect.

### Record the mediator, not just the outcome

Throughput alone can say a model was faster; it cannot say *why*. Every point
records the quantities the hypothesis is about:

- `ic_imbalance`, `jc_imbalance` — `ceil(blocks/workers)·workers/blocks`
- `packedB_bytes` — the resident packed-B working set
- `mc`, `nc`, `nib`, `njb`, `ic_nt`, `jc_nt` — **as the nest actually stepped
  them**, not as configured

That last point is not pedantry. #430's committed data records `mc=213` for runs
whose nest stepped 210 serially and 128 on eight threads, which left the
parameter under test unrecoverable from its own record.

### Negative controls are part of the design

Every shape list includes shapes where the effect **cannot** appear — square
(no jc partition), tall/thin (ic-dominated), `T=1` (every balancing model is a
no-op by construction). They are recorded so their absence is never assumed.

---

## 4. Failure modes this project has actually hit

Each of these produced a wrong or unfalsifiable result at least once.

### Controls that cannot fail

The `nc`-model harness first defined a control as *a shape where every model
agrees*. On every machine whose detected L3 differs from the compile-time
figure, no shape qualifies — so the control count was **zero**, the spread
printed `0.00%`, and "is the effect above the noise" became a comparison against
nothing that no run could fail. **It read like a clean result.**

The control is now **per arm**: any arm whose parameter equals the baseline's is
running identical code, whatever the others did.

**Countermeasure:** report the control *count* beside the floor, and emit
`effect_above_noise=unmeasured` — three states, because a boolean cannot
distinguish "no effect" from "no measurement".

### Two arms that were secretly one arm

In #470 the GEMM kernel was selected by *inferring* from element types, and
`(i8,i8)` is valid input to both kernels — so the "control" arm and the "quad"
arm had become the same kernel. A full green test suite did not notice, because
the two kernels agree bit-for-bit. It was caught only when the two arms reported
the same number to three digits.

**Countermeasure:** selection is an explicit argument, and a **structural** test
compares function addresses. Verified by sabotage: making the selector ignore its
argument fails the structural test while the bit-identity test still passes.

### Verifications that could not fail

Four instances, including: a reproduction run against a profile that did not
exist on the branch, where the command failed silently under a redirect and the
diff compared each file against its own unchanged self and printed `IDENTICAL`.

**Countermeasure:** before trusting a check, ask *what would make this fail?* and
where feasible, break the thing deliberately and confirm the check fires. Every
guard in the `nc` work was verified this way, and it caught two defects in the
#488 fix that reasoning had missed.

### Single-machine generalisation

#426 shipped runtime cache detection on the argument that measured sizes must
beat constants tuned for a Haswell core. It lost on every machine that ran it.

**Countermeasure:** a parameter change that affects blocking requires evidence
from **≥3 microarchitectures** before becoming a default.

### Fitting the guard to the losses

The first candidate guard for #429 — "decline when the thread grid changes" —
separated 44 arms perfectly. But three of its four target arms were a *different*
defect (#488), and once that was fixed the same guard would have declined two
real gains to prevent one regression.

**Countermeasure:** prefer a predicate that states the **mechanism** over one
that correlates with the outcome, and re-check any fitted rule after fixing
anything it might be a proxy for.

---

## 5. Hypothesis register

Status: **✅ validated** · **❌ falsified** · **⚠️ partial** · **🔬 open**

| # | Hypothesis | Status | Evidence |
|---|---|---|---|
| #426 | Detected cache sizes beat compile-time constants | ❌ | Regressed on every machine; reverted to a per-level opt-in |
| #426 | `kc` from detected L1 improves throughput (H₁) | ❌ | Refuted, direction reversed; see [cache-blocking A/B](cache-blocking-ab-study.md) |
| #408 | `mc` must balance the ic partition | ✅ | `nib` 16→18 cost a 1.41× critical path, turning +21.5% into −7.4% |
| #441 | `mc` must also be capped so `nib` fills the thread budget | ✅ | i7 ran 5 of 8 threads (0.551), Zen 4 4 of 8 (0.590) |
| #441 | Greedy ic-first grid factorisation wastes cores | ✅ | Jetson 0.760 where the grid was the only difference |
| #453 | Add the C strip to the compile-time `mc` model | ❌ | Falsified by its own calibration point; needs 8.5·L2 on one family, 2.5 on another |
| #453 | Bound `mc` at runtime by the C strip, charging A | ❌ | Every win was a shape where C alone exceeded L2; A is streamed, not resident |
| #451 | A VNNI/SDOT quad micro-kernel beats widen-then-FMA | ✅ | 1.25–3.64× depending on machine; kernel *shape* alone worth 1.25–3.64× with no new silicon |
| #451 | `has_native_quad_dot` is a single machine-wide property | ❌ | Per-*pairing*, and the ISAs mirror each other: x86 does `u8×i8`, ARM does the symmetric pairs |
| #429 | `balanced_nc` evens the jc partition and wins | ⚠️ | Wins on 42/44 arms, median ×1.161 — **but only with the packed-B guard**; plain M1 loses 18% on Zen 4 |
| #429 | Then apply the detected L3 to `nc` | ❌ | **Up to 45% slower** (×0.548), 37 regressions — and it already includes `balanced_nc`, so balancing does not rescue it |
| #426/#429 | Per-team L3 budget "models the wrong cause" | ❌ | **Reversed by measurement.** M4's speedup does *not* track imbalance (r=0.29), does track `nc` reduction. Capacity was the dominant cause on those shapes |
| #492 | The per-sharer budget can be confined to where it wins | ✅ | **`jc_nt >= 2`, read from the baseline grid, separates the classes exactly** (105/105 admitted, 0/35 controls). Pre-registered, then confirmed |
| #492 | The confined model beats the shipped default | ✅ | **Adopted as the default.** vs M6: Xeon +74%, i7 +108–116%, Ryzen +73–76%, Jetson +15–22%. 0 arms worse, 0 regressions, 40/40 controls unchanged, 1280/1280 bit-identical |
| #429/#492 | Balancing the jc partition is the main effect | ❌ | Real but **second-order**: M6 is ×1.16 where M7 is ×1.7–2.2. `nc` being too *large* dominates being *ragged* |
| #479 | The jc imbalance metric mediates the throughput change | ⚠️ | True for M1 (r=0.957 Xeon, 0.71 pooled) but it recovers only ~40% of the theoretical saving; **not** true for M4 (r=0.29) |
| #488 | `balanced_mc`'s rounding buys real balance | ❌ | Padded critical path **identical** in all 383,934 changed cases; it bought nothing and cost 2.28× total work |
| #486 | `nc` may be sized from the accumulator type | ❌ | The packed-B panel is in *operand* precision; overstated 2× for fp32→fp64, 4× for i8→i32. **Accounting corrected**, cross-checked against the packers |
| #486 | Enlarging `nc` to match the true panel is worth throughput | 🔬 | **Unmeasured, and it points the wrong way.** It would enlarge `nc` 2–4× — the direction M2 was falsified in. Accounting fixed; **budget policy deliberately unchanged** |
| #429 | The packed-B/L3 guard generalises | 🔬 | Exact on 44 arms, but both confirmed regressions are **one machine**. A host with L3 between 16 and 25 MB would test it properly |

---

## 6. Reproducing a study

```bash
# 1. Plan — pure enumeration, seconds, machine may be busy
bash benchmarks/machines/<machine>-nc-sweep.sh

#    Read one line:  m1_balanced  N of 20
#    N = 0 means this machine cannot discriminate. That is a result.

# 2. Measure — needs a quiet box, tens of minutes
sudo cpupower frequency-set -g performance      # if available
bash benchmarks/machines/<machine>-nc-bench.sh --dtypes "double float"

# 3. Read the summary: noise floor, best gain, convergence, warmup.
#    A gain not comfortably above the floor is a measurement of the box.

# 4. Commit the CSV and its .sysinfo together. CI rejects a bare CSV.
```

### Convergence and warmup are separate questions

The summary reports both, and they answer different things:

```
Convergence: the last third of rounds improved the best time by 4.86%
             over 84 arm(s)  (converged)
Warmup:      round 0 sat 29.72% above the eventual minimum, worst arm.
```

**Convergence** is what `--rounds` answers: the best time over all rounds against
the best over the earlier two thirds — what the tail actually bought. Judged
against the session's own noise floor, not a hardcoded threshold, so "still
moving" means *moving by more than this box's noise*. Below three rounds there is
no tail and it reports **`UNMEASURED`**, which is not the same as converged.

**Warmup** is a diagnostic with no advice attached: a large value says the untimed
warmup pass did not fully warm. Raising `--rounds` cannot change round 0.

> These were one statistic until #493. It computed round-0 warmth and printed
> convergence advice against it — telling an operator to double their machine
> time on a session whose 24 control arms agreed to 1.48%. The first fix then
> reported `(converged)` at `--rounds 2`, where no tail exists: a clean verdict
> from an absent measurement, i.e. the *same* defect as a `0.00%` floor over zero
> control arms, rebuilt inside its own repair. Both are fixed; the episode is
> kept here because the pattern recurs.

---

## See also

- [Benchmark systems](../benchmarks/systems.md) — topology, caches, pinning
- [Cache blocking A/B study](cache-blocking-ab-study.md) — the worked example
- [Blocking and scheduling assessment](blocking-and-scheduling-assessment.md)
- [Hardware expansion plan](hardware-expansion-plan.md) — what each machine can
  and cannot answer
- `benchmarks/README.md` — the commands
