# Hardware expansion: which processors to pursue, and what each would settle

**Status:** proposal. No hardware acquired, no runs scheduled.
**Scope:** which machines would move the dense-kernel programme forward, what
each is expected to teach, and what MTL5 must build before a given machine can
say anything at all.

This is a design document, not a shopping list. A processor earns a place here
by being able to **falsify something we currently believe**, not by being fast.
Three of the nine below are interesting precisely because MTL5 would perform
*badly* on them.

## Where the programme actually stands

Four machines have produced data. What they have in common is more limiting than
what separates them.

| machine | ISA reached | int8 quad dot | vector width | cores |
|---|---|---|---|---|
| Xeon E5-2420 v2 (2013) | SSE4 | decomposed | 128-bit | 6, homogeneous |
| i7-12700K (Alder Lake) | AVX2 | decomposed *(has AVX-VNNI, unreachable)* | 256-bit | 8P + 4E, pinned to P |
| Ryzen 9 8945HS (Zen 4) | AVX3_DL | **native `vpdpbusd`** | 512-bit ISA, 256-bit datapath | 8, homogeneous |
| Jetson Orin Nano (A78AE) | NEON | *(untested for int)* | 128-bit | 6, homogeneous |

Every open question in
[the assessment](blocking-and-scheduling-assessment.md) is limited by that
table:

- **§6 — operand width beats instruction specialisation (~18× vs ~1.2×).** One
  native-VNNI datapoint, and it is on hardware whose AVX-512 is *double-pumped
  through a 256-bit datapath*. We have never measured the split on a full-width
  implementation.
- **§7 — hardware you own is not hardware you can reach.** Established as a
  capability fact. Never *sized*: nobody has measured what the unreachable
  AVX-VNNI on the i7 would have been worth.
- **§2 / #458 — the `mc` trough is a partition effect.** One shape, one machine,
  6 cores. The acceptance criterion asks for a rule that holds on three
  machines; we cannot even state the rule.
- **The GEMM prediction.** §6 says a dot is the worst case for showing an
  instruction off, and predicts the balance moves toward the instruction when
  operands are reused O(n) times. There is no integer GEMM, so this is
  unfalsified.

## Selection criteria

A candidate is worth pursuing if it satisfies at least one:

1. **It varies one axis against a machine we already have.** Same-vendor,
   next-generation parts are worth more than a new vendor at the same design
   point, because the confound structure is known.
2. **It reaches a capability MTL5 cannot currently express.** These tell us what
   to *build*, which is often worth more than a number.
3. **It breaks an assumption the code makes.** Three core types instead of two;
   a scalable vector length; a memory system that is not DDR.

And a candidate is **not** worth pursuing merely because it is faster. A faster
machine at the same design point re-measures a known quantity.

## Intel

### I1. Xeon 6 (Granite Rapids) or Xeon 4th/5th gen (Sapphire/Emerald Rapids) — *highest value*

**Varies:** vector datapath (full 512-bit vs Zen 4's double-pumped), core count,
and it is the only widely available part with a **matrix engine (AMX)**.

**What it would settle**

- **The §6 split at full width.** Our only native-VNNI number comes from a
  256-bit datapath executing 512-bit instructions. If the instruction's ~1.2×
  share is really bounded by the datapath rather than the ISA, a full-width part
  should move it and a doubling of lane count should not. That is a clean,
  falsifiable prediction and we cannot currently test it.
- **Whether a matrix engine changes the question rather than the answer.** AMX
  is not a wider SIMD unit; it is a tile register file with an outer-product
  instruction, so blocking becomes *tile scheduling*. That is the closest thing
  in commodity hardware to the KPU's "schedule as an input" model that
  [the assessment's final section](blocking-and-scheduling-assessment.md)
  reserves judgement on. Measuring AMX against our own blocked GEMM is the most
  direct evidence we could get about which of our CPU findings are artifacts.

**What MTL5 must build first:** Highway has **no AMX support at all** — verified,
no `amx` anywhere in its ops headers. AMX would need intrinsics behind the
`batch<>` seam or beside it, plus tile configuration state that has no analogue
in the current design. This is a substantial piece of work and should be scoped
before the hardware is acquired, not after.

**Risk:** AMX is bf16/int8 only, so it does not touch the fp64 path at all. If
the goal is the dense fp64 kernels, this machine teaches less than its price
suggests.

### I2. Core Ultra 200 series (Arrow Lake) — *sizes the §7 loss*

**Varies:** no AVX-512 at all, AVX-VNNI present, **no hyperthreading**, P+E
hybrid.

**What it would settle**

- **What §7 costs.** The i7 established that AVX-VNNI is unreachable through
  Highway. It did not establish what that is *worth*. A hand-written
  AVX-VNNI path measured against the current decomposition on the same machine
  would put a number on the gap — and that number decides whether "add an
  AVX-VNNI target" is worth raising upstream or is a rounding error.
- **The partition question without SMT.** `plan_gemm_grid` and `balanced_mc`
  reason about a thread budget. Every machine we have has SMT, and we always pin
  one thread per physical core to sidestep it. Arrow Lake has no SMT, so the
  budget *is* the core count — which removes a confound from #458 rather than
  adding one.

**What MTL5 must build first:** an AVX-VNNI path, or a documented decision not
to have one. Nothing else.

### I3. Core Ultra 200V (Lunar Lake) — *probes the bytes claim*

**Varies:** on-package LPDDR5X, a fundamentally different memory system.

**What it would settle:** §6's central claim is that ~18× of the int8 win is
**bytes moved**. That claim is only as good as the range of memory systems it
has been tested against, and we have tested DDR3 (2013), DDR5, and DDR5. A part
with on-package memory and much lower latency at moderate bandwidth is the most
different memory system available in a laptop-class CPU. If the operand-width
ratio is really the driver, the large-*n* speedups should track the
**bytes-per-element ratio** on this machine too, not the bandwidth figure.

**Priority:** lower. Interesting, not decisive.

## AMD

### A1. Zen 5 (Ryzen 9000 desktop, or EPYC Turin) — *the cleanest single experiment*

**Varies:** the AVX-512 datapath width, against a machine we already have data
from, with everything else close.

**What it would settle:** this is the **direct A/B for §6**. Zen 4 executes
AVX-512 on a 256-bit datapath; Zen 5's is 512-bit wide on the parts that
implement it fully. Same vendor, same ISA, same compiler path, same
`AVX3_DL`/`AVX3_ZEN4` target — one axis moved. If the instruction's share of the
int8 win rises from ~1.2× toward the ~3× instruction-count ratio, the current
conclusion is datapath-bound and generalises poorly. If it does not move, the
conclusion is memory-bound and is a property of the *algorithm*, not the
hardware — which would make it far more transferable, including to the KPU.

Of the nine, this is the one I would run first. It is the only candidate that
tests an existing conclusion by moving exactly one variable.

**What MTL5 must build first:** nothing. Existing `run_int_bench.sh` and the
Zen 4 profile work unchanged.

### A2. EPYC Genoa / Turin (high core count, multi-CCD) — *for #458*

**Varies:** core count (8 → 64+), and a chiplet topology where L3 is per-CCD, so
"the L3" that `nc` is derived from is not shared by all threads.

**What it would settle**

- **#458's acceptance criterion**, which asks whether the `mc` optimum is
  predictable from static parameters. Every partition finding we have comes from
  ≤ 8 threads. `balanced_mc` rounds a block count to a multiple of the ic-thread
  count; at 64 threads the rounding is a far larger fraction, and the
  [defect found analytically](https://github.com/stillwater-sc/mtl5/issues/458)
  — that the balanced count is discarded by the caller's recomputation — should
  be much more visible.
- **Whether `nc = f(L3)` survives chiplets.** §1 already doubts the model's
  premises. A per-CCD L3 breaks the assumption that L3 is a single shared
  resource, which is the strongest available test of §1 short of a different
  architecture.

**Risk:** many-core NUMA effects could dominate and make the partition signal
unreadable. Would need the run contract extended with NUMA pinning first.

### A3. Zen 4c (Bergamo) or a Zen 5c part — *tests the cache premise directly*

**Varies:** cache hierarchy only. Same ISA, same VNNI, deliberately smaller L3
per core.

**What it would settle:** §1 says the analytical model's premise — that the
packed A block should occupy about half of L2 — is not universal, and §4 says
cache-size *detection* never wins. Both are inferences from machines whose cache
hierarchies are conventional. A dense-core part is a cache hierarchy chosen for
a different objective, and it is the cheapest way to test §1 without changing
vendor, ISA or compiler.

**Priority:** lower, but it is the most *scientifically* clean of the AMD three.

## Qualcomm

Qualcomm's value here is not throughput. It is that Snapdragon parts break
assumptions the x86 machines all share, and they are where MTL5's ARM story is
currently weakest.

### Q1. Snapdragon X Elite (X1E, Oryon) — *the ARM int8 story*

**Varies:** vendor, ISA family, OS, and toolchain all at once — which is a
weakness for controlled comparison and a strength for finding assumptions we did
not know we had.

**What it would settle**

- **The ARM half of §6.** ARMv8.6 `I8MM` provides `vusdot` — a `uint8 × int8`
  dot that Highway *does* implement (unlike AMX), so the native path is
  reachable. That makes Snapdragon the only non-x86 machine that can produce the
  same `native` vs `decomposed` within-machine comparison the Zen 4 gave us. If
  the instruction's share lands near ~1.2× there too, on a completely different
  microarchitecture and memory system, §6 becomes a claim about *algorithms*
  rather than about x86.
- **Whether the toolchain story repeats.** §7 was a Windows/MSVC finding.
  Snapdragon X is a Windows-on-ARM part, so the same class of question —
  can the shipping compiler even reach the instruction? — arises again with
  different answers. We should expect to be surprised.

**What MTL5 must build first:** likely nothing for the kernels, but the
**benchmark contract has no ARM64-Windows path**. The machine profiles are
`bash` (Linux/WSL) or PowerShell/MSVC (x64). This is plumbing, not research, but
it is not zero.

### Q2. Snapdragon X Plus — *core count on an identical microarchitecture*

**Varies:** core count only, against Q1.

**What it would settle:** #458 wants to know whether the partition optimum is
predictable from static parameters. Two parts with the *same core design*, the
same caches per core and different core counts is the cleanest possible
partition experiment — cleaner than anything available on x86, where core-count
variation usually comes bundled with cache and frequency changes. If the `mc`
trough moves in a way predicted by the thread count alone, that is the rule
#458 asks for.

**Priority:** only worth acquiring *with* Q1; alone it teaches little.

### Q3. A mobile Snapdragon (8-series) via a dev kit — *sustained vs peak*

**Varies:** thermal envelope, aggressively.

**What it would settle:** the benchmark contract has a thermal gate that
[fails only where a sensor and a limit are both readable](../benchmarks/systems.md),
and it has never been exercised on a part that actually throttles under a dense
kernel. Every performance claim in this programme is implicitly "at sustained
clocks". A machine that cannot sustain them tests whether our *methodology*
survives, which is a different and rarer kind of result than a throughput
number.

**Priority:** lowest for throughput; highest for validating the measurement
apparatus.

## Ranked

If only three machines can be pursued:

1. **Zen 5** (A1) — tests an existing conclusion by moving one variable. Zero
   new code.
2. **Snapdragon X Elite** (Q1) — tests whether §6 is about algorithms or about
   x86. Modest plumbing.
3. **Xeon with AMX** (I1) — the only one that tests whether our whole framing
   survives a matrix engine. Substantial new code, and worth scoping before
   buying.

The first two are experiments. The third is a research programme.

## Computational dynamics this would expose

The point of nine machines is not nine numbers. It is that several quantities we
currently treat as constants would become *functions*, and functions can be
falsified.

### 1. The arithmetic-intensity crossover, per operand width

A dot has intensity ~1 op/byte and is bandwidth-bound almost everywhere; a GEMM
is compute-bound almost everywhere. The interesting quantity is the **knee**:
the *n* at which each arm crosses from compute- to bandwidth-bound. We can
already see it in the int suite's footprint sweep — the int8 curve flattens far
later than fp64's, because it moves an eighth of the bytes.

With enough machines the knee becomes a prediction: it should scale with
(bytes per element) ÷ (achievable bandwidth), and any machine where it does not
is telling us something about its memory system that the roofline model does
not capture. **One machine cannot distinguish a real knee from a cache size.**

### 2. Where operand width stops paying

§6 measured ~18× from narrowing operands 8 bytes → 1. That cannot continue: at
some width the kernel stops being bandwidth-bound and the ratio collapses toward
the instruction's ~1.2×. Where that happens is a **property of the machine's
balance**, and sweeping fp64 → fp32 → int32 → int16 → int8 on machines of very
different balance would locate it. The int suite already runs exactly those five
widths, which is why it sweeps by footprint rather than by round numbers.

### 3. Whether the partition trough is universal or parochial

#458 found a **W-shaped** `mc` response with the shipped value in the trough, on
one machine at one shape. The three hypotheses — block-count quantisation,
register-tile raggedness, shared-panel re-reads — make *different* predictions as
core count and cache size vary, and a machine set that spans 6 to 64 cores would
separate them. Notably, the quantisation hypothesis has an analytically derived
signature already; it needs hardware only to confirm it, not to find it.

### 4. What a matrix engine does to blocking

This is the one that connects to the KPU work. On AMX or SME the register file
holds *tiles*, and the loop structure that `derive_blocking` produces has no
obvious meaning. Two outcomes, both valuable:

- the blocking model transfers with the tile as the new `mr × nr`, and our
  partition findings carry over — in which case the CPU work generalises further
  than expected;
- or the model does not transfer, and scheduling a matrix engine is a different
  problem — which is exactly the assessment's hypothesis about the KPU, tested
  on hardware we can actually buy.

### 5. The capability map, which is a result in itself

Three cliffs are already known and none has been measured:

| cliff | consequence | machines affected |
|---|---|---|
| Highway has no AVX-VNNI path | 256-bit VNNI silicon unreachable | Alder Lake, Raptor Lake, Arrow Lake |
| Highway has no AMX or SME | matrix engines unreachable | Sapphire Rapids+, Apple M4, some ARM |
| `HWY_HAVE_SCALABLE` ⇒ **scalar fallback** | MTL5 has *no SIMD at all* on generic SVE builds | any SVE part not pinned to a fixed vector length |

The third is the sharpest and the least known: on an SVE machine, a default
MTL5 build gets the **scalar** `batch<T>`, not a slow vector path — a silent
capability cliff, not a performance one. Highway offers fixed-width SVE targets
(`HWY_SVE_256`, `HWY_SVE2_128`) that are *not* scalable and would work, but only
if the build pins the vector length. Nothing in MTL5 currently does, or warns.

Snapdragon X is NEON-only and so does **not** test this — which is worth saying
plainly, because it is the kind of thing a hardware list gets wrong by assuming
"ARM" implies "SVE".

## Protocol

Any machine added here inherits the existing contract without amendment:
preflight gates, per-machine `OUTDIR`, provenance sidecars, and the integer
suite's guard that
[refuses to time a build lacking the native instruction](../../benchmarks/README.md).
Three additions are required before the first run on new hardware:

1. **A machine profile** in `benchmarks/machines/`, which *is* the invocation
   and refuses to run on an unrecognised host.
2. **A capability check recorded, not assumed.** `build_isa` and
   `simd::backend_name()` already do this; a new ISA needs its keys added, as
   `AVXVNNI(unused)` was.
3. **A control arm on the same machine wherever possible.** The single most
   valuable measurement this programme has produced — the ~1.2× instruction
   share — came from comparing a native and an emulated path *within one run*,
   and the cross-machine version of the same comparison was wrong by 20%.

## What would make this a waste

Stated up front, so it can be checked later:

- **Buying for throughput.** A faster machine at a design point we already have
  produces a number nobody can act on.
- **Running the dot suite and stopping.** §6 says a dot is the worst case for
  showing an instruction off. Without an integer GEMM, an AMX or I8MM machine
  cannot demonstrate what it is for — so the **GEMM kernel should precede the
  hardware**, not follow it.
- **Cross-machine comparison without a within-machine control.** Already burned
  us once, at 20% on the headline number.

## See also

- [Assessment: blocking, scheduling, and what transfers](blocking-and-scheduling-assessment.md)
- [Cache blocking A/B study](cache-blocking-ab-study.md)
- [Benchmark systems and the run contract](../benchmarks/systems.md)
