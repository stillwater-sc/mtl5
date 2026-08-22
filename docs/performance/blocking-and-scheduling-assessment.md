# Assessment: blocking, scheduling, and what transfers to other backends

**Status:** living document. Updated as experiments land.
**Scope:** what the MTL5 performance experiments have established about *how a
dense kernel's schedule gets chosen*, how much of it is CPU-specific, and what
must be re-derived when a GPU or KPU backend arrives.

This is the accumulated verdict, not an experiment log. The runs behind each
claim are on [Cache blocking A/B](cache-blocking-ab-study.md) and
[Multi-core scaling](multicore-scaling-investigation.md); the integer-lane
numbers in §6 come from `benchmarks/data/ryzen-9-8945hs/int_arms.csv`. This page
records what survived them.

## What has been established

### 1. The analytical model's central premise is not universal

MTL5 derives `kc`, `mc`, `nc` from an analytical model (BLIS's) whose core
assumption is that the packed A block occupies about half of L2. That assumption
is not shared by the most widely deployed BLAS. For Haswell, double precision:

| | mr×nr | mc | kc | nc | A block | vs 256 KB L2 |
|---|---|---|---|---|---|---|
| BLIS | 6×8 | 72 | 256 | 4080 | 147 KB | 57% |
| **OpenBLAS** | 4×8 | **512** | 256 | **13824** | **1024 KB** | **400%** |
| MTL5 (AVX2) | 6×8 | 64 | 256 | 4096 | 131 KB | 51% |

OpenBLAS oversizes the A block by 2–4× on Haswell, Sandy Bridge, Nehalem and
Zen, and fits it inside L2 only on SkylakeX, where L2 is 1 MB. So "A block in
L2" is one design point among several, and a model that treats it as a law will
mis-generalise on hardware unlike its calibration target — which is exactly what
we measured when detection replaced the assumed 256 KB L2 with a real 1.25 MB
one (#426, #430).

### 2. `mc` is dominated by the thread partition, not by cache residency

Measured response curve, 2048³ on a 6-core Xeon E5-2420 v2, `mc` overridden
directly:

| spread across `mc` ∈ [12, 256] | |
|---|---|
| single-threaded | **3.6%** |
| six threads | **13.0%** |

A residency effect cannot be four times stronger with six threads on identical
footprints. The curve is **W-shaped**, and the shipped `mc = 32` sits in the
trough at **88.5%** of the best; both 12 and 144 are ~12% faster. The shape
reproduces independently at `kc = 256` and `kc = 512`.

Two rounds of analytical repair failed against this. Charging the C strip
alongside A (`ccap`) won 15% on large square shapes and lost 15% elsewhere;
charging C alone (`ccap2`) removed every loss and kept only 1–2.5% of the win.
No threshold on residency produces both outcomes (#453).

### 3. Partition arithmetic has produced the largest measured wins

| change | effect |
|---|---|
| Grid factorization instead of greedy fill (#441) | i7 1024³ T=8 **0.551 → 0.707**, Zen 4 **0.590 → 0.928** |
| `mc` decoupled from the register tile, `balanced_mc` (#408) | parallel efficiency 62.7% → **81.8%** at N=1024 |
| Packing the shared B panel across the team (#348) | efficiency 79% → **91%** |

Every one is about *which thread does which block*, not about what fits in which
cache. That is the strongest signal in the whole programme.

### 4. Cache-size detection never wins

Across four machines and 30 informative points: 1 faster, 8 slower, 21
indistinguishable, with the single win at 2.7% (#430). `MTL5_ENABLE_CACHE_DETECTION`
ships **off**. The machinery is retained because it is how the model's
generalisation gets tested, not because it pays.

### 5. Parameters are coupled, and the coupling is where defects hide

`kc = (L1/2)/(nr·sizeof)` charges the **B micro-panel only**, so a narrow `nr`
inflates `kc` — the Xeon build derives 512 where both references use 256. Since
`mc = (L2/2)/(kc·sizeof)`, that halves `mc`. Measured, our 512 is *better* here
than the reference 256 at nearly every `mc`, so this is not a defect on this
machine — but it is a reminder that no parameter in this model can be judged
alone. #408 is the same lesson: coupling a cache quantity (`mc`) to a register
quantity (`mr`) cost 7.4% at eight threads.

### 6. Narrowing the operand beats optimising the instruction, by an order of magnitude

The integer-lane work (#451) produced the largest speedups in this programme,
and almost none of it came from the instruction everyone reaches for.

Measured on a Ryzen 9 8945HS (Zen 4, AVX3_DL, **native `vpdpbusd`** — confirmed
in the sidecar, not assumed), `dot` with `uint8 × int8` operands against an fp64
baseline:

| n | 1 024 | 65 536 | 1 048 576 | 4 194 304 |
|---|---|---|---|---|
| speedup vs fp64 | 2.3× | 9.2× | 17.9× | **26.0×** |

That 26× is mostly **bytes**: an int8 dot moves one byte per element where fp64
moves eight, and a dot product is a streaming reduction with no reuse, so at
large *n* it is bandwidth-bound and the ratio approaches the operand-width
ratio. The instruction is a small part of it.

Three machines, same kernel, same arms — the middle column has VNNI silicon it
cannot reach (§7), so it isolates the bandwidth effect on a *modern* memory
system rather than a 2013 one:

| n | Xeon E5-2420 v2 (2013, no VNNI) | i7-12700K (decomposed) | Zen 4 (native VNNI) |
|---|---|---|---|
| 65 536 | 2.5× | 4.6× | 9.2× |
| 1 048 576 | 3.7× | 7.8× | 17.9× |
| 4 194 304 | 7.3× | 15.4× | 26.0× |

The i7 reaching 15.4× **without a usable VNNI instruction** is the clearest
statement of the point: most of the win is the operand width, and it is
available on hardware that cannot execute the specialised op at all.

How small can be measured **without leaving the machine**. Zen 4 has VNNI but
not AVX10.2, so in the same run, over the same data, `uint8 × int8` is a native
`vpdpbusd` while symmetric `int8 × int8` is emulated (two `vpdpbusd` plus a
shift and a subtract):

| n | 1 024 | 16 384 | 262 144 | 4 194 304 | median |
|---|---|---|---|---|---|
| native ÷ emulated | 1.33× | 1.50× | 1.44× | 1.28× | **1.42×** |

**That 1.42× is an over-estimate, and the i7 is what showed it.** The two arms
differ in more than one respect: they also take *different decompositions*
(`PromoteEven`/`PromoteOdd` for the symmetric form against a shorter route for
the mixed one). On the i7, where **both** are decomposed, the same pair should
therefore come out level — and it does not:

| | median `u8×i8 ÷ i8×i8` |
|---|---|
| Zen 4 — one native, one emulated | 1.42× |
| i7 — **both emulated** | **1.17×** |
| instruction, net of the shape difference | **≈ 1.22×** |

So roughly 1.17× of the Zen 4 ratio is the shape of the decomposition, not the
instruction. The honest figure is **~1.2×**, with ~1.4× as the upper bound if
the shape difference does not transfer between AVX2 and AVX3_DL — which cannot
be measured directly, because on Zen 4 the mixed form is never decomposed.

**~1.2× from the instruction; ~18× from the operand width.** That the native
form is barely faster despite issuing about a third of the instructions says the
kernel is not instruction-throughput-bound even at L1-resident sizes — the same
conclusion §2 reached about `mc` by a different route.

This is what the control was for. The claim was written from the Zen 4 run alone
and stood for a day before the i7 corrected it; nothing in the Zen 4 data could
have exposed it, because the confound is only visible where the instruction is
absent.

This is the cleanest result the programme has, and it is a **data-movement**
result. It also carries a caveat: a *dot* is the worst case for showing an
instruction off, because there is no operand reuse to amortise the traffic.

**The GEMM now exists, and it settles that caveat more sharply than predicted.**
A GEMM reuses each operand O(n) times, so the panels are packed once and read
from cache thereafter and their width in memory stops mattering. Measured on the
Xeon (SSE4, no VNNI), n=512:

| `gemm_f32` | `gemm_i32` | `gemm_i16_i32` | `gemm_i8_i32` |
|---|---|---|---|
| **18.8** | 13.5 | 12.8 | 13.3 GOP/s |

Narrowing the operand buys **nothing** — `gemm_i8_i32` is no faster than
`gemm_i32` — and every integer arm is slower than fp32, because the widening
path promotes into int32 lanes and inherits int32's arithmetic cost, and integer
multiply has no FMA on this ISA.

So the prediction was right in direction and wrong in magnitude. The balance
does not merely *move* toward the instruction in a GEMM; at this point the
operand width contributes **zero** and the instruction is the only thing left.

#### 6a. The quad micro-kernel, and the part of that which was wrong

The paragraph that used to close this section said: *on a machine without VNNI
there is no int8 GEMM win at all.* That was a statement about the
**widen-on-load** kernel, which was the only int8 GEMM that existed when it was
written, and it does not survive building the other one.

`vpdpbusd` consumes four k-values per instruction, so it needs a
quad-interleaved pack layout (`Ap[…][i*4+q] == A(i, 4g+q)`, and the same for B)
and a micro-kernel whose left operand is a **broadcast quad** rather than a
broadcast scalar. Both now exist. Measured on the same Xeon, same n=512, still
`int8 quad dot: decomposed`:

| `gemm_f32` | `gemm_i8_i32` | `gemm_i8_i32_quad` | `gemm_u8i8_i32_quad` |
|---|---|---|---|
| 19.7 | 13.2 | **16.5** | **21.8** GOP/s |

From `benchmarks/data/xeon-e5-2420/int_arms.csv`, produced by `run_int_bench.sh`
(preflight, native-quad guard, `.sysinfo` sidecar; `label=native-int-decomposed`,
`build_isa=SSE2 AVX`, `pin=0,1,2,3,4,5`).

**The controlled comparison is ~1.25×** — `gemm_i8_i32_quad` against
`gemm_i8_i32`, same operands, only the kernel changing — and it GROWS WITH n,
which is the shape a register-blocking change should have:

| n | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|
| kernel, operands fixed | 1.19× | 1.24× | **1.25×** | **1.27×** |
| signedness, kernel fixed | 1.27× | 1.29× | **1.32×** | **1.28×** |

The small sizes never leave the caches, so the operand-traffic reduction the
quad layout buys has nothing to pay for yet. Holding the kernel fixed and
changing the operand signedness instead is a further ~1.30×, because `u8 × i8`
is the shape the decomposition handles most cheaply.

The 1.64× obtained by dividing the last arm by the first moves **both**
variables and is not a result about the kernel. It is quotable only as "the best
int8 GEMM available after this change against the best available before", and it
exists at all because there is no `u8 × i8` widen-on-load arm to compare with:
`load_widen` requires matching signedness, so that pairing previously fell to the
generic scalar loop.

That care is owed to §6's own history. The 1.42× → 1.17× correction recorded
above came from precisely this mistake — two arms differing in the instruction
*and* in the decomposition shape, read as though only the instruction had moved.
Reproducing it one section later would be hard to excuse.

The `u8 × i8` arm nonetheless beats fp32 outright, on a 2013 part with no VNNI
silicon in it.

The mechanism is checkable in the disassembly, and was checked: Highway's
decomposition of the quad accumulate is a pair of `vpmaddwd` plus
sign-extension shifts, which still folds four products into each accumulator
lane in a handful of instructions, where widen-on-load runs four independent
promote-multiply-add chains. The symmetric form carries more of that shift work
than the mixed one, which is the gap between the two quad arms.

#### 6b. The instruction is per pairing, and the two ISAs disagree about which

A cross-machine reading of the arms above needs one more fact, and it inverts
the obvious interpretation on half the hardware. Support for the quad
multiply-accumulate is **per operand pairing**, and x86 and ARM implement
*opposite* pairings first (read from Highway's own gates, not inferred):

| pairing | x86 AVX3_DL | x86 AVX10.2 | NEON + DotProd | NEON + I8MM |
|---|---|---|---|---|
| `u8 × i8` | **native** `vpdpbusd` | native | emulated | **native** `USDOT` |
| `i8 × i8` | emulated | native `vpdpbssd` | **native** `SDOT` | native |
| `u8 × u8` | emulated | native `vpdpbuud` | **native** `UDOT` | native |

So `gemm_u8i8_i32_quad` — the fastest arm on every x86 measured, and the shape
VNNI exists for — is the **emulated** one on a Cortex-A78, where the symmetric
pairings are native. **A comparison that pairs arms across machines by name
rather than by whether they were native gets the sign of the effect wrong.**

Two consequences worth stating separately:

- **No machine in the fleet is a baseline for another's arms.** x86 gets exactly
  one pairing before AVX10.2; ARM gets exactly two before I8MM. Neither is a
  superset.
- **`PARTIAL` is the normal state, not an edge case.** Zen 4 is partial. A
  Cortex-A78 is partial, for the complementary half. Only AVX10.2 and NEON+I8MM
  are fully native, and MTL5 has measured neither. The committed Zen 4 CSV is
  labelled `native-int`, which over-claimed for its symmetric arms; runs after
  this change are labelled `native-int-partial` and `bench_all` prints the
  per-pairing line.

This is the same failure mode as §7 one level down — a capability treated as a
single bit when the hardware exposes it in pieces.

**What survives and what does not.** The operand *width* still contributes
nothing in a GEMM — that finding is untouched, and the preceding conclusion
holds as a statement about the **widen-on-load** kernel, which is the only one
that existed when it was written. What was wrong was equating "the instruction"
with "VNNI silicon": the ~1.25× measured here is the four-products-per-lane
**kernel shape**, which a machine without the instruction can partly express
anyway. A VNNI or AMX machine's GEMM number is still an almost pure measurement
of arithmetic rather than bandwidth — the opposite end of the axis from the dot
— but its baseline is now `gemm_i8_i32_quad` on the same machine, not the
widening arm, and the increment attributable to the silicon is correspondingly
smaller than this section previously implied.

This is the second time a claim in this document has been narrowed by supplying
the control it lacked, and the pattern is the same both times: the original
statement was true of everything that had been measured, and false of the thing
that had not been built yet.

### 7. Hardware you own is not hardware you can reach

The i7-12700K (Alder Lake) **has** VNNI silicon — `__AVXVNNI__`, the VEX-encoded
256-bit form — and MTL5 cannot use it. Highway implements the quad
multiply-accumulate only in its `HWY_AVX3_DL` target, which is gated on AVX-512
macros; Alder Lake's AVX-512 is fused off, and Highway carries no AVX-VNNI path
at all. Compiled at `-march=alderlake` the kernel emits `vpmaddwd` and
`has_native_quad_dot` is false — and a run on the machine confirms it, labelling
itself `native-int-decomposed` with `build_isa=SSE2 AVX AVX2 FMA` and
`SIMD backend: AVX2`.

Two lessons, both of which generalise past this instance:

- **A feature bit is not a capability.** Three separate gates sit between the
  silicon and the kernel — the CPU's fuses, the compiler's macros, and the SIMD
  library's target taxonomy — and any one of them can be the binding constraint.
  MSVC is a fourth: it defines none of the seven macros Highway requires, so a
  Windows-native build cannot reach VNNI even on a Zen 4.
- **Which is why the benchmark records the derived answer.** `build_isa` now
  carries `AVX512VNNI` and `AVX3_DL`, and the runner refuses to time a
  decomposed build unless told to. Without that, an i7 run would have produced a
  plausible CSV measuring the emulation, and nothing in it could have said so.

## What makes these claims checkable

The measurement apparatus is as much a result as the numbers, and it transfers
to any backend:

- **A run contract** — preflight gates, pinning, interleaving with rotated arm
  order, min-of-N, per-machine output directories ([systems.md](../benchmarks/systems.md))
- **Provenance in every sidecar** — commit, dirty state, `build_isa` from the
  compiler's own macros, governor, power mode, thermal before/after, and the
  *effective* schedule (`mc_used`, `nib`, `njb`, `ic_nt`, `jc_nt`), not the
  configured one
- **Integrity gates** — identical-configuration arms that disagree fail the
  analysis rather than being reported, which is what disqualified a Jetson
  session whose round-to-round spread exceeded the effect under test

Three separate results in this programme were initially attributed to the wrong
cause and corrected only because a control existed. Any new backend needs the
same discipline before its numbers mean anything.

## Contrast: CPU, GPU, KPU

The three differ in **where the schedule comes from**, which is why results do
not transfer directly.

| | who decides the schedule | how it is tuned | what our findings say |
|---|---|---|---|
| **CPU** (today) | an analytical model at compile time, plus runtime heuristics for the partition | measured, per machine, because the model's premises do not generalise | the partition dominates; the cache model is secondary and its premises are contested |
| **GPU** | the hardware scheduler over a launch configuration the programmer picks | occupancy and tile-size search, usually empirical | the same "measure the response curve" problem reappears, one level up: tile size, launch shape, shared-memory budget |
| **KPU** | a **system-level schedule specification** supplied as input | the schedule is stated, not inferred | this is the interesting case — see below |

### Why the KPU case is different, and what to revisit

On the KPU the schedule is an *input to the machine* rather than an emergent
property of a cache hierarchy the code is trying to guess. That should dissolve
the specific problem this programme has been fighting — a model predicting
residency it cannot observe, and a partition heuristic whose optimum is W-shaped
and machine-specific.

When the KPU fsim backend is connected, these are the questions this work leaves
ready to ask:

1. **Which findings were CPU artifacts?** The W-shaped `mc` curve and the
   trough at the shipped default are almost certainly artifacts of *this*
   partition heuristic. The claim that "the partition matters more than the cache
   model" may well survive; the specific numbers will not.
2. **What is the invariant?** Data movement per operand — how many times A, B and
   C cross each level — is the quantity that is architecture-independent. Our
   sidecars record the schedule that determined it, so past runs remain
   interpretable in those terms. §6 is the sharpest evidence for this framing
   the programme has: narrowing the operand from 8 bytes to 1 bought ~18×, while
   the specialised instruction bought ~1.4×. If one quantity is going to survive
   the move to a GPU or KPU, bytes-per-operand is the candidate — and it is the
   one a schedule specification can state directly.
3. **Does an explicit schedule remove the need to measure, or move it?** If the
   KPU's schedule specification is complete, the response curve should be
   predictable rather than discovered. That is a falsifiable claim, and the
   harness here is the way to test it: same shapes, same contract, compare a
   predicted schedule against a swept one.
4. **What is the common comparison metric?** GFLOP/s is not enough to compare
   across a CPU, a GPU and a KPU fairly. Bytes moved per useful flop, and the
   fraction of peak each backend's schedule achieves, are the terms in which the
   three can be set beside each other.

Until then, the CPU results above should be read as *what it costs to infer a
schedule that the hardware never told you* — which is the baseline any
schedule-specified architecture has to beat.
