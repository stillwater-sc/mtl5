# Assessment: blocking, scheduling, and what transfers to other backends

**Status:** living document. Updated as experiments land.
**Scope:** what the MTL5 performance experiments have established about *how a
dense kernel's schedule gets chosen*, how much of it is CPU-specific, and what
must be re-derived when a GPU or KPU backend arrives.

This is the accumulated verdict, not an experiment log. The runs behind each
claim are on [Cache blocking A/B](cache-blocking-ab-study.md) and
[Multi-core scaling](multicore-scaling-investigation.md); this page records what
survived them.

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
   interpretable in those terms.
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
