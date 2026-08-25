# The BLIS five-loop GEMM: panels, micro-panels, and why they are shaped that way

MTL5's dense `mult()` routes matrix-matrix multiplication through a blocked GEMM
built on the GotoBLAS/BLIS design. This page is the algorithmic description of
that design — the loop nest, the packed data layouts, the parameter model, and
the parallelisation — collected in one place. It is deliberately implementation-
independent where it can be, and points at the source where it cannot.

> The material lived only as commentary in a dozen headers
> (`gemm_blocked.hpp`, `gemm_pack.hpp`, `gemm_quad_pack.hpp`,
> `gemm_microkernel.hpp`, `simd/blocking.hpp`, `nc_model.hpp`, …). Those
> comments explain *decisions*; this page explains the *algorithm* they are
> decisions about.

**Reference:** Van Zee & van de Geijn, *BLIS: A Framework for Rapidly
Instantiating BLAS Functionality*, ACM TOMS 41(3), 2015; and Low et al.,
*Analytical Modeling Is Enough for High-Performance BLIS*, ACM TOMS 43(2), 2016.

---

## 1. The problem: GEMM is not compute-bound by default

For `C := C + A·B` with `A` being `m×k`, `B` being `k×n`:

- **Arithmetic:** `2·m·n·k` flops
- **Data:** `m·k + k·n + m·n` elements

The ratio grows with problem size, so GEMM *can* be compute-bound. A textbook
triple loop is not, for two reasons that have nothing to do with flops:

1. **Every element is re-read many times, from wherever it happens to live.**
   With `C(i,j) += A(i,p)·B(p,j)` in the inner loop, each `A` row is streamed
   once per column of `B`. At `n = 4096` that is 4096 traversals of `A`.
2. **The elements a single instruction needs are not adjacent.** A row-major `B`
   read down a column strides by `n`. Every access is a separate cache line, and
   the hardware prefetcher cannot help.

Blocking fixes (1). **Packing** fixes (2), and it is the part most descriptions
under-explain — so it gets the most space here.

---

## 2. Three ideas, three levels

| idea | what it exploits | unit |
|---|---|---|
| **Register blocking** | a small `C` tile can live in vector registers for the whole `k` loop | `mr × nr` **micro-tile** |
| **Cache blocking** | a block of `A` or a panel of `B` can be made to fit a chosen cache level | `mc × kc`, `kc × nc` |
| **Packing** | copying into a kernel-order layout is cheaper than strided access, if the copy is amortised | **panel** / **micro-panel** |

The third is the one that turns the first two from an idea into a fast program.

### Residency: what is supposed to live where

```
   L1  ──  B micro-panel   kc × nr    re-read for every ir step
   L2  ──  A block         mc × kc    re-read for every jr step
   L3  ──  B panel         kc × nc    re-read for every ic step
   regs ── C micro-tile    mr × nr    resident across the whole kc loop
```

Each operand is placed at the level whose *reuse distance* matches. That
correspondence — not raw capacity — is what the parameter model in §6 encodes.

---

## 3. The five loops

Outermost to innermost, with the quantity each partitions:

```
for jc = 0 … n step nc            ← partition columns of C and B      [L3]
  for pc = 0 … k step kc          ← partition the k dimension
      pack B(pc:pc+kc, jc:jc+nc) → Bp                                 ← packing
    for ic = 0 … m step mc        ← partition rows of C and A         [L2]
        pack A(ic:ic+mc, pc:pc+kc) → Ap                               ← packing
      for jr = 0 … nc step nr     ← macro-kernel over B micro-panels  [L1]
        for ir = 0 … mc step mr   ← macro-kernel over A micro-panels
            MICRO-KERNEL: C(ir,jr) += Ap(ir) · Bp(jr)                 [registers]
```

Two properties follow from the ordering, and both matter later:

- **`B` is packed once per `(jc, pc)`**, then reused by every `ic` iteration.
  Packing cost is amortised over `m/mc` uses. This is why `B`'s packing is
  hoisted above the `ic` loop, and why the threaded version can share one packed
  `B` across a team (§7).
- **`A` is packed once per `(jc, pc, ic)`** and reused by every `jr`. Amortised
  over `nc/nr` uses.

The `pc` loop is **sequential and never parallelised**. Each iteration
accumulates into the same `C` elements, so its order fixes the floating-point
summation order. Holding it fixed is what lets MTL5 guarantee that threaded
results are **bit-identical** to serial ones (§7).

`k` is partitioned at all only because the packed `B` panel must fit L3; a full
`k` would make it unbounded.

---

## 4. Panels and micro-panels

This is the heart of the design.

**Definitions.**

- A **panel** is what one packing call produces: a `mc × kc` block of `A`, or a
  `kc × nc` panel of `B`, copied into kernel order.
- A **micro-panel** is the slice of that panel one micro-kernel invocation
  consumes: `mr × kc` of `A`, or `kc × nr` of `B`.

A panel is stored as a **contiguous sequence of micro-panels**. That is the
whole trick: the macro-kernel walks micro-panels by adding a constant stride,
and the micro-kernel reads each one from front to back with **unit stride**.

### 4.1 The `A` layout — column-major inside each micro-panel

```
Ap[q·MR·k + p·MR + i]  =  A(q·MR + i, p)          i ∈ [0,MR)  p ∈ [0,k)
   └─panel q─┘ └─ k step ─┘ └row┘
```

Panel `q` covers rows `[q·MR, q·MR+MR)`. Within it, storage runs **column-major
over `k`**: all `MR` rows for `p = 0`, then all `MR` rows for `p = 1`, and so on.

```
   A block (mc × kc)                packed Ap
   ┌───────────────┐                ┌──────────────────────────────┐
   │ ░░░░░░░░░░░░░ │ MR rows        │ ░░░░ ░░░░ ░░░░ …             │ panel 0
   │ ▒▒▒▒▒▒▒▒▒▒▒▒▒ │ MR rows   ──▶  │ ▒▒▒▒ ▒▒▒▒ ▒▒▒▒ …             │ panel 1
   │ ▓▓▓▓▓▓▓▓▓▓▓▓▓ │ …              │ ▓▓▓▓ ▓▓▓▓ ▓▓▓▓ …             │ panel 2
   └───────────────┘                └──────────────────────────────┘
      kc columns                      p=0  p=1  p=2 …  (MR values each)
```

The micro-kernel therefore reads `Ap[p·MR + i]` — the `MR` values it needs at
step `p` are **adjacent**, one vector load or one broadcast per row.

### 4.2 The `B` layout — row-major inside each micro-panel

```
Bp[q·NR·k + p·NR + j]  =  B(p, q·NR + j)          j ∈ [0,NR)  p ∈ [0,k)
```

The mirror image: panel `q` covers columns `[q·NR, q·NR+NR)`, and within it
storage runs **row-major over `k`**. The micro-kernel reads `Bp[p·NR + j]`, so
the `NR` columns it needs at step `p` are adjacent — and since `NR` is a
multiple of the SIMD width, that is a contiguous vector load.

**The strided access has been paid for exactly once**, during packing, in a loop
that does nothing else and whose cost is amortised over many kernel calls.

### 4.3 Edges

`m` and `n` are not multiples of `mr` and `nr` in general. Both packers **pad
with zeros** to a whole micro-panel:

```
packed_A_size(m,k,MR) = ⌈m/MR⌉ · MR · k
packed_B_size(k,n,NR) = ⌈n/NR⌉ · NR · k
```

Zeros contribute nothing to a sum, so the kernel needs no edge logic *inside*
the `k` loop. The remaining edge — a `C` tile shorter than `mr` or narrower than
`nr` — is handled by accumulating through a zeroed `mr × nr` temporary, so the
micro-kernel only ever performs full-tile loads and stores.

> The padding is not free, and it is measurable. It is why `mc` must never fall
> below `mr`: at `mc < mr` **every** block is a ragged tile, not just the last
> one, and every block pays the edge path. See the note in `balanced_mc`.

---

## 5. The micro-kernel

```
C(i,j) += Σ_{p<kc} A(i,p) · B(p,j)      0 ≤ i < MR,  0 ≤ j < NR
```

The `MR × NR` tile of `C` is held **entirely in vector registers** for the whole
`kc` loop. Each step is a rank-1 update: broadcast one `A` value across a
vector, load one `B` vector, fused-multiply-add into an accumulator.

`MR` and `NR` are **compile-time** template parameters. That is deliberate: it
lets the compiler fully unroll the tile and keep the accumulators in registers,
which is what hand-written assembly kernels achieve by construction.

`NR` is the vectorised dimension and must be a multiple of the SIMD width.

**Mixed precision** is a property of the loads, not of the loop. When the
operand type is narrower than the accumulator, the operands are read with a
widening load, so e.g. `float` panels accumulate in `double` registers with no
separate kernel.

---

## 6. Choosing the parameters

MTL5 uses the BLIS *analytical* model — parameters derived from documented
hardware characteristics — rather than autotuning.

### 6.1 The register tile `mr × nr`

Two constraints, and **which one binds is the point**:

- **Floor:** `mr·nr ≥ N_vec · L_vfma · N_vfma` — enough independent accumulators
  to cover dependent-FMA latency at the issue width. (Eq. 1 of the BLIS model.)
- **Ceiling:** accumulators *plus* the operands they multiply must fit the
  architectural vector register file. Exceed it and the kernel spills every step.

MTL5 budgets **about ¾ of the register file** for accumulators and holds the `B`
micro-panel in a small fixed number of vectors. Sizing at the floor alone is a
measured mistake — on an AVX2 core it yields a `4×8` tile using 8 of 16 `ymm`
registers, leaving half the file idle:

| tile | accumulator vectors | N=1024 | N=2048 |
|---|---|---|---|
| 4×8 | 8 | 57.90 | 56.90 |
| 5×8 | 10 | 63.41 | 64.16 |
| **6×8** | **12** | **67.57** | **66.96** |
| 8×8 | 16 | 57.68 | 57.03 |
| 8×12 | 24 | 55.31 | 55.72 |

Both directions cost: too few accumulators cannot hide latency, too many leave
nothing for operands. The peak at ~¾ of the file is also where BLIS's
hand-written Haswell kernel sits (6×8), and its AVX-512 kernel (8×24 of 32).

### 6.2 The cache blocks

```
kc = (L1 / 2) / (nr · sizeof)      B micro-panel (kc × nr) ≈ ½ L1
mc = (L2 / 2) / (kc · sizeof)      A block       (mc × kc) ≈ ½ L2
nc = ⌊ L3 / (kc · sizeof) ⌋_nr     B panel       (kc × nc) ≈   L3
```

Each is the capacity of the level whose reuse distance that operand has. The
halving leaves room for the other traffic crossing the same cache.

**`mc` is deliberately not rounded to a multiple of `mr`.** That coupling makes
a *cache* quantity depend on the *register tile*, and the dependency is
destructive: a register-tile change once moved `mc` 64 → 60, which moved the
block count at `m=1024` from 16 (exactly 2.00 per thread on 8 threads) to 18
(2.25) — a 1.41× critical path that turned a +21.5% single-thread win into a
−7.4% eight-thread regression. Block-count effects dominate block-size effects
once threads are involved.

### 6.3 Where the model stops and measurement starts

`nc` sets the **jc block count**, `njb = ⌈n/nc⌉`, and the threaded nest hands
those blocks to teams round-robin — so `nc` is a *partition* quantity, not only
a capacity one. MTL5 therefore treats its sizing as an empirical question, with
several candidate models measured against each other across machines.

Two results from that programme are worth knowing before touching `nc`:

- **Applying a detected (larger) L3 to `nc` is falsified** — up to 45% slower,
  even with partition balancing applied. A larger cache block is not a faster one.
- **The dominant effect on jc-parallel shapes is `nc` being too *large*,** not
  merely ragged; models that shrink it win.

See [the benchmarking methodology](../performance/benchmarking-methodology.md)
for the hypothesis register.

---

## 7. Parallelisation: a 2D grid over two of the five loops

MTL5 parallelises the `jc` and `ic` loops over a `jc_nt × ic_nt` thread grid
(BLIS "multi-loop" parallelism), using the C++ standard concurrency runtime.

```
        jc teams  ─────────────▶
   ic   ┌────┬────┬────┬────┐
  thr   │ T0 │ T2 │ T4 │ T6 │   each COLUMN is a team sharing one packed B
   │    ├────┼────┼────┼────┤
   ▼    │ T1 │ T3 │ T5 │ T7 │   each thread owns disjoint rows of C
        └────┴────┴────┴────┘
```

- The `n`-dimension blocks are partitioned across **teams**; within a team, the
  `m`-dimension blocks are partitioned across **members**.
- A team **cooperatively packs** its shared `B` panel — each member takes a
  disjoint range of micro-panels. Panels are independent and land at computable
  disjoint offsets, so the packed bytes are identical however the range is split.
- Members synchronise on a per-team barrier: pack, publish, compute, and a
  second barrier before the leader repacks for the next `pc`.
- Each thread writes **disjoint rows of `C`**, so there is no race and no
  reduction.

The grid degenerates to pure `ic` parallelism for square and tall problems
(where `njb = 1`) and to pure `jc` parallelism for wide, short ones.

### The bit-identity guarantee

Every `C` macro-block receives **the same FMAs in the same order** regardless of
which thread runs it, because the `pc` loop stays sequential per block and
blocking only decides *grouping*, never summation order.

So MTL5's threaded GEMM is **bit-identical to its serial GEMM for any grid
shape and any thread count** — a property worth more than it first appears: it
makes threading invisible to numerical debugging, and it means a blocking change
can be validated by an exact comparison rather than a tolerance.

> This is also why the benchmark harness compares arms **element-wise** rather
> than by checksum: the guarantee is exact, so the check should be too.

---

## 8. A different layout for the same nest: the quad kernel

Integer dot-product instructions (x86 VNNI `vpdpbusd`, ARM `SDOT`/`UDOT`)
consume **four `k` values per instruction**: lane `j` receives
`Σ_{q<4} A(i, p+q) · B(p+q, j)`.

That requires the four `k` values for one `(row, lane)` to sit in four **adjacent
bytes** — which is a *different layout*, not a different traversal of the same
one:

```
standard    Ap[q·MR·k  + p·MR   + i    ] = A(i, p)
quad        Ap[q·MR·KP + g·MR·4 + i·4+t] = A(i, 4g+t)      KP = ⌈k/4⌉·4

standard    Bp[q·NR·k  + p·NR   + j    ] = B(p, j)
quad        Bp[q·NR·KP + g·NR·4 + j·4+t] = B(4g+t, j)
```

with `g` the k-group index. `k` is zero-padded to a multiple of four, so the
padding argument of §4.3 carries over unchanged.

**Everything else in the nest is shared** — the five loops, the cache blocks, the
thread grid, the cooperative pack, the exception handling. Only the packer and
the micro-kernel differ, which is the payoff of separating layout from schedule.

The kernel is selected by an **explicit template argument**, never inferred from
the element types: an `(i8,i8)` pair is valid input to *both* kernels, so
inference would silently reroute the existing path — which it once did, making
two benchmark arms secretly the same kernel.

---

## 9. Where each piece lives

| concern | file |
|---|---|
| the five loops, threading, barriers | `include/mtl/detail/gemm_blocked.hpp` |
| standard packing (`pack_A`, `pack_B`) | `include/mtl/detail/gemm_pack.hpp` |
| quad packing | `include/mtl/detail/gemm_quad_pack.hpp` |
| micro-kernel | `include/mtl/detail/gemm_microkernel.hpp` |
| quad micro-kernel | `include/mtl/detail/gemm_quad_microkernel.hpp` |
| parameter model (`mr,nr,kc,mc,nc`) | `include/mtl/simd/blocking.hpp` |
| `nc` candidate models | `include/mtl/detail/nc_model.hpp` |
| cache detection | `include/mtl/util/cache_info.hpp` |
| SIMD abstraction | `include/mtl/simd/batch.hpp` |
| thread pool | `include/mtl/detail/thread_pool.hpp` |
| entry point (`mult`) | `include/mtl/operation/mult.hpp` |

---

## See also

- [Benchmarking methodology](../performance/benchmarking-methodology.md) — how
  blocking changes are validated, and the hypothesis register
- [Benchmark systems](../benchmarks/systems.md) — per-machine cache hierarchies
- [On-node threading](on-node-threading.md)
- [Mixed-precision kernels](mixed-precision-kernels.md)
