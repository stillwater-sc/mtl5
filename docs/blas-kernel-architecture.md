# BLAS kernel architecture: blocking and scheduling

How MTL5's native dense kernels are structured, level by level, and *why* the
structure differs at each level. The single organizing principle is **arithmetic
intensity** -- flops performed per byte moved from memory -- which decides
whether a kernel is bound by memory bandwidth or by the FMA pipeline, and
therefore how it must be blocked and scheduled.

This document accompanies the native-BLAS performance epic (#82). Implemented
pieces: L1 in `include/mtl/simd/algorithm.hpp` (#86), the GEMM micro-kernel in
`include/mtl/detail/gemm_microkernel.hpp` (#88), blocking params in
`include/mtl/simd/blocking.hpp` (#85), all over the SIMD layer
`include/mtl/simd/batch.hpp` (#83). GEMV/GER (#87) and the GEMM macro-kernel
(#90) are designs here until landed.

---

## The taxonomy at a glance

| Level | Shape | Work | Data | Intensity (fp64) | Bound by | Operations |
|-------|-------|------|------|------------------|----------|------------|
| **L1** | vector-vector | O(n)   | O(n)   | ~0.1-0.25 flop/byte | memory bandwidth | `dot`, `nrm2`, `axpy`, `scal` |
| **L2** | matrix-vector | O(n^2) | O(n^2) | ~0.25 flop/byte     | memory bandwidth | `gemv`, **`ger` (rank-1)**, `trsv`, `symv` |
| **L3** | matrix-matrix | O(n^3) | O(n^2) | **O(n)** (grows!)   | FMA throughput   | `gemm`, `syrk` (rank-k), `trsm`, `symm` |

The jump that matters is L2 -> L3: only at L3 does work (n^3) outgrow data
(n^2), so intensity grows with n and the kernel can be made **compute-bound**.
L1 and L2 move as many (or more) bytes than they do flops, so the best a kernel
can do is **saturate memory bandwidth** -- no amount of blocking changes that.

```
   flop/byte (log)                                     L3 gemm
        ^                                             /   (compute-bound;
        |                                            /     ~93% FMA peak)
   O(n) |                                        ___/
        |                                    ___/
        |                                ___/   <- "roofline" knee
   ~0.25|====L1 dot/axpy==L2 gemv/ger====            (bandwidth-bound flat line)
        +-------------------------------------------------> problem size n
```

**Legend used in the diagrams**

```
  [b] or |a a a a|   one SIMD batch = W lanes (W = simd::batch<T>::size:
                     2 for SSE, 4 for AVX2, 8 for AVX-512, all fp64)
  fma(x,y,z)         fused multiply-add  x*y + z   (one rounding)
  bcast(s)           broadcast scalar s into all W lanes
  accK               an independent accumulator register (hides FMA latency)
  -->                stream direction (unit-stride memory sweep)
```

---

## Level 1 -- vector-vector (bandwidth-bound)

Two shapes: **reductions** (dot, nrm2 -> a scalar) and **maps** (axpy, scal ->
a vector). Both stream their operands once; the only kernel lever is SIMD width
plus, for reductions, *multiple accumulators* to hide the FP add/FMA latency.

### dot / nrm2 -- reduction with K independent accumulators

```
  dot:  s = SUM_i a[i]*b[i]            nrm2: s = sqrt( SUM_i a[i]*a[i] )

  a: |....|....|....|....| |....| ...   unit-stride, W lanes per batch
  b: |....|....|....|....| |....| ...
      \    \    \    \
       fma  fma  fma  fma   <- 4 batches per iteration, one per accumulator
        v    v    v    v
      acc0 acc1 acc2 acc3   K=4 INDEPENDENT chains (no dependency between them)

  per step:  acc0=fma(a0,b0,acc0)  acc1=fma(a1,b1,acc1) ...   (nrm2: b==a)
  finish:    s = reduce_add( (acc0+acc1)+(acc2+acc3) )  +  scalar tail
```

Why 4 accumulators: a single `acc = fma(a,b,acc)` chain stalls `Lfma` cycles per
step (each FMA waits for the previous result). K chains keep `K*Nfma` FMAs in
flight so the unit issues every cycle -- but since L1 is bandwidth-bound, this
only matters until the loads saturate the bus. (`reduce_dot`/`reduce_sum_squares`
in `simd/algorithm.hpp`.)

### axpy / scal -- map / stream (no reduction)

```
  axpy: y[i] += alpha*x[i]                 scal: x[i] *= alpha
  va = bcast(alpha)

  x: |....|....|....| ...      for each batch:
  y: |....|....|....| ...         yb = fma(va, xb, yb);  store yb     (axpy)
       |    |    |                xb = va * xb;          store xb     (scal)
     fma  fma  fma   ----> store back in place / into y

  no accumulator chain (each lane independent);  1-2 streams in, 1 out.
```

Both are pure bandwidth: ~2 flops per 16-24 bytes moved. The kernel's whole job
is to issue wide aligned loads/stores and reach the DRAM ceiling.

---

## Level 2 -- matrix-vector (bandwidth-bound)

The matrix `A` (O(n^2) bytes) is read **exactly once** -- unavoidable, and the
bottleneck. Blocking only removes re-reads of the *small reused vector*; it
cannot remove the A traffic, so L2 also tops out at the memory ceiling.
Orientation decides which axis is unit-stride and therefore the kernel shape.

### gemv, row-major A -- inner-product form, register-block MR rows

`y[i] = dot(A_row_i, x)`. A rows are unit-stride -> SIMD-dot each row with x.
Block **MR rows** so one `x` batch is loaded once and reused MR times.

```
        x   (loaded once per j-batch, reused across MR rows)
        |....|....|....| ...  -->
          |  (reused MR times)
          v
   A (row-major)                         y
 i0|A0,jb..|......|   fma(A0,xb,acc0) --> |y0| <- reduce(acc0)
 i1|A1,jb..|......|   fma(A1,xb,acc1) --> |y1| <- reduce(acc1)
 i2|A2,jb..|......|   fma(A2,xb,acc2) --> |y2| <- reduce(acc2)
 i3|A3,jb..|......|   fma(A3,xb,acc3) --> |y3| <- reduce(acc3)
    \_W_/                                 MR=4 accumulators (registers)
   sweep j: 0,W,2W,..,N                   reduce after the j-sweep
```

### gemv, col-major A -- axpy-of-columns form, y-strip resident

`y += x[j]*A[:,j]`. A columns are unit-stride -> keep a strip of `y` resident in
registers and stream columns; y is written once.

```
   A (col-major)     x[j]            y-strip (resident in registers)
   col j unit-stride (scalar)        |....|....| ... |
 ^ |a|a|a|a| ...      |xj| --bcast-->   per column j:
 M |a|a|a|a|          +--+                yb_k = fma( bcast(xj), Acol_j[k], yb_k )
 v |a|a|a|a|        stream j ---->     write y-strip ONCE at the end
   +-+-+-+-+                           (no horizontal reduce: accumulate in-lane)
```

### ger -- rank-1 update (the L2 outer-product)

`A := A + alpha * x * y^T`, i.e. `A[i,j] += alpha*x[i]*y[j]`. This is GEMV's
twin: GEMV contracts a matrix and a vector into a vector (inner product); GER
expands two vectors into a matrix (outer product).

```
   x (M)         y^T (N)               A (M x N)  +=  alpha * x (x) y^T
   |x0|     |y0 y1 y2 ... |
   |x1|  (x)              =       row-major, vectorize over j:
   |x2|                             for i:  ax = bcast(alpha * x[i])
   |..|                               for jb: A[i,jb] = fma(ax, y[jb], A[i,jb])
   +--+                            reads AND writes all of A once  ->  bandwidth.
```

GER touches every element of A once (read+write) for 2 flops each -> intensity
~0.25, **memory-bound, squarely Level 2**. Note the inner line
`A[i,jb] = fma(ax, y[jb], A[i,jb])` -- hold that thought; it reappears at L3.

---

## Level 3 -- matrix-matrix (compute-bound)

Now work (n^3) dwarfs data (n^2): each element of A and B is *reused* O(n) times,
so with the right blocking the kernel becomes FMA-bound and can approach machine
peak. The cost is a 5-loop nest that stages operands through the cache hierarchy
plus **packing** them into contiguous, SIMD-friendly panels.

### gemm -- the GotoBLAS/BLIS 5-loop nest + packing + micro-kernel

```
 Loop 5  jc: N step nc     B,C columns ---------------> nc panel targets L3
   Loop 4  pc: K step kc                                pack B(pc,jc) -> Bc
     Loop 3  ic: M step mc                              pack A(ic,pc) -> Ac (L2)
       Loop 2  jr: nc step nr     -- macro-kernel -->   Bc micro-panel (L1)
         Loop 1  ir: mc step mr
            +------------------------- MICRO-KERNEL (#88) ------------------+
            |  C[ir:+mr, jr:+nr] += Ac panel (mr x kc) * Bc panel (kc x nr) |
            |  mr x nr C microtile lives in REGISTERS across the kc loop    |
            +---------------------------------------------------------------+

   packing makes the hot loop read contiguous, aligned bytes (and minimizes
   TLB misses); kc->L1, mc x kc->L2, kc x nc->L3 keeps each reused operand in
   the cache level whose bandwidth matches its reuse distance.
```

### the micro-kernel = kc rank-1 updates, accumulated in registers

Here is the punchline. The GEMM micro-kernel's inner loop is **a sequence of
rank-1 updates** of the `mr x nr` C microtile -- the *same* outer-product step
as GER -- except the tile stays in vector registers across all `kc` of them, so
C is written to memory only **once** instead of every update:

```
  C_tile  (mr x nr, in REGISTERS)   for p = 0 .. kc-1:
                                        a_p = mr values of A (one packed column)
   p=0:  C_tile += a_p (x) b_p          b_p = nr values of B (one packed row)
   p=1:  C_tile += a_p (x) b_p          C_tile += a_p (x) b_p   <- rank-1 update
   p=2:  C_tile += a_p (x) b_p
    ...  (kc rank-1 updates)         realized as mr*nr FMAs into the tile:
   flush: C += C_tile  (ONCE)           for i: ai=bcast(A[p,i])
                                          for jb: c[i][jb]=fma(ai,b[jb],c[i][jb])

         mr=4, nr=8 (AVX2 fp64):  C tile = 4 x 2 batches = 8 registers
```

Compare to GER's inner line `A[i,jb] = fma(ax, y[jb], A[i,jb])`: byte-for-byte
the same FMA, but GER reloads/stores A every update (bandwidth-bound, L2) while
the micro-kernel accumulates `kc` updates in registers and stores once
(compute-bound, L3). **The difference between L2 and L3 is not the arithmetic --
it is where the accumulator lives.**

Measured: this micro-kernel reaches ~72.5 GFLOP/s on one AVX2 P-core, ~93% of
the FMA peak (#88).

---

## Where do rank-1 updates fit? (the taxonomy answer)

A **rank-1 update** is `A += alpha * x * y^T` -- an outer product accumulated
into a matrix. By the work/data rule it is a **Level-2** operation: it is the
BLAS routine `ger` (`sger`/`dger`), O(n^2) work over O(n^2) data, memory-bound.

But the rank-1 update is also the **atomic step of the Level-3 micro-kernel**:
GEMM is built as `kc` rank-1 updates of a register-resident tile, repeated over
the blocking nest. So rank-1 updates sit exactly on the **L2 <-> L3 boundary**:

```
   rank-1 update   A += x (x) y^T
        |
        +--- as a standalone op, writing A to memory each time ........ L2  (ger)
        |
        +--- accumulated kc-deep in registers, written once ........... L3  (gemm micro-kernel)

   and one step up:
   rank-k update   C += A (x) B  (k columns)  = kc rank-1 updates ...... L3  (syrk/gemm)
```

The progression is clean:
- **rank-1, to memory** = GER = **L2**.
- **rank-1, in registers, x kc** = the GEMM micro-kernel inner loop.
- **rank-k** (a block of rank-1s) = GEMM / SYRK = **L3**.

Equivalently: GEMV/GER are the inner-/outer-product *primitives*; GEMM is what
you get when you make the outer product deep enough (`kc`) that the operands are
reused enough times to keep the FMA units, not the memory bus, busy.

---

## Summary: what each level optimizes for

| | L1 (dot/nrm2/axpy/scal) | L2 (gemv/ger) | L3 (gemm) |
|---|---|---|---|
| Bound by | DRAM bandwidth | DRAM bandwidth | FMA throughput |
| Blocking | SIMD width only | register-block the reused vector | 5-loop cache nest (kc/mc/nc) |
| Packing | none | none | yes (A, B panels) |
| Reuse target | x, y in registers | x or y strip in registers | mr x nr C tile in registers |
| Accumulators | K (reductions) | MR rows / y-strip | mr x nr microtile |
| Kernel goal | hit the memory ceiling | hit the memory ceiling | ~peak FMA (~93%) |
| MTL5 file | `simd/algorithm.hpp` | `mult.hpp` (#87) | `gemm_microkernel.hpp` + macro (#90) |
