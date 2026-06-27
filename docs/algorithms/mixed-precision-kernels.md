# Mixed-Precision Kernels: Why, What, and How

This is the entry point to MTL5's reason for existing. MTL5 is a linear algebra
library built for **mixed-precision algorithm design** — the deliberate use of
*different* numeric precisions for storage, for computation, and for the result of
a single operation. This page explains why that matters, what MTL5 gives you to
express it, and how the fast path is actually built — using the SIMD *widening*
GEMM as the worked example. No prior numerical-analysis background is assumed.

For the deep architecture this page links out to two companion design notes:
[BLAS kernel architecture](../design/blas-kernel-architecture.md) (the blocking
and scheduling of the kernels) and
[Custom number types through the SIMD BLAS](../design/mixed-precision-custom-types-SIMD.md)
(how an arbitrary storage type flows through the SIMD seam).

---

## Why: the precision you store in is not the precision you compute in

Every floating-point format trades range and accuracy for size and speed. A
narrow type — `float`, `bfloat16`, IEEE `half`, an 8-bit float, a `posit<16,2>` —
is *cheaper* than `double` in every way that a modern machine cares about:

- **Memory bandwidth.** Half the bytes per element means half the traffic from
  DRAM and cache — and most linear algebra is bandwidth-bound, not compute-bound.
- **Cache footprint.** Twice as many elements fit in L1/L2, so blocked kernels
  reuse data more.
- **SIMD throughput.** A vector register holds twice as many `float` lanes as
  `double` lanes, so each instruction does twice the work.
- **Energy.** Narrower multipliers and fewer bytes moved cost less power.

The catch is accuracy. But here is the key observation that the whole field turns
on: **for the reductions that dominate linear algebra — dot products, matrix
products, norms — the error comes overwhelmingly from the *accumulation*, not
from the *storage*.**

Consider a dot product of `n` values. Storing each operand as `float` costs one
rounding of ~6 decimal digits per value — usually harmless. But summing `n`
products *in* `float` rounds after every addition, and those errors accumulate;
worse, if the terms have mixed signs (**catastrophic cancellation**) the relative
error can blow up to `O(n · ε)` or far more. The fix is not to store wider — it is
to **accumulate** wider:

```text
sum stored in float  :  error grows with n, cancellation amplifies it
sum stored in double :  ~16 digits of headroom — the rounding noise floor drops
                        by ~10 orders of magnitude, at almost no storage cost
```

So the winning move is: **store narrow, accumulate wide.** You pay the narrow
storage cost (bandwidth, cache, SIMD lanes) and you keep the wide accumulation
accuracy. That is the entire thesis of mixed precision, and it is why MTL5 treats
the accumulator as a first-class, independent choice. (For the formal error
picture — backward vs forward error, the condition number — see
[Measuring Solver Accuracy](measuring-solver-accuracy.md).)

---

## What: the three precisions — Element → Accumulate → Result

MTL5 models every reduction-style operation as having **three** independent
precisions, not one:

| Precision | Role | Example |
|---|---|---|
| **Element** | how operands are *stored* | `float` matrices `A`, `B` |
| **Accumulator** | how the inner product is *computed* | `double` (or a quire, or a custom type) |
| **Result** | how the answer is *serialized* | whatever `C`'s element type is |

The accumulate→result conversion is **fused into the store** — there is no
separate down-convert pass over the output. This is the *Element → Accumulate →
Result* model, the headline of the mixed-precision tensor-ops work (epic #157).

It is expressed once, generically, through a small policy:

```cpp
// mtl/math/accumulator_traits.hpp
template <typename Acc, typename Value>
struct accumulator_traits {
    static void clear(Acc&);                       // acc = 0
    static void add_product(Acc&, Value a, Value b); // acc += a*b  (in Acc precision)
    template <typename Result> static Result value(const Acc&); // round out to Result
};
```

`add_product` is the seam where the wide accumulation happens; `value<Result>` is
the fused round-out. Because the operation is named (not hard-wired to a hardware
FMA), the accumulator can be a software type — a [posit/quire](../design/mixed-precision-custom-types-SIMD.md),
a logarithmic number system, an exact super-accumulator — not just a wider IEEE
float.

### The API

Pass the accumulator as an explicit template argument; omit it (`void`) for the
unchanged, hardware-default behavior:

```cpp
mtl::mult(A, B, C);                 // default: accumulate in C's type (unchanged)
mtl::mult<double>(Af, Bf, C);       // float operands, fp64 accumulation
mtl::dot<double>(xf, yf);           // float vectors, fp64 dot
mtl::two_norm<double>(xf);          // float vector, fp64 sum-of-squares
```

Two guarantees make this safe to adopt incrementally:

- **`Accumulator = void` is byte-identical** to the previous behavior and still
  dispatches to external BLAS / the native fast kernels when types qualify.
- **A custom accumulator forces the native kernel.** External BLAS uses a
  hardware-fixed accumulator and cannot honor a custom one, so MTL5 routes any
  non-default accumulator to its own kernels (`interface::accumulator_allows_blas_v`).

### Where mixed precision pays off at the algorithm level

The accumulator policy is the building block; the *algorithm* is where it earns
its keep. The canonical pattern is **mixed-precision iterative refinement**:
factor a matrix in a low precision (fast, small), then refine the solution with a
residual computed in a high precision. The low-precision factor is "wrong" by a
controlled amount; a few high-precision residual corrections recover full
accuracy. MTL5's `sparse::iterative_refine` is the reusable, dependency-free core
of exactly this loop, and the supernodal/KLU solvers expose a low-precision factor
with a high-precision refine path on top of it.

---

## How: the SIMD widening GEMM, step by step

The most instructive worked example is matrix–matrix multiply (GEMM), the
compute-bound Level-3 kernel where mixed precision is both hardest and most
valuable. We want `mult<double>(A_float, B_float, C_double)` — float operands,
fp64 accumulation — to run at *SIMD speed*, not as a scalar loop.

### Start from the same-type blocked GEMM

The native same-type GEMM is the classic GotoBLAS/BLIS design (covered in detail
in [BLAS kernel architecture](../design/blas-kernel-architecture.md)):

1. **A five-loop cache-blocking nest** partitions `C`, `A`, `B` into blocks sized
   to L3 / L2 / L1, so each byte fetched from memory is reused many times before
   eviction.
2. **Packing** copies each block into a contiguous, canonically-ordered scratch
   panel, so the innermost kernel sees unit-stride data regardless of the
   matrices' storage orientation.
3. **A register micro-kernel** holds a small `MR × NR` tile of `C` entirely in
   vector registers and accumulates `kc` rank-1 updates into it with fused
   multiply-adds (FMAs) — this is where the FLOPs happen at peak rate.

These three optimizations — *blocking* for cache reuse, *packing* for unit
stride, *register tiling* for FLOP throughput — are independent of precision.

### The naive mixed path, and why it is slow

Before widening, the mixed-accumulator case fell back to the **generic scalar
kernel**: a triple loop calling `accumulator_traits::add_product` per element.
Correct, but it abandons all three optimizations above — no SIMD, poor reuse. On
a 512×512 problem it is **~16× slower** than it needs to be. The accuracy was
right; the speed was left on the table.

### The widening trick: widen on load, accumulate in wide registers

The insight is that the micro-kernel does not need the operands and the
accumulator to be the *same* type. It needs:

- operand panels that are cheap to stream (so: keep them **`float`**), and
- an accumulator tile that is accurate (so: make it **`double`** registers).

The bridge is a **widening SIMD load**: read a vector of `float` lanes and widen
them to `double` lanes in one instruction (`batch<double>::load_widen<float>`,
the same primitive that gave `dot` its widening fast path, #165). The micro-kernel
becomes parameterized on two types — `TC` (accumulator/C) and `TAB` (operands):

```cpp
template <typename TC, typename TAB, std::size_t MR, std::size_t NR>
void gemm_microkernel(std::size_t kc, const TAB* Ap, const TAB* Bp,
                      TC* C, std::size_t ldc) {
    using B = simd::batch<TC>;                 // accumulate in TC-wide registers
    constexpr std::size_t W  = B::size;        // SIMD lanes per batch
    constexpr std::size_t NB = NR / W;         // batches spanning the NR columns
    constexpr bool widen = sizeof(TAB) < sizeof(TC);

    B c[MR][NB] /* = 0 */;
    for (std::size_t p = 0; p < kc; ++p) {
        B b[NB];                               // load one B row, widening on the way in
        for (std::size_t jb = 0; jb < NB; ++jb)
            b[jb] = widen ? B::load_widen<TAB>(Bp + p * NR + jb * W)   // float -> double, one op
                          : B::load_unaligned (Bp + p * NR + jb * W);
        const TAB* ap = Ap + p * MR;
        for (std::size_t i = 0; i < MR; ++i) {
            const B a_i(static_cast<TC>(ap[i]));                       // broadcast, widened
            for (std::size_t jb = 0; jb < NB; ++jb)
                c[i][jb] = fma(a_i, b[jb], c[i][jb]);                  // FMA in TC
        }
    }
    /* flush c[i][jb] into C */
}
```

Two properties make this a clean change rather than a risky fork:

- **The same-type path is byte-identical.** With `TAB == TC` the `if constexpr`
  selects the ordinary `load_unaligned` and the cast is the identity, so the
  compiler emits the *exact* original kernel. No regression to the heavily-used
  `float×float→float` / `double×double→double` GEMM.
- **Blocking follows the accumulator.** The `MR × NR` tile and block sizes are
  chosen for `TC` (double) so the C microtile maps to double registers; the
  operand panels are simply packed in the narrower `TAB`.

The dispatcher routes only the SIMD-eligible case — `Accumulator = double`,
`float` operands, `double` result, dense and contiguous — to this widening kernel;
everything else (software accumulators, other widenings) keeps the generic kernel.

### What it buys

Measured on a Highway (SIMD) build, single-threaded:

| operation | mixed-precision speedup vs scalar generic |
|---|---|
| `dot` float→double (#165) | ~2.6× |
| `gemm` float→double (#176), N=256 | ~10× |
| `gemm` float→double (#176), N=512 | ~16× |

…and the result matches a `double` GEMM run on the exact widened operands to
within summation-order rounding — full fp64 accumulation accuracy, at fp32
storage cost, at SIMD speed. The same-type kernels are untouched.

---

## Designing your own mixed-precision algorithm

The takeaways that generalize beyond GEMM:

1. **Separate storage from compute precision on purpose.** Ask, per operation,
   "what must be stored accurately?" and "where does error concentrate?" — they
   are rarely the same answer. Error concentrates in *accumulation* and in
   *residuals*.
2. **Spend precision where the error is, not uniformly.** A wide accumulator on a
   narrow-stored reduction is almost free and recovers most of the lost accuracy;
   widening the *storage* would cost bandwidth for little gain.
3. **Recover accuracy with iterative refinement.** A low-precision factorization
   plus a high-precision residual loop reaches high-precision answers at
   low-precision factor cost — the dominant pattern for direct solvers.
4. **The seam is type-generic.** The `batch<T>` SIMD seam and `accumulator_traits`
   work for `posit`/LNS/quire and hypothetical 8- and 12-bit formats, not only
   IEEE `float`/`double` — see
   [Custom number types through the SIMD BLAS](../design/mixed-precision-custom-types-SIMD.md).
5. **Make the fast path opt-in and the default path unchanged.** Mixed precision
   is a tool you reach for deliberately; the `Accumulator = void` default keeps
   existing code bit-for-bit identical and full-speed.

## See also

- [BLAS kernel architecture](../design/blas-kernel-architecture.md) — the blocking and scheduling of the L1/L2/L3 kernels.
- [Custom number types through the SIMD BLAS](../design/mixed-precision-custom-types-SIMD.md) — how an arbitrary storage type flows through the `batch<T>` seam.
- [Measuring Solver Accuracy](measuring-solver-accuracy.md) — residuals, norms, and the error model that motivates wide accumulation.
