# Custom number types through the SIMD BLAS: how an FP12 would flow

MTL5 is built for mixed-precision algorithm design, so its number-type state
space (posit8/16, FP8 e4m3/e5m2, a hypothetical **FP12**, LNS, fixpnt, bf16, …)
is far larger than the lane types any hardware vector ISA provides. x86 (AVX2/
AVX-512) gives you f32/f64 and a little bf16/f16; NEON adds f16; even a custom
RISC-V Vector (RVV) extension only has FMA units for a handful of element
widths. This document explains how a type the hardware has never heard of —
FP12 is the running example — flows through the native BLAS kernels, what
already supports it, and what would have to change (especially for a
vector-length-agnostic ISA like RVV).

It builds on the [BLAS kernel architecture](blas-kernel-architecture.md) doc
(the L1/L2/L3 taxonomy) and the native-BLAS performance epic (#82).

---

## 1. The reframing: storage type is not lane type

The type zoo is unbounded; the set of types the ISA can do FMAs in is tiny. You
reconcile them by **never requiring the kernel to vectorize the exotic type**.
Keep two type layers:

- **Storage / algebraic layer** — the full zoo. Lives in the containers
  (`dense2D<FP12>`, `dense_vector<FP12>`), i.e. in memory. This is where
  footprint and bandwidth are won.
- **Execution / lane layer** — a *small* set of types for which
  `mtl::simd::batch<T>` is actually implemented. This is where FLOP/s happen
  (f32/f64 today; plus whatever a custom extension adds).

The bridge between the layers is a **conversion policy**, which already exists
in MTL5 as `project_onto<Target>` / `embed_into<Target>`
(`include/mtl/operation/projection.hpp`), including a scaled (quantizing)
overload:

- `embed_into<Compute>(fp12)` — widen FP12 → f32 (narrower → wider, lossless).
- `project_onto<FP12>(accum, scale)` — narrow accumulator → FP12 (lossy, scaled).

The only real design question is *where on the dataflow those conversions sit*.
The BLAS taxonomy answers that differently at each level.

```text
   memory (storage type)         registers (lane type)        memory
   ┌───────────────┐   embed_into   ┌──────────────┐  project_onto  ┌────────┐
   │ FP12  zoo type │ ───────────▶ │ batch<f32>    │ ───────────▶ │ FP12   │
   │ (bandwidth)    │   (widen)     │ FMA / reduce  │   (narrow)    │        │
   └───────────────┘               └──────────────┘               └────────┘
        ▲ unbounded variety            ▲ small fixed set
```

---

## 2. The single seam: `batch<T>`

Everything above the SIMD layer — the L1 kernels (`simd/algorithm.hpp`), the
GEMM micro-kernel (`detail/gemm_microkernel.hpp`), and the packing routines
(`detail/gemm_pack.hpp`) — is written **once** against `mtl::simd::batch<T>`. A
new ISA is, in principle, "just" a new `batch<T>` backend: `load_aligned/
unaligned`, `store_*`, `+ - *`, `fma`, `reduce_*`. Implement those over your
intrinsics and the kernels above light up unchanged.

The catch is what `batch.hpp` is already explicit about — `size` is a
**compile-time constant**:

```cpp
// include/mtl/simd/batch.hpp
#  if HWY_HAVE_SCALABLE
#    define MTL5_SIMD_USE_HIGHWAY 0   // scalable (SVE/RVV): no constexpr lane count yet -> scalar
#  else
#    define MTL5_SIMD_USE_HIGHWAY 1
#  endif
...
static constexpr std::size_t size = HWY_MAX_LANES_D(D);
```

That `constexpr` propagates through the whole stack:

- `width<T> = batch<T>::size` (constexpr) → `derive_blocking<T>(width<T>)` is
  constexpr (`simd/blocking.hpp`), so `mr/nr/kc/mc/nc` are compile-time.
- The micro-kernel builds a **compile-time-shaped register tile**:

```cpp
// include/mtl/detail/gemm_microkernel.hpp
constexpr std::size_t W  = simd::batch<T>::size;
static_assert(NR % W == 0, "NR must be a multiple of the SIMD width");
constexpr std::size_t NB = NR / W;
simd::batch<T> c[MR][NB];          // mr x (nr/W) tile, register-resident
```

This is exactly what RVV's **vector-length-agnostic (VLA)** model does not give
you: `vl` is a runtime value (set by `vsetvli`, tunable via LMUL). So today RVV
and SVE fall back to the scalar `batch` (`size == 1`) — correct, but slow.
Supporting a custom V-extension *well* is therefore not only a new backend; it
is a second, length-agnostic flavor of the kernels (see §5).

---

## 3. How FP12 flows, level by level

Mapping onto the L1/L2/L3 taxonomy:

### L1 (dot / nrm2 / axpy / scal) — bandwidth-bound

FP12 is a pure win and trivial to support. L1 is limited by bytes moved, and
FP12 is ~⅔ the bytes of f32 (~3/8 of f64), so the same kernel moves more
elements per second. Mechanism: **widen on load** (`embed_into` or a hardware
unpack) into f32 lanes, reduce/axpy in f32, narrow on store. No packing, so the
conversion is per-load — you want it cheap.

### L2 (gemv / ger) — bandwidth-bound

Same story: the matrix `A` is read once and dominates, so storing `A` as FP12
directly cuts the bottleneck. Convert on load, accumulate wide.

### L3 (gemm) — compute-bound; the interesting case

Two independent wins, two mechanisms:

1. **Footprint.** FP12 panels let the `mc·kc` and `kc·nc` cache blocks hold more
   elements, so blocking should be derived from `sizeof(storage_type)`, not the
   compute type.
2. **Conversion is essentially free here.** The packing layer (#89) is the
   natural choke point: `pack_A`/`pack_B` already touch every element exactly
   once and emit contiguous panels. Fold the FP12 → compute-type `embed_into`
   **into the pack copy**. Because each packed element is reused O(n) times by
   the micro-kernel, the one-time widening amortizes to ~nothing. This is
   precisely why production mixed-precision GEMM converts during packing, not in
   the inner loop.
3. **Wide accumulator is mandatory.** Even with FP12 inputs, the `mr × nr` C
   microtile stays in a *wide* type (f32/f64) in registers across the whole `kc`
   loop; you `project_onto<FP12>` only at the final flush. The
   rank-1-updates-in-registers structure is unchanged — only the *operand* width
   shrinks.

```text
   L3 GEMM, mixed precision:

   A:FP12 ─pack+embed_into─▶ Ap:f32 ┐
                                    ├─▶ micro-kernel: C_tile:f32 += Ap ⊗ Bp   (kc deep)
   B:FP12 ─pack+embed_into─▶ Bp:f32 ┘                     │
                                                          ▼
                                          project_onto<FP12>(C_tile)  →  C:FP12   (once)
```

---

## 4. The generalization: one type becomes three

The kernels currently template on a single element type `T`. Mixed precision
needs a triple — **storage**, **compute** (the `batch<>` lane type the FMAs run
in), and **accumulate** (the C tile / reduction type, `>=` compute):

```cpp
// storage  : FP12 in memory
// compute  : the batch<> lane type (e.g. f32)
// accum    : C tile / reduction type (>= compute), e.g. f32 or f64
template <class Storage, class Compute, class Accum, std::size_t MR, std::size_t NR>
void gemm_microkernel(std::size_t kc, const Storage* Ap, const Storage* Bp,
                      Accum* C, std::size_t ldc);
```

and the pack hook becomes a convert-on-copy, `pack_A<Storage, Compute>`. The
`requires mtl::Scalar<T>` constraint already on `pack_A`/`pack_B` is the
placeholder for this: `Scalar` admits arithmetic **and** custom number types
(it requires `+ - * unary- T{0}`, which the zero-padding uses), so a single
`std::is_floating_point` constraint was deliberately avoided — it would exclude
exactly the custom types this design exists to serve.

---

## 5. Two FP12 hardware scenarios

### (a) No native FP12 ALU — expressible in MTL5 today

FP12 never enters a vector register as FP12; it is a storage/transport format
only. `batch<FP12>` does not exist — you use `batch<f32>` and convert at the
seams (pack for L3, load/store for L1/L2). The only new code is the conversion
hook in packing and the `(Storage, Compute, Accum)` plumbing. This reuses the
entire existing kernel set and is the right first cut: pure software.

### (b) Custom RVV with FP12 lanes / FMA

Now FP12 graduates to an execution-layer type: implement `batch<FP12>` over the
custom intrinsics, and L3 can run FMAs *in* FP12 (still accumulating wide). But
a hardware-shaped wall appears immediately: **12 is not a power of two.** RVV's
`SEW` is 8/16/32/64; a 512-bit register holds 42.67 FP12s — non-integral,
non-byte-addressable lanes. Realistic options:

- **Storage-12 / register-16** (pragmatic): keep FP12 packed at 12 bits in DRAM
  for bandwidth, but a custom load-unpack widens to 16-bit lanes (`SEW=16`) in
  the register file. You get the bandwidth win and clean lanes; the FMA unit is
  "FP12-precision in a 16-bit lane."
- **True 12-bit packed lanes**: requires custom load/store-unpack functional
  units and a non-standard lane model — more invasive, and it breaks the
  assumption that `load_aligned` is a plain vector load.

Either way the decision is *contained inside* `batch<FP12>`: its `load`/`store`
encode the (un)packing and its `size` reports the lane count. The kernels above
do not care.

---

## 6. What would have to change in MTL5 (in dependency order)

1. **`batch<T>` VLA mode** — the gating change. Either (i) *pin* RVV to a fixed
   width at startup (one `vsetvlmax`, treat `size` as a runtime constant —
   minimal, loses some VLA benefit), or (ii) make `size` a runtime `vl` and
   convert the kernels to `vsetvli` strip-mining. Option (i) is the cheap unblock
   that lets RVV/SVE stop falling back to scalar.
2. **`gemm_microkernel`** — replace the compile-time `c[MR][NB]` tile +
   `static_assert(NR % W == 0)` with a VLA-tolerant shape (tile sized from
   runtime `vl`, tail handled by shrinking `vl` — which is actually *cleaner*
   than the x86 masked-tail dance).
3. **`derive_blocking`** — take `nvec` from runtime `vlmax` and
   `sizeof(Storage)` instead of constexpr `width<T>`; keep it a pure function so
   it can run once at init.
4. **Kernels → `(Storage, Compute, Accum)` triple** — `pack_A`/`pack_B` gain a
   convert-on-copy (`embed_into`); the L1 kernels gain widen-on-load; the C
   tile / reduction type becomes `Accum`.
5. **`interface/dispatch_traits.hpp`** — the one place the *policy* lives: given
   `Storage = FP12`, pick `Compute`/`Accum` (e.g. FP12 → f32 → f32, or FP12 →
   FP12-lane → f64 on custom hardware) and decide whether a native `batch<FP12>`
   exists. Everything else stays generic.

---

## 7. Bottom line

The state-space blow-up is contained because the exotic type only has to be a
**memory format plus a conversion**, not a vector lane type:

- `project_onto` / `embed_into` are the conversions;
- `pack_A` / `pack_B` (L3) and load/store (L1/L2) are the seams;
- `batch<T>` is the one place an ISA — including a custom RVV — plugs in;
- the **wide-accumulator** rule keeps numerics honest no matter how narrow the
  inputs get.

The genuinely hard part is not FP12's 12 bits — it is that **VLA breaks the
compile-time-lane-count assumption** baked through `batch::size` →
`derive_blocking` → the micro-kernel tile. That refactor (steps 1–2) is the real
prerequisite, and it is worth doing for SVE anyway; FP12 then rides in as a
`(Storage, Compute, Accum)` policy on top. The cleanest moment to factor the
triple in is when building the GEMM **macro-kernel** (#90), rather than
retrofitting it later.
