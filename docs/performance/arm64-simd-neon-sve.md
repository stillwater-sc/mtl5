# ARM64 SIMD: NEON, fixed-length SVE2 and scalable SVE2

**Status:** NEON and fixed-length SVE2 are built and tested in CI (#717). Scalable
SVE2 is still demoted to scalar code; that is #427. **No performance number in
this document has been measured.** Every claim about speed is a hypothesis with
the experiment that would test it.

**Scope:** what MTL5 does on each of the three ARM64 vector targets, the
requirements that #716 and #717 uncovered, and the performance engineering each
target needs.

## Why there are three targets, not one

On AArch64 the build flags decide which vector ISA Highway compiles for, and
MTL5's `simd::batch<T>` reacts to each one differently:

| target | lane count known at | what `batch<T>` does | typical flags |
|---|---|---|---|
| **NEON** | compile time (128 bits) | Highway, `float_lanes == 4` | the default (`-march=armv8-a`) |
| **fixed-length SVE2** (`HWY_SVE2_128`, `HWY_SVE_256`) | compile time (`-msve-vector-bits=N`) | Highway, stored as a fixed-length ACLE type (#717) | `-mcpu=neoverse-n2 -msve-vector-bits=128` |
| **scalable SVE2** (`HWY_SVE2`, `HWY_SVE`) | run time only | **scalar fallback, `size == 1`** | `-march=armv9-a`, `-mcpu=neoverse-n2`, `-march=native` on SVE hardware |

`batch<T>::size` is a `constexpr` that the GEMM tile shape, the packing layout
and the blocking parameters are derived from at compile time. A scalable
target has no such constant, so `batch.hpp` turns Highway off for it
(`HWY_HAVE_SCALABLE` → `MTL5_SIMD_USE_HIGHWAY 0`).

## What we found (#716, #717)

### The CI hardware

The `ubuntu-24.04-arm` runners are **Neoverse N2**, with NEON, SVE, SVE2, I8MM and
BF16, at a **128-bit** SVE vector length (`/proc/sys/abi/sve_default_vector_length`
= 16). Both the GCC 13 and Clang 18 lanes report it; the probe added in #716
prints it as a check annotation on every run.

### What Highway 1.4.0 selects

These were found by compiling Highway 1.4.0's headers and reading `HWY_TARGET`
and `HWY_HAVE_SCALABLE` (GCC 15 cross compiler and Clang 18). The CI lanes in
#717 now assert the GCC 13 and Clang 18 results on the runner itself.

| flags | GCC | Clang 18 |
|---|---|---|
| `-march=armv8-a` (default) | `NEON_WITHOUT_AES` | `NEON_WITHOUT_AES` |
| `-march=armv9-a` or `-mcpu=neoverse-n2` | **`SVE2`, scalable** | `NEON_WITHOUT_AES` |
| `-mcpu=neoverse-n2 -msve-vector-bits=128` | `SVE2_128` | `NEON_WITHOUT_AES` |
| `-march=armv8.2-a+sve -msve-vector-bits=256` | `SVE_256` | `NEON_WITHOUT_AES` |

Clang never gets an SVE target in Highway 1.4.0. `hwy/detect_targets.h` marks
`SVE2_128` broken before Clang 21, and `SVE`, `SVE_256` and `SVE2` broken before
Clang 22. Highway then falls back to NEON with no diagnostic, which is why every
CI lane now names the target it expects.

### Requirements this uncovered

**R1. `batch` cannot hold a sizeless SVE type.** Highway's SVE vectors are ACLE
types such as `svfloat32_t`. These are *sizeless*: they can't be class members or
array elements, and `sizeof` doesn't apply to them. Before #717, `batch` didn't
compile for *any* SVE target. The scalable targets never reached that code only
because the scalar demotion came first. #717's initial hypothesis was that the
`SVE2_128` target would work with the existing `batch`. It wouldn't have:

```
batch.hpp:181: error: member variables cannot have SVE type 'svfloat32_t'
```

On the fixed-length targets ACLE lets the width be attached to the type, so
`batch` stores
`svfloat32_t __attribute__((arm_sve_vector_bits(__ARM_FEATURE_SVE_BITS)))`
(`simd::detail::vec_storage`). That is an ordinary sized type, and it converts
implicitly to and from the sizeless one. The scalable targets have no width to
attach, so this fix doesn't reach them.

**R2. `SVE2_128` needs I8MM and BF16 in the compiler flags.** Highway compiles
its `SVE2_128` functions with `+sve2,+i8mm,+bf16` (plus `+sve2-aes` where
available). `-march=armv9-a -msve-vector-bits=128` selects the target but lacks
I8MM and BF16. Every inlined Highway call from MTL5 code then fails with
`inlining failed in call to 'always_inline' ...: target specific option
mismatch`. Use `-mcpu=neoverse-n2` (or `-march=armv9-a+i8mm+bf16`) together with
`-msve-vector-bits=128`.

**R3. A fixed-length binary is correct only at its vector length.** Code built
with `-msve-vector-bits=128` assumes a 128-bit SVE vector length. On a machine
with a different vector length it computes wrong answers. It does not trap.
- Neoverse N2, V2 and Grace (Graviton 4) are 128 bits.
- Neoverse V1 (Graviton 3) is 256 bits.
- A64FX is 512 bits.

CI guards this with a runner check. A library user has no such guard. MTL5 should check
`svcntb() * 8 == __ARM_FEATURE_SVE_BITS` once at startup in a
fixed-length build, and refuse to run on a mismatch. This is cheap and turns a
silent wrong answer into a diagnosis.

**R4. `MTL5_NATIVE_ARCH=ON` on an SVE machine with GCC removes SIMD entirely.**
`-march=native` on N2, V1, V2 or Grace selects a scalable target, and `batch`
falls to one lane. The user who asks for the *most* tuned build gets the
*least* vectorized one, slower than the portable default (NEON). Clang users
don't hit this, but only because Highway refuses SVE for Clang (R2's table). The
#717 `native` lane pins the behaviour (`size == 1`) so it can't change
unnoticed. Mitigations, in increasing cost:
1. A configure-time warning when `MTL5_NATIVE_ARCH=ON`, `MTL5_WITH_HIGHWAY=ON`
   and the compiler defines `__ARM_FEATURE_SVE` without
   `__ARM_FEATURE_SVE_BITS`.
2. Append `-msve-vector-bits=<detected>` to the native flags when configuring
   on the machine that will run the binary, which is what "native" already
   means. This must also apply R2 and R3.
3. Prefer NEON for a scalable target until #427 lands, by defining
   `HWY_DISABLED_TARGETS=HWY_ALL_SVE` in native builds. Preprocessing confirms
   that with GCC and `-mcpu=neoverse-n2` this selects `NEON_WITHOUT_AES`. It
   always works and loses only what SVE adds over NEON at 128 bits (see below).

This needs its own issue.

**R5. Every SIMD lane must assert its target.** A lane that builds the wrong
target still passes the suite, because every target computes correct answers.
`test_batch` now fails to *compile* when the lane's
`MTL5_TEST_EXPECT_HWY_TARGET` differs from `HWY_TARGET`. It also requires
`size > 1` on AArch64 as it already did on x86-64, except on a scalable target,
where it requires `size == 1` until #427.

## Performance engineering, per target

Nothing below has been measured. Each subsection ends with the experiment that
would settle it.

### NEON (128-bit, 32 registers)

**Where it stands.** This is the target every ARM64 Highway build gets by
default, and the only ARM target with measurements. The register tile is
**8×4 fp64 / 8×8 fp32**, with 16 accumulators, half of the 32-register file.
It was measured on a Cortex-A78C (`simd/blocking.hpp`, `target_hw_traits`). The
int8 quad kernel uses `SDOT`/`UDOT` natively and `USDOT` when I8MM is present.

**What it needs:**

- **Per-core FMA throughput.** `hw_traits` carries one `fma_latency` and
  `fma_units` for all of aarch64. The BLIS bound
  `mr * nr >= Nvec * Lvfma * Nvfma` changes sharply across cores:
  - A78 and N2: 2 FP/ASIMD pipes, so the 16-accumulator tile covers the
    pipeline.
  - V1 and V2: 4 pipes, so the same tile covers half of it. fp64 needs
    `2 * 4 * 4 = 32` lane-accumulators, i.e. 16 registers at 2 lanes each, with
    no headroom left.

  V-class cores probably want `acc_vectors` near 24 and a tile like 12×4 fp64.
  That is the candidate that *lost* on the A78C. So the tile has to become a
  per-core decision, not a per-ISA one.
- **FMA by element.** The micro-kernel broadcasts `A(i, p)` into a register and
  then issues an FMA. NEON has `FMLA Vd.2D, Vn.2D, Vm.D[lane]`, which reads the
  multiplier straight from a lane. If the packed A panel is loaded one q-register
  at a time (2 fp64 or 4 fp32 values), the broadcasts disappear: on a core
  with two load ports, they are the instruction slots competing with the B-panel
  loads. Check that the compiler already fuses `dup` + `fmla`. If it doesn't, the fix
  is a NEON-specific micro-kernel: Highway 1.4.0 has no lane-indexed FMA
  operation.
- **Paired loads.** The B micro-panel row should be loaded with `LDP q, q` (32
  bytes per instruction). Check the disassembly of `gemm_microkernel<double,
  double, 8, 4>`.
- **int8 and bf16 matrix instructions.** I8MM's `SMMLA`/`UMMLA`/`USMMLA` and
  BF16's `BFMMLA` compute a 2×8 by 8×2 product into a 2×2 int32 (or fp32) tile
  per instruction. That is **twice the multiply-accumulates of `SDOT`** per
  instruction, but it needs a different packing: pairs of rows of A and pairs of
  columns of B, each 8 bytes deep. This is the ARM counterpart of the x86 quad
  kernel (#451) and probably the largest single int8 gain available on N2, V2
  and Grace. N2 has both features.

**Experiment:** `bench_blocking_ab` on an N2 or Graviton 4 (V2) host. Sweep
`mr` and `acc_vectors` exactly as the A78C sweep did. Add the `SMMLA` kernel as
an arm against the `SDOT` quad kernel at n = 1024.

### Fixed-length SVE2 (`SVE2_128` on N2/V2/Grace, `SVE_256` on V1)

**Where it stands.** It compiles and runs in CI as of #717. The `sve2-128` lane
runs the full suite on N2. `batch` is identical in shape to NEON at 128 bits:
`float_lanes == 4`, the same tile and the same packing.

**What SVE adds at 128 bits** is not width. On N2, V2 and Grace, SVE and NEON
share the same 128-bit datapath, so dense GEMM *throughput* should be the same.
What SVE adds:

- **Predicated tails.** `WHILELT` builds a lane mask for the last partial
  vector, so a loop needs no scalar remainder and no zero-padded temporary.
  MTL5's edge tiles go through "a zeroed mr×nr temporary so the micro-kernel
  only ever writes full tiles" (`gemm_blocked.hpp`). Masked loads and stores
  would remove that copy. That matters for skinny and odd shapes, not for large
  square ones. Highway exposes it as `FirstN` + `MaskedLoad` / `BlendedStore`,
  which also compile on NEON (emulated), so one source can serve both.
- **Gather/scatter.** `LD1D` with a vector of indices. This is relevant to the
  CRS kernels (`mult_sparse_crs_rows`, CRS×dense), whose `x[col[k]]` access is
  a gather. Whether hardware gathers beat scalar loads on N2 is unknown and
  often negative; measure it before building on it.
- **Wider integer dot products.** SVE2's `SDOT`/`UDOT` and `USDOT` (with I8MM)
  match NEON's. SVE's `SMMLA` exists too. No width gain at 128 bits.

**What `SVE_256` adds** on V1 (Graviton 3) *is* width: 8 fp32 or 4 fp64 lanes,
twice NEON. The tile follows `batch::size` automatically (`nr` must be a
multiple of the width). But the accumulator budget, the B-panel `kc` and the
packing alignment were tuned at 128 bits and need re-deriving. V1 has SVE but
not SVE2, so the target is `SVE_256` (`-mcpu=neoverse-v1
-msve-vector-bits=256`), not `SVE2_*`.

**Requirements before shipping fixed-length SVE to users:** R2 (flags), R3
(vector-length check at startup), and recording the target in every benchmark
sidecar. `simd::backend_name()` already reports `SVE2_128`, so a NEON number
and an SVE number can't be confused afterwards.

**Experiment:** on the same N2 host, build the same commit twice: NEON (default)
and `SVE2_128`. Compare GEMM n = 256..2048, GEMV, dot and the CRS kernels.
The prediction is parity on large GEMM, an SVE win on odd and skinny shapes
from predication, and an unknown result for CRS gathers. On a V1 host,
`SVE_256` against NEON on GEMM is the one place a width gain should appear.

### Scalable SVE2 (#427)

**Where it stands.** It's demoted to scalar code: `size == 1`, no SIMD at all.
This is what `-march=armv9-a`, `-mcpu=neoverse-n2` and GCC's `-march=native` on
SVE hardware produce today (R4).

**Why it is hard.** The vector length is a run-time value, and MTL5's dense
kernels are built on it being a compile-time one:
- `gemm_microkernel<TC, TAB, MR, NR>` static-asserts `NR % W == 0`, and its C
  tile is `batch c[MR][NB]`. That is an *array* of vectors, and arrays of
  sizeless types are ill-formed (`array elements cannot have SVE type`). R1's
  fix is unavailable because there is no width to attach.
- `simd::default_blocking<T>` derives `mr × nr` at compile time from
  `simd::width<T>`.
- The packed-panel layouts (`Ap[p*MR + i]`, `Bp[p*NR + j]`) bake `NR` into the
  index arithmetic.

**Two designs:**

1. **Multi-versioned fixed-length kernels.** Compile the hot kernels (GEMM
   micro-kernel and packing, GEMV, dot, axpy) once per vector length that ships
   in hardware: 128 (N2, V2, Grace), 256 (V1) and 512 (A64FX). Each copy uses
   R1's fixed-length storage. Pick one at run time from `svcntb()`. Highway's
   dynamic dispatch (`HWY_DYNAMIC_DISPATCH`, `foreach_target.h`) already treats
   `SVE2_128` and `SVE_256` as separate targets. Everything above the kernel
   (blocking, packing, threading) stays `constexpr` per copy.
   - *Cost:* about three copies of the SIMD kernels in the binary, and a
     dispatch layer that MTL5's static-dispatch design doesn't have yet.
   - *Gain:* SIMD on every SVE machine that exists, and it fixes R4 without
     the R3 hazard, since the copy is chosen *by* the vector length.
2. **True vector-length-agnostic kernels.** `size` becomes `Lanes(d)` at run
   time. The C tile becomes `MR × NB` named locals produced by template
   unrolling, because an array is not allowed. `nr = NB * Lanes(d)` becomes a
   run-time blocking parameter, and packing takes `nr` as an argument.
   - *Cost:* the blocking model, the packing, the edge-tile logic and every
     `constexpr` consumer of `simd::width` get a run-time path, and the
     compiler can no longer fully unroll on `NR`.
   - *Gain:* one binary for any length, including future ones and RISC-V RVV,
     which has the same problem.

**Recommendation:** design 1 first. It reuses everything #717 built and covers
every SVE implementation that ships. It also removes the native-build cliff
(R4). Design 2 is the end state #427 describes, but it rewrites the dense core
for hardware (VL > 512, RVV) MTL5 has no user on yet. The scalable CI lane
(`native`) turns into the dispatch test: it should assert `size > 1` once the
dispatcher picks the `SVE2_128` copy on N2.

**Experiment:** none possible until one design lands. At that point, the
`native` CI lane becomes the correctness test. For performance, compare against
the `SVE2_128` build on the same host; dispatch overhead should be invisible
above n = 64.

## Hardware that would settle these

Shared CI runners give correctness, never timing (see
[benchmarking methodology](benchmarking-methodology.md)). The numbers have to
come from dedicated hosts, ideally one of each width:

| host | core | SVE | VL | settles |
|---|---|---|---|---|
| Graviton 4 / Grace | Neoverse V2 | SVE2 + I8MM + BF16 | 128 | V-class tile (4 FP pipes), `SMMLA`, NEON vs `SVE2_128` |
| Azure Cobalt 100 | Neoverse N2 | SVE2 + I8MM + BF16 | 128 | the CI core, timed; predication on odd shapes |
| Graviton 3 | Neoverse V1 | SVE + I8MM + BF16 | 256 | `SVE_256` width gain, R3's mismatch hazard against 128 |
| A64FX | Fujitsu | SVE | 512 | the dispatch design at the widest length |

The [hardware expansion plan](hardware-expansion-plan.md) already lists the
scalar-fallback cliff as an open capability gap. This document is the
engineering plan for closing it.

## Related

- #427: vector-length-agnostic `batch<T>` (scalable SVE / RVV).
- #717: ARM64 Highway CI lanes and fixed-length SVE storage.
- #716: the runner SVE probe.
- #330: the x64 Highway lane.
- #643: the NEON register-file budget.
- #451: quad (int8) kernels.
