# Performance Engineering Case Study: Multi-Core Scaling of MTL5's Threaded Kernels

This is a worked performance-engineering investigation: a scaling result that
looked like a failure, the experiment designed to explain it, and what the data
actually showed. It doubles as a template for how to run — and how *not* to run —
a multi-core scaling experiment. The reference for the threading feature itself is
[On-Node Threading and Multi-Core Scaling](../algorithms/on-node-threading.md);
this page is the *why/what/how* of measuring it.

## TL;DR

- A first scaling sweep showed GEMM going from 3.35× (4 threads) to only 4.3×
  (8 threads) — apparently a wall at 4 cores.
- The cause was **the experiment, not the code**: the CPU-affinity mask
  (`taskset -c 0-7`) covered only **four physical cores hyperthreaded**, so
  "8 threads" ran on 4 cores.
- Re-run with one thread per **distinct physical core**, GEMM scales **6.32× on
  8 cores** (N=2048) and **6.85×** at N=4096 (~86% parallel efficiency).
- A problem-size sweep **rules out a memory-bandwidth ceiling for GEMM** (scaling
  *improves* with N). The Level-1/2 kernels and SpMV plateau at ~2–2.5× — that
  *is* a real DDR bandwidth roofline.

## 1. Why — the question

MTL5 added on-node threading (a persistent pool behind GEMM, GEMV, the L1 kernels,
and sparse SpMV; issue #221). The obvious question for any parallel library is:
**does it scale with cores?** An initial indicative sweep answered "not past 4":

| threads | GEMM GFLOP/s | speedup |
|---|---|---|
| 1 | 58.7 | 1.00× |
| 2 | 109.5 | 1.87× |
| 4 | 196.6 | 3.35× |
| 8 | 252.3 | **4.30×** |

Doubling the thread count from 4 to 8 bought only 1.28×. Either the software has a
scaling limiter around 4 cores, or the measurement is wrong. "It failed, ship the
number" is not acceptable — a 4.3×-on-8-cores claim is a strong statement about
the library, so it needs a root cause.

## 2. What — the system under test

- **CPU:** Intel Core i7-12700K — a **hybrid** part: 8 Performance cores (with
  2-way SMT → 16 logical CPUs) + 4 Efficiency cores (no SMT, 4 logical CPUs),
  20 logical CPUs total. P-cores ~4.9–5.0 GHz, E-cores ~3.8 GHz.
- **Kernels:** the native (no-external-BLAS) threaded paths — blocked GEMM,
  SIMD GEMV, `dot`/`nrm2`/`axpy`/`scal`, and sparse SpMV.
- **Metric:** fp64 GFLOP/s (or ms/matvec for SpMV), best of a short repeated run,
  and speedup vs a single thread. Thread count set with `MTL5_NUM_THREADS`.
- **Note:** single-run, shared developer machine → numbers are **indicative**
  (±10–15%). The *shape* of the scaling curve, not the absolute peak, is the
  object of study.

## 3. How — the experiment

### 3.1 Hypotheses

| # | Hypothesis | How to isolate it |
|---|---|---|
| H1 | Threads landed on **SMT siblings**, not distinct cores | pin one thread per physical core; compare against an all-SMT control |
| H2 | Threads landed on the slower **E-cores** | exclude E-cores from the affinity mask |
| H3 | **Memory bandwidth** saturates | sweep problem size N; bandwidth-bound scaling gets *worse* with N |
| H4 | **Thread-pool / per-region sync** overhead dominates | if compute-bound GEMM scales ~linearly, sync is not the wall; check N-dependence |
| H6 | **All-core turbo** lowers per-core clock under load | measure effective clock, or attribute the residual after H1–H4 |

### 3.2 The flaw: an affinity mask that lies

The original sweep pinned with `taskset -c 0-7`. On this CPU the logical→physical
map is:

```text
lscpu -e=CPU,CORE,MAXMHZ
CPU CORE  MAXMHZ      CPU CORE  MAXMHZ
  0    0  4900         8    4  4900
  1    0  4900         9    4  4900
  2    1  4900        10    5  4900
  3    1  4900        11    5  4900
  4    2  5000        12    6  4900
  5    2  5000        13    6  4900
  6    3  5000        14    7  4900
  7    3  5000        16..19  8..11  3800   (E-cores)
```

Logical CPUs `0–7` are **four physical cores (0–3), each doubled by SMT**. So
`MTL5_NUM_THREADS=8 taskset -c 0-7` runs eight software threads on **four physical
cores** — the extra four threads are SMT siblings, worth ~1.25–1.3× on
compute-bound code. The measured 4→8 gain of 1.28× is exactly that. *The
experiment never tested eight cores.*

### 3.3 The corrected design

Pin each software thread to its **own physical P-core** — one logical CPU per
core, no SMT siblings, no E-cores. The primary logical CPU of each P-core is the
even id: `0,2,4,6,8,10,12,14`.

```bash
MTL5_NUM_THREADS=1 taskset -c 0                        ./bench --suite gemm --sizes 2048
MTL5_NUM_THREADS=2 taskset -c 0,2                      ./bench --suite gemm --sizes 2048
MTL5_NUM_THREADS=4 taskset -c 0,2,4,6                  ./bench --suite gemm --sizes 2048
MTL5_NUM_THREADS=8 taskset -c 0,2,4,6,8,10,12,14       ./bench --suite gemm --sizes 2048
```

Controls:

- **SMT control** — repeat T=8 on `0-7` (four cores hyperthreaded) to confirm it
  reproduces the original number.
- **Size sweep** — GEMM at N = 1024, 2048, 4096 to separate compute-bound from
  bandwidth-bound behavior (H3 vs H4).

## 4. Results

### 4.1 GEMM, one thread per distinct physical core (N=2048)

| threads | GFLOP/s | speedup | efficiency |
|---|---|---|---|
| 1 | 57.4 | 1.00× | — |
| 2 | 110.3 | 1.92× | 96% |
| 4 | 205.1 | 3.57× | 89% |
| **8** | **362.5** | **6.32×** | **79%** |

**SMT control** — T=8 on `taskset -c 0-7` (4 cores hyperthreaded): **252.1
GFLOP/s (4.39×)**, reproducing the original "wall" almost exactly. H1 confirmed.

### 4.2 GEMM size sweep at 8 physical cores

| N | 1T | 8T | speedup |
|---|---|---|---|
| 1024 | 58.0 | 294.9 | 5.08× |
| 2048 | 57.2 | 360.6 | 6.30× |
| 4096 | 57.4 | 393.7 | **6.85×** |

Scaling **improves** with N. A memory-bandwidth ceiling would do the opposite, so
**H3 is rejected for GEMM**. Bigger tiles do more arithmetic per parallel region,
so the fixed per-`(jc,pc)`-region synchronization amortizes better — pointing at
**H4** as the small-N limiter.

### 4.3 Memory-bound kernels, distinct physical cores (8T speedup)

| Kernel | 1T | 2T | 4T | 8T | speedup |
|---|---|---|---|---|---|
| GEMV (N=8192) | 9.4 | 13.2 | 17.1 | 20.6 | 2.19× |
| dot (N=8M) | 4.6 | 7.6 | 8.6 | 11.1 | 2.43× |
| axpy (N=8M) | 3.4 | 5.3 | 6.4 | 6.9 | 2.02× |
| scal (N=8M) | 2.6 | 4.2 | 5.6 | 6.6 | 2.48× |
| SpMV (490k-row 2D Laplacian) | — | 1.33× | 1.79× | 1.92× | 1.92× |

These plateau at ~2–2.5× regardless of correct pinning.

## 5. Analysis

- **GEMM is compute-bound and scales with cores.** 6.32× at 8 cores (N=2048),
  6.85× at N=4096. The ~14–21% shortfall from ideal splits between (a) the
  per-region pool synchronization — dominant at small N, where there are few
  cache blocks to spread over 8 threads (5.08× at N=1024) — and (b) the all-core
  turbo drop (H6), the residual few percent after H1–H4 are accounted for.
- **The Level-1/2 kernels and SpMV are memory-bandwidth-bound.** Two DDR5 channels
  are saturated by a couple of P-cores, so more threads cannot help — the ~2–2.5×
  plateau is the roofline, not a defect. Threading them is still worth the ~2× a
  few cores extract over one, and (in a solver) it overlaps with the compute-bound
  GEMM/preconditioner work.
- **No software scaling wall at 4 cores exists.** The apparent wall was an
  affinity-mask artifact.

## 6. Conclusions

1. **H1 (SMT-sibling pinning): the cause.** Eight real cores give 6.3–6.9×, not
   4.3×.
2. **H3 (bandwidth) rejected for GEMM**, confirmed for the memory-bound kernels.
3. **H4 (per-region sync)** is the small-N limiter for GEMM; larger problems
   amortize it away.
4. **H6 (turbo)** accounts for the last few percent.

MTL5's on-node threading scales as a from-scratch implementation should: the
compute-bound kernel tracks cores, and the bandwidth-bound kernels track memory
channels.

## 7. Reproduction

Build the native-fast benchmark and run the pinned sweep:

```bash
cmake -B build-nf -DMTL5_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
      -DMTL5_NATIVE_FAST_GEMM=ON -DMTL5_WITH_HIGHWAY=ON -DMTL5_NATIVE_ARCH=ON
cmake --build build-nf --target bench_all

# Confirm your topology first, then pin to one logical CPU per physical core:
lscpu -e=CPU,CORE,MAXMHZ
for T in 1 2 4 8; do
  CPUS=$(...)   # even ids of the first T P-cores, e.g. 0,2,4,6,8,10,12,14
  MTL5_NUM_THREADS=$T taskset -c "$CPUS" ./build-nf/benchmarks/bench_all \
      --suite gemm --sizes 2048 --label "t$T"
done
```

## 8. Lessons — a checklist for scaling experiments

- **Verify the affinity mask against the core topology.** `taskset -c 0-7` is not
  "8 cores" on an SMT machine. Pin one thread per *physical* core; keep SMT
  siblings and heterogeneous (E-)cores out unless they are the subject.
- **Sweep the problem size.** It separates compute-bound from bandwidth-bound
  behavior and exposes fixed per-region overhead at small sizes.
- **Keep a control.** An all-SMT run confirmed the artifact and quantified the SMT
  gain in one measurement.
- **Attribute the residual.** Frequency (turbo), sync, and imbalance each explain
  a slice; name them rather than lumping them into "overhead."
- **Distrust a too-clean failure.** "Exactly the SMT speedup" was the tell that the
  cores, not the code, were the story.
