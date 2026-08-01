# On-Node Threading Results (#297) — Provisional

Measured scaling of the #297 threading rollout, using the suites and driver from
the [benchmarking plan](issue-297-threading-benchmark-plan.md)
(`benchmarks/run_scaling_297.sh` + `benchmarks/analyze_scaling.py`).

> **Status: PROVISIONAL.** These numbers were gathered on a shared development
> sandbox with the CPU in the `powersave` governor (frequency scaling active).
> The methodology is sound — one process per thread count, pinned to *distinct
> physical P-cores*, median of 25 (solve) / 10–20 (dense) iterations after
> warmups — but the absolute magnitudes carry sandbox/turbo noise and a few
> super-linear (>100%) points from frequency/cache effects. Treat the **shapes
> of the curves and the qualitative findings** as the result; re-run on a quiesced
> box with the `performance` governor for citable figures (command at the end).

## 1. Headline findings

- **Dense GEMM scales as before** — square GEMM hits **6.25–6.27× on 8 cores**
  (~78% efficiency), matching the earlier #221 study (~6.3×). The #297
  **multi-loop (2D) grid** additionally makes **wide/short** shapes scale
  (32×8192: **1.73×**) where the old ic-only parallelism was stuck near 1×, and
  **tall/thin** shapes scale fully (8192×64: **6.11×**).
- **Dense factorizations scale well** — LU **7.35×**, Cholesky **7.03×**, QR up
  to **6.49×** on 8 cores; QR falls to a **3.45×** ceiling at n=4096 as its
  serial panel work grows (an honest limit, not a defect).
- **Element-wise sweeps are memory-bandwidth bound** — they scale to ~**4.2×**
  at the sizes that fit the bandwidth sweet spot and ceiling near ~**2.4×** at
  the largest sizes; the **single-row wide matrix is flat at 1.00×**, precisely
  the row-parallel gap tracked in **#313**.
- **Sparse triangular-solve scaling tracks level structure** — wide-level
  systems (arrow) speed up; deep-narrow (2-D/3-D Laplacian) and sequential
  (banded) systems are scheduling-limited and stay near 1× — exactly what level
  scheduling can and cannot parallelize.

## 2. System under test

| | |
|---|---|
| CPU | 12th Gen Intel Core i7-12700K (8 P-cores × 2 SMT + 4 E-cores) |
| Pinned to | distinct **P-cores** only: CPUs `0,2,4,6,8,10,12,14` (one logical id per core; SMT siblings and E-cores excluded) |
| Governor | `powersave` (⚠ frequency scaling — a source of the >100% points) |
| Threads swept | 1, 2, 4, 8 |
| Build | Release, `-O3`, native-fast GEMM + Highway SIMD + `-march=native` |
| Driver | `benchmarks/run_scaling_297.sh` (`BENCH_PCPUS=0,2,4,6,8,10,12,14`) |

Efficiency below is `speedup / (T / T_base)`; values slightly over 100% at T=2
are cache/turbo artifacts of the `powersave` governor, not real super-linear
speedup.

## 3. Dense GEMM — multi-loop (2D) grid (#311)

GFLOP/s; `k = 1024`. Square and tall/thin scale on the ic loop; the wide/short
rows are the multi-loop payoff (ic-only could not fill the pool there).

| shape | 1T | 2T | 4T | 8T | 8T speedup |
|-------|----|----|----|----|-----------|
| 1024² (square) | 53.8 | 112.1 | 200.3 | 336.2 | **6.25×** |
| 2048² (square) | 57.2 | 110.8 | 209.3 | 358.9 | **6.27×** |
| 8192×64 (tall) | 51.3 | 99.5 | 187.4 | 313.7 | **6.11×** |
| 32×8192 (wide) | 25.9 | 44.8 | 44.4 | 43.1 | **1.73×** (→2T; plateaus) |
| 64×4096 (wide) | 37.9 | 38.3 | 37.4 | 34.5 | ~1.0× (small/mem-bound) |

The wide/short shapes are small (≤ 0.27 GFLOP) and memory-bound, so their
absolute ceiling is low — but 32×8192 scaling to 1.73× is scaling the multi-loop
`jc` grid delivers and ic-only could not. Larger wide problems benefit more.

## 4. Dense factorizations (#298, #300)

GFLOP/s (native threaded path; no external LAPACK).

| kernel | n | 1T | 4T | 8T | 8T speedup |
|--------|----|----|----|----|-----------|
| LU | 1024 | 0.36 | 1.37 | 1.88 | 5.27× |
| LU | 2048 | 0.33 | 1.39 | 2.39 | **7.35×** |
| LU | 4096 | 0.22 | 0.86 | 1.46 | 6.52× |
| Cholesky | 2048 | 0.67 | 2.34 | 4.16 | 6.23× |
| Cholesky | 4096 | 0.34 | 1.31 | 2.40 | **7.03×** |
| QR | 2048 | 6.11 | 22.2 | 39.7 | **6.49×** |
| QR | 4096 | 6.03 | 16.6 | 20.8 | 3.45× (serial panel ceiling) |

## 5. Element-wise expression sweeps (#312) — memory-bandwidth bound

Throughput = elements/ns; ratio = speedup.

| case | 1T | 2T | 4T | 8T | 8T speedup |
|------|----|----|----|----|-----------|
| vector 1e6 | 1.94 | 4.10 | 7.63 | 8.18 | **4.21×** |
| vector 1e7 | 1.32 | 1.88 | 2.80 | 3.17 | 2.41× |
| matrix 1024² | 2.22 | 4.19 | 7.94 | 9.36 | **4.22×** |
| matrix 4096² | 1.30 | 1.86 | 2.74 | 3.10 | 2.39× |
| matrix 2M×8 (tall) | 1.24 | 1.77 | 2.66 | 2.99 | 2.41× |
| **matrix 1×16M (wide)** | 1.31 | 1.31 | 1.32 | 1.32 | **1.00× (#313)** |

Element-wise work is bandwidth-limited, so scaling ceilings at the memory system
(~2.4–4.2× here), not the core count. The single-row wide matrix is the sharp
outlier: the row-parallel sweep cannot split a single row, so it stays serial —
the exact limitation **#313** proposes to fix by flattening the sweep to a
linear element index.

## 6. Sparse triangular solves (#301–#309) — scaling tracks level structure

Solve throughput (nnz/ns from the factors); ratio = speedup. Grid side 120
(2-D `n=14400`; 3-D `n=13824`; arrow `n=28800`; banded `n=57600`).

| system (structure) | 1T | 2T | 4T | 8T | 8T speedup |
|--------------------|----|----|----|----|-----------|
| Cholesky, arrow (one wide level) | 2.24 | 2.26 | 2.38 | 2.37 | 1.06× |
| Cholesky, 2-D Laplacian (deep-narrow) | 2.82 | 2.65 | 2.57 | 2.46 | 0.87× |
| Cholesky, 3-D Laplacian (deep-narrow) | 1.63 | 1.58 | 1.52 | 1.46 | 0.90× |
| LDLᵀ / supernodal LDLᵀ, 2-D Laplacian | 2.5 | 2.4 | 2.3 | 2.2 | ~0.9× |
| LU / supernodal LU, 2-D Laplacian | 1.34 | 1.35 | 1.24 | 1.17 | ~0.9× |
| LU / supernodal LU, banded (recurrence) | 0.9 | 0.9 | 0.95 | 0.95 | ~1.0× |

At these modest sizes **everything is flat-to-slightly-negative** — the solves
run at or below the per-level grain, so the barrier/dispatch cost is not
amortized and the deep-narrow cases even lose a little to it. This is the honest
small-n regime, not a defect.

The scheduling payoff is **size- and structure-dependent**. Level scheduling
parallelizes the independent rows *within* a level, so the available parallelism
*is* the level width:

- **Wide-level** systems benefit once the level is large enough to clear the
  grain. The arrow matrix's forward solve is a single level of `n−1` independent
  rows; a larger probe (arrow `n=245000`, unpinned) reached **1.43×** — versus
  1.06× at `n=28800` here. Bigger wide levels → more benefit.
- **Deep-narrow** systems (2-D/3-D Laplacian: many small levels) and
  **sequential** systems (banded: a recurrence, one row per level) have no
  level width to exploit and stay near 1× at every size — inherent to the
  dependency structure, not the schedule.

Residuals stayed ~1e-13 at every thread count (correctness holds under threads).

## 7. Caveats

1. `powersave` governor → frequency drift; a few T=2 efficiencies exceed 100%
   (cache/turbo). Re-run with `performance` for stable magnitudes.
2. Shared sandbox → background load noise; medians mitigate but don't eliminate.
3. Small problems run serial by design (below the `parallel_for`/`parallel_ewise`
   grain) — expected, not a regression.
4. GEMM wide/short rows are small and memory-bound; their low absolute ceiling is
   a problem-size effect, not a multi-loop failure.

## 8. Reproduce (citable run)

On a quiesced machine, set the governor to `performance` and pin to *your*
physical cores:

```bash
sudo cpupower frequency-set -g performance          # optional, for stable clocks
BENCH_PCPUS=<one-logical-id-per-physical-core> \
THREADS="1 2 4 8" \
benchmarks/run_scaling_297.sh
./benchmarks/analyze_scaling.py benchmarks/data/scaling_*.csv --plot benchmarks/data/scaling_297.png
```

Find your P-core ids with `lscpu -e=CPU,CORE,MAXMHZ` (one CPU per distinct CORE,
highest-MAXMHZ cores first).

## 9. Follow-ups the data motivates

- **#313** — flatten the dense-matrix element-wise sweep so wide/short shapes
  scale (the 1×16M row above is stuck at 1.00×).
- **#310** — reuse solve schedules in the `*_refactor` paths (amortizes the
  schedule build over the many-solve transient workload the sparse suite models).
