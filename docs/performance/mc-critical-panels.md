# The threaded ic step, counted in micro-panels (#458)

The multi-threaded `mc` response curve is W-shaped (#458). This page shows what
causes it, the change that follows, and its measured effect on three machines.

## The finding

**The serial path steps `mc` in whole micro-panels and the threaded path does
not.** `balanced_mc` rounds a serial `mc` down to a multiple of the register
tile's `mr`. In the threaded path it rounded the ic block *count* up to a
multiple of the thread count and kept whatever `mc` came out. With `mr = 6`:

| machine (Highway build) | kc | serial `mc` | threaded `mc` | panels per threaded block |
|---|---|---|---|---|
| Xeon E5-2420 v2 (AVX) | 512 | 30 | 32 | 5.33, so 6 are computed |
| i7-12700K (AVX2) | 256 | 60 | 64 | 10.67, so 11 are computed |

The micro-kernel computes whole `mr × nr` tiles. A block of `mc` rows therefore
costs `ceil(mc / mr)` panels, and in the threaded path every block carries a
partial one.

This explains all three of #458's observations:

- **The spread is 3.6% at T=1 and 13.0% at T=6.** The two thread counts ran
  different blocks. T=1 always ran whole panels.
- **The curve is W-shaped.** A count of whole panels is quantised, so the
  critical path rises and falls as `mc` moves through multiples of `mr`.
- **It is independent of kc.** Panel counts don't depend on kc.

The earlier "`mc = 8` lost 17%" result is the same effect: 1.33 panels per block.

## The model, against the measured curve

The cost is the number of micro-panels on the busiest thread of the round-robin
partition (`detail::critical_panels`). The table fits each candidate to #458's
Xeon curve (2048³, T=6, `benchmarks/data/xeon-e5-2420-sweep`), as correlation
with the best of 3 rounds:

| cost model | kc = 512 | kc = 256 |
|---|---|---|
| block-count balance, `ceil(nib/T)/(nib/T)`: what `balanced_mc` optimised | +0.23 | 0.00 |
| critical rows | +0.21 | +0.02 |
| **critical micro-panels** | **+0.83** | **+0.74** |

On the i7-12700K, an `mc` sweep at the shipped kc = 256 fits the panel model at
r = +0.99 on three of four shapes at T=8. On the fourth, 4096², the whole range
is inside the noise floor.

**The model has one known residual.** On the Xeon, `mc = 171` measures about 5%
below the model's prediction. Its A block (171 × 512 × 8 bytes) is much larger
than the 256 KB L2, so this is plausibly a cache effect the panel count does not
see.

## The change

The block-count result is kept as the incumbent. A block size in
`[ceil(mc_max / 2), mc_max]` replaces it only when it gives the busiest thread
strictly fewer micro-panels.

- **Ties keep the incumbent.** Plans that were already panel-optimal don't move.
  That includes #408's `mc = 64` at m = 1024 on 8 threads, and the i7's shipped
  plans at T=8.
- **The window guards against small blocks.** A panel count alone always prefers
  a one-panel block, and that is wrong: each ic block is another pass over the
  shared packed B panel. On the i7 at kc = 1024, `mc = 6` ran at ×0.61–0.87 of
  `mc = 16` on eight threads. Searching no lower than half the cache bound allows
  at most twice the blocks.
- **Unchanged:** the serial path, the #488 sub-tile fallback, and callers that
  pass `mr = 1`.

## Measured: i7-12700K

The run used `benchmarks/machines/i7-12700k-mc.sh`:

- pinned to the 8 P-cores;
- 4 interleaved rounds with alternating order, min of 5 reps;
- `default` (this change) against `legacymc` (the old rule), both built from the
  same tree;
- data in `benchmarks/data/i7-12700k-mc458`.

These are the points where the plan changed:

| shape | T | `mc` old → new | old GF/s | new GF/s | ratio |
|---|---|---|---|---|---|
| 1000×1000×1000 | 2 | 63 → 42 | 133.2 | 139.5 | ×1.048 |
| 1000×1000×1000 | 4 | 63 → 42 | 251.0 | 261.8 | ×1.043 |
| 1000×1000×1000 | 6 | 56 → 42 | 347.3 | 369.1 | ×1.063 |
| 1000×1000×1000 | 8 | 63 → 42 | 449.9 | 463.8 | ×1.031 |
| 1024×1024×1024 | 2 | 64 → 42 | 135.3 | 137.6 | ×1.017 |
| 2048×2048×2048 | 2 | 64 → 54 | 132.1 | 136.6 | ×1.034 |
| 3000×3000×1000 | 2 | 63 → 60 | 128.1 | 134.2 | ×1.047 |
| 3000×3000×1000 | 4 | 63 → 54 | 248.7 | 258.8 | ×1.041 |
| 3000×3000×1000 | 6 | 63 → 42 | 352.3 | 367.5 | ×1.043 |
| 3000×3000×1000 | 8 | 63 → 54 | 445.3 | 463.6 | ×1.041 |
| 4096×4096×1024 | 2 | 64 → 54 | 129.1 | 137.8 | ×1.067 |
| 4096×4096×1024 | 4 | 64 → 54 | 247.7 | 265.5 | ×1.072 |
| 4096×4096×1024 | 6 | 63 → 36 | 340.3 | 371.7 | ×1.092 |

- **Faster:** 12 of the 13 changed plans, by 3.1–9.2%. The 13th, 1024² at T=2, is
  ×1.017, below the 2% threshold the analyzer calls.
- **Slower:** none.
- **Control:** the 17 points whose plan did not change agree within 2%. That
  includes every T=1 point, since the serial path is untouched.

## Measured: Xeon E5-2420 v2

The run used `benchmarks/machines/xeon-e5-2420-mc.sh`:

- pinned to the six physical cores 0–5;
- the same protocol as the i7;
- data in `benchmarks/data/xeon-e5-2420-mc458`.

The governor was `ondemand`, which the sidecar records. The median round-to-round
spread was still 0.9% (maximum 2.5%).

This is the machine #458's curve was measured on. The table lists every point
where the plan changed:

| shape | T | `mc` old → new | critical panels | predicted | old GF/s | new GF/s | measured |
|---|---|---|---|---|---|---|---|
| 512×8192×1024 | 2 | 32 → 24 | 48 → 44 | ×1.091 | 15.6 | 17.4 | **×1.115** |
| 512×8192×1024 | 4 | 32 → 18 | 24 → 23 | ×1.043 | 29.5 | 31.9 | **×1.082** |
| 1000×1000×1000 | 2 | 32 → 24 | 96 → 84 | ×1.143 | 15.9 | 18.4 | **×1.155** |
| 1000×1000×1000 | 4 | 32 → 18 | 48 → 42 | ×1.143 | 30.7 | 35.5 | **×1.158** |
| 1000×1000×1000 | 6 | 28 → 24 | 30 → 28 | ×1.071 | 47.2 | 50.7 | **×1.075** |
| 1024×1024×1024 | 2 | 32 → 30 | 96 → 86 | ×1.116 | 16.2 | 18.2 | **×1.121** |
| 1024×1024×1024 | 4 | 32 → 24 | 48 → 44 | ×1.091 | 31.0 | 34.3 | **×1.107** |
| 2048×2048×2048 | 2 | 32 → 18 | 192 → 171 | ×1.123 | 16.3 | 18.6 | **×1.146** |
| 2048×2048×2048 | 4 | 32 → 30 | 96 → 87 | ×1.103 | 31.6 | 35.5 | **×1.123** |
| 2048×2048×2048 | 6 | 32 → 18 | 66 → 57 | ×1.158 | 45.3 | 52.6 | **×1.162** |
| 3000×3000×1000 | 2 | 32 → 30 | 282 → 250 | ×1.128 | 16.4 | 18.6 | **×1.133** |
| 3000×3000×1000 | 4 | 32 → 30 | 144 → 125 | ×1.152 | 31.4 | 35.6 | **×1.133** |
| 3000×3000×1000 | 6 | 32 → 24 | 96 → 84 | ×1.143 | 46.4 | 52.5 | **×1.133** |
| 4096×4096×1024 | 2 | 32 → 18 | 384 → 342 | ×1.123 | 16.3 | 18.6 | **×1.140** |
| 4096×4096×1024 | 4 | 32 → 18 | 192 → 171 | ×1.123 | 31.5 | 36.0 | **×1.143** |
| 4096×4096×1024 | 6 | 32 → 18 | 132 → 114 | ×1.158 | 46.1 | 53.3 | **×1.158** |

- **Faster:** all 16 changed plans, by 7.5–16.2%. The issue's own point, 2048³
  at T=6, went from 45.3 to 52.6 GF/s. That is above the best point of the
  issue's override sweep, 51.2 GF/s at `mc = 12`, which was measured in a
  different session.
- **Slower:** none.
- **Control:** the 8 unchanged points, including all four T=1 points, agree
  within 1%.
- **Per point, the panel model is a prediction rather than just a fit:**
  - Xeon: r = +0.91 between predicted and measured ratio, and measured gains
    average ×1.010 of predicted.
  - i7: the predicted range is narrow compared with the noise, so r is only
    +0.42, but the calibration is ×1.005.

## Measured: Ryzen 9 8945HS (Zen 4)

The run used `benchmarks/machines/ryzen-9-8945hs-mc.sh`:

- under WSL2, built with Clang 18 for AVX-512 (`build_isa` includes `AVX3_DL`);
- pinned to one logical CPU per core, 0,2,…,14;
- the same protocol as the other two machines;
- data in `benchmarks/data/ryzen-9-8945hs-mc458`.

Its blocking is `mr = 6`, `nr = 16`, `kc = 128` and `mc_max = 128`, not the
i7's 64, so 128 rows is 21.33 panels per block. The rule moves fewer plans here
than on the other machines: at T = 6 the old rule's `mc` (84, 86, 114) is
already panel-optimal, and at T = 8 nothing in the window gives fewer panels.

| shape | T | `mc` old → new | critical panels | predicted | old GF/s | new GF/s | best | median |
|---|---|---|---|---|---|---|---|---|
| 1024×1024×1024 | 2 | 128 → 102 | 88 → 86 | ×1.023 | 127.8 | 132.2 | **×1.034** | ×1.031 |
| 2048×2048×2048 | 2 | 128 → 114 | 176 → 171 | ×1.029 | 114.2 | 119.8 | **×1.049** | ×1.009 |
| 2048×2048×2048 | 4 | 128 → 102 | 88 → 87 | ×1.011 | 208.4 | 216.0 | **×1.037** | ×1.013 |
| 4096×4096×1024 | 2 | 128 → 114 | 352 → 342 | ×1.029 | 108.0 | 114.9 | **×1.064** | ×1.038 |
| 4096×4096×1024 | 4 | 128 → 114 | 176 → 171 | ×1.029 | 193.1 | 202.4 | **×1.048** | ×1.048 |

GF/s is the best of 4 rounds; "median" is the ratio of the round medians.

- **Faster:** all 5 changed plans, on both measures. At 1024² T=2 and at both
  4096² points, the two arms' round ranges don't overlap. At the two 2048³
  points they do, so those two are directional rather than resolved.
- **Slower:** none.
- **Control, T ≤ 4:** the 13 unchanged points agree within 1.7%, including
  every T=1 point.
- **Control, T = 6 and 8:** the unchanged points scatter from ×0.91 to ×1.12
  between arms that run the same plan. The round-to-round spread there reaches
  23%, against a median of 2.4% over the whole run. This is the noise floor of
  a WSL2 guest at high thread counts, where Windows places the virtual CPUs; it
  is not a property of either rule, and no changed plan sits in that range.
- **Model:** the gains are small, ×1.01–1.03 predicted, so the per-point
  correlation is weak (r = +0.66). Measured gains average ×1.021 of predicted.

## Reproducing

```bash
benchmarks/machines/i7-12700k-mc.sh   # or xeon-e5-2420-mc.sh, ryzen-9-8945hs-mc.sh
machine=i7-12700k                     # or xeon-e5-2420, ryzen-9-8945hs
python3 benchmarks/analyze_blocking_ab.py \
    "benchmarks/data/${machine}-mc458/blocking_ab_legacymc.csv" \
    "benchmarks/data/${machine}-mc458/blocking_ab_default.csv"
```

In the analyzer's output, a ratio below 1 means the new rule (`default`) is
faster. The column headed "detection helped" is the analyzer's generic wording
from #430.
