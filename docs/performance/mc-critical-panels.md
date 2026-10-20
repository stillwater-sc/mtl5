# The threaded ic step, counted in micro-panels (#458)

The multi-threaded `mc` response curve is W-shaped (#458). This page shows what
causes it, the change that follows, and what has been measured so far.

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

## Not yet measured

**Xeon E5-2420 v2.** This is where the curve was found and where the largest
gain is predicted:

- predicted: ×1.04–1.16 at T=2..6, from the panel count;
- already in the issue's curve: the shipped `mc = 32` at 2048³ T=6 measured
  ×0.885 of the best point;
- to run: `benchmarks/machines/xeon-e5-2420-mc.sh`, writing to
  `benchmarks/data/xeon-e5-2420-mc458`.

**Ryzen 9 8945HS (Zen 4).** This is #458's third acceptance machine. Run
`ARMS="default legacymc"` with its existing profile, after giving that profile its
own `OUTDIR`.

Until both are in, this is a measured improvement on one machine and a
prediction on the others. It is not yet the "rule that predicts the optimum on
all three machines" that #458 asks for.

## Reproducing

```bash
benchmarks/machines/i7-12700k-mc.sh          # or xeon-e5-2420-mc.sh
python3 benchmarks/analyze_blocking_ab.py \
    benchmarks/data/<machine>-mc458/blocking_ab_legacymc.csv \
    benchmarks/data/<machine>-mc458/blocking_ab_default.csv
```

In the analyzer's output, a ratio below 1 means the new rule (`default`) is
faster. The column headed "detection helped" is the analyzer's generic wording
from #430.
