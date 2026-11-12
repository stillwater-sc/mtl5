# Register-tile sweep (#643)

Which micro-kernel register tile (mr × nr) the native GEMM uses on each ISA, and
the sweeps that chose it. Each machine's hardware and pinning policy are on the
[benchmark systems](systems.md) page.

## Ryzen 9 8945HS: no tile clears the noise, so AVX-512 keeps 6×16

`bash benchmarks/machines/ryzen-9-8945hs-tile.sh` under WSL, fp64, 5 rotated
rounds, data in `benchmarks/data/ryzen-9-8945hs-tile-double/`. Only `mr` changes;
nr = 16, kc = 128, mc = 128 and nc = 8192 are identical across the arms.
Geometric-mean ratio of the median against `default` (mr = 6):

| Tile | Accumulators (of 32) | T=1 | T=8 |
|---|---|---|---|
| 8×16 | 16 | 0.995 | 1.046 |
| 10×16 | 20 | 0.996 | 1.036 |
| 12×16 | 24 | 1.032 | 1.021 |
| 14×16 | 28 | 1.002 | 0.994 |

The default arm's round-to-round spread was 5–41%, as expected inside a VM
(Windows places the virtual CPUs). 12×16 is nominally ahead at both thread counts
(×1.032 and ×1.021), but neither gain, nor any other tile's, clears that spread,
so AVX-512 builds keep the AVX2 description. That is a
measured reason to keep it, not a measured loss for a larger tile. A bare-metal
Linux run would be the way to settle it.

## Arduino Ventuno Q: 8×4 wins, and 12×4 loses at four threads

`benchmarks/machines/arduino-ventuno-q-tile.sh`, `schedutil`, 5 rotated rounds,
data in `benchmarks/data/arduino-ventuno-q-schedutil-tile-{double,float}/`. Every
arm is the same source with only `mr` changed; nr, kc, mc and nc are identical
across the arms (fp64 nr = 4, kc = 512, mc = 32; fp32 nr = 8, kc = 512, mc = 64),
and the checksums agree across all arms. The ratios are median GFLOP/s against
`default` (mr = 6 in both runs, which predate the change), as a geometric mean
over the shapes:

| Tile (fp64 / fp32) | Accumulators (of 32) | fp64 T=1 | fp64 T=4 | fp32 T=1 | fp32 T=4 | Worst shape (fp64 / fp32) |
|---|---|---|---|---|---|---|
| 8×4 / 8×8 | 16 | **1.174** | **1.102** | **1.170** | **1.219** | 1.060 / 1.156 |
| 10×4 / 10×8 | 20 | 1.156 | 1.045 | 1.171 | 1.160 | 0.967 / 1.064 |
| 12×4 / 12×8 | 24 | 1.061 | 0.930 | 1.063 | 1.206 | 0.840 / 1.048 |
| 14×4 / 14×8 | 28 | 1.049 | 0.878 | 1.030 | 1.028 | 0.736 / 0.930 |

* **The peak is at half the register file**, not at the 3/4 that #351 measured on
  AVX2. 12×4 is what the 3/4 rule would give on 32 registers, and it loses.
* **fp32 agrees on 8.** 8×8 is best at T=4 on every shape and ties 10×8 at T=1.
  fp32 has no 12-row loss: its wide shapes are 128 rows (mc = 64), so each
  thread's 32-row block takes 12-row panels at 36 rows of work, not 24 for 16.
* **The T=4 losses are a divisibility effect, not register pressure.** They fall on
  the wide 64-row shapes, where each of the four threads gets a 16-row block. 8-row
  panels tile it exactly; 12-row panels cover it with 24 rows of work and 14-row
  panels with 28. It is the #408 family again: a register-tile change that moves
  how a block divides.
* **Shipped as a result:** aarch64 builds now derive 8×4 fp64 and 8×8 fp32
  (`target_hw_traits`, 32 registers with a measured budget of 16 accumulators).
  In these sessions that took one A78C from 8.3 to 9.7 GFLOP/s fp64, 57% of
  the 16.9 GFLOP/s peak at 2112 MHz, and from 17.1 to 19.9 GFLOP/s fp32, 59% of
  33.8. Four cores went from 35.5 to 37.6 GFLOP/s fp64 and from 61.9 to
  77.2 GFLOP/s fp32 at 2048³.
* **Noise.** The clocks were unpinned, so the default arm spread 12–14% between
  rounds at T=1 (2–3% at T=4 on the squares). The 8×4 gain is consistent across
  every shape at both thread counts, which noise of that size would not produce.
  The absolute numbers are lower than the Ventuno's [cache-blocking A/B session](cache-blocking-ab.md#arduino-ventuno-q-a-null-run-because-there-was-nothing-to-detect) (8.3 against
  9.1 GFLOP/s for the same 6×4 tile, both under `schedutil`). This data does not
  say why; compare ratios within a session, not GFLOP/s across them.

**Not measured, and covered by the same change:** the Jetson's A78AE and every
other aarch64 core (Apple M-series, the Neoverse cores of the CI
runners). The tile is a compile-time choice per ISA, so all of them moved
together on one board's fp64 data.
