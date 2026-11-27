# Wall-clock tracking

When the lab machines are on, each runs `mtl5_bench` on `main` (on demand), and each session
is compared with that machine's previous one (#944). The instruction-count gate on
every PR (`bench-regression.yml`) catches extra work. This page catches what
instruction counts cannot see: caches, vectorization and threading. It is a report and
gates nothing.

A ratio is new time / previous time, so above 1 is slower. A run's ratio is the median
over its entries, which repeats to about 0.5% between two sessions of one commit (the
cards' repeatability notes). A run is marked *slower* or *faster* outside its report's
run band. How the sessions are run and compared is in
[`benchmarks/tracking/README.md`](https://github.com/stillwater-sc/mtl5/blob/main/benchmarks/tracking/README.md).

## arduino-ventuno-q

Latest: session `20261127T125653Z` (commit `f98777e6c6f7`) against `20261113T174232Z` (commit `bf315e4604cf`), class medium. [Full report](https://github.com/stillwater-sc/mtl5/blob/main/benchmarks/data/arduino-ventuno-q/tracking/20261127T125653Z.md).

| run | median ratio | outside ±10% | verdict |
|---|---:|---:|---|
| blas-t1 | 1.000 | 20 | steady |
| blas-t4 | 0.997 | 21 | steady |
| native-fast-t1 | 0.999 | 23 | steady |
| native-fast-t4 | 1.000 | 27 | steady |
| native-t1 | 0.994 | 24 | steady |
| native-t4 | 0.998 | 23 | steady |

## i7-12700k

Latest: session `20261127T125653Z` (commit `f98777e6c6f7`) against `20261113T060838Z` (commit `8867b44b3f52`), class medium. [Full report](https://github.com/stillwater-sc/mtl5/blob/main/benchmarks/data/i7-12700k/tracking/20261127T125653Z.md).

| run | median ratio | outside ±10% | verdict |
|---|---:|---:|---|
| blas-t1 | 1.001 | 20 | steady |
| blas-t8 | 0.999 | 22 | steady |
| native-fast-t1 | 1.001 | 20 | steady |
| native-fast-t8 | 0.998 | 26 | steady |
| native-t1 | 1.001 | 21 | steady |
| native-t8 | 1.001 | 24 | steady |

Changed besides the commit: `os_version`, `preflight_kernel`.

## jetson-orin

Latest: session `20261127T125653Z` (commit `f98777e6c6f7`) against `20261113T140622Z` (commit `967d20ac6b04`), class medium. [Full report](https://github.com/stillwater-sc/mtl5/blob/main/benchmarks/data/jetson-orin/tracking/20261127T125653Z.md).

| run | median ratio | outside ±10% | verdict |
|---|---:|---:|---|
| blas-t1 | 0.999 | 19 | steady |
| blas-t6 | 1.000 | 27 | steady |
| native-fast-t1 | 1.000 | 22 | steady |
| native-fast-t6 | 1.001 | 22 | steady |
| native-t1 | 1.000 | 23 | steady |
| native-t6 | 1.000 | 27 | steady |

## xeon-e5-2420v2

Latest: session `20261127T125653Z` (commit `f98777e6c6f7`) against `20261111T193510Z` (commit `bfa1c2822688`), class medium. [Full report](https://github.com/stillwater-sc/mtl5/blob/main/benchmarks/data/xeon-e5-2420v2/tracking/20261127T125653Z.md).

| run | median ratio | outside ±10% | verdict |
|---|---:|---:|---|
| blas-t1 | 1.002 | 12 | steady |
| blas-t6 | 0.999 | 15 | steady |
| native-fast-t1 | 0.995 | 17 | steady |
| native-fast-t6 | 0.994 | 23 | steady |
| native-t1 | 0.999 | 19 | steady |
| native-t6 | 0.995 | 26 | steady |

Changed besides the commit: `os_version`, `preflight_kernel`.
