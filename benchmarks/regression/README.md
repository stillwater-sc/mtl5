# The benchmark regression gate (#944)

MTL5 is a performance library, so a PR that makes a kernel slower should fail
review as a PR that breaks a result does. Shared CI runners cannot time anything
reliably (the same binary varied by 31% on an idle runner, #454), so the gate
does not measure time. It measures **instruction counts**, which repeat exactly
on one binary whatever the runner's load.

| Check | Where | Blocks a merge | What it catches |
|---|---|---|---|
| Instruction counts (`bench-regression.yml`) | Linux x64 and Linux arm64, every PR | yes, above 2% | an extra copy, a temporary an expression template started building, a worse algorithm, a fast path that stopped being taken |
| Dispatch tests (`tests/unit/operation/test_mult_dispatch*.cpp`) | the unit tier, all 8 platforms | yes | `mult` no longer reaching BLAS, the blocked GEMM or the SIMD GEMV |
| Wall-clock runs on the benchmark machines | on demand, by hand, when the lab machines are on; published into `benchmarks/data/` ([`../tracking/`](../tracking/README.md)) | no: a report | cache, SIMD-width and threading effects, which an instruction count cannot see |

## The instruction-count gate

`bench-regression.yml` builds `mtl5_bench` in four configurations:

| configuration | runner | build |
|---|---|---|
| `native` | `ubuntu-24.04` (x64) | the portable kernels |
| `native-fast` | `ubuntu-24.04` (x64) | the blocked GEMM and SIMD GEMV over Highway |
| `native-arm64` | `ubuntu-24.04-arm` | the portable kernels |
| `native-fast-arm64` | `ubuntu-24.04-arm` | the blocked GEMM and SIMD GEMV over Highway, on NEON |

The arm64 pair covers the code that only an AArch64 build compiles: the NEON
paths of the SIMD kernels, which the Jetson and the Ventuno Q run. Neither
build uses `-march=native`, so both target baseline armv8-a. Valgrind does
not support SVE, which the hosted arm64 runners' cores have.

Each configuration is compared only with main's counts for the same
configuration: its own baseline artifact, `ir-counts-<configuration>`.

For each kernel in [`kernels.txt`](kernels.txt), `ir_gate.py count` runs

```text
valgrind --tool=callgrind --toggle-collect=mtl5_bench_counted_region \
    mtl5_bench --count <area/op/dtype> --class small
```

`mtl5_bench --count` builds the registry entry's problem and runs it once as a
warm-up. It then runs it once more inside `mtl5_bench_counted_region()`, the only
call callgrind counts, and finally checks the result. A kernel whose check fails
fails the job. Setup, the warm-up and the check are not counted.

`ir_gate.py compare` then compares the counts with those of main's latest
successful run of the workflow, which each push to main uploads as the artifact
`ir-counts-<config>`. Each kernel gets one of these results:

| status | meaning | the gate |
|---|---|---|
| `ok` | within ±2% of main | passes |
| `faster` | more than 2% below main | passes |
| `REGRESSION` | more than 2% above main | **fails** |
| `accepted` | more than 2% above main, and at or below its `max_ir` in `accepted.json` | passes |
| `new` / `removed` | in only one of the two runs | passes |

The table goes to the job summary. On a push to main the comparison with the
previous main run is a report only: the PR was gated already.

### Pinned so that the counts measure the code, not the runner

`ir_gate.py` runs every count in a fixed environment (`PINNED_ENV`):

- `MTL5_NUM_THREADS=1`: no count depends on a scheduler.
- `MTL5_BLOCKING_KC=256`, `MTL5_BLOCKING_MC=96`: the blocked GEMM derives kc and
  mc from the detected caches. Valgrind passes the host's cache description
  through, so two runners with different caches would count different
  blockings. The pins need a build with `-DMTL5_ENABLE_BLOCKING_OVERRIDE`.
  `mtl5_bench --count` reports whether the override is compiled in, and
  `ir_gate.py` refuses a build without it.
- `MTL5_NC_MODEL=m0_default`: the nc model that does not read the L3 size.

So the gate counts the GEMM at one fixed blocking, not at the one a user's
machine derives. That is the point: how the blocking responds to a cache is a
wall-clock question.

The pins are recorded in the counts file, and a baseline taken with other pins,
another configuration or another class is not compared.

### Accepting an intended increase

A new algorithm or a deliberate trade-off may cost instructions. The PR that
makes the change commits the acceptance in
[`accepted.json`](accepted.json):

```json
{
  "kernels": {
    "blas3/gemm/f64": {
      "max_ir": {"native-fast": 4812000, "native-fast-arm64": 5104000},
      "reason": "#NNN: packs B once more for the edge tile"
    }
  }
}
```

- `max_ir` maps each configuration that the change makes costlier to the count
  that configuration's job measured, which its job summary prints. The kernel
  passes up to that count and no further, so a later regression on top of the
  accepted one still fails.
- **One count per configuration.** Each build counts a kernel differently, so a
  configuration that is not listed is not accepted: its rise still fails. A bare
  number for all configurations is refused.
- `reason` is required, and should name the issue or PR.
- After the merge, main's baseline carries the new count, and the gate reports
  the acceptance as stale. Remove it in a later PR. A stale acceptance doesn't
  fail anything, but it keeps that kernel's gate loose up to its `max_ir`.

### Running it locally

```bash
cmake -B build-ir -DCMAKE_BUILD_TYPE=Release -DMTL5_BUILD_BENCHMARKS=ON \
      -DMTL5_BUILD_TESTS=OFF -DMTL5_BUILD_EXAMPLES=OFF \
      -DCMAKE_CXX_FLAGS=-DMTL5_ENABLE_BLOCKING_OVERRIDE
cmake --build build-ir --target mtl5_bench -j4
python3 benchmarks/regression/ir_gate.py count --bench build-ir/benchmarks/mtl5_bench \
    --kernels benchmarks/regression/kernels.txt --config native --out before.json
# ... change the code, rebuild, count again into after.json ...
python3 benchmarks/regression/ir_gate.py compare --baseline before.json --current after.json \
    --accepted benchmarks/regression/accepted.json
python3 benchmarks/regression/test_ir_gate.py     # the script's own tests
```

The 14 kernels take about 3 seconds under callgrind. Two counts of one binary are
identical. A different compiler version gives different counts, so compare a
local run only with another local run, never with CI's artifact.

### Choosing the kernels

`kernels.txt` lists `mtl5_bench` registry ids (`mtl5_bench --list`). The set is
small on purpose. It covers:

- dense: GEMM, GEMV, dot, the norm, axpy;
- the dense factorizations: LU, Cholesky, QR;
- sparse: SpMV, the level-scheduled triangular solve, native KLU;
- iterative: CG and BiCGSTAB.

Adding a kernel shows it as `new` in its first PR, which passes.

### What the gate does not catch

An instruction count is blind to cache misses, to vector width and to threads.
A kernel that runs the same instructions with a worse access pattern is not a
regression to this gate. Neither is one that stopped using all cores. Those are
what the wall-clock runs on the benchmark machines measure (the follow-up of
#944).

## The dispatch tests

`mult` chooses its kernel at compile time: BLAS, the native blocked GEMM and
SIMD GEMV, or the portable loop. A choice that silently falls through to the
portable loop gives the right answer slowly, so no result test sees it.

`include/mtl/detail/dispatch_trace.hpp` is a hook that a test opts into by
defining `MTL5_DISPATCH_TRACE`. Each `mult` branch then counts its hits, and the
test reads which path a product took. In every other build the hook compiles to
nothing.

- `test_mult_dispatch.cpp` turns the native-fast path on and checks dense float
  and double, both orientations, `C = A * B`, matrix-vector, and the widening
  float-to-double accumulator. It expects BLAS where `MTL5_HAS_BLAS` is defined,
  the blocked GEMM elsewhere, and the generic kernel for a type no fast path
  supports.
- `test_mult_dispatch_generic.cpp` turns the native-fast path off, and checks
  the default build's path.
