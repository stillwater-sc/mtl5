# Example benchmark data

Small, committed `bench_all` CSVs so `../plot_results.py` and the documented
GFLOP/s-vs-N curves reproduce without re-running the suite. **These are
illustrative reference numbers, not an official performance claim** — re-run on
your own hardware for anything you depend on.

## Platform

All CSVs in this directory were produced on:

| | |
|---|---|
| CPU | 12th Gen Intel Core i7-12700K |
| Logical cores | 20 (run pinned to a single thread) |
| OS | Ubuntu 24.04.4 LTS (Noble), kernel 6.8.0-117 |
| Compiler | GCC 13, `-O3 -DNDEBUG` (CMake `Release`) |
| OpenBLAS | 0.3.26 (`libopenblas-dev` 0.3.26+ds-1ubuntu0.1, pthreads build) |
| Intel MKL | oneAPI 2026.0 (`BLA_VENDOR=Intel10_64lp`) |

Single-threaded for stable, comparable per-size numbers
(`OMP_NUM_THREADS=1`, plus `OPENBLAS_NUM_THREADS=1` / `MKL_NUM_THREADS=1`).

## Files

| File | Backends | How it was generated |
|------|----------|----------------------|
| `blas_sweep_openblas.csv` | native, blas (OpenBLAS) | OpenBLAS build, command below |
| `blas_sweep_mkl.csv`      | native, blas (MKL)      | MKL build, command below |

Both are the odd-size BLAS sweep `65:1025:80` (all odd, non-power-of-2 sizes)
across L1/L2/L3 (`dot`, `nrm2`, `gemv`, `gemm`).

```bash
# OpenBLAS build
cmake -B build-openblas -DMTL5_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
      -DMTL5_WITH_BLAS=ON -DMTL5_WITH_LAPACK=ON
cmake --build build-openblas --target bench_all
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ./build-openblas/benchmarks/bench_all --suite blas --sweep 65:1025:80 \
    --csv benchmarks/data/blas_sweep_openblas.csv

# MKL build
source /opt/intel/oneapi/setvars.sh
cmake -B build-mkl -DMTL5_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release \
      -DMTL5_WITH_BLAS=ON -DMTL5_WITH_LAPACK=ON -DBLA_VENDOR=Intel10_64lp
cmake --build build-mkl --target bench_all
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./build-mkl/benchmarks/bench_all --suite blas --sweep 65:1025:80 \
    --csv benchmarks/data/blas_sweep_mkl.csv
```

## Plot

```bash
./benchmarks/plot_results.py \
    benchmarks/data/blas_sweep_openblas.csv \
    benchmarks/data/blas_sweep_mkl.csv \
    --labels openblas,mkl --op gemm --out gemm_gflops.png
```
