# mtl5_bench capability card -- ryzen-7-2700x

<!-- Written before benchmarks/publish_platform.cmake existed; its raw session is
     not yet in benchmarks/data/. Publishing that session with the script
     replaces this page. -->

The platform benchmark's capability cards for the `ryzen-7-2700x` machine described on the
[benchmark systems](systems.md) page: every operation family, measured against this
machine's FMA peak and memory bandwidth (#721).

> **Preliminary.** Measured outside #737's methodology: cores not pinned, the machine's
> clock policy not applied, one session, so no repeatability band. The session's raw
> results (JSON and sidecars) are not yet committed under `benchmarks/data/ryzen-7-2700x/mtl5_bench/`.

| | |
|---|---|
| session | `20261108T224456Z` (UTC) |
| commit | `e096dbbc6788` |
| class | `medium` |
| runs | native, native-fast, at 1, 16 threads |
| failed checks | 0 |
| raw data | not yet committed |

## AMD Ryzen 7 2700X Eight-Core Processor -- native, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 2700X Eight-Core Processor (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.51.36256 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `e096dbbc6788` |
| FMA peak (ceiling) | f64 31.9 GFLOP/s, f32 63.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 31.9, f32 63.6 GFLOP/s |
| ... this build, scalar | f64 1.2, f32 1.2 GFLOP/s |
| bandwidth | read 21.1, update 23.4, triad 19.4 GB/s (ceiling 23.4) |
| last-level cache | 8 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 40% | 80% | 0 |
| blas2 | 6 | 40% | 48% | 0 |
| blas3 | 10 | 26% | 32% | 0 |
| factor | 12 | 4% | 9% | 0 |
| eigen | 8 | 7% | 13% | 0 |
| elementwise | 5 | 10% | 21% | 0 |
| sparse | 5 | 37% | 86% | 0 |
| direct | 7 | 1% | 1% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 53% | 62% | 0 |
| mg | 1 | 51% | 51% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 20% | 33% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 14.194 ms | 1.18 | 9.46 | 40% | ok |
| blas1 | dot | f32 | n=8388608 | 14.387 ms | 1.17 | 4.66 | 20% | ok |
| blas1 | axpy | f64 | n=8388608 | 16.203 ms | 1.04 | 12.43 | 53% | ok |
| blas1 | axpy | f32 | n=8388608 | 14.882 ms | 1.13 | 6.76 | 29% | ok |
| blas1 | scale | f64 | n=8388608 | 7.206 ms | 1.16 | 18.63 | 80% | ok |
| blas1 | two_norm | f64 | n=8388608 | 14.603 ms | 1.15 | 4.60 | 20% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.570 ms | 1.28 | 10.21 | 44% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 8.846 ms | 0.95 | 7.59 | 32% | ok |
| blas1 | sum | f64 | n=8388608 | 6.636 ms | 1.26 | 10.11 | 43% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 116.474 ms | 1.58 | 1.15 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 14.252 ms | 2.35 | 9.42 | 40% | ok |
| blas2 | gemv | f32 | n=4096 | 13.097 ms | 2.56 | 5.13 | 22% | ok |
| blas2 | symv | f64 | n=4096 | 9.995 ms | 3.36 | 6.72 | 29% | ok |
| blas2 | trmv | f64 | n=4096 | 7.160 ms | 2.34 | 9.38 | 40% | ok |
| blas2 | trsv | f64 | n=4096 | 6.939 ms | 2.42 | 9.68 | 41% | ok |
| blas2 | ger | f64 | n=4096 | 24.014 ms | 1.40 | 11.18 | 48% | ok |
| blas3 | gemm | f64 | 512x512x512 | 28.436 ms | 9.44 | 0.22 | 30% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.307 ms | 20.17 | 0.24 | 32% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 15.912 ms | 8.43 | 0.40 | 26% | ok |
| blas3 | symm | f64 | 512x512x512 | 28.809 ms | 9.32 | 0.22 | 29% | ok |
| blas3 | syrk | f64 | 512x512x512 | 18.498 ms | 7.26 | 0.23 | 23% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 29.410 ms | 9.13 | 0.21 | 29% | ok |
| blas3 | trmm | f64 | 512x512x512 | 45.065 ms | 2.98 | 0.12 | 9% | ok |
| blas3 | trsm | f64 | 512x512x512 | 45.826 ms | 2.93 | 0.11 | 9% | ok |
| blas3 | trsm | f32 | 512x512x512 | 50.882 ms | 2.64 | 0.05 | 4% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 19.056 ms | 0.28 | 2.23 | 10% | ok |
| factor | lu | f64 | 512x512 | 148.661 ms | 0.60 | 0.03 | 2% | ok |
| factor | lu | f32 | 512x512 | 150.947 ms | 0.59 | 0.01 | 1% | ok |
| factor | cholesky | f64 | 512x512 | 16.425 ms | 2.72 | 0.26 | 9% | ok |
| factor | cholesky | f32 | 512x512 | 16.857 ms | 2.65 | 0.12 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 16.489 ms | 2.71 | 0.25 | 9% | ok |
| factor | ldlt_bk | f64 | 512x512 | 159.439 ms | 0.28 | 0.03 | 1% | ok |
| factor | qr | f64 | 512x512 | 512.832 ms | 0.35 | 0.01 | 1% | ok |
| factor | lq | f64 | 512x512 | 154.419 ms | 1.16 | 0.03 | 4% | ok |
| factor | hessenberg | f64 | 512x512 | 685.062 ms | 0.65 | 0.01 | 2% | ok |
| factor | householder | f64 | 512x512 | 3.286 ms | 0.32 | 1.28 | 5%\* | ok |
| factor | inv | f64 | 512x512 | 171.541 ms | 1.56 | 0.02 | 5% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 64.490 ms | 1.39 | 0.07 | 4% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 33.554 ms | 0.67 | 0.03 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 111.391 ms | 1.36 | 0.01 | 4% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 586.004 ms | 0.26 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 75.484 ms | 2.22 | 0.01 | 7% | ok |
| eigen | schur | f64 | 256x256 | 173.144 ms | 2.42 | 0.01 | 8% | ok |
| eigen | svd | f64 | 256x256 | 179.009 ms | 2.06 | 0.01 | 6% | ok |
| eigen | expm | f64 | 256x256 | 61.162 ms | 4.11 | 0.02 | 13% | ok |
| eigen | sylvester | f64 | 256x256 | 434.842 ms | 2.31 | 0.00 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 57.329 ms | 0.15 | 2.34 | 10% | ok |
| elementwise | log | f64 | n=8388608 | 75.034 ms | 0.11 | 1.79 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 56.559 ms | 0.15 | 2.37 | 10% | ok |
| elementwise | sqrt | f64 | n=8388608 | 27.251 ms | 0.31 | 4.93 | 21% | ok |
| elementwise | erf | f64 | n=8388608 | 60.953 ms | 0.14 | 2.20 | 9% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.359 ms | 1.93 | 20.04 | 86% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.144 ms | 0.83 | 8.66 | 37% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 7.615 ms | 5.50 | 11.56 | 49% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 21.180 ms | 0.15 | 0.54 | 2% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 2.961 ms | 0.68 | 5.62 | 24% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 185.719 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 53.487 ms | 0.21 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 34.282 ms | 0.33 | 0.02 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 112.160 ms | 0.20 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 121.169 ms | 0.18 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 46.807 ms | 0.48 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 353.341 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 5.922 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 9.345 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.779 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 27.218 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 53.128 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 67.238 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 44.273 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 22.740 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 33.048 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 23.994 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 78.125 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 48.771 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 27.843 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 44.966 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 55.468 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 84.429 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 72.799 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.061 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.382 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 46.639 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.400 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 3.559 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.049 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.133 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.358 ms | 1.11 | 12.44 | 53% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.032 ms | 1.29 | 14.43 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.376 ms | 1.10 | 12.35 | 53% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 24.197 ms | 1.59 | 12.01 | 51% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 63.165 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 69.704 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 126.893 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.852 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 14.221 ms | 1.18 | 4.72 | 20% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 53.319 ms | 5.03 | 0.08 | 16% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.300 ms | 7.29 | 7.75 | 33% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.358 ms | 3.85 | 15.40 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.988 ms | 4.21 | 8.41 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.635 ms | 6.37 | 6.37 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.236 ms | 7.50 | 7.50 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 18.349 ms | 14.63 | 0.17 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 47.401 ms | 5.66 | 0.04 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 38.913 ms | 6.90 | 0.04 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 39.613 ms | 6.78 | 0.04 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 38.405 ms | 6.99 | 0.04 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## AMD Ryzen 7 2700X Eight-Core Processor -- native, 16 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 2700X Eight-Core Processor (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.51.36256 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `e096dbbc6788` |
| FMA peak (ceiling) | f64 248.9 GFLOP/s, f32 498.5 GFLOP/s |
| ... hardware, AVX2+FMA | f64 248.9, f32 498.5 GFLOP/s |
| ... this build, scalar | f64 14.9, f32 17.0 GFLOP/s |
| bandwidth | read 26.3, update 27.4, triad 21.2 GB/s (ceiling 27.4) |
| last-level cache | 8 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 91% | 112% | 0 | 14% |
| blas2 | 6 | 41% | 108% | 0 | 6% |
| blas3 | 10 | 4% | 30% | 0 | 6% |
| factor | 12 | 1% | 20% | 0 | 10% |
| eigen | 8 | 1% | 3% | 0 | 6% |
| elementwise | 5 | 8% | 17% | 0 | 6% |
| sparse | 5 | 32% | 103% | 0 | 6% |
| direct | 7 | 0% | 0% | 0 | 6% |
| ordering | 4 | n/a | n/a | 0 | 6% |
| krylov | 13 | n/a | n/a | 0 | 6% |
| pc | 7 | n/a | n/a | 0 | 6% |
| smoother | 3 | 47% | 54% | 0 | 6% |
| mg | 1 | 41% | 41% | 0 | 6% |
| sparse_eig | 4 | n/a | n/a | 0 | 5% |
| mixed | 3 | 16% | 21% | 0 | 6% |
| integer | 9 | n/a | n/a | 0 | 29% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.706 ms | 3.57 | 28.52 | 104% | ok | 3.0x (19%) |
| blas1 | dot | f32 | n=8388608 | 2.809 ms | 5.97 | 23.89 | 87% | ok | 5.1x (32%) |
| blas1 | axpy | f64 | n=8388608 | 7.158 ms | 2.34 | 28.13 | 103% | ok | 2.3x (14%) |
| blas1 | axpy | f32 | n=8388608 | 3.499 ms | 4.79 | 28.77 | 105% | ok | 4.3x (27%) |
| blas1 | scale | f64 | n=8388608 | 4.374 ms | 1.92 | 30.69 | 112% | ok | 1.6x (10%) |
| blas1 | two_norm | f64 | n=8388608 | 2.687 ms | 6.25 | 24.98 | 91% | ok | 5.4x (34%) |
| blas1 | one_norm | f64 | n=8388608 | 6.604 ms | 1.27 | 10.16 | 37% | ok | 1.0x (6%) |
| blas1 | infinity_norm | f64 | n=8388608 | 8.787 ms | 0.95 | 7.64 | 28% | ok | 1.0x (6%) |
| blas1 | sum | f64 | n=8388608 | 6.611 ms | 1.27 | 10.15 | 37% | ok | 1.0x (6%) |
| blas1 | dot_k3 | f64 | n=8388608 | 115.573 ms | 1.60 | 1.16 | 4% | ok | 1.0x (6%) |
| blas2 | gemv | f64 | n=4096 | 4.541 ms | 7.39 | 29.57 | 108% | ok | 3.1x (20%) |
| blas2 | gemv | f32 | n=4096 | 2.621 ms | 12.80 | 25.61 | 94% | ok | 5.0x (31%) |
| blas2 | symv | f64 | n=4096 | 10.069 ms | 3.33 | 6.67 | 24% | ok | 1.0x (6%) |
| blas2 | trmv | f64 | n=4096 | 7.113 ms | 2.36 | 9.44 | 35% | ok | 1.0x (6%) |
| blas2 | trsv | f64 | n=4096 | 7.081 ms | 2.37 | 9.49 | 35% | ok | 1.0x (6%) |
| blas2 | ger | f64 | n=4096 | 23.955 ms | 1.40 | 11.21 | 41% | ok | 1.0x (6%) |
| blas3 | gemm | f64 | 512x512x512 | 3.622 ms | 74.11 | 1.74 | 30% | ok | 7.9x (49%) |
| blas3 | gemm | f32 | 512x512x512 | 2.715 ms | 98.88 | 1.16 | 20% | ok | 4.9x (31%) |
| blas3 | gemmt | f64 | 512x512x512 | 15.902 ms | 8.44 | 0.40 | 3% | ok | 1.0x (6%) |
| blas3 | symm | f64 | 512x512x512 | 28.242 ms | 9.50 | 0.22 | 4% | ok | 1.0x (6%) |
| blas3 | syrk | f64 | 512x512x512 | 18.815 ms | 7.13 | 0.22 | 3% | ok | 1.0x (6%) |
| blas3 | syr2k | f64 | 512x512x512 | 28.495 ms | 9.42 | 0.22 | 4% | ok | 1.0x (6%) |
| blas3 | trmm | f64 | 512x512x512 | 44.256 ms | 3.03 | 0.12 | 1% | ok | 1.0x (6%) |
| blas3 | trsm | f64 | 512x512x512 | 46.392 ms | 2.89 | 0.11 | 1% | ok | 1.0x (6%) |
| blas3 | trsm | f32 | 512x512x512 | 44.436 ms | 3.02 | 0.06 | 1% | ok | 1.1x (7%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 17.021 ms | 0.31 | 2.50 | 9% | ok | 1.1x (7%) |
| factor | lu | f64 | 512x512 | 50.534 ms | 1.77 | 0.08 | 1% | ok | 2.9x (18%) |
| factor | lu | f32 | 512x512 | 58.603 ms | 1.53 | 0.04 | 0% | ok | 2.6x (16%) |
| factor | cholesky | f64 | 512x512 | 16.523 ms | 2.71 | 0.25 | 1% | ok | 1.0x (6%) |
| factor | cholesky | f32 | 512x512 | 16.316 ms | 2.74 | 0.13 | 1% | ok | 1.0x (6%) |
| factor | ldlt | f64 | 512x512 | 15.902 ms | 2.81 | 0.26 | 1% | ok | 1.0x (6%) |
| factor | ldlt_bk | f64 | 512x512 | 159.553 ms | 0.28 | 0.03 | 0% | ok | 1.0x (6%) |
| factor | qr | f64 | 512x512 | 310.657 ms | 0.58 | 0.01 | 0% | ok | 1.7x (10%) |
| factor | lq | f64 | 512x512 | 72.861 ms | 2.46 | 0.06 | 1% | ok | 2.1x (13%) |
| factor | hessenberg | f64 | 512x512 | 418.485 ms | 1.07 | 0.01 | 0% | ok | 1.6x (10%) |
| factor | householder | f64 | 512x512 | 0.778 ms | 1.35 | 5.39 | 20%\* | ok | 4.2x (26%) |
| factor | inv | f64 | 512x512 | 159.077 ms | 1.69 | 0.03 | 1% | ok | 1.1x (7%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 63.831 ms | 1.40 | 0.07 | 1% | ok | 1.0x (6%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 33.550 ms | 0.67 | 0.03 | 0% | ok | 1.0x (6%) |
| eigen | eigen_symmetric | f64 | 256x256 | 104.711 ms | 1.44 | 0.01 | 1% | ok | 1.1x (7%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 583.927 ms | 0.26 | 0.00 | 0% | ok | 1.0x (6%) |
| eigen | eigenvalue | f64 | 256x256 | 68.055 ms | 2.47 | 0.02 | 1% | ok | 1.1x (7%) |
| eigen | schur | f64 | 256x256 | 233.611 ms | 1.80 | 0.00 | 1% | ok | 0.7x (5%) |
| eigen | svd | f64 | 256x256 | 173.886 ms | 2.12 | 0.01 | 1% | ok | 1.0x (6%) |
| eigen | expm | f64 | 256x256 | 36.989 ms | 6.80 | 0.03 | 3% | ok | 1.7x (10%) |
| eigen | sylvester | f64 | 256x256 | 445.949 ms | 2.26 | 0.00 | 1% | ok | 1.0x (6%) |
| elementwise | exp | f64 | n=8388608 | 54.608 ms | 0.15 | 2.46 | 9% | ok | 1.0x (7%) |
| elementwise | log | f64 | n=8388608 | 68.645 ms | 0.12 | 1.96 | 7% | ok | 1.1x (7%) |
| elementwise | sin | f64 | n=8388608 | 58.953 ms | 0.14 | 2.28 | 8% | ok | 1.0x (6%) |
| elementwise | sqrt | f64 | n=8388608 | 28.626 ms | 0.29 | 4.69 | 17% | ok | 1.0x (6%) |
| elementwise | erf | f64 | n=8388608 | 64.030 ms | 0.13 | 2.10 | 8% | ok | 1.0x (6%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.969 ms | 2.70 | 28.10 | 103% | ok | 1.4x (9%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.153 ms | 0.83 | 8.64 | 32% | ok | 1.0x (6%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.561 ms | 9.18 | 19.30 | 71% | ok | 1.7x (10%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.681 ms | 0.16 | 0.56 | 2% | ok | 1.0x (6%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 3.881 ms | 0.52 | 4.29 | 16% | ok | 0.8x (5%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 174.271 ms | 0.06 | 0.00 | 0% | ok | 1.1x (7%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 53.205 ms | 0.21 | 0.01 | 0% | ok | 1.0x (6%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 33.784 ms | 0.33 | 0.02 | 0% | ok | 1.0x (6%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 109.599 ms | 0.20 | 0.01 | 0% | ok | 1.0x (6%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 134.579 ms | 0.17 | 0.01 | 0% | ok | 0.9x (6%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 50.168 ms | 0.45 | 0.01 | 0% | ok | 0.9x (6%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 420.798 ms | 0.00 | 0.00 | 0% | ok | 0.8x (5%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 6.478 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 10.788 ms | 0.00 | 0.00 | n/a | ok | 0.9x (5%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.866 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 32.297 ms | 0.00 | 0.00 | n/a | ok | 0.8x (5%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 56.006 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 70.935 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 46.187 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 25.129 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 32.520 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 25.480 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 87.694 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.748 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.736 ms | 0.00 | 0.00 | n/a | ok | 0.9x (5%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 48.518 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 53.815 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 93.213 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 79.002 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.076 ms | 0.00 | 0.00 | n/a | ok | 0.8x (5%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.579 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| pc | ilut | f64 | 128^2 grid | 49.207 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| pc | ic_0 | f64 | 128^2 grid | 3.460 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| pc | ildl | f64 | 128^2 grid | 3.748 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| pc | ssor | f64 | 128^2 grid | 0.029 ms | 0.00 | 0.00 | n/a | ok | 1.7x (11%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.145 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.272 ms | 1.15 | 12.91 | 47% | ok | 1.0x (6%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.002 ms | 1.31 | 14.65 | 54% | ok | 1.0x (6%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.406 ms | 1.09 | 12.19 | 45% | ok | 1.0x (6%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 25.987 ms | 1.48 | 11.18 | 41% | ok | 0.9x (6%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 55.326 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 101.304 ms | 0.00 | 0.00 | n/a | ok | 0.7x (4%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 152.520 ms | 0.00 | 0.00 | n/a | ok | 0.8x (5%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 14.543 ms | 0.00 | 0.00 | n/a | ok | 0.8x (5%) |
| mixed | dot | f32>f64 | n=8388608 | 15.628 ms | 1.07 | 4.29 | 16% | ok | 0.9x (6%) |
| mixed | gemm | f32>f64 | 512x512x512 | 7.016 ms | 38.26 | 0.60 | 15% | ok | 7.6x (47%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 3.078 ms | 5.45 | 5.79 | 21% | ok | 0.7x (5%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.302 ms | 3.90 | 15.60 | n/a | ok | 1.0x (6%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.894 ms | 4.31 | 8.62 | n/a | ok | 1.0x (6%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.174 ms | 7.72 | 7.72 | n/a | ok | 1.2x (8%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.165 ms | 7.75 | 7.75 | n/a | ok | 1.0x (6%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.704 ms | 99.26 | 1.16 | n/a | ok | 6.8x (42%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 8.559 ms | 31.36 | 0.25 | n/a | ok | 5.5x (35%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 8.393 ms | 31.98 | 0.19 | n/a | ok | 4.6x (29%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 7.651 ms | 35.08 | 0.21 | n/a | ok | 5.2x (32%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 7.734 ms | 34.71 | 0.20 | n/a | ok | 5.0x (31%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## AMD Ryzen 7 2700X Eight-Core Processor -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 2700X Eight-Core Processor (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.51.36256 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `e096dbbc6788` |
| FMA peak (ceiling) | f64 31.9 GFLOP/s, f32 64.2 GFLOP/s |
| ... hardware, AVX2+FMA | f64 31.9, f32 64.2 GFLOP/s |
| ... this build, SSE2 | f64 31.9, f32 63.9 GFLOP/s |
| bandwidth | read 25.1, update 21.0, triad 18.3 GB/s (ceiling 25.1) |
| last-level cache | 8 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 85% | 106% | 0 |
| blas2 | 6 | 89% | 98% | 0 |
| blas3 | 10 | 27% | 42% | 0 |
| factor | 12 | 3% | 9% | 0 |
| eigen | 8 | 6% | 16% | 0 |
| elementwise | 5 | 9% | 19% | 0 |
| sparse | 5 | 34% | 85% | 0 |
| direct | 7 | 1% | 1% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 52% | 58% | 0 |
| mg | 1 | 47% | 47% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 79% | 84% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.268 ms | 3.18 | 25.48 | 101% | ok |
| blas1 | dot | f32 | n=8388608 | 2.517 ms | 6.67 | 26.66 | 106% | ok |
| blas1 | axpy | f64 | n=8388608 | 8.908 ms | 1.88 | 22.60 | 90% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.689 ms | 3.58 | 21.47 | 85% | ok |
| blas1 | scale | f64 | n=8388608 | 6.799 ms | 1.23 | 19.74 | 79% | ok |
| blas1 | two_norm | f64 | n=8388608 | 2.823 ms | 5.94 | 23.77 | 95% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.628 ms | 1.27 | 10.12 | 40% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 22.107 ms | 0.38 | 3.04 | 12% | ok |
| blas1 | sum | f64 | n=8388608 | 6.638 ms | 1.26 | 10.11 | 40% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 119.442 ms | 1.55 | 1.12 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 5.478 ms | 6.12 | 24.51 | 98% | ok |
| blas2 | gemv | f32 | n=4096 | 2.811 ms | 11.94 | 23.89 | 95% | ok |
| blas2 | symv | f64 | n=4096 | 10.340 ms | 3.25 | 6.50 | 26% | ok |
| blas2 | trmv | f64 | n=4096 | 2.997 ms | 5.60 | 22.41 | 89% | ok |
| blas2 | trsv | f64 | n=4096 | 3.172 ms | 5.29 | 21.18 | 84% | ok |
| blas2 | ger | f64 | n=4096 | 24.293 ms | 1.38 | 11.05 | 44% | ok |
| blas3 | gemm | f64 | 512x512x512 | 21.348 ms | 12.57 | 0.29 | 39% | ok |
| blas3 | gemm | f32 | 512x512x512 | 9.929 ms | 27.04 | 0.32 | 42% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 13.463 ms | 9.97 | 0.47 | 31% | ok |
| blas3 | symm | f64 | 512x512x512 | 22.346 ms | 12.01 | 0.28 | 38% | ok |
| blas3 | syrk | f64 | 512x512x512 | 19.065 ms | 7.04 | 0.22 | 22% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 30.963 ms | 8.67 | 0.20 | 27% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.742 ms | 7.56 | 0.30 | 24% | ok |
| blas3 | trsm | f64 | 512x512x512 | 21.624 ms | 6.21 | 0.24 | 19% | ok |
| blas3 | trsm | f32 | 512x512x512 | 11.764 ms | 11.41 | 0.22 | 18% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 16.838 ms | 0.32 | 2.52 | 10% | ok |
| factor | lu | f64 | 512x512 | 149.146 ms | 0.60 | 0.03 | 2% | ok |
| factor | lu | f32 | 512x512 | 51.625 ms | 1.73 | 0.04 | 3% | ok |
| factor | cholesky | f64 | 512x512 | 16.437 ms | 2.72 | 0.26 | 9% | ok |
| factor | cholesky | f32 | 512x512 | 16.436 ms | 2.72 | 0.13 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 16.202 ms | 2.76 | 0.26 | 9% | ok |
| factor | ldlt_bk | f64 | 512x512 | 159.488 ms | 0.28 | 0.03 | 1% | ok |
| factor | qr | f64 | 512x512 | 511.967 ms | 0.35 | 0.01 | 1% | ok |
| factor | lq | f64 | 512x512 | 172.004 ms | 1.04 | 0.02 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 678.472 ms | 0.66 | 0.01 | 2% | ok |
| factor | householder | f64 | 512x512 | 3.289 ms | 0.32 | 1.28 | 5%\* | ok |
| factor | inv | f64 | 512x512 | 206.977 ms | 1.30 | 0.02 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 153.007 ms | 0.58 | 0.03 | 2% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.822 ms | 0.56 | 0.03 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 113.651 ms | 1.33 | 0.01 | 4% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 610.571 ms | 0.25 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 68.393 ms | 2.45 | 0.02 | 8% | ok |
| eigen | schur | f64 | 256x256 | 220.307 ms | 1.90 | 0.00 | 6% | ok |
| eigen | svd | f64 | 256x256 | 183.620 ms | 2.01 | 0.01 | 6% | ok |
| eigen | expm | f64 | 256x256 | 50.017 ms | 5.03 | 0.02 | 16% | ok |
| eigen | sylvester | f64 | 256x256 | 393.788 ms | 2.56 | 0.00 | 8% | ok |
| elementwise | exp | f64 | n=8388608 | 61.654 ms | 0.14 | 2.18 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 70.100 ms | 0.12 | 1.91 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 59.464 ms | 0.14 | 2.26 | 9% | ok |
| elementwise | sqrt | f64 | n=8388608 | 28.556 ms | 0.29 | 4.70 | 19% | ok |
| elementwise | erf | f64 | n=8388608 | 61.858 ms | 0.14 | 2.17 | 9% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.270 ms | 2.06 | 21.44 | 85% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.167 ms | 0.83 | 8.60 | 34% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 7.504 ms | 5.58 | 11.73 | 47% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 21.023 ms | 0.15 | 0.55 | 2% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 2.982 ms | 0.67 | 5.58 | 22% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 172.342 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 52.812 ms | 0.21 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 32.804 ms | 0.34 | 0.02 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 100.850 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 127.068 ms | 0.18 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 53.139 ms | 0.42 | 0.01 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 402.178 ms | 0.00 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 6.007 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 9.421 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.877 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 26.042 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 43.186 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 51.599 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 38.306 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 18.357 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 25.135 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 21.382 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 53.390 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 38.915 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.594 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 34.267 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 36.995 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 54.484 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 45.719 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.074 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.586 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 43.890 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.183 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 3.554 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.039 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.180 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.253 ms | 1.16 | 13.02 | 52% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.010 ms | 1.30 | 14.59 | 58% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.393 ms | 1.09 | 12.26 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 24.617 ms | 1.56 | 11.81 | 47% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 51.213 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 70.644 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 119.792 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 10.019 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 3.196 ms | 5.25 | 21.00 | 84% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 25.600 ms | 10.49 | 0.16 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.898 ms | 18.67 | 19.84 | 79% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.816 ms | 5.96 | 23.83 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.203 ms | 13.94 | 27.89 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.857 ms | 19.57 | 19.57 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.677 ms | 24.76 | 24.76 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 27.443 ms | 9.78 | 0.11 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 27.521 ms | 9.75 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 29.075 ms | 9.23 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 7.638 ms | 35.14 | 0.21 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 6.695 ms | 40.09 | 0.23 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## AMD Ryzen 7 2700X Eight-Core Processor -- native-fast, 16 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 2700X Eight-Core Processor (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.51.36256 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `e096dbbc6788` |
| FMA peak (ceiling) | f64 246.6 GFLOP/s, f32 494.3 GFLOP/s |
| ... hardware, AVX2+FMA | f64 243.3, f32 488.7 GFLOP/s |
| ... this build, SSE2 | f64 246.6, f32 494.3 GFLOP/s |
| bandwidth | read 25.4, update 26.2, triad 20.9 GB/s (ceiling 26.2) |
| last-level cache | 8 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 112% | 118% | 0 | 8% |
| blas2 | 6 | 81% | 116% | 0 | 6% |
| blas3 | 10 | 5% | 23% | 0 | 10% |
| factor | 12 | 1% | 24% | 0 | 10% |
| eigen | 8 | 1% | 3% | 0 | 7% |
| elementwise | 5 | 9% | 19% | 0 | 6% |
| sparse | 5 | 33% | 107% | 0 | 7% |
| direct | 7 | 0% | 0% | 0 | 6% |
| ordering | 4 | n/a | n/a | 0 | 6% |
| krylov | 13 | n/a | n/a | 0 | 6% |
| pc | 7 | n/a | n/a | 0 | 7% |
| smoother | 3 | 49% | 55% | 0 | 6% |
| mg | 1 | 46% | 46% | 0 | 6% |
| sparse_eig | 4 | n/a | n/a | 0 | 6% |
| mixed | 3 | 76% | 77% | 0 | 6% |
| integer | 9 | n/a | n/a | 0 | 18% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.357 ms | 3.85 | 30.81 | 117% | ok | 1.2x (8%) |
| blas1 | dot | f32 | n=8388608 | 2.250 ms | 7.46 | 29.83 | 114% | ok | 1.1x (7%) |
| blas1 | axpy | f64 | n=8388608 | 6.836 ms | 2.45 | 29.45 | 112% | ok | 1.3x (8%) |
| blas1 | axpy | f32 | n=8388608 | 3.264 ms | 5.14 | 30.84 | 118% | ok | 1.4x (9%) |
| blas1 | scale | f64 | n=8388608 | 4.458 ms | 1.88 | 30.11 | 115% | ok | 1.5x (10%) |
| blas1 | two_norm | f64 | n=8388608 | 2.295 ms | 7.31 | 29.24 | 111% | ok | 1.2x (8%) |
| blas1 | one_norm | f64 | n=8388608 | 6.608 ms | 1.27 | 10.16 | 39% | ok | 1.0x (6%) |
| blas1 | infinity_norm | f64 | n=8388608 | 14.559 ms | 0.58 | 4.61 | 18% | ok | 1.5x (9%) |
| blas1 | sum | f64 | n=8388608 | 6.648 ms | 1.26 | 10.09 | 38% | ok | 1.0x (6%) |
| blas1 | dot_k3 | f64 | n=8388608 | 117.447 ms | 1.57 | 1.14 | 4% | ok | 1.0x (6%) |
| blas2 | gemv | f64 | n=4096 | 4.405 ms | 7.62 | 30.48 | 116% | ok | 1.2x (8%) |
| blas2 | gemv | f32 | n=4096 | 2.266 ms | 14.81 | 29.63 | 113% | ok | 1.2x (8%) |
| blas2 | symv | f64 | n=4096 | 10.024 ms | 3.35 | 6.70 | 26% | ok | 1.0x (6%) |
| blas2 | trmv | f64 | n=4096 | 3.159 ms | 5.31 | 21.26 | 81% | ok | 0.9x (6%) |
| blas2 | trsv | f64 | n=4096 | 3.365 ms | 4.99 | 19.96 | 76% | ok | 0.9x (6%) |
| blas2 | ger | f64 | n=4096 | 24.890 ms | 1.35 | 10.79 | 41% | ok | 1.0x (6%) |
| blas3 | gemm | f64 | 512x512x512 | 4.760 ms | 56.40 | 1.32 | 23% | ok | 4.5x (28%) |
| blas3 | gemm | f32 | 512x512x512 | 3.837 ms | 69.97 | 0.82 | 14% | ok | 2.6x (16%) |
| blas3 | gemmt | f64 | 512x512x512 | 14.098 ms | 9.52 | 0.45 | 4% | ok | 1.0x (6%) |
| blas3 | symm | f64 | 512x512x512 | 5.832 ms | 46.02 | 1.08 | 19% | ok | 3.8x (24%) |
| blas3 | syrk | f64 | 512x512x512 | 18.837 ms | 7.13 | 0.22 | 3% | ok | 1.0x (6%) |
| blas3 | syr2k | f64 | 512x512x512 | 28.438 ms | 9.44 | 0.22 | 4% | ok | 1.1x (7%) |
| blas3 | trmm | f64 | 512x512x512 | 11.427 ms | 11.75 | 0.46 | 5% | ok | 1.6x (10%) |
| blas3 | trsm | f64 | 512x512x512 | 11.385 ms | 11.79 | 0.46 | 5% | ok | 1.9x (12%) |
| blas3 | trsm | f32 | 512x512x512 | 8.993 ms | 14.93 | 0.29 | 3% | ok | 1.3x (8%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 16.964 ms | 0.31 | 2.50 | 10% | ok | 1.0x (6%) |
| factor | lu | f64 | 512x512 | 49.409 ms | 1.81 | 0.08 | 1% | ok | 3.0x (19%) |
| factor | lu | f32 | 512x512 | 82.944 ms | 1.08 | 0.03 | 0% | ok | 0.6x (4%) |
| factor | cholesky | f64 | 512x512 | 16.451 ms | 2.72 | 0.25 | 1% | ok | 1.0x (6%) |
| factor | cholesky | f32 | 512x512 | 16.409 ms | 2.73 | 0.13 | 1% | ok | 1.0x (6%) |
| factor | ldlt | f64 | 512x512 | 15.829 ms | 2.83 | 0.26 | 1% | ok | 1.0x (6%) |
| factor | ldlt_bk | f64 | 512x512 | 160.108 ms | 0.28 | 0.03 | 0% | ok | 1.0x (6%) |
| factor | qr | f64 | 512x512 | 312.080 ms | 0.57 | 0.01 | 0% | ok | 1.6x (10%) |
| factor | lq | f64 | 512x512 | 109.561 ms | 1.63 | 0.04 | 1% | ok | 1.6x (10%) |
| factor | hessenberg | f64 | 512x512 | 434.708 ms | 1.03 | 0.01 | 0% | ok | 1.6x (10%) |
| factor | householder | f64 | 512x512 | 0.673 ms | 1.56 | 6.24 | 24%\* | ok | 4.9x (31%) |
| factor | inv | f64 | 512x512 | 159.895 ms | 1.68 | 0.03 | 1% | ok | 1.3x (8%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 53.593 ms | 1.67 | 0.08 | 1% | ok | 2.9x (18%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 43.571 ms | 0.51 | 0.02 | 0% | ok | 0.9x (6%) |
| eigen | eigen_symmetric | f64 | 256x256 | 119.587 ms | 1.26 | 0.01 | 1% | ok | 1.0x (6%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 584.781 ms | 0.26 | 0.00 | 0% | ok | 1.0x (7%) |
| eigen | eigenvalue | f64 | 256x256 | 68.048 ms | 2.47 | 0.02 | 1% | ok | 1.0x (6%) |
| eigen | schur | f64 | 256x256 | 185.943 ms | 2.26 | 0.01 | 1% | ok | 1.2x (7%) |
| eigen | svd | f64 | 256x256 | 174.965 ms | 2.11 | 0.01 | 1% | ok | 1.0x (7%) |
| eigen | expm | f64 | 256x256 | 37.893 ms | 6.64 | 0.03 | 3% | ok | 1.3x (8%) |
| eigen | sylvester | f64 | 256x256 | 390.383 ms | 2.58 | 0.00 | 1% | ok | 1.0x (6%) |
| elementwise | exp | f64 | n=8388608 | 58.003 ms | 0.14 | 2.31 | 9% | ok | 1.1x (7%) |
| elementwise | log | f64 | n=8388608 | 74.233 ms | 0.11 | 1.81 | 7% | ok | 0.9x (6%) |
| elementwise | sin | f64 | n=8388608 | 58.646 ms | 0.14 | 2.29 | 9% | ok | 1.0x (6%) |
| elementwise | sqrt | f64 | n=8388608 | 27.444 ms | 0.31 | 4.89 | 19% | ok | 1.0x (7%) |
| elementwise | erf | f64 | n=8388608 | 62.164 ms | 0.13 | 2.16 | 8% | ok | 1.0x (6%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.967 ms | 2.71 | 28.17 | 107% | ok | 1.3x (8%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.138 ms | 0.83 | 8.68 | 33% | ok | 1.0x (6%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.519 ms | 9.27 | 19.48 | 74% | ok | 1.7x (10%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 19.535 ms | 0.17 | 0.59 | 2% | ok | 1.1x (7%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 3.252 ms | 0.62 | 5.12 | 20% | ok | 0.9x (6%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 166.490 ms | 0.07 | 0.00 | 0% | ok | 1.0x (6%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 49.649 ms | 0.23 | 0.01 | 0% | ok | 1.1x (7%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 28.916 ms | 0.39 | 0.03 | 0% | ok | 1.1x (7%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 107.379 ms | 0.21 | 0.01 | 0% | ok | 0.9x (6%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 131.631 ms | 0.17 | 0.01 | 0% | ok | 1.0x (6%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 51.293 ms | 0.44 | 0.01 | 0% | ok | 1.0x (6%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 388.801 ms | 0.00 | 0.00 | 0% | ok | 1.0x (6%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 6.058 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 9.506 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.856 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 26.683 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 43.101 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 51.286 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 36.871 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.931 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 25.154 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 21.456 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 47.519 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 43.657 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 26.641 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 34.188 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 37.655 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 53.020 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 47.587 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.061 ms | 0.00 | 0.00 | n/a | ok | 1.2x (8%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.770 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| pc | ilut | f64 | 128^2 grid | 43.149 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| pc | ic_0 | f64 | 128^2 grid | 3.839 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| pc | ildl | f64 | 128^2 grid | 3.596 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| pc | ssor | f64 | 128^2 grid | 0.029 ms | 0.00 | 0.00 | n/a | ok | 1.3x (8%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.118 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.286 ms | 1.15 | 12.83 | 49% | ok | 1.0x (6%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.034 ms | 1.29 | 14.42 | 55% | ok | 1.0x (6%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.442 ms | 1.07 | 12.01 | 46% | ok | 1.0x (6%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 24.290 ms | 1.58 | 11.97 | 46% | ok | 1.0x (6%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 48.740 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 74.842 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 127.283 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.789 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| mixed | dot | f32>f64 | n=8388608 | 3.302 ms | 5.08 | 20.32 | 77% | ok | 1.0x (6%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.258 ms | 63.04 | 0.99 | 26% | ok | 6.0x (38%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.893 ms | 18.79 | 19.97 | 76% | ok | 1.0x (6%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.023 ms | 5.55 | 22.20 | n/a | ok | 0.9x (6%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.229 ms | 13.65 | 27.31 | n/a | ok | 1.0x (6%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.885 ms | 18.95 | 18.95 | n/a | ok | 1.0x (6%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.757 ms | 22.18 | 22.18 | n/a | ok | 0.9x (6%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 4.165 ms | 64.44 | 0.76 | n/a | ok | 6.6x (41%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.555 ms | 40.95 | 0.32 | n/a | ok | 4.2x (26%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.432 ms | 60.56 | 0.35 | n/a | ok | 6.6x (41%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 2.109 ms | 127.31 | 0.75 | n/a | ok | 3.6x (23%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 2.294 ms | 117.03 | 0.69 | n/a | ok | 2.9x (18%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

