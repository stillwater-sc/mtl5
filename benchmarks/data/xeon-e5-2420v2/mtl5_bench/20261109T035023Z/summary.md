# mtl5_bench platform characterization

Class `medium`, variants native native-fast, threads 1,12, 2026-11-09T04:03:14Z.

Not measured:
- blas: no BLAS/LAPACK found

## Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz -- native, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-42-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `73d11400a0e3` |
| FMA peak (ceiling) | f64 9.6 GFLOP/s, f32 19.5 GFLOP/s |
| ... hardware, SSE2 | f64 9.6, f32 19.5 GFLOP/s |
| ... this build, scalar | f64 0.1, f32 0.6 GFLOP/s |
| bandwidth | read 9.8, update 9.7, triad 7.8 GB/s (ceiling 9.8) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 24% | 102% | 0 |
| blas2 | 6 | 66% | 93% | 0 |
| blas3 | 10 | 41% | 46% | 0 |
| factor | 12 | 18% | 40% | 0 |
| eigen | 8 | 16% | 27% | 0 |
| elementwise | 5 | 9% | 15% | 0 |
| sparse | 5 | 56% | 108% | 0 |
| direct | 7 | 2% | 5% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 79% | 92% | 0 |
| mg | 1 | 80% | 80% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 24% | 75% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 198.243 ms | 0.08 | 0.68 | 7% | ok |
| blas1 | dot | f32 | n=8388608 | 28.236 ms | 0.59 | 2.38 | 24% | ok |
| blas1 | axpy | f64 | n=8388608 | 224.022 ms | 0.07 | 0.90 | 9% | ok |
| blas1 | axpy | f32 | n=8388608 | 95.701 ms | 0.18 | 1.05 | 11% | ok |
| blas1 | scale | f64 | n=8388608 | 13.455 ms | 0.62 | 9.98 | 102% | ok |
| blas1 | two_norm | f64 | n=8388608 | 190.803 ms | 0.09 | 0.35 | 4% | ok |
| blas1 | one_norm | f64 | n=8388608 | 10.098 ms | 0.83 | 6.65 | 68% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 13.683 ms | 0.61 | 4.90 | 50% | ok |
| blas1 | sum | f64 | n=8388608 | 10.133 ms | 0.83 | 6.62 | 68% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 289.856 ms | 0.64 | 0.46 | 7% | ok |
| blas2 | gemv | f64 | n=4096 | 20.321 ms | 1.65 | 6.61 | 68% | ok |
| blas2 | gemv | f32 | n=4096 | 19.943 ms | 1.68 | 3.37 | 34% | ok |
| blas2 | symv | f64 | n=4096 | 16.725 ms | 2.01 | 4.02 | 41% | ok |
| blas2 | trmv | f64 | n=4096 | 10.726 ms | 1.56 | 6.26 | 64% | ok |
| blas2 | trsv | f64 | n=4096 | 10.480 ms | 1.60 | 6.41 | 66% | ok |
| blas2 | ger | f64 | n=4096 | 29.615 ms | 1.13 | 9.07 | 93% | ok |
| blas3 | gemm | f64 | 512x512x512 | 66.564 ms | 4.03 | 0.09 | 42% | ok |
| blas3 | gemm | f32 | 512x512x512 | 35.988 ms | 7.46 | 0.09 | 38% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 30.421 ms | 4.41 | 0.21 | 46% | ok |
| blas3 | symm | f64 | 512x512x512 | 62.056 ms | 4.33 | 0.10 | 45% | ok |
| blas3 | syrk | f64 | 512x512x512 | 39.830 ms | 3.37 | 0.11 | 35% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 68.388 ms | 3.93 | 0.09 | 41% | ok |
| blas3 | trmm | f64 | 512x512x512 | 35.637 ms | 3.77 | 0.15 | 39% | ok |
| blas3 | trsm | f64 | 512x512x512 | 33.551 ms | 4.00 | 0.16 | 42% | ok |
| blas3 | trsm | f32 | 512x512x512 | 16.855 ms | 7.96 | 0.16 | 41% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 39.116 ms | 0.14 | 1.09 | 11% | ok |
| factor | lu | f64 | 512x512 | 32.285 ms | 2.77 | 0.13 | 29% | ok |
| factor | lu | f32 | 512x512 | 17.280 ms | 5.18 | 0.12 | 27% | ok |
| factor | cholesky | f64 | 512x512 | 25.843 ms | 1.73 | 0.16 | 18% | ok |
| factor | cholesky | f32 | 512x512 | 24.279 ms | 1.84 | 0.09 | 9% | ok |
| factor | ldlt | f64 | 512x512 | 26.204 ms | 1.71 | 0.16 | 18% | ok |
| factor | ldlt_bk | f64 | 512x512 | 142.446 ms | 0.31 | 0.03 | 3% | ok |
| factor | qr | f64 | 512x512 | 531.062 ms | 0.34 | 0.01 | 4% | ok |
| factor | lq | f64 | 512x512 | 77.074 ms | 2.32 | 0.05 | 24% | ok |
| factor | hessenberg | f64 | 512x512 | 646.905 ms | 0.69 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.395 ms | 0.31 | 1.24 | 13%\* | ok |
| factor | inv | f64 | 512x512 | 200.828 ms | 1.34 | 0.02 | 14% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 23.386 ms | 3.83 | 0.18 | 40% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 62.970 ms | 0.36 | 0.02 | 4% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 183.661 ms | 0.82 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1027.840 ms | 0.15 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 110.626 ms | 1.52 | 0.01 | 16% | ok |
| eigen | schur | f64 | 256x256 | 281.036 ms | 1.49 | 0.00 | 16% | ok |
| eigen | svd | f64 | 256x256 | 325.255 ms | 1.13 | 0.00 | 12% | ok |
| eigen | expm | f64 | 256x256 | 95.632 ms | 2.63 | 0.01 | 27% | ok |
| eigen | sylvester | f64 | 256x256 | 642.453 ms | 1.57 | 0.00 | 16% | ok |
| elementwise | exp | f64 | n=8388608 | 156.889 ms | 0.05 | 0.86 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 144.485 ms | 0.06 | 0.93 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 175.782 ms | 0.05 | 0.76 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 92.492 ms | 0.09 | 1.45 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 2665.074 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.583 ms | 1.01 | 10.54 | 108% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.284 ms | 0.80 | 8.29 | 85% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 16.108 ms | 2.60 | 5.47 | 56% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.856 ms | 0.19 | 0.68 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 6.355 ms | 0.32 | 2.62 | 27% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 290.392 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 71.977 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 27.667 ms | 0.40 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 119.953 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 159.967 ms | 0.14 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 50.015 ms | 0.45 | 0.01 | 5% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 241.148 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.241 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.700 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.627 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.300 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 241.432 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 408.723 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 132.261 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 105.410 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 126.490 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 82.109 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 535.829 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 179.758 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 94.660 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 168.989 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 144.262 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 176.494 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 260.328 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.129 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.638 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 34.116 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.976 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.822 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.075 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.145 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.798 ms | 0.69 | 7.72 | 79% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.259 ms | 0.80 | 9.00 | 92% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.955 ms | 0.66 | 7.42 | 76% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 36.974 ms | 1.04 | 7.86 | 80% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 146.977 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 197.334 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 153.608 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 44.124 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 195.240 ms | 0.09 | 0.34 | 4% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 117.350 ms | 2.29 | 0.04 | 24% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.427 ms | 6.91 | 7.34 | 75% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.709 ms | 2.50 | 10.00 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.257 ms | 5.15 | 10.30 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.511 ms | 3.04 | 3.04 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 5.366 ms | 3.13 | 3.13 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 59.607 ms | 4.50 | 0.05 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 46.793 ms | 5.74 | 0.04 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 24.456 ms | 10.98 | 0.06 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 24.497 ms | 10.96 | 0.06 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 74.304 ms | 3.61 | 0.02 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz -- native, 12 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-42-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `73d11400a0e3` |
| FMA peak (ceiling) | f64 56.2 GFLOP/s, f32 117.7 GFLOP/s |
| ... hardware, SSE2 | f64 56.2, f32 117.7 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.1 GFLOP/s |
| bandwidth | read 11.3, update 10.6, triad 8.4 GB/s (ceiling 11.3) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 54% | 101% | 0 | 40% |
| blas2 | 6 | 82% | 107% | 0 | 9% |
| blas3 | 10 | 7% | 46% | 0 | 8% |
| factor | 12 | 3% | 37% | 0 | 9% |
| eigen | 8 | 2% | 10% | 0 | 9% |
| elementwise | 5 | 8% | 13% | 0 | 8% |
| sparse | 5 | 66% | 102% | 0 | 9% |
| direct | 7 | 0% | 1% | 0 | 8% |
| ordering | 4 | n/a | n/a | 0 | 8% |
| krylov | 13 | n/a | n/a | 0 | 8% |
| pc | 7 | n/a | n/a | 0 | 8% |
| smoother | 3 | 69% | 80% | 0 | 8% |
| mg | 1 | 69% | 69% | 0 | 8% |
| sparse_eig | 4 | n/a | n/a | 0 | 8% |
| mixed | 3 | 23% | 56% | 0 | 8% |
| integer | 9 | n/a | n/a | 0 | 43% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 29.704 ms | 0.56 | 4.52 | 40% | ok | 6.7x (56%) |
| blas1 | dot | f32 | n=8388608 | 5.843 ms | 2.87 | 11.48 | 101% | ok | 4.8x (40%) |
| blas1 | axpy | f64 | n=8388608 | 32.837 ms | 0.51 | 6.13 | 54% | ok | 6.8x (57%) |
| blas1 | axpy | f32 | n=8388608 | 11.495 ms | 1.46 | 8.76 | 77% | ok | 8.3x (69%) |
| blas1 | scale | f64 | n=8388608 | 12.484 ms | 0.67 | 10.75 | 95% | ok | 1.1x (9%) |
| blas1 | two_norm | f64 | n=8388608 | 28.924 ms | 0.58 | 2.32 | 20% | ok | 6.6x (55%) |
| blas1 | one_norm | f64 | n=8388608 | 11.744 ms | 0.71 | 5.71 | 50% | ok | 0.9x (7%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.548 ms | 0.62 | 4.95 | 44% | ok | 1.0x (8%) |
| blas1 | sum | f64 | n=8388608 | 10.156 ms | 0.83 | 6.61 | 58% | ok | 1.0x (8%) |
| blas1 | dot_k3 | f64 | n=8388608 | 293.663 ms | 0.63 | 0.46 | 4% | ok | 1.0x (8%) |
| blas2 | gemv | f64 | n=4096 | 11.115 ms | 3.02 | 12.08 | 107% | ok | 1.8x (15%) |
| blas2 | gemv | f32 | n=4096 | 5.740 ms | 5.85 | 11.70 | 103% | ok | 3.5x (29%) |
| blas2 | symv | f64 | n=4096 | 34.082 ms | 0.98 | 1.97 | 17% | ok | 0.5x (4%) |
| blas2 | trmv | f64 | n=4096 | 10.767 ms | 1.56 | 6.24 | 55% | ok | 1.0x (8%) |
| blas2 | trsv | f64 | n=4096 | 11.587 ms | 1.45 | 5.80 | 51% | ok | 0.9x (8%) |
| blas2 | ger | f64 | n=4096 | 28.784 ms | 1.17 | 9.33 | 82% | ok | 1.0x (9%) |
| blas3 | gemm | f64 | 512x512x512 | 10.443 ms | 25.70 | 0.60 | 46% | ok | 6.4x (53%) |
| blas3 | gemm | f32 | 512x512x512 | 6.027 ms | 44.54 | 0.52 | 38% | ok | 6.0x (50%) |
| blas3 | gemmt | f64 | 512x512x512 | 35.633 ms | 3.77 | 0.18 | 7% | ok | 0.9x (7%) |
| blas3 | symm | f64 | 512x512x512 | 62.577 ms | 4.29 | 0.10 | 8% | ok | 1.0x (8%) |
| blas3 | syrk | f64 | 512x512x512 | 39.642 ms | 3.39 | 0.11 | 6% | ok | 1.0x (8%) |
| blas3 | syr2k | f64 | 512x512x512 | 69.485 ms | 3.86 | 0.09 | 7% | ok | 1.0x (8%) |
| blas3 | trmm | f64 | 512x512x512 | 34.180 ms | 3.93 | 0.15 | 7% | ok | 1.0x (9%) |
| blas3 | trsm | f64 | 512x512x512 | 33.895 ms | 3.96 | 0.15 | 7% | ok | 1.0x (8%) |
| blas3 | trsm | f32 | 512x512x512 | 17.061 ms | 7.87 | 0.15 | 7% | ok | 1.0x (8%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.623 ms | 0.14 | 1.10 | 10% | ok | 1.0x (8%) |
| factor | lu | f64 | 512x512 | 26.055 ms | 3.43 | 0.16 | 6% | ok | 1.2x (10%) |
| factor | lu | f32 | 512x512 | 15.740 ms | 5.68 | 0.13 | 5% | ok | 1.1x (9%) |
| factor | cholesky | f64 | 512x512 | 55.426 ms | 0.81 | 0.08 | 1% | ok | 0.5x (4%) |
| factor | cholesky | f32 | 512x512 | 24.554 ms | 1.82 | 0.09 | 2% | ok | 1.0x (8%) |
| factor | ldlt | f64 | 512x512 | 26.380 ms | 1.70 | 0.16 | 3% | ok | 1.0x (8%) |
| factor | ldlt_bk | f64 | 512x512 | 140.140 ms | 0.32 | 0.03 | 1% | ok | 1.0x (8%) |
| factor | qr | f64 | 512x512 | 346.492 ms | 0.52 | 0.01 | 1% | ok | 1.5x (13%) |
| factor | lq | f64 | 512x512 | 54.204 ms | 3.30 | 0.08 | 6% | ok | 1.4x (12%) |
| factor | hessenberg | f64 | 512x512 | 467.108 ms | 0.96 | 0.01 | 2% | ok | 1.4x (12%) |
| factor | householder | f64 | 512x512 | 0.988 ms | 1.06 | 4.25 | 37%\* | ok | 3.4x (29%) |
| factor | inv | f64 | 512x512 | 194.016 ms | 1.38 | 0.02 | 2% | ok | 1.0x (9%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 23.095 ms | 3.87 | 0.18 | 7% | ok | 1.0x (8%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 63.615 ms | 0.35 | 0.02 | 1% | ok | 1.0x (8%) |
| eigen | eigen_symmetric | f64 | 256x256 | 177.025 ms | 0.85 | 0.01 | 2% | ok | 1.0x (9%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1047.826 ms | 0.14 | 0.00 | 0% | ok | 1.0x (8%) |
| eigen | eigenvalue | f64 | 256x256 | 186.422 ms | 0.90 | 0.01 | 2% | ok | 0.6x (5%) |
| eigen | schur | f64 | 256x256 | 277.157 ms | 1.51 | 0.00 | 3% | ok | 1.0x (8%) |
| eigen | svd | f64 | 256x256 | 313.352 ms | 1.18 | 0.00 | 2% | ok | 1.0x (9%) |
| eigen | expm | f64 | 256x256 | 42.970 ms | 5.86 | 0.02 | 10% | ok | 2.2x (19%) |
| eigen | sylvester | f64 | 256x256 | 594.847 ms | 1.69 | 0.00 | 3% | ok | 1.1x (9%) |
| elementwise | exp | f64 | n=8388608 | 157.376 ms | 0.05 | 0.85 | 8% | ok | 1.0x (8%) |
| elementwise | log | f64 | n=8388608 | 144.081 ms | 0.06 | 0.93 | 8% | ok | 1.0x (8%) |
| elementwise | sin | f64 | n=8388608 | 175.478 ms | 0.05 | 0.76 | 7% | ok | 1.0x (8%) |
| elementwise | sqrt | f64 | n=8388608 | 92.801 ms | 0.09 | 1.45 | 13% | ok | 1.0x (8%) |
| elementwise | erf | f64 | n=8388608 | 2552.635 ms | 0.00 | 0.05 | 0% | ok | 1.0x (9%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.367 ms | 1.11 | 11.51 | 102% | ok | 1.1x (9%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.296 ms | 0.79 | 8.26 | 73% | ok | 1.0x (8%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.853 ms | 3.53 | 7.43 | 66% | ok | 1.4x (11%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 18.726 ms | 0.17 | 0.61 | 5%\* | ok | 0.9x (8%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 6.037 ms | 0.33 | 2.76 | 24% | ok | 1.1x (9%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 283.581 ms | 0.04 | 0.00 | 0% | ok | 1.0x (9%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 71.766 ms | 0.16 | 0.01 | 0% | ok | 1.0x (8%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 58.662 ms | 0.19 | 0.01 | 0% | ok | 0.5x (4%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 119.379 ms | 0.19 | 0.01 | 0% | ok | 1.0x (8%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 160.539 ms | 0.14 | 0.00 | 0% | ok | 1.0x (8%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 55.346 ms | 0.40 | 0.01 | 1% | ok | 0.9x (8%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 238.642 ms | 0.01 | 0.00 | 0% | ok | 1.0x (8%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.325 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.526 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.626 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 41.503 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 238.580 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 407.430 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 130.672 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 106.152 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 134.780 ms | 0.00 | 0.00 | n/a | ok | 0.9x (8%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 81.901 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 536.164 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 179.608 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 93.759 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 140.135 ms | 0.00 | 0.00 | n/a | ok | 1.2x (10%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 142.570 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 175.695 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 257.813 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.132 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.576 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | ilut | f64 | 128^2 grid | 34.005 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | ic_0 | f64 | 128^2 grid | 2.992 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | ildl | f64 | 128^2 grid | 2.823 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | ssor | f64 | 128^2 grid | 0.078 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.264 ms | 0.00 | 0.00 | n/a | ok | 0.9x (8%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.736 ms | 0.70 | 7.85 | 69% | ok | 1.0x (8%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.240 ms | 0.81 | 9.05 | 80% | ok | 1.0x (8%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.075 ms | 0.64 | 7.20 | 64% | ok | 1.0x (8%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 36.987 ms | 1.04 | 7.86 | 69% | ok | 1.0x (8%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 149.920 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 197.737 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 152.407 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 44.059 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| mixed | dot | f32>f64 | n=8388608 | 195.020 ms | 0.09 | 0.34 | 3% | ok | 1.0x (8%) |
| mixed | gemm | f32>f64 | 512x512x512 | 20.586 ms | 13.04 | 0.20 | 23% | ok | 5.7x (48%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.821 ms | 5.95 | 6.32 | 56% | ok | 0.9x (7%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.725 ms | 2.49 | 9.98 | n/a | ok | 1.0x (8%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.100 ms | 5.41 | 10.82 | n/a | ok | 1.1x (9%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.208 ms | 3.22 | 3.22 | n/a | ok | 1.1x (9%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 5.202 ms | 3.22 | 3.22 | n/a | ok | 1.0x (9%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 9.831 ms | 27.31 | 0.32 | n/a | ok | 6.1x (51%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 9.129 ms | 29.40 | 0.23 | n/a | ok | 5.1x (43%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.506 ms | 59.57 | 0.35 | n/a | ok | 5.4x (45%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.229 ms | 63.47 | 0.37 | n/a | ok | 5.8x (48%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.913 ms | 19.29 | 0.11 | n/a | ok | 5.3x (45%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-42-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `73d11400a0e3` |
| FMA peak (ceiling) | f64 9.9 GFLOP/s, f32 19.5 GFLOP/s |
| ... hardware, SSE2 | f64 8.0, f32 14.8 GFLOP/s |
| ... this build, SSE4 | f64 9.9, f32 19.5 GFLOP/s |
| bandwidth | read 10.0, update 10.0, triad 8.0 GB/s (ceiling 10.0) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 102% | 110% | 0 |
| blas2 | 6 | 94% | 101% | 0 |
| blas3 | 10 | 74% | 98% | 0 |
| factor | 12 | 17% | 39% | 0 |
| eigen | 8 | 16% | 44% | 0 |
| elementwise | 5 | 9% | 14% | 0 |
| sparse | 5 | 56% | 104% | 0 |
| direct | 7 | 2% | 4% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 78% | 90% | 0 |
| mg | 1 | 78% | 78% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 74% | 89% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 17.391 ms | 0.96 | 7.72 | 77% | ok |
| blas1 | dot | f32 | n=8388608 | 6.172 ms | 2.72 | 10.87 | 108% | ok |
| blas1 | axpy | f64 | n=8388608 | 19.382 ms | 0.87 | 10.39 | 103% | ok |
| blas1 | axpy | f32 | n=8388608 | 9.143 ms | 1.83 | 11.01 | 110% | ok |
| blas1 | scale | f64 | n=8388608 | 13.137 ms | 0.64 | 10.22 | 102% | ok |
| blas1 | two_norm | f64 | n=8388608 | 6.156 ms | 2.73 | 10.90 | 109% | ok |
| blas1 | one_norm | f64 | n=8388608 | 10.097 ms | 0.83 | 6.65 | 66% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 13.581 ms | 0.62 | 4.94 | 49% | ok |
| blas1 | sum | f64 | n=8388608 | 10.074 ms | 0.83 | 6.66 | 66% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 279.873 ms | 0.66 | 0.48 | 7% | ok |
| blas2 | gemv | f64 | n=4096 | 13.337 ms | 2.52 | 10.07 | 100% | ok |
| blas2 | gemv | f32 | n=4096 | 6.590 ms | 5.09 | 10.19 | 101% | ok |
| blas2 | symv | f64 | n=4096 | 16.680 ms | 2.01 | 4.03 | 40% | ok |
| blas2 | trmv | f64 | n=4096 | 7.101 ms | 2.36 | 9.46 | 94% | ok |
| blas2 | trsv | f64 | n=4096 | 7.091 ms | 2.37 | 9.47 | 94% | ok |
| blas2 | ger | f64 | n=4096 | 28.529 ms | 1.18 | 9.41 | 94% | ok |
| blas3 | gemm | f64 | 512x512x512 | 28.668 ms | 9.36 | 0.22 | 94% | ok |
| blas3 | gemm | f32 | 512x512x512 | 14.044 ms | 19.11 | 0.22 | 98% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 17.554 ms | 7.65 | 0.36 | 77% | ok |
| blas3 | symm | f64 | 512x512x512 | 29.721 ms | 9.03 | 0.21 | 91% | ok |
| blas3 | syrk | f64 | 512x512x512 | 82.373 ms | 1.63 | 0.05 | 16% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 66.080 ms | 4.06 | 0.10 | 41% | ok |
| blas3 | trmm | f64 | 512x512x512 | 18.295 ms | 7.34 | 0.29 | 74% | ok |
| blas3 | trsm | f64 | 512x512x512 | 19.126 ms | 7.02 | 0.27 | 71% | ok |
| blas3 | trsm | f32 | 512x512x512 | 9.501 ms | 14.13 | 0.28 | 72% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 40.070 ms | 0.13 | 1.06 | 11% | ok |
| factor | lu | f64 | 512x512 | 31.290 ms | 2.86 | 0.13 | 29% | ok |
| factor | lu | f32 | 512x512 | 16.977 ms | 5.27 | 0.12 | 27% | ok |
| factor | cholesky | f64 | 512x512 | 27.160 ms | 1.65 | 0.15 | 17% | ok |
| factor | cholesky | f32 | 512x512 | 25.112 ms | 1.78 | 0.08 | 9% | ok |
| factor | ldlt | f64 | 512x512 | 26.609 ms | 1.68 | 0.16 | 17% | ok |
| factor | ldlt_bk | f64 | 512x512 | 143.159 ms | 0.31 | 0.03 | 3% | ok |
| factor | qr | f64 | 512x512 | 524.296 ms | 0.34 | 0.01 | 3% | ok |
| factor | lq | f64 | 512x512 | 76.016 ms | 2.35 | 0.06 | 24% | ok |
| factor | hessenberg | f64 | 512x512 | 637.825 ms | 0.70 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.365 ms | 0.31 | 1.25 | 12%\* | ok |
| factor | inv | f64 | 512x512 | 200.749 ms | 1.34 | 0.02 | 13% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 23.019 ms | 3.89 | 0.18 | 39% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 63.668 ms | 0.35 | 0.02 | 4% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 179.221 ms | 0.84 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1037.015 ms | 0.15 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 107.641 ms | 1.56 | 0.01 | 16% | ok |
| eigen | schur | f64 | 256x256 | 267.883 ms | 1.57 | 0.00 | 16% | ok |
| eigen | svd | f64 | 256x256 | 269.510 ms | 1.37 | 0.00 | 14% | ok |
| eigen | expm | f64 | 256x256 | 57.453 ms | 4.38 | 0.02 | 44% | ok |
| eigen | sylvester | f64 | 256x256 | 593.183 ms | 1.70 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 156.544 ms | 0.05 | 0.86 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 144.499 ms | 0.06 | 0.93 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 176.626 ms | 0.05 | 0.76 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 92.843 ms | 0.09 | 1.45 | 14% | ok |
| elementwise | erf | f64 | n=8388608 | 2799.777 ms | 0.00 | 0.05 | 0% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.597 ms | 1.01 | 10.49 | 104% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 4.281 ms | 0.61 | 6.36 | 63% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.696 ms | 2.67 | 5.61 | 56% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 32.682 ms | 0.10 | 0.35 | 4%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 6.036 ms | 0.33 | 2.76 | 27% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 246.604 ms | 0.05 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 72.679 ms | 0.15 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 27.540 ms | 0.41 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 120.700 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 160.899 ms | 0.14 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 51.642 ms | 0.43 | 0.01 | 4% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 232.798 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.443 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.741 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.657 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 41.641 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 82.764 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 114.845 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 66.817 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 33.552 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 39.374 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 38.276 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 67.700 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 76.907 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 43.064 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 45.014 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 43.152 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 46.333 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 62.590 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.137 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.728 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 34.412 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.016 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.886 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.077 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.245 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.722 ms | 0.70 | 7.88 | 78% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.252 ms | 0.80 | 9.02 | 90% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.977 ms | 0.66 | 7.37 | 73% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 37.260 ms | 1.03 | 7.80 | 78% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 146.073 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 198.300 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 162.211 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 16.832 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 7.535 ms | 2.23 | 8.91 | 89% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 64.258 ms | 4.18 | 0.07 | 42% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.400 ms | 6.99 | 7.43 | 74% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.190 ms | 2.71 | 10.84 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.101 ms | 5.41 | 10.82 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.175 ms | 7.71 | 7.71 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.735 ms | 9.67 | 9.67 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 19.562 ms | 13.72 | 0.16 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 22.994 ms | 11.67 | 0.09 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 22.744 ms | 11.80 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 16.758 ms | 16.02 | 0.09 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 17.379 ms | 15.45 | 0.09 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz -- native-fast, 12 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-42-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `73d11400a0e3` |
| FMA peak (ceiling) | f64 58.9 GFLOP/s, f32 117.9 GFLOP/s |
| ... hardware, SSE2 | f64 58.9, f32 117.9 GFLOP/s |
| ... this build, SSE4 | f64 58.9, f32 117.6 GFLOP/s |
| bandwidth | read 11.3, update 10.5, triad 8.4 GB/s (ceiling 11.3) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 98% | 106% | 0 | 9% |
| blas2 | 6 | 84% | 107% | 0 | 8% |
| blas3 | 10 | 25% | 75% | 0 | 17% |
| factor | 12 | 3% | 14% | 0 | 9% |
| eigen | 8 | 3% | 11% | 0 | 8% |
| elementwise | 5 | 7% | 13% | 0 | 8% |
| sparse | 5 | 56% | 102% | 0 | 9% |
| direct | 7 | 0% | 1% | 0 | 8% |
| ordering | 4 | n/a | n/a | 0 | 8% |
| krylov | 13 | n/a | n/a | 0 | 8% |
| pc | 7 | n/a | n/a | 0 | 8% |
| smoother | 3 | 71% | 81% | 0 | 9% |
| mg | 1 | 70% | 70% | 0 | 9% |
| sparse_eig | 4 | n/a | n/a | 0 | 9% |
| mixed | 3 | 45% | 79% | 0 | 8% |
| integer | 9 | n/a | n/a | 0 | 41% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 11.165 ms | 1.50 | 12.02 | 106% | ok | 1.6x (13%) |
| blas1 | dot | f32 | n=8388608 | 5.580 ms | 3.01 | 12.03 | 106% | ok | 1.1x (9%) |
| blas1 | axpy | f64 | n=8388608 | 18.153 ms | 0.92 | 11.09 | 98% | ok | 1.1x (9%) |
| blas1 | axpy | f32 | n=8388608 | 8.691 ms | 1.93 | 11.58 | 103% | ok | 1.1x (9%) |
| blas1 | scale | f64 | n=8388608 | 12.399 ms | 0.68 | 10.83 | 96% | ok | 1.1x (9%) |
| blas1 | two_norm | f64 | n=8388608 | 5.582 ms | 3.01 | 12.02 | 106% | ok | 1.1x (9%) |
| blas1 | one_norm | f64 | n=8388608 | 21.395 ms | 0.39 | 3.14 | 28% | ok | 0.5x (4%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.463 ms | 0.62 | 4.98 | 44% | ok | 1.0x (8%) |
| blas1 | sum | f64 | n=8388608 | 10.048 ms | 0.83 | 6.68 | 59% | ok | 1.0x (8%) |
| blas1 | dot_k3 | f64 | n=8388608 | 280.320 ms | 0.66 | 0.48 | 4% | ok | 1.0x (8%) |
| blas2 | gemv | f64 | n=4096 | 11.159 ms | 3.01 | 12.03 | 107% | ok | 1.2x (10%) |
| blas2 | gemv | f32 | n=4096 | 5.611 ms | 5.98 | 11.97 | 106% | ok | 1.2x (10%) |
| blas2 | symv | f64 | n=4096 | 16.717 ms | 2.01 | 4.02 | 36% | ok | 1.0x (8%) |
| blas2 | trmv | f64 | n=4096 | 7.089 ms | 2.37 | 9.48 | 84% | ok | 1.0x (8%) |
| blas2 | trsv | f64 | n=4096 | 7.136 ms | 2.35 | 9.41 | 83% | ok | 1.0x (8%) |
| blas2 | ger | f64 | n=4096 | 34.334 ms | 0.98 | 7.82 | 69% | ok | 0.8x (7%) |
| blas3 | gemm | f64 | 512x512x512 | 6.438 ms | 41.70 | 0.98 | 71% | ok | 4.5x (37%) |
| blas3 | gemm | f32 | 512x512x512 | 3.049 ms | 88.04 | 1.03 | 75% | ok | 4.6x (38%) |
| blas3 | gemmt | f64 | 512x512x512 | 6.806 ms | 19.72 | 0.92 | 33% | ok | 2.6x (21%) |
| blas3 | symm | f64 | 512x512x512 | 7.594 ms | 35.35 | 0.83 | 60% | ok | 3.9x (33%) |
| blas3 | syrk | f64 | 512x512x512 | 82.434 ms | 1.63 | 0.05 | 3% | ok | 1.0x (8%) |
| blas3 | syr2k | f64 | 512x512x512 | 67.088 ms | 4.00 | 0.09 | 7% | ok | 1.0x (8%) |
| blas3 | trmm | f64 | 512x512x512 | 8.963 ms | 14.97 | 0.58 | 25% | ok | 2.0x (17%) |
| blas3 | trsm | f64 | 512x512x512 | 9.364 ms | 14.33 | 0.56 | 24% | ok | 2.0x (17%) |
| blas3 | trsm | f32 | 512x512x512 | 5.049 ms | 26.58 | 0.52 | 23% | ok | 1.9x (16%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.710 ms | 0.14 | 1.10 | 10% | ok | 1.0x (9%) |
| factor | lu | f64 | 512x512 | 25.881 ms | 3.46 | 0.16 | 6% | ok | 1.2x (10%) |
| factor | lu | f32 | 512x512 | 15.390 ms | 5.81 | 0.14 | 5% | ok | 1.1x (9%) |
| factor | cholesky | f64 | 512x512 | 50.825 ms | 0.88 | 0.08 | 1% | ok | 0.5x (4%) |
| factor | cholesky | f32 | 512x512 | 24.991 ms | 1.79 | 0.08 | 2% | ok | 1.0x (8%) |
| factor | ldlt | f64 | 512x512 | 26.832 ms | 1.67 | 0.16 | 3% | ok | 1.0x (8%) |
| factor | ldlt_bk | f64 | 512x512 | 139.721 ms | 0.32 | 0.03 | 1% | ok | 1.0x (9%) |
| factor | qr | f64 | 512x512 | 397.471 ms | 0.45 | 0.01 | 1% | ok | 1.3x (11%) |
| factor | lq | f64 | 512x512 | 54.070 ms | 3.31 | 0.08 | 6% | ok | 1.4x (12%) |
| factor | hessenberg | f64 | 512x512 | 500.747 ms | 0.89 | 0.01 | 2% | ok | 1.3x (11%) |
| factor | householder | f64 | 512x512 | 2.636 ms | 0.40 | 1.59 | 14%\* | ok | 1.3x (11%) |
| factor | inv | f64 | 512x512 | 199.453 ms | 1.35 | 0.02 | 2% | ok | 1.0x (8%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 21.839 ms | 4.10 | 0.19 | 7% | ok | 1.1x (9%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 62.756 ms | 0.36 | 0.02 | 1% | ok | 1.0x (8%) |
| eigen | eigen_symmetric | f64 | 256x256 | 178.976 ms | 0.84 | 0.01 | 1% | ok | 1.0x (8%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1020.022 ms | 0.15 | 0.00 | 0% | ok | 1.0x (8%) |
| eigen | eigenvalue | f64 | 256x256 | 107.862 ms | 1.56 | 0.01 | 3% | ok | 1.0x (8%) |
| eigen | schur | f64 | 256x256 | 269.317 ms | 1.56 | 0.00 | 3% | ok | 1.0x (8%) |
| eigen | svd | f64 | 256x256 | 268.422 ms | 1.38 | 0.00 | 2% | ok | 1.0x (8%) |
| eigen | expm | f64 | 256x256 | 38.904 ms | 6.47 | 0.03 | 11% | ok | 1.5x (12%) |
| eigen | sylvester | f64 | 256x256 | 590.308 ms | 1.71 | 0.00 | 3% | ok | 1.0x (8%) |
| elementwise | exp | f64 | n=8388608 | 170.874 ms | 0.05 | 0.79 | 7% | ok | 0.9x (8%) |
| elementwise | log | f64 | n=8388608 | 144.267 ms | 0.06 | 0.93 | 8% | ok | 1.0x (8%) |
| elementwise | sin | f64 | n=8388608 | 176.593 ms | 0.05 | 0.76 | 7% | ok | 1.0x (8%) |
| elementwise | sqrt | f64 | n=8388608 | 93.034 ms | 0.09 | 1.44 | 13% | ok | 1.0x (8%) |
| elementwise | erf | f64 | n=8388608 | 2553.402 ms | 0.00 | 0.05 | 0% | ok | 1.1x (9%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.368 ms | 1.11 | 11.50 | 102% | ok | 1.1x (9%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 4.306 ms | 0.61 | 6.32 | 56% | ok | 1.0x (8%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.754 ms | 3.56 | 7.49 | 66% | ok | 1.3x (11%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 19.547 ms | 0.17 | 0.59 | 5%\* | ok | 1.7x (14%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 6.083 ms | 0.33 | 2.74 | 24% | ok | 1.0x (8%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 245.467 ms | 0.05 | 0.00 | 0% | ok | 1.0x (8%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 72.722 ms | 0.15 | 0.01 | 0% | ok | 1.0x (8%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 27.621 ms | 0.41 | 0.03 | 1% | ok | 1.0x (8%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 120.587 ms | 0.19 | 0.01 | 0% | ok | 1.0x (8%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 161.130 ms | 0.14 | 0.00 | 0% | ok | 1.0x (8%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 50.815 ms | 0.44 | 0.01 | 1% | ok | 1.0x (8%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 236.222 ms | 0.01 | 0.00 | 0% | ok | 1.0x (8%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.413 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 27.104 ms | 0.00 | 0.00 | n/a | ok | 0.5x (4%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.644 ms | 0.00 | 0.00 | n/a | ok | 1.0x (9%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 41.798 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 83.190 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 117.464 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 66.618 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 33.176 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 39.129 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 38.379 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 68.705 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 77.051 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 39.212 ms | 0.00 | 0.00 | n/a | ok | 1.1x (9%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 45.162 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 43.389 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 46.756 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 56.278 ms | 0.00 | 0.00 | n/a | ok | 1.1x (9%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.136 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.725 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | ilut | f64 | 128^2 grid | 35.056 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | ic_0 | f64 | 128^2 grid | 2.996 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | ildl | f64 | 128^2 grid | 2.849 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | ssor | f64 | 128^2 grid | 0.077 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.260 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.632 ms | 0.72 | 8.08 | 71% | ok | 1.0x (9%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.185 ms | 0.82 | 9.21 | 81% | ok | 1.0x (9%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.185 ms | 0.63 | 7.01 | 62% | ok | 1.0x (8%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 36.516 ms | 1.05 | 7.96 | 70% | ok | 1.0x (9%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 144.512 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 185.478 ms | 0.00 | 0.00 | n/a | ok | 1.1x (9%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 144.212 ms | 0.00 | 0.00 | n/a | ok | 1.1x (9%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 16.669 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| mixed | dot | f32>f64 | n=8388608 | 7.512 ms | 2.23 | 8.93 | 79% | ok | 1.0x (8%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.251 ms | 21.91 | 0.34 | 37% | ok | 5.2x (44%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 3.537 ms | 4.74 | 5.04 | 45% | ok | 0.7x (6%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.208 ms | 2.70 | 10.81 | n/a | ok | 1.0x (8%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.091 ms | 5.43 | 10.85 | n/a | ok | 1.0x (8%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.168 ms | 7.74 | 7.74 | n/a | ok | 1.0x (8%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.720 ms | 9.76 | 9.76 | n/a | ok | 1.0x (8%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 3.922 ms | 68.45 | 0.80 | n/a | ok | 5.0x (42%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.450 ms | 60.33 | 0.47 | n/a | ok | 5.2x (43%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.377 ms | 61.32 | 0.36 | n/a | ok | 5.2x (43%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.417 ms | 78.55 | 0.46 | n/a | ok | 4.9x (41%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 2.965 ms | 90.54 | 0.53 | n/a | ok | 5.9x (49%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

