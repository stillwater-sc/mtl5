# mtl5_bench platform characterization

Class `medium`, variants native native-fast, threads 1,4, CPUs 0,1,2,3, clock policy governor performance on the A78C policies; performance 902-2112 MHz on cpus 0,1; performance 940-2361 MHz on cpus 2,3, 2026-11-11T18:57:21Z.

Not measured:
- blas: no BLAS/LAPACK found

## arduino-ventuno-q -- native, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.6 GFLOP/s |
| ... this build, scalar | f64 8.4, f32 8.4 GFLOP/s |
| bandwidth | read 22.8, update 24.8, triad 23.5 GB/s (ceiling 24.8) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 86% | 104% | 0 |
| blas2 | 10 | 53% | 70% | 0 |
| blas3 | 10 | 31% | 33% | 0 |
| factor | 12 | 12% | 25% | 0 |
| eigen | 8 | 15% | 22% | 0 |
| elementwise | 5 | 5% | 7% | 0 |
| sparse | 5 | 29% | 67% | 0 |
| direct | 7 | 1% | 3% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 57% | 0 |
| mg | 1 | 36% | 36% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 34% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.645 ms | 2.97 | 23.78 | 96% | ok |
| blas1 | dot | f32 | n=8388608 | 4.301 ms | 3.90 | 15.60 | 63% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.825 ms | 2.14 | 25.73 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.959 ms | 4.24 | 25.42 | 102% | ok |
| blas1 | scale | f64 | n=8388608 | 5.317 ms | 1.58 | 25.24 | 102% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.156 ms | 5.32 | 21.26 | 86% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.303 ms | 1.01 | 8.08 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.389 ms | 0.48 | 3.86 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.437 ms | 0.99 | 7.95 | 32% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 197.318 ms | 0.94 | 0.68 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 9.490 ms | 3.54 | 14.15 | 57% | ok |
| blas2 | gemv | f32 | n=4096 | 7.043 ms | 4.76 | 9.53 | 38% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 9.812 ms | 3.42 | 13.69 | 55% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.578 ms | 7.33 | 14.67 | 59% | ok |
| blas2 | symv | f64 | n=4096 | 17.375 ms | 1.93 | 3.87 | 16% | ok |
| blas2 | trmv | f64 | n=4096 | 5.128 ms | 3.27 | 13.10 | 53% | ok |
| blas2 | trmv | f32 | n=4096 | 6.863 ms | 2.44 | 4.89 | 20% | ok |
| blas2 | trsv | f64 | n=4096 | 5.160 ms | 3.25 | 13.02 | 52% | ok |
| blas2 | trsv | f32 | n=4096 | 20.807 ms | 0.81 | 1.61 | 7% | ok |
| blas2 | ger | f64 | n=4096 | 15.354 ms | 2.19 | 17.49 | 70% | ok |
| blas3 | gemm | f64 | 512x512x512 | 51.977 ms | 5.16 | 0.12 | 31% | ok |
| blas3 | gemm | f32 | 512x512x512 | 24.970 ms | 10.75 | 0.13 | 32% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 38.900 ms | 3.45 | 0.16 | 21% | ok |
| blas3 | symm | f64 | 512x512x512 | 53.379 ms | 5.03 | 0.12 | 30% | ok |
| blas3 | syrk | f64 | 512x512x512 | 40.900 ms | 3.28 | 0.10 | 20% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 79.004 ms | 3.40 | 0.08 | 20% | ok |
| blas3 | trmm | f64 | 512x512x512 | 25.960 ms | 5.17 | 0.20 | 31% | ok |
| blas3 | trsm | f64 | 512x512x512 | 26.104 ms | 5.14 | 0.20 | 31% | ok |
| blas3 | trsm | f32 | 512x512x512 | 12.129 ms | 11.07 | 0.22 | 33% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 30.489 ms | 0.17 | 1.39 | 6% | ok |
| factor | lu | f64 | 512x512 | 25.441 ms | 3.52 | 0.16 | 21% | ok |
| factor | lu | f32 | 512x512 | 13.191 ms | 6.78 | 0.16 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 22.212 ms | 2.01 | 0.19 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 19.621 ms | 2.28 | 0.11 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 21.430 ms | 2.09 | 0.20 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 68.692 ms | 0.65 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 249.357 ms | 0.72 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 60.909 ms | 2.94 | 0.07 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 347.275 ms | 1.29 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.542 ms | 0.68 | 2.72 | 11% | ok |
| factor | inv | f64 | 512x512 | 147.412 ms | 1.82 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 21.071 ms | 4.25 | 0.20 | 25% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.815 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.044 ms | 1.56 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 591.530 ms | 0.26 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.208 ms | 2.50 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 150.393 ms | 2.79 | 0.01 | 17% | ok |
| eigen | svd | f64 | 256x256 | 230.525 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 67.537 ms | 3.73 | 0.02 | 22% | ok |
| eigen | sylvester | f64 | 256x256 | 370.132 ms | 2.72 | 0.00 | 16% | ok |
| elementwise | exp | f64 | n=8388608 | 87.629 ms | 0.10 | 1.53 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 111.960 ms | 0.07 | 1.20 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 148.325 ms | 0.06 | 0.90 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 72.164 ms | 0.12 | 1.86 | 7% | ok |
| elementwise | erf | f64 | n=8388608 | 117.684 ms | 0.07 | 1.14 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.640 ms | 1.60 | 16.61 | 67% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.157 ms | 1.21 | 12.63 | 51% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.101 ms | 3.46 | 7.28 | 29% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.314 ms | 0.24 | 0.86 | 3% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.783 ms | 0.23 | 1.90 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 181.733 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 59.014 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.603 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 103.876 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 140.222 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.142 ms | 0.52 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 200.762 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.602 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.216 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.696 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.376 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.548 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 104.223 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 53.959 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 28.889 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.305 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 33.932 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 63.901 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.711 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.359 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 37.477 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 36.213 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.707 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 47.575 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.113 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.833 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 30.575 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.053 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.917 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.064 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.743 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.077 ms | 1.26 | 14.12 | 57% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.482 ms | 1.05 | 11.82 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 5.024 ms | 0.52 | 5.84 | 24% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.454 ms | 1.18 | 8.96 | 36% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 231.574 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 270.304 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 161.327 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.405 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 8.066 ms | 2.08 | 8.32 | 34% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 48.644 ms | 5.52 | 0.09 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.939 ms | 5.71 | 6.06 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.853 ms | 5.88 | 23.52 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.384 ms | 12.12 | 24.25 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.649 ms | 4.60 | 4.60 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.181 ms | 14.20 | 14.20 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 24.668 ms | 10.88 | 0.13 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 26.359 ms | 10.18 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 20.297 ms | 13.23 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 20.294 ms | 13.23 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 38.432 ms | 6.98 | 0.04 | n/a | ok |

## arduino-ventuno-q -- native, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 67.7 GFLOP/s, f32 135.6 GFLOP/s |
| ... hardware, NEON | f64 67.7, f32 135.6 GFLOP/s |
| ... this build, scalar | f64 26.9, f32 33.7 GFLOP/s |
| bandwidth | read 22.8, update 24.6, triad 22.6 GB/s (ceiling 24.6) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 91% | 107% | 0 | 28% |
| blas2 | 10 | 54% | 73% | 0 | 28% |
| blas3 | 10 | 9% | 16% | 0 | 29% |
| factor | 12 | 2% | 16% | 0 | 28% |
| eigen | 8 | 4% | 9% | 0 | 28% |
| elementwise | 5 | 5% | 8% | 0 | 27% |
| sparse | 5 | 55% | 72% | 0 | 27% |
| direct | 7 | 0% | 1% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 28% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 28% |
| smoother | 3 | 53% | 62% | 0 | 27% |
| mg | 1 | 33% | 33% | 0 | 23% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 30% | 38% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 51% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.985 ms | 2.80 | 22.42 | 91% | ok | 0.9x (24%) |
| blas1 | dot | f32 | n=8388608 | 3.022 ms | 5.55 | 22.21 | 90% | ok | 1.4x (36%) |
| blas1 | axpy | f64 | n=8388608 | 8.305 ms | 2.02 | 24.24 | 99% | ok | 0.9x (24%) |
| blas1 | axpy | f32 | n=8388608 | 4.322 ms | 3.88 | 23.29 | 95% | ok | 0.9x (23%) |
| blas1 | scale | f64 | n=8388608 | 5.094 ms | 1.65 | 26.35 | 107% | ok | 1.0x (26%) |
| blas1 | two_norm | f64 | n=8388608 | 2.868 ms | 5.85 | 23.40 | 95% | ok | 1.1x (28%) |
| blas1 | one_norm | f64 | n=8388608 | 7.299 ms | 1.15 | 9.19 | 37% | ok | 1.1x (28%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.514 ms | 0.54 | 4.33 | 18% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.870 ms | 1.07 | 8.53 | 35% | ok | 1.1x (27%) |
| blas1 | dot_k3 | f64 | n=8388608 | 177.417 ms | 1.04 | 0.76 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 7.803 ms | 4.30 | 17.21 | 70% | ok | 1.2x (30%) |
| blas2 | gemv | f32 | n=4096 | 4.002 ms | 8.39 | 16.78 | 68% | ok | 1.8x (44%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 8.170 ms | 4.11 | 16.44 | 67% | ok | 1.2x (30%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.143 ms | 6.52 | 13.05 | 53% | ok | 0.9x (22%) |
| blas2 | symv | f64 | n=4096 | 16.181 ms | 2.07 | 4.15 | 17% | ok | 1.1x (27%) |
| blas2 | trmv | f64 | n=4096 | 5.076 ms | 3.31 | 13.23 | 54% | ok | 1.0x (25%) |
| blas2 | trmv | f32 | n=4096 | 6.115 ms | 2.74 | 5.49 | 22% | ok | 1.1x (28%) |
| blas2 | trsv | f64 | n=4096 | 5.052 ms | 3.32 | 13.30 | 54% | ok | 1.0x (26%) |
| blas2 | trsv | f32 | n=4096 | 18.653 ms | 0.90 | 1.80 | 7% | ok | 1.1x (28%) |
| blas2 | ger | f64 | n=4096 | 14.996 ms | 2.24 | 17.90 | 73% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 27.283 ms | 9.84 | 0.23 | 15% | ok | 1.9x (48%) |
| blas3 | gemm | f32 | 512x512x512 | 12.039 ms | 22.30 | 0.26 | 16% | ok | 2.1x (52%) |
| blas3 | gemmt | f64 | 512x512x512 | 36.081 ms | 3.72 | 0.17 | 5% | ok | 1.1x (27%) |
| blas3 | symm | f64 | 512x512x512 | 51.602 ms | 5.20 | 0.12 | 8% | ok | 1.0x (26%) |
| blas3 | syrk | f64 | 512x512x512 | 37.254 ms | 3.60 | 0.11 | 5% | ok | 1.1x (27%) |
| blas3 | syr2k | f64 | 512x512x512 | 75.659 ms | 3.55 | 0.08 | 5% | ok | 1.0x (26%) |
| blas3 | trmm | f64 | 512x512x512 | 22.329 ms | 6.01 | 0.23 | 9% | ok | 1.2x (29%) |
| blas3 | trsm | f64 | 512x512x512 | 22.892 ms | 5.86 | 0.23 | 9% | ok | 1.1x (29%) |
| blas3 | trsm | f32 | 512x512x512 | 10.829 ms | 12.39 | 0.24 | 9% | ok | 1.1x (28%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 25.407 ms | 0.21 | 1.67 | 7% | ok | 1.2x (30%) |
| factor | lu | f64 | 512x512 | 110.776 ms | 0.81 | 0.04 | 1% | ok | 0.2x (6%) |
| factor | lu | f32 | 512x512 | 90.738 ms | 0.99 | 0.02 | 1% | ok | 0.1x (4%) |
| factor | cholesky | f64 | 512x512 | 19.931 ms | 2.24 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 17.579 ms | 2.55 | 0.12 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 19.155 ms | 2.34 | 0.22 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 60.709 ms | 0.74 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 198.339 ms | 0.90 | 0.02 | 1% | ok | 1.3x (31%) |
| factor | lq | f64 | 512x512 | 129.678 ms | 1.38 | 0.03 | 2% | ok | 0.5x (12%) |
| factor | hessenberg | f64 | 512x512 | 438.860 ms | 1.02 | 0.01 | 2% | ok | 0.8x (20%) |
| factor | householder | f64 | 512x512 | 1.095 ms | 0.96 | 3.83 | 16% | ok | 1.4x (35%) |
| factor | inv | f64 | 512x512 | 223.001 ms | 1.20 | 0.02 | 2% | ok | 0.7x (17%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 104.699 ms | 0.85 | 0.04 | 1% | ok | 0.2x (5%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.514 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 87.746 ms | 1.72 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 533.152 ms | 0.28 | 0.00 | 0% | ok | 1.1x (28%) |
| eigen | eigenvalue | f64 | 256x256 | 60.552 ms | 2.77 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 134.448 ms | 3.12 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.690 ms | 1.79 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 42.363 ms | 5.94 | 0.02 | 9% | ok | 1.6x (40%) |
| eigen | sylvester | f64 | 256x256 | 319.788 ms | 3.15 | 0.00 | 5% | ok | 1.2x (29%) |
| elementwise | exp | f64 | n=8388608 | 80.467 ms | 0.10 | 1.67 | 7% | ok | 1.1x (27%) |
| elementwise | log | f64 | n=8388608 | 99.791 ms | 0.08 | 1.34 | 5% | ok | 1.1x (28%) |
| elementwise | sin | f64 | n=8388608 | 135.635 ms | 0.06 | 0.99 | 4% | ok | 1.1x (27%) |
| elementwise | sqrt | f64 | n=8388608 | 66.721 ms | 0.13 | 2.01 | 8% | ok | 1.1x (27%) |
| elementwise | erf | f64 | n=8388608 | 106.810 ms | 0.08 | 1.26 | 5% | ok | 1.1x (28%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.535 ms | 1.70 | 17.73 | 72% | ok | 1.1x (27%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.970 ms | 1.33 | 13.82 | 56% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.486 ms | 6.46 | 13.57 | 55% | ok | 1.9x (47%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.783 ms | 0.28 | 0.98 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 10.136 ms | 0.20 | 1.64 | 7% | ok | 0.9x (22%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 163.032 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 53.227 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.190 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 92.385 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 126.087 ms | 0.18 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.094 ms | 0.57 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 181.535 ms | 0.01 | 0.00 | 0% | ok | 1.1x (28%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.689 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 11.986 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.633 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.432 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 63.630 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 97.271 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.420 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 25.990 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.313 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.581 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.214 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.159 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.081 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.715 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.371 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.541 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.538 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.099 ms | 0.00 | 0.00 | n/a | ok | 1.1x (29%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.357 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilut | f64 | 128^2 grid | 27.176 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.776 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.696 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.062 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.555 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.936 ms | 1.35 | 15.15 | 62% | ok | 1.1x (27%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.264 ms | 1.16 | 12.95 | 53% | ok | 1.1x (27%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.512 ms | 0.58 | 6.50 | 26% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.993 ms | 1.07 | 8.08 | 33% | ok | 0.9x (23%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 209.179 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 243.163 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 151.555 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.892 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.202 ms | 2.33 | 9.32 | 38% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 13.187 ms | 20.36 | 0.32 | 30% | ok | 3.7x (92%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.637 ms | 6.36 | 6.76 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.809 ms | 5.97 | 23.89 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.365 ms | 12.29 | 24.58 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.271 ms | 5.13 | 5.13 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.044 ms | 16.07 | 16.07 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 12.094 ms | 22.20 | 0.26 | n/a | ok | 2.0x (51%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 7.463 ms | 35.97 | 0.28 | n/a | ok | 3.5x (88%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.909 ms | 45.43 | 0.27 | n/a | ok | 3.4x (86%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.879 ms | 45.66 | 0.27 | n/a | ok | 3.5x (86%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 10.475 ms | 25.63 | 0.15 | n/a | ok | 3.7x (92%) |

## arduino-ventuno-q -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON_WITHOUT_AES |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.5 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 16.8, f32 33.6 GFLOP/s |
| bandwidth | read 22.6, update 24.9, triad 23.2 GB/s (ceiling 24.9) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 94% | 104% | 0 |
| blas2 | 10 | 54% | 69% | 0 |
| blas3 | 10 | 45% | 58% | 0 |
| factor | 12 | 12% | 25% | 0 |
| eigen | 8 | 15% | 31% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 29% | 67% | 0 |
| direct | 7 | 1% | 3% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 47% | 56% | 0 |
| mg | 1 | 37% | 37% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 38% | 66% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.676 ms | 2.96 | 23.65 | 95% | ok |
| blas1 | dot | f32 | n=8388608 | 2.874 ms | 5.84 | 23.35 | 94% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.802 ms | 2.15 | 25.80 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.944 ms | 4.25 | 25.52 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 5.252 ms | 1.60 | 25.56 | 103% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.078 ms | 5.45 | 21.80 | 88% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.376 ms | 1.00 | 8.01 | 32% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.558 ms | 0.48 | 3.82 | 15% | ok |
| blas1 | sum | f64 | n=8388608 | 8.148 ms | 1.03 | 8.24 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 198.924 ms | 0.93 | 0.67 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 8.371 ms | 4.01 | 16.04 | 65% | ok |
| blas2 | gemv | f32 | n=4096 | 4.064 ms | 8.26 | 16.52 | 66% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 49.616 ms | 0.68 | 2.71 | 11% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 24.640 ms | 1.36 | 2.72 | 11% | ok |
| blas2 | symv | f64 | n=4096 | 17.224 ms | 1.95 | 3.90 | 16% | ok |
| blas2 | trmv | f64 | n=4096 | 5.033 ms | 3.33 | 13.35 | 54% | ok |
| blas2 | trmv | f32 | n=4096 | 2.617 ms | 6.41 | 12.83 | 52% | ok |
| blas2 | trsv | f64 | n=4096 | 4.868 ms | 3.45 | 13.80 | 56% | ok |
| blas2 | trsv | f32 | n=4096 | 2.653 ms | 6.32 | 12.66 | 51% | ok |
| blas2 | ger | f64 | n=4096 | 15.595 ms | 2.15 | 17.22 | 69% | ok |
| blas3 | gemm | f64 | 512x512x512 | 28.161 ms | 9.53 | 0.22 | 57% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.831 ms | 19.41 | 0.23 | 58% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 17.801 ms | 7.54 | 0.35 | 45% | ok |
| blas3 | symm | f64 | 512x512x512 | 28.985 ms | 9.26 | 0.22 | 55% | ok |
| blas3 | syrk | f64 | 512x512x512 | 42.563 ms | 3.15 | 0.10 | 19% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 81.656 ms | 3.29 | 0.08 | 20% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.715 ms | 7.58 | 0.30 | 45% | ok |
| blas3 | trsm | f64 | 512x512x512 | 17.905 ms | 7.50 | 0.29 | 45% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.532 ms | 15.73 | 0.31 | 47% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 30.647 ms | 0.17 | 1.39 | 6% | ok |
| factor | lu | f64 | 512x512 | 24.687 ms | 3.62 | 0.17 | 22% | ok |
| factor | lu | f32 | 512x512 | 13.224 ms | 6.77 | 0.16 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 22.256 ms | 2.01 | 0.19 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 19.635 ms | 2.28 | 0.11 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 22.613 ms | 1.98 | 0.19 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 68.770 ms | 0.65 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 252.740 ms | 0.71 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 60.617 ms | 2.95 | 0.07 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 352.106 ms | 1.27 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.578 ms | 0.67 | 2.66 | 11% | ok |
| factor | inv | f64 | 512x512 | 145.579 ms | 1.84 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 20.939 ms | 4.27 | 0.20 | 25% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.872 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.007 ms | 1.56 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 612.887 ms | 0.25 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.202 ms | 2.50 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 149.842 ms | 2.80 | 0.01 | 17% | ok |
| eigen | svd | f64 | 256x256 | 230.314 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 47.953 ms | 5.25 | 0.02 | 31% | ok |
| eigen | sylvester | f64 | 256x256 | 356.540 ms | 2.82 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 87.845 ms | 0.10 | 1.53 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 109.993 ms | 0.08 | 1.22 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 147.246 ms | 0.06 | 0.91 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 71.384 ms | 0.12 | 1.88 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 116.753 ms | 0.07 | 1.15 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.639 ms | 1.60 | 16.61 | 67% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.138 ms | 1.22 | 12.73 | 51% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.018 ms | 3.48 | 7.33 | 29% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.391 ms | 0.24 | 0.86 | 3% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.409 ms | 0.24 | 1.98 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 182.608 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 61.413 ms | 0.18 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.042 ms | 0.43 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 102.862 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 139.992 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.281 ms | 0.52 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 204.914 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.357 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.072 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.695 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.222 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 71.016 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 106.816 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 54.363 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 28.813 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.541 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 33.785 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 63.297 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.395 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.466 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 37.371 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.568 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.672 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 47.214 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.111 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.841 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 30.723 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.057 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.941 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.064 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.736 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.090 ms | 1.25 | 14.03 | 56% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.498 ms | 1.05 | 11.74 | 47% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.948 ms | 0.89 | 9.95 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 31.860 ms | 1.21 | 9.12 | 37% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 235.103 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 273.155 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 160.436 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.199 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.110 ms | 4.08 | 16.33 | 66% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 41.736 ms | 6.43 | 0.10 | 38% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.656 ms | 6.32 | 6.71 | 38% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.842 ms | 5.90 | 23.62 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.388 ms | 12.09 | 24.18 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.331 ms | 7.20 | 7.20 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.256 ms | 3.94 | 3.94 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.322 ms | 15.50 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 16.664 ms | 16.11 | 0.13 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 16.562 ms | 16.21 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 23.628 ms | 11.36 | 0.07 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 50.783 ms | 5.29 | 0.03 | n/a | ok |

## arduino-ventuno-q -- native-fast, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON_WITHOUT_AES |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 67.5 GFLOP/s, f32 135.6 GFLOP/s |
| ... hardware, NEON | f64 67.5, f32 135.6 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 67.5, f32 135.5 GFLOP/s |
| bandwidth | read 23.1, update 24.8, triad 22.8 GB/s (ceiling 24.8) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 91% | 106% | 0 | 27% |
| blas2 | 10 | 56% | 82% | 0 | 28% |
| blas3 | 10 | 18% | 51% | 0 | 40% |
| factor | 12 | 2% | 18% | 0 | 27% |
| eigen | 8 | 4% | 11% | 0 | 28% |
| elementwise | 5 | 5% | 8% | 0 | 28% |
| sparse | 5 | 55% | 71% | 0 | 27% |
| direct | 7 | 0% | 1% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 28% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 28% |
| smoother | 3 | 51% | 60% | 0 | 27% |
| mg | 1 | 33% | 33% | 0 | 22% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 36% | 73% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 87% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.969 ms | 2.81 | 22.49 | 91% | ok | 1.0x (24%) |
| blas1 | dot | f32 | n=8388608 | 3.100 ms | 5.41 | 21.65 | 87% | ok | 0.9x (23%) |
| blas1 | axpy | f64 | n=8388608 | 8.309 ms | 2.02 | 24.23 | 98% | ok | 0.9x (23%) |
| blas1 | axpy | f32 | n=8388608 | 4.312 ms | 3.89 | 23.34 | 94% | ok | 0.9x (23%) |
| blas1 | scale | f64 | n=8388608 | 5.083 ms | 1.65 | 26.40 | 106% | ok | 1.0x (26%) |
| blas1 | two_norm | f64 | n=8388608 | 2.872 ms | 5.84 | 23.36 | 94% | ok | 1.1x (27%) |
| blas1 | one_norm | f64 | n=8388608 | 7.655 ms | 1.10 | 8.77 | 35% | ok | 1.1x (27%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.425 ms | 0.54 | 4.35 | 18% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.311 ms | 1.15 | 9.18 | 37% | ok | 1.1x (28%) |
| blas1 | dot_k3 | f64 | n=8388608 | 178.051 ms | 1.04 | 0.75 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 6.915 ms | 4.85 | 19.42 | 78% | ok | 1.2x (30%) |
| blas2 | gemv | f32 | n=4096 | 3.282 ms | 10.22 | 20.46 | 82% | ok | 1.2x (31%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 29.809 ms | 1.13 | 4.50 | 18% | ok | 1.7x (42%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 12.554 ms | 2.67 | 5.35 | 22% | ok | 2.0x (49%) |
| blas2 | symv | f64 | n=4096 | 15.526 ms | 2.16 | 4.33 | 17% | ok | 1.1x (28%) |
| blas2 | trmv | f64 | n=4096 | 4.822 ms | 3.48 | 13.93 | 56% | ok | 1.0x (26%) |
| blas2 | trmv | f32 | n=4096 | 2.432 ms | 6.90 | 13.81 | 56% | ok | 1.1x (27%) |
| blas2 | trsv | f64 | n=4096 | 4.867 ms | 3.45 | 13.80 | 56% | ok | 1.0x (25%) |
| blas2 | trsv | f32 | n=4096 | 2.453 ms | 6.84 | 13.69 | 55% | ok | 1.1x (27%) |
| blas2 | ger | f64 | n=4096 | 15.059 ms | 2.23 | 17.83 | 72% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 7.809 ms | 34.38 | 0.81 | 51% | ok | 3.6x (90%) |
| blas3 | gemm | f32 | 512x512x512 | 4.209 ms | 63.77 | 0.75 | 47% | ok | 3.3x (82%) |
| blas3 | gemmt | f64 | 512x512x512 | 11.703 ms | 11.47 | 0.54 | 17% | ok | 1.5x (38%) |
| blas3 | symm | f64 | 512x512x512 | 8.687 ms | 30.90 | 0.72 | 46% | ok | 3.3x (83%) |
| blas3 | syrk | f64 | 512x512x512 | 42.008 ms | 3.20 | 0.10 | 5% | ok | 1.0x (25%) |
| blas3 | syr2k | f64 | 512x512x512 | 78.979 ms | 3.40 | 0.08 | 5% | ok | 1.0x (26%) |
| blas3 | trmm | f64 | 512x512x512 | 11.071 ms | 12.12 | 0.47 | 18% | ok | 1.6x (40%) |
| blas3 | trsm | f64 | 512x512x512 | 11.199 ms | 11.98 | 0.47 | 18% | ok | 1.6x (40%) |
| blas3 | trsm | f32 | 512x512x512 | 6.745 ms | 19.90 | 0.39 | 15% | ok | 1.3x (32%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 28.290 ms | 0.19 | 1.50 | 6% | ok | 1.1x (27%) |
| factor | lu | f64 | 512x512 | 112.439 ms | 0.80 | 0.04 | 1% | ok | 0.2x (5%) |
| factor | lu | f32 | 512x512 | 97.358 ms | 0.92 | 0.02 | 1% | ok | 0.1x (3%) |
| factor | cholesky | f64 | 512x512 | 20.268 ms | 2.21 | 0.21 | 3% | ok | 1.1x (27%) |
| factor | cholesky | f32 | 512x512 | 17.564 ms | 2.55 | 0.12 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 20.545 ms | 2.18 | 0.20 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 60.987 ms | 0.73 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 184.780 ms | 0.97 | 0.02 | 1% | ok | 1.4x (34%) |
| factor | lq | f64 | 512x512 | 131.583 ms | 1.36 | 0.03 | 2% | ok | 0.5x (12%) |
| factor | hessenberg | f64 | 512x512 | 436.050 ms | 1.03 | 0.01 | 2% | ok | 0.8x (20%) |
| factor | householder | f64 | 512x512 | 0.939 ms | 1.12 | 4.46 | 18% | ok | 1.7x (42%) |
| factor | inv | f64 | 512x512 | 229.129 ms | 1.17 | 0.02 | 2% | ok | 0.6x (16%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 107.501 ms | 0.83 | 0.04 | 1% | ok | 0.2x (5%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.696 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.836 ms | 1.74 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 526.929 ms | 0.29 | 0.00 | 0% | ok | 1.2x (29%) |
| eigen | eigenvalue | f64 | 256x256 | 60.071 ms | 2.79 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 133.863 ms | 3.13 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.711 ms | 1.79 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 32.484 ms | 7.75 | 0.03 | 11% | ok | 1.5x (37%) |
| eigen | sylvester | f64 | 256x256 | 313.948 ms | 3.21 | 0.01 | 5% | ok | 1.1x (28%) |
| elementwise | exp | f64 | n=8388608 | 80.754 ms | 0.10 | 1.66 | 7% | ok | 1.1x (27%) |
| elementwise | log | f64 | n=8388608 | 98.588 ms | 0.09 | 1.36 | 5% | ok | 1.1x (28%) |
| elementwise | sin | f64 | n=8388608 | 133.527 ms | 0.06 | 1.01 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 64.853 ms | 0.13 | 2.07 | 8% | ok | 1.1x (28%) |
| elementwise | erf | f64 | n=8388608 | 105.065 ms | 0.08 | 1.28 | 5% | ok | 1.1x (28%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.542 ms | 1.70 | 17.66 | 71% | ok | 1.1x (27%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.994 ms | 1.31 | 13.65 | 55% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.347 ms | 6.60 | 13.87 | 56% | ok | 1.9x (47%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.909 ms | 0.27 | 0.97 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 9.458 ms | 0.21 | 1.76 | 7% | ok | 0.9x (22%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 164.559 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 55.491 ms | 0.20 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.452 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 92.784 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 126.331 ms | 0.18 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.130 ms | 0.57 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 186.887 ms | 0.01 | 0.00 | 0% | ok | 1.1x (27%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.685 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 11.836 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.628 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.403 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.203 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 97.389 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.261 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 25.966 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.258 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.594 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.125 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.170 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.264 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.672 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.379 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.984 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.589 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.100 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.415 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.345 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.778 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ildl | f64 | 128^2 grid | 2.669 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ssor | f64 | 128^2 grid | 0.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.627 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.973 ms | 1.33 | 14.86 | 60% | ok | 1.1x (26%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.299 ms | 1.14 | 12.76 | 51% | ok | 1.1x (27%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.677 ms | 0.98 | 10.96 | 44% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.592 ms | 1.08 | 8.17 | 33% | ok | 0.9x (22%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 209.923 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 243.995 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 150.429 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.799 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 3.687 ms | 4.55 | 18.20 | 73% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 11.186 ms | 24.00 | 0.37 | 36% | ok | 3.7x (93%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.380 ms | 7.05 | 7.49 | 30% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.793 ms | 6.01 | 24.02 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.360 ms | 12.34 | 24.67 | n/a | ok | 1.0x (26%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.100 ms | 7.99 | 7.99 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 3.812 ms | 4.40 | 4.40 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 5.006 ms | 53.63 | 0.63 | n/a | ok | 3.5x (87%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.816 ms | 55.74 | 0.44 | n/a | ok | 3.5x (87%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.705 ms | 57.06 | 0.33 | n/a | ok | 3.5x (88%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 6.002 ms | 44.72 | 0.26 | n/a | ok | 3.9x (98%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.234 ms | 20.28 | 0.12 | n/a | ok | 3.8x (96%) |

