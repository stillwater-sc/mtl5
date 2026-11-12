# mtl5_bench platform characterization

Class `medium`, variants native native-fast, threads 1,4, CPUs 0,1,2,3, clock policy governor performance on the A78C policies; performance 902-2112 MHz on cpus 0,1; performance 940-2361 MHz on cpus 2,3, 2026-11-11T19:16:03Z.

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
| bandwidth | read 23.1, update 24.8, triad 23.4 GB/s (ceiling 24.8) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 83% | 104% | 0 |
| blas2 | 10 | 55% | 71% | 0 |
| blas3 | 10 | 30% | 33% | 0 |
| factor | 12 | 12% | 25% | 0 |
| eigen | 8 | 15% | 22% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 30% | 66% | 0 |
| direct | 7 | 1% | 3% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 47% | 56% | 0 |
| mg | 1 | 36% | 36% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 33% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.648 ms | 2.97 | 23.77 | 96% | ok |
| blas1 | dot | f32 | n=8388608 | 4.517 ms | 3.71 | 14.86 | 60% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.810 ms | 2.15 | 25.78 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.952 ms | 4.25 | 25.47 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 5.350 ms | 1.57 | 25.09 | 101% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.237 ms | 5.18 | 20.73 | 83% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.187 ms | 1.02 | 8.20 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.287 ms | 0.49 | 3.88 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.362 ms | 1.00 | 8.03 | 32% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 196.176 ms | 0.94 | 0.68 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 9.257 ms | 3.62 | 14.51 | 58% | ok |
| blas2 | gemv | f32 | n=4096 | 6.915 ms | 4.85 | 9.71 | 39% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 9.817 ms | 3.42 | 13.68 | 55% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.329 ms | 7.75 | 15.51 | 62% | ok |
| blas2 | symv | f64 | n=4096 | 16.936 ms | 1.98 | 3.97 | 16% | ok |
| blas2 | trmv | f64 | n=4096 | 4.955 ms | 3.39 | 13.56 | 55% | ok |
| blas2 | trmv | f32 | n=4096 | 6.805 ms | 2.47 | 4.94 | 20% | ok |
| blas2 | trsv | f64 | n=4096 | 4.954 ms | 3.39 | 13.56 | 55% | ok |
| blas2 | trsv | f32 | n=4096 | 20.718 ms | 0.81 | 1.62 | 7% | ok |
| blas2 | ger | f64 | n=4096 | 15.256 ms | 2.20 | 17.60 | 71% | ok |
| blas3 | gemm | f64 | 512x512x512 | 53.873 ms | 4.98 | 0.12 | 30% | ok |
| blas3 | gemm | f32 | 512x512x512 | 25.232 ms | 10.64 | 0.12 | 32% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 41.049 ms | 3.27 | 0.15 | 20% | ok |
| blas3 | symm | f64 | 512x512x512 | 52.486 ms | 5.11 | 0.12 | 31% | ok |
| blas3 | syrk | f64 | 512x512x512 | 43.366 ms | 3.10 | 0.10 | 18% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 79.913 ms | 3.36 | 0.08 | 20% | ok |
| blas3 | trmm | f64 | 512x512x512 | 25.870 ms | 5.19 | 0.20 | 31% | ok |
| blas3 | trsm | f64 | 512x512x512 | 26.327 ms | 5.10 | 0.20 | 30% | ok |
| blas3 | trsm | f32 | 512x512x512 | 12.115 ms | 11.08 | 0.22 | 33% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 32.062 ms | 0.17 | 1.32 | 5% | ok |
| factor | lu | f64 | 512x512 | 26.158 ms | 3.42 | 0.16 | 20% | ok |
| factor | lu | f32 | 512x512 | 13.186 ms | 6.79 | 0.16 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 22.459 ms | 1.99 | 0.19 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 19.631 ms | 2.28 | 0.11 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 21.788 ms | 2.05 | 0.19 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 69.248 ms | 0.65 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 249.655 ms | 0.72 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 60.831 ms | 2.94 | 0.07 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 347.591 ms | 1.29 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.545 ms | 0.68 | 2.71 | 11% | ok |
| factor | inv | f64 | 512x512 | 145.030 ms | 1.85 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 21.397 ms | 4.18 | 0.20 | 25% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.945 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.285 ms | 1.55 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 592.811 ms | 0.25 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.319 ms | 2.49 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 151.903 ms | 2.76 | 0.01 | 16% | ok |
| eigen | svd | f64 | 256x256 | 231.130 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 67.411 ms | 3.73 | 0.02 | 22% | ok |
| eigen | sylvester | f64 | 256x256 | 379.393 ms | 2.65 | 0.00 | 16% | ok |
| elementwise | exp | f64 | n=8388608 | 89.134 ms | 0.09 | 1.51 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 110.964 ms | 0.08 | 1.21 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 149.831 ms | 0.06 | 0.90 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 71.381 ms | 0.12 | 1.88 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 117.077 ms | 0.07 | 1.15 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.669 ms | 1.57 | 16.32 | 66% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.179 ms | 1.20 | 12.50 | 50% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.007 ms | 3.49 | 7.33 | 30% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.369 ms | 0.24 | 0.86 | 3% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.765 ms | 0.23 | 1.90 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 181.876 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 58.962 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.333 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 103.207 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 139.816 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.235 ms | 0.52 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 207.110 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.643 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.782 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.705 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.875 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.463 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 103.856 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 53.914 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 29.059 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.443 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 33.705 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 63.674 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.987 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.765 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 36.992 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.997 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.313 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 47.442 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.112 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 5.231 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 30.495 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.058 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.952 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.063 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.731 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.109 ms | 1.24 | 13.91 | 56% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.496 ms | 1.05 | 11.75 | 47% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 5.035 ms | 0.52 | 5.82 | 23% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.432 ms | 1.19 | 8.96 | 36% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 231.698 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 270.586 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 164.754 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.426 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 8.135 ms | 2.06 | 8.25 | 33% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 48.743 ms | 5.51 | 0.09 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.942 ms | 5.70 | 6.06 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.846 ms | 5.89 | 23.58 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.399 ms | 11.99 | 23.99 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.654 ms | 4.59 | 4.59 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.150 ms | 14.58 | 14.58 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 26.726 ms | 10.04 | 0.12 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 26.354 ms | 10.19 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 20.313 ms | 13.22 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 20.297 ms | 13.23 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 38.404 ms | 6.99 | 0.04 | n/a | ok |

## arduino-ventuno-q -- native, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 67.5 GFLOP/s, f32 135.5 GFLOP/s |
| ... hardware, NEON | f64 67.5, f32 135.5 GFLOP/s |
| ... this build, scalar | f64 33.7, f32 33.9 GFLOP/s |
| bandwidth | read 23.6, update 25.1, triad 22.9 GB/s (ceiling 25.1) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 90% | 107% | 0 | 27% |
| blas2 | 10 | 52% | 72% | 0 | 27% |
| blas3 | 10 | 9% | 16% | 0 | 29% |
| factor | 12 | 2% | 18% | 0 | 28% |
| eigen | 8 | 4% | 9% | 0 | 28% |
| elementwise | 5 | 5% | 8% | 0 | 27% |
| sparse | 5 | 54% | 70% | 0 | 27% |
| direct | 7 | 0% | 1% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 28% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 28% |
| smoother | 3 | 51% | 59% | 0 | 27% |
| mg | 1 | 33% | 33% | 0 | 23% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 30% | 37% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 55% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.958 ms | 2.82 | 22.53 | 90% | ok | 0.9x (24%) |
| blas1 | dot | f32 | n=8388608 | 3.072 ms | 5.46 | 21.85 | 87% | ok | 1.5x (37%) |
| blas1 | axpy | f64 | n=8388608 | 8.226 ms | 2.04 | 24.48 | 98% | ok | 0.9x (24%) |
| blas1 | axpy | f32 | n=8388608 | 4.217 ms | 3.98 | 23.87 | 95% | ok | 0.9x (23%) |
| blas1 | scale | f64 | n=8388608 | 5.022 ms | 1.67 | 26.72 | 107% | ok | 1.1x (27%) |
| blas1 | two_norm | f64 | n=8388608 | 2.859 ms | 5.87 | 23.47 | 94% | ok | 1.1x (28%) |
| blas1 | one_norm | f64 | n=8388608 | 7.692 ms | 1.09 | 8.72 | 35% | ok | 1.1x (27%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.459 ms | 0.54 | 4.34 | 17% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.897 ms | 1.06 | 8.50 | 34% | ok | 1.1x (26%) |
| blas1 | dot_k3 | f64 | n=8388608 | 178.871 ms | 1.03 | 0.75 | 3% | ok | 1.1x (27%) |
| blas2 | gemv | f64 | n=4096 | 7.760 ms | 4.32 | 17.31 | 69% | ok | 1.2x (30%) |
| blas2 | gemv | f32 | n=4096 | 3.957 ms | 8.48 | 16.97 | 68% | ok | 1.7x (44%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 8.003 ms | 4.19 | 16.78 | 67% | ok | 1.2x (31%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.188 ms | 6.47 | 12.94 | 52% | ok | 0.8x (21%) |
| blas2 | symv | f64 | n=4096 | 15.842 ms | 2.12 | 4.24 | 17% | ok | 1.1x (27%) |
| blas2 | trmv | f64 | n=4096 | 5.749 ms | 2.92 | 11.68 | 47% | ok | 0.9x (22%) |
| blas2 | trmv | f32 | n=4096 | 6.104 ms | 2.75 | 5.50 | 22% | ok | 1.1x (28%) |
| blas2 | trsv | f64 | n=4096 | 5.354 ms | 3.13 | 12.55 | 50% | ok | 0.9x (23%) |
| blas2 | trsv | f32 | n=4096 | 19.346 ms | 0.87 | 1.74 | 7% | ok | 1.1x (27%) |
| blas2 | ger | f64 | n=4096 | 14.897 ms | 2.25 | 18.02 | 72% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 27.034 ms | 9.93 | 0.23 | 15% | ok | 2.0x (50%) |
| blas3 | gemm | f32 | 512x512x512 | 12.044 ms | 22.29 | 0.26 | 16% | ok | 2.1x (52%) |
| blas3 | gemmt | f64 | 512x512x512 | 35.650 ms | 3.76 | 0.18 | 6% | ok | 1.2x (29%) |
| blas3 | symm | f64 | 512x512x512 | 54.711 ms | 4.91 | 0.11 | 7% | ok | 1.0x (24%) |
| blas3 | syrk | f64 | 512x512x512 | 35.106 ms | 3.82 | 0.12 | 6% | ok | 1.2x (31%) |
| blas3 | syr2k | f64 | 512x512x512 | 78.992 ms | 3.40 | 0.08 | 5% | ok | 1.0x (25%) |
| blas3 | trmm | f64 | 512x512x512 | 22.576 ms | 5.95 | 0.23 | 9% | ok | 1.1x (29%) |
| blas3 | trsm | f64 | 512x512x512 | 22.810 ms | 5.88 | 0.23 | 9% | ok | 1.2x (29%) |
| blas3 | trsm | f32 | 512x512x512 | 10.830 ms | 12.39 | 0.24 | 9% | ok | 1.1x (28%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 28.360 ms | 0.19 | 1.50 | 6% | ok | 1.1x (28%) |
| factor | lu | f64 | 512x512 | 109.537 ms | 0.82 | 0.04 | 1% | ok | 0.2x (6%) |
| factor | lu | f32 | 512x512 | 91.533 ms | 0.98 | 0.02 | 1% | ok | 0.1x (4%) |
| factor | cholesky | f64 | 512x512 | 19.947 ms | 2.24 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 17.565 ms | 2.55 | 0.12 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 19.210 ms | 2.33 | 0.22 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 61.333 ms | 0.73 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 213.764 ms | 0.84 | 0.02 | 1% | ok | 1.2x (29%) |
| factor | lq | f64 | 512x512 | 138.218 ms | 1.29 | 0.03 | 2% | ok | 0.4x (11%) |
| factor | hessenberg | f64 | 512x512 | 447.266 ms | 1.00 | 0.01 | 1% | ok | 0.8x (19%) |
| factor | householder | f64 | 512x512 | 0.936 ms | 1.12 | 4.48 | 18% | ok | 1.7x (41%) |
| factor | inv | f64 | 512x512 | 220.125 ms | 1.22 | 0.02 | 2% | ok | 0.7x (16%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 109.426 ms | 0.82 | 0.04 | 1% | ok | 0.2x (5%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.739 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.841 ms | 1.74 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 525.379 ms | 0.29 | 0.00 | 0% | ok | 1.1x (28%) |
| eigen | eigenvalue | f64 | 256x256 | 60.025 ms | 2.80 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 134.076 ms | 3.13 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.438 ms | 1.80 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 42.850 ms | 5.87 | 0.02 | 9% | ok | 1.6x (39%) |
| eigen | sylvester | f64 | 256x256 | 320.048 ms | 3.15 | 0.00 | 5% | ok | 1.2x (30%) |
| elementwise | exp | f64 | n=8388608 | 81.364 ms | 0.10 | 1.65 | 7% | ok | 1.1x (27%) |
| elementwise | log | f64 | n=8388608 | 100.498 ms | 0.08 | 1.34 | 5% | ok | 1.1x (28%) |
| elementwise | sin | f64 | n=8388608 | 135.677 ms | 0.06 | 0.99 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 65.958 ms | 0.13 | 2.03 | 8% | ok | 1.1x (27%) |
| elementwise | erf | f64 | n=8388608 | 107.588 ms | 0.08 | 1.25 | 5% | ok | 1.1x (27%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.555 ms | 1.68 | 17.51 | 70% | ok | 1.1x (27%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.990 ms | 1.32 | 13.68 | 55% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.449 ms | 6.49 | 13.65 | 54% | ok | 1.9x (47%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.795 ms | 0.28 | 0.98 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 9.907 ms | 0.20 | 1.68 | 7% | ok | 0.9x (22%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 163.561 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 53.182 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.388 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 91.661 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 125.560 ms | 0.18 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.527 ms | 0.57 | 0.02 | 1% | ok | 1.1x (27%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 181.881 ms | 0.01 | 0.00 | 0% | ok | 1.1x (28%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.049 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.033 ms | 0.00 | 0.00 | n/a | ok | 1.1x (26%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.630 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.715 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.123 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 97.680 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.571 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.093 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.451 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.546 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.815 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 56.981 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.116 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.528 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.188 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 36.045 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.209 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.099 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.374 ms | 0.00 | 0.00 | n/a | ok | 1.2x (30%) |
| pc | ilut | f64 | 128^2 grid | 27.633 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.797 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.675 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ssor | f64 | 128^2 grid | 0.062 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.552 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.977 ms | 1.32 | 14.83 | 59% | ok | 1.1x (27%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.288 ms | 1.14 | 12.82 | 51% | ok | 1.1x (27%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.472 ms | 0.59 | 6.56 | 26% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.349 ms | 1.09 | 8.22 | 33% | ok | 0.9x (23%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 209.007 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 243.701 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 133.262 ms | 0.00 | 0.00 | n/a | ok | 1.2x (31%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.968 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.221 ms | 2.32 | 9.29 | 37% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 13.140 ms | 20.43 | 0.32 | 30% | ok | 3.7x (93%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.657 ms | 6.31 | 6.71 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.811 ms | 5.97 | 23.87 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.373 ms | 12.22 | 24.45 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.269 ms | 5.13 | 5.13 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.036 ms | 16.20 | 16.20 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 12.256 ms | 21.90 | 0.26 | n/a | ok | 2.2x (55%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 7.445 ms | 36.06 | 0.28 | n/a | ok | 3.5x (89%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.901 ms | 45.49 | 0.27 | n/a | ok | 3.4x (86%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.889 ms | 45.58 | 0.27 | n/a | ok | 3.4x (86%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 10.483 ms | 25.61 | 0.15 | n/a | ok | 3.7x (92%) |

## arduino-ventuno-q -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON_WITHOUT_AES |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.6 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 16.8, f32 33.6 GFLOP/s |
| bandwidth | read 22.7, update 24.9, triad 23.3 GB/s (ceiling 24.9) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 94% | 104% | 0 |
| blas2 | 10 | 55% | 71% | 0 |
| blas3 | 10 | 45% | 58% | 0 |
| factor | 12 | 12% | 26% | 0 |
| eigen | 8 | 15% | 31% | 0 |
| elementwise | 5 | 5% | 7% | 0 |
| sparse | 5 | 29% | 68% | 0 |
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
| blas1 | dot | f64 | n=8388608 | 5.701 ms | 2.94 | 23.54 | 95% | ok |
| blas1 | dot | f32 | n=8388608 | 2.878 ms | 5.83 | 23.31 | 94% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.801 ms | 2.15 | 25.81 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.941 ms | 4.26 | 25.54 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 5.332 ms | 1.57 | 25.17 | 101% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.161 ms | 5.31 | 21.23 | 85% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.590 ms | 0.98 | 7.81 | 31% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.349 ms | 0.48 | 3.87 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.755 ms | 0.96 | 7.67 | 31% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 198.557 ms | 0.93 | 0.68 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 8.251 ms | 4.07 | 16.28 | 65% | ok |
| blas2 | gemv | f32 | n=4096 | 4.052 ms | 8.28 | 16.57 | 67% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 49.182 ms | 0.68 | 2.73 | 11% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 25.699 ms | 1.31 | 2.61 | 11% | ok |
| blas2 | symv | f64 | n=4096 | 16.930 ms | 1.98 | 3.97 | 16% | ok |
| blas2 | trmv | f64 | n=4096 | 4.952 ms | 3.39 | 13.56 | 55% | ok |
| blas2 | trmv | f32 | n=4096 | 2.592 ms | 6.47 | 12.96 | 52% | ok |
| blas2 | trsv | f64 | n=4096 | 4.788 ms | 3.50 | 14.03 | 56% | ok |
| blas2 | trsv | f32 | n=4096 | 2.593 ms | 6.47 | 12.95 | 52% | ok |
| blas2 | ger | f64 | n=4096 | 15.218 ms | 2.20 | 17.64 | 71% | ok |
| blas3 | gemm | f64 | 512x512x512 | 28.189 ms | 9.52 | 0.22 | 57% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.842 ms | 19.39 | 0.23 | 58% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 17.957 ms | 7.47 | 0.35 | 44% | ok |
| blas3 | symm | f64 | 512x512x512 | 29.177 ms | 9.20 | 0.22 | 55% | ok |
| blas3 | syrk | f64 | 512x512x512 | 41.220 ms | 3.26 | 0.10 | 19% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 79.548 ms | 3.37 | 0.08 | 20% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.649 ms | 7.60 | 0.30 | 45% | ok |
| blas3 | trsm | f64 | 512x512x512 | 18.067 ms | 7.43 | 0.29 | 44% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.526 ms | 15.74 | 0.31 | 47% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 31.380 ms | 0.17 | 1.35 | 5% | ok |
| factor | lu | f64 | 512x512 | 24.602 ms | 3.64 | 0.17 | 22% | ok |
| factor | lu | f32 | 512x512 | 13.196 ms | 6.78 | 0.16 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 22.260 ms | 2.01 | 0.19 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 19.666 ms | 2.27 | 0.11 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 22.616 ms | 1.98 | 0.19 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 67.975 ms | 0.66 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 249.647 ms | 0.72 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 60.589 ms | 2.95 | 0.07 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 347.760 ms | 1.29 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.553 ms | 0.68 | 2.70 | 11% | ok |
| factor | inv | f64 | 512x512 | 143.459 ms | 1.87 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 20.862 ms | 4.29 | 0.20 | 26% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.882 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 96.701 ms | 1.56 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 604.340 ms | 0.25 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.220 ms | 2.50 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 149.906 ms | 2.80 | 0.01 | 17% | ok |
| eigen | svd | f64 | 256x256 | 230.064 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 48.004 ms | 5.24 | 0.02 | 31% | ok |
| eigen | sylvester | f64 | 256x256 | 355.257 ms | 2.83 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 89.862 ms | 0.09 | 1.49 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 110.900 ms | 0.08 | 1.21 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 149.958 ms | 0.06 | 0.90 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 71.961 ms | 0.12 | 1.87 | 7% | ok |
| elementwise | erf | f64 | n=8388608 | 117.191 ms | 0.07 | 1.15 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.620 ms | 1.62 | 16.81 | 68% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.156 ms | 1.21 | 12.63 | 51% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.029 ms | 3.48 | 7.32 | 29% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.383 ms | 0.24 | 0.86 | 3% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.460 ms | 0.24 | 1.97 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 181.889 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 61.650 ms | 0.18 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.561 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 103.552 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 141.067 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.271 ms | 0.52 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 203.997 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.748 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 14.105 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.712 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 44.046 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.892 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 104.593 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 53.819 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 28.929 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.449 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 33.866 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 63.542 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.128 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.728 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 37.230 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.682 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.597 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 47.039 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.114 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.882 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 31.046 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.058 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.905 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.064 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.722 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.102 ms | 1.25 | 13.95 | 56% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.489 ms | 1.05 | 11.78 | 47% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.941 ms | 0.89 | 9.97 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.006 ms | 1.20 | 9.08 | 37% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 232.202 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 270.564 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 149.555 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.228 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.079 ms | 4.11 | 16.45 | 66% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 41.762 ms | 6.43 | 0.10 | 38% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.652 ms | 6.33 | 6.72 | 38% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.825 ms | 5.94 | 23.76 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.388 ms | 12.08 | 24.17 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.320 ms | 7.23 | 7.23 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.255 ms | 3.94 | 3.94 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.316 ms | 15.50 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 16.639 ms | 16.13 | 0.13 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 16.573 ms | 16.20 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 23.623 ms | 11.36 | 0.07 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 50.869 ms | 5.28 | 0.03 | n/a | ok |

## arduino-ventuno-q -- native-fast, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON_WITHOUT_AES |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 67.8 GFLOP/s, f32 135.6 GFLOP/s |
| ... hardware, NEON | f64 67.8, f32 135.5 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 67.4, f32 135.6 GFLOP/s |
| bandwidth | read 23.0, update 24.8, triad 22.5 GB/s (ceiling 24.8) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 90% | 109% | 0 | 28% |
| blas2 | 10 | 57% | 90% | 0 | 27% |
| blas3 | 10 | 18% | 51% | 0 | 40% |
| factor | 12 | 2% | 16% | 0 | 28% |
| eigen | 8 | 4% | 12% | 0 | 28% |
| elementwise | 5 | 5% | 8% | 0 | 27% |
| sparse | 5 | 55% | 87% | 0 | 28% |
| direct | 7 | 0% | 1% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 29% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 52% | 59% | 0 | 27% |
| mg | 1 | 33% | 33% | 0 | 22% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 35% | 74% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 85% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 6.001 ms | 2.80 | 22.37 | 90% | ok | 1.0x (24%) |
| blas1 | dot | f32 | n=8388608 | 3.137 ms | 5.35 | 21.40 | 86% | ok | 0.9x (23%) |
| blas1 | axpy | f64 | n=8388608 | 8.208 ms | 2.04 | 24.53 | 99% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 4.249 ms | 3.95 | 23.69 | 96% | ok | 0.9x (23%) |
| blas1 | scale | f64 | n=8388608 | 4.971 ms | 1.69 | 27.00 | 109% | ok | 1.1x (27%) |
| blas1 | two_norm | f64 | n=8388608 | 2.872 ms | 5.84 | 23.37 | 94% | ok | 1.1x (28%) |
| blas1 | one_norm | f64 | n=8388608 | 7.510 ms | 1.12 | 8.94 | 36% | ok | 1.1x (29%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.554 ms | 0.54 | 4.31 | 17% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.808 ms | 1.07 | 8.60 | 35% | ok | 1.1x (28%) |
| blas1 | dot_k3 | f64 | n=8388608 | 179.772 ms | 1.03 | 0.75 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 6.591 ms | 5.09 | 20.38 | 82% | ok | 1.3x (31%) |
| blas2 | gemv | f32 | n=4096 | 2.997 ms | 11.20 | 22.40 | 90% | ok | 1.4x (34%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 22.955 ms | 1.46 | 5.85 | 24% | ok | 2.1x (54%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 12.495 ms | 2.69 | 5.37 | 22% | ok | 2.1x (51%) |
| blas2 | symv | f64 | n=4096 | 15.586 ms | 2.15 | 4.31 | 17% | ok | 1.1x (27%) |
| blas2 | trmv | f64 | n=4096 | 4.778 ms | 3.51 | 14.06 | 57% | ok | 1.0x (26%) |
| blas2 | trmv | f32 | n=4096 | 2.543 ms | 6.60 | 13.21 | 53% | ok | 1.0x (25%) |
| blas2 | trsv | f64 | n=4096 | 4.783 ms | 3.51 | 14.04 | 57% | ok | 1.0x (25%) |
| blas2 | trsv | f32 | n=4096 | 2.412 ms | 6.96 | 13.92 | 56% | ok | 1.1x (27%) |
| blas2 | ger | f64 | n=4096 | 14.838 ms | 2.26 | 18.10 | 73% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 7.818 ms | 34.34 | 0.80 | 51% | ok | 3.6x (90%) |
| blas3 | gemm | f32 | 512x512x512 | 4.213 ms | 63.71 | 0.75 | 47% | ok | 3.3x (82%) |
| blas3 | gemmt | f64 | 512x512x512 | 10.402 ms | 12.90 | 0.60 | 19% | ok | 1.7x (43%) |
| blas3 | symm | f64 | 512x512x512 | 8.675 ms | 30.94 | 0.73 | 46% | ok | 3.4x (84%) |
| blas3 | syrk | f64 | 512x512x512 | 36.562 ms | 3.67 | 0.11 | 5% | ok | 1.1x (28%) |
| blas3 | syr2k | f64 | 512x512x512 | 77.579 ms | 3.46 | 0.08 | 5% | ok | 1.0x (26%) |
| blas3 | trmm | f64 | 512x512x512 | 10.954 ms | 12.25 | 0.48 | 18% | ok | 1.6x (40%) |
| blas3 | trsm | f64 | 512x512x512 | 11.394 ms | 11.78 | 0.46 | 17% | ok | 1.6x (40%) |
| blas3 | trsm | f32 | 512x512x512 | 6.689 ms | 20.06 | 0.39 | 15% | ok | 1.3x (32%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 29.173 ms | 0.18 | 1.46 | 6% | ok | 1.1x (27%) |
| factor | lu | f64 | 512x512 | 109.071 ms | 0.82 | 0.04 | 1% | ok | 0.2x (6%) |
| factor | lu | f32 | 512x512 | 93.127 ms | 0.96 | 0.02 | 1% | ok | 0.1x (4%) |
| factor | cholesky | f64 | 512x512 | 20.069 ms | 2.23 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 17.536 ms | 2.55 | 0.12 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 20.330 ms | 2.20 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 60.247 ms | 0.74 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 198.816 ms | 0.90 | 0.02 | 1% | ok | 1.3x (31%) |
| factor | lq | f64 | 512x512 | 126.414 ms | 1.42 | 0.03 | 2% | ok | 0.5x (12%) |
| factor | hessenberg | f64 | 512x512 | 440.059 ms | 1.02 | 0.01 | 1% | ok | 0.8x (20%) |
| factor | householder | f64 | 512x512 | 1.033 ms | 1.02 | 4.06 | 16% | ok | 1.5x (38%) |
| factor | inv | f64 | 512x512 | 224.090 ms | 1.20 | 0.02 | 2% | ok | 0.6x (16%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 108.003 ms | 0.83 | 0.04 | 1% | ok | 0.2x (5%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.471 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.506 ms | 1.75 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 525.246 ms | 0.29 | 0.00 | 0% | ok | 1.2x (29%) |
| eigen | eigenvalue | f64 | 256x256 | 59.907 ms | 2.80 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 133.826 ms | 3.13 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.719 ms | 1.79 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 31.967 ms | 7.87 | 0.03 | 12% | ok | 1.5x (38%) |
| eigen | sylvester | f64 | 256x256 | 312.866 ms | 3.22 | 0.01 | 5% | ok | 1.1x (28%) |
| elementwise | exp | f64 | n=8388608 | 81.760 ms | 0.10 | 1.64 | 7% | ok | 1.1x (27%) |
| elementwise | log | f64 | n=8388608 | 100.712 ms | 0.08 | 1.33 | 5% | ok | 1.1x (28%) |
| elementwise | sin | f64 | n=8388608 | 135.791 ms | 0.06 | 0.99 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 67.033 ms | 0.13 | 2.00 | 8% | ok | 1.1x (27%) |
| elementwise | erf | f64 | n=8388608 | 107.826 ms | 0.08 | 1.24 | 5% | ok | 1.1x (27%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.265 ms | 2.07 | 21.52 | 87% | ok | 1.3x (32%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.987 ms | 1.32 | 13.71 | 55% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.438 ms | 6.51 | 13.68 | 55% | ok | 1.9x (47%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.814 ms | 0.28 | 0.97 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 9.438 ms | 0.21 | 1.76 | 7% | ok | 0.9x (22%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 164.695 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 55.546 ms | 0.20 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.573 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 92.323 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 126.886 ms | 0.18 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.188 ms | 0.57 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 185.829 ms | 0.01 | 0.00 | 0% | ok | 1.1x (27%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.659 ms | 0.00 | 0.00 | n/a | ok | 1.1x (29%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.239 ms | 0.00 | 0.00 | n/a | ok | 1.2x (29%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.636 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 39.374 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.234 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 97.497 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.728 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.066 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.456 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.710 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.007 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.268 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.429 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.708 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.446 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.720 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.625 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.102 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.437 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilut | f64 | 128^2 grid | 27.683 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.782 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.695 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.571 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.017 ms | 1.30 | 14.54 | 59% | ok | 1.0x (26%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.278 ms | 1.15 | 12.88 | 52% | ok | 1.1x (27%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.662 ms | 0.98 | 11.02 | 44% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.794 ms | 1.07 | 8.12 | 33% | ok | 0.9x (22%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 210.080 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 244.974 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 149.863 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.893 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 3.680 ms | 4.56 | 18.24 | 74% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 11.197 ms | 23.97 | 0.37 | 35% | ok | 3.7x (93%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.357 ms | 7.12 | 7.56 | 31% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.784 ms | 6.03 | 24.11 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.361 ms | 12.33 | 24.66 | n/a | ok | 1.0x (26%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.111 ms | 7.95 | 7.95 | n/a | ok | 1.1x (27%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 3.810 ms | 4.40 | 4.40 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 5.067 ms | 52.97 | 0.62 | n/a | ok | 3.4x (85%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.867 ms | 55.15 | 0.43 | n/a | ok | 3.4x (85%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.289 ms | 62.59 | 0.37 | n/a | ok | 3.9x (97%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 6.509 ms | 41.24 | 0.24 | n/a | ok | 3.6x (91%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.316 ms | 20.16 | 0.12 | n/a | ok | 3.8x (96%) |

