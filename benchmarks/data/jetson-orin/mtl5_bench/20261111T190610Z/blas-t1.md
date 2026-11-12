## jetson-orin -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 10.7 GFLOP/s, f32 21.5 GFLOP/s |
| ... hardware, NEON | f64 10.7, f32 21.5 GFLOP/s |
| ... this build, scalar | f64 5.4, f32 5.4 GFLOP/s |
| bandwidth | read 11.5, update 16.2, triad 14.8 GB/s (ceiling 16.2) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 71% | 131% | 0 |
| blas2 | 10 | 64% | 123% | 0 |
| blas3 | 10 | 89% | 95% | 0 |
| factor | 12 | 41% | 61% | 0 |
| eigen | 8 | 37% | 80% | 0 |
| elementwise | 5 | 8% | 15% | 0 |
| sparse | 5 | 32% | 74% | 0 |
| direct | 7 | 1% | 3% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 49% | 60% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 33% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.633 ms | 1.44 | 11.54 | 71% | ok |
| blas1 | dot | f32 | n=8388608 | 6.587 ms | 2.55 | 10.19 | 63% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.267 ms | 1.37 | 16.41 | 101% | ok |
| blas1 | axpy | f32 | n=8388608 | 5.963 ms | 2.81 | 16.88 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 6.355 ms | 1.32 | 21.12 | 131% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.600 ms | 3.00 | 11.98 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.751 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.502 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 192.384 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 12.620 ms | 2.66 | 10.64 | 66% | ok |
| blas2 | gemv | f32 | n=4096 | 5.885 ms | 5.70 | 11.41 | 71% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 12.953 ms | 2.59 | 10.37 | 64% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.832 ms | 5.75 | 11.51 | 71% | ok |
| blas2 | symv | f64 | n=4096 | 15.986 ms | 2.10 | 4.20 | 26% | ok |
| blas2 | trmv | f64 | n=4096 | 8.649 ms | 1.94 | 7.77 | 48% | ok |
| blas2 | trmv | f32 | n=4096 | 4.868 ms | 3.45 | 6.90 | 43% | ok |
| blas2 | trsv | f64 | n=4096 | 8.637 ms | 1.94 | 7.78 | 48% | ok |
| blas2 | trsv | f32 | n=4096 | 4.851 ms | 3.46 | 6.92 | 43% | ok |
| blas2 | ger | f64 | n=4096 | 13.491 ms | 2.49 | 19.90 | 123% | ok |
| blas3 | gemm | f64 | 512x512x512 | 26.731 ms | 10.04 | 0.24 | 94% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.122 ms | 20.46 | 0.24 | 95% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 16.622 ms | 8.07 | 0.38 | 75% | ok |
| blas3 | symm | f64 | 512x512x512 | 26.694 ms | 10.06 | 0.24 | 94% | ok |
| blas3 | syrk | f64 | 512x512x512 | 14.433 ms | 9.30 | 0.29 | 87% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 28.179 ms | 9.53 | 0.22 | 89% | ok |
| blas3 | trmm | f64 | 512x512x512 | 13.634 ms | 9.84 | 0.38 | 92% | ok |
| blas3 | trsm | f64 | 512x512x512 | 14.984 ms | 8.96 | 0.35 | 83% | ok |
| blas3 | trsm | f32 | 512x512x512 | 7.581 ms | 17.70 | 0.35 | 83% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.800 ms | 0.54 | 4.33 | 27% | ok |
| factor | lu | f64 | 512x512 | 13.587 ms | 6.59 | 0.31 | 61% | ok |
| factor | lu | f32 | 512x512 | 8.464 ms | 10.57 | 0.25 | 49% | ok |
| factor | cholesky | f64 | 512x512 | 7.870 ms | 5.68 | 0.53 | 53% | ok |
| factor | cholesky | f32 | 512x512 | 5.051 ms | 8.86 | 0.42 | 41% | ok |
| factor | ldlt | f64 | 512x512 | 32.350 ms | 1.38 | 0.13 | 13% | ok |
| factor | ldlt_bk | f64 | 512x512 | 85.727 ms | 0.52 | 0.05 | 5% | ok |
| factor | qr | f64 | 512x512 | 28.432 ms | 6.29 | 0.15 | 59% | ok |
| factor | lq | f64 | 512x512 | 93.379 ms | 1.92 | 0.04 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 479.353 ms | 0.93 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 2.000 ms | 0.53 | 2.10 | 13% | ok |
| factor | inv | f64 | 512x512 | 173.614 ms | 1.55 | 0.02 | 14% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 17.699 ms | 5.06 | 0.24 | 47% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 10.351 ms | 2.16 | 0.10 | 20% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 145.168 ms | 1.04 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 899.228 ms | 0.17 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 62.978 ms | 2.66 | 0.02 | 25% | ok |
| eigen | schur | f64 | 256x256 | 90.345 ms | 4.64 | 0.01 | 43% | ok |
| eigen | svd | f64 | 256x256 | 43.220 ms | 8.54 | 0.02 | 80% | ok |
| eigen | expm | f64 | 256x256 | 57.335 ms | 4.39 | 0.02 | 41% | ok |
| eigen | sylvester | f64 | 256x256 | 253.673 ms | 3.97 | 0.01 | 37% | ok |
| elementwise | exp | f64 | n=8388608 | 75.836 ms | 0.11 | 1.77 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.342 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 178.313 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.075 ms | 0.15 | 2.44 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 126.498 ms | 0.07 | 1.06 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.279 ms | 1.15 | 11.95 | 74% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.085 ms | 0.85 | 8.83 | 55% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.136 ms | 2.44 | 5.14 | 32% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.222 ms | 0.16 | 0.57 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.891 ms | 0.17 | 1.40 | 9% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 280.496 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 101.423 ms | 0.11 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.396 ms | 0.30 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 130.286 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 189.415 ms | 0.12 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 61.944 ms | 0.36 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 291.222 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.182 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.421 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.138 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 67.378 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 105.576 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 147.448 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.008 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.451 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.700 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.607 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.655 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.107 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.799 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.822 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.664 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.629 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 68.485 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.166 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.683 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 45.083 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.494 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.229 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.090 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 3.939 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.003 ms | 0.87 | 9.77 | 60% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.735 ms | 0.70 | 7.85 | 49% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.502 ms | 0.58 | 6.51 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 44.941 ms | 0.86 | 6.47 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 351.388 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 356.687 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 194.035 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 21.238 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 12.499 ms | 1.34 | 5.37 | 33% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 75.646 ms | 3.55 | 0.06 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 4.557 ms | 3.68 | 3.91 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.600 ms | 3.00 | 11.98 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.687 ms | 6.24 | 12.49 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.642 ms | 2.97 | 2.97 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.757 ms | 9.55 | 9.55 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 38.114 ms | 7.04 | 0.08 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 41.110 ms | 6.53 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 31.870 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 31.862 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 60.142 ms | 4.46 | 0.03 | n/a | ok |

