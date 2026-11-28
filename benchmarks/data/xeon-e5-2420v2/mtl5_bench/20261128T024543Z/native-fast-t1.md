## xeon-e5-2420v2 -- native-fast, 1 thread, class large

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 21.0 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 21.0 GFLOP/s |
| ... this build, SSE4 | f64 10.5, f32 21.0 GFLOP/s |
| bandwidth | read 10.4, update 10.3, triad 8.1 GB/s (ceiling 10.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 96% | 108% | 0 |
| blas2 | 10 | 91% | 98% | 0 |
| blas3 | 10 | 82% | 95% | 0 |
| factor | 12 | 17% | 49% | 0 |
| eigen | 8 | 11% | 46% | 0 |
| elementwise | 5 | 8% | 14% | 0 |
| sparse | 5 | 55% | 99% | 0 |
| direct | 19 | 4% | 17% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 15% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 75% | 83% | 0 |
| mg | 1 | 62% | 62% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 72% | 89% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=33554432 | 47.785 ms | 1.40 | 11.24 | 108% | ok |
| blas1 | dot | f32 | n=33554432 | 23.885 ms | 2.81 | 11.24 | 108% | ok |
| blas1 | axpy | f64 | n=33554432 | 80.456 ms | 0.83 | 10.01 | 96% | ok |
| blas1 | axpy | f32 | n=33554432 | 38.287 ms | 1.75 | 10.52 | 101% | ok |
| blas1 | scale | f64 | n=33554432 | 56.327 ms | 0.60 | 9.53 | 91% | ok |
| blas1 | two_norm | f64 | n=33554432 | 24.123 ms | 2.78 | 11.13 | 107% | ok |
| blas1 | one_norm | f64 | n=33554432 | 39.199 ms | 0.86 | 6.85 | 66% | ok |
| blas1 | infinity_norm | f64 | n=33554432 | 52.227 ms | 0.64 | 5.14 | 49% | ok |
| blas1 | sum | f64 | n=33554432 | 39.135 ms | 0.86 | 6.86 | 66% | ok |
| blas1 | dot_k3 | f64 | n=33554432 | 1089.589 ms | 0.68 | 0.49 | 6% | ok |
| blas2 | gemv | f64 | n=8192 | 55.310 ms | 2.43 | 9.71 | 93% | ok |
| blas2 | gemv | f32 | n=8192 | 27.204 ms | 4.93 | 9.87 | 95% | ok |
| blas2 | gemv_colmajor | f64 | n=8192 | 52.270 ms | 2.57 | 10.27 | 98% | ok |
| blas2 | gemv_colmajor | f32 | n=8192 | 26.215 ms | 5.12 | 10.24 | 98% | ok |
| blas2 | symv | f64 | n=8192 | 53.794 ms | 2.50 | 4.99 | 48% | ok |
| blas2 | trmv | f64 | n=8192 | 28.740 ms | 2.33 | 9.34 | 90% | ok |
| blas2 | trmv | f32 | n=8192 | 15.077 ms | 4.45 | 8.91 | 85% | ok |
| blas2 | trsv | f64 | n=8192 | 29.114 ms | 2.31 | 9.22 | 88% | ok |
| blas2 | trsv | f32 | n=8192 | 14.115 ms | 4.75 | 9.51 | 91% | ok |
| blas2 | ger | f64 | n=8192 | 119.910 ms | 1.12 | 8.96 | 86% | ok |
| blas3 | gemm | f64 | 1024x1024x1024 | 225.315 ms | 9.53 | 0.11 | 91% | ok |
| blas3 | gemm | f32 | 1024x1024x1024 | 107.910 ms | 19.90 | 0.12 | 95% | ok |
| blas3 | gemmt | f64 | 1024x1024x1024 | 131.375 ms | 8.17 | 0.19 | 78% | ok |
| blas3 | symm | f64 | 1024x1024x1024 | 233.407 ms | 9.20 | 0.11 | 88% | ok |
| blas3 | syrk | f64 | 1024x1024x1024 | 124.320 ms | 8.64 | 0.13 | 82% | ok |
| blas3 | syr2k | f64 | 1024x1024x1024 | 247.051 ms | 8.69 | 0.10 | 83% | ok |
| blas3 | trmm | f64 | 1024x1024x1024 | 149.868 ms | 7.16 | 0.14 | 68% | ok |
| blas3 | trsm | f64 | 1024x1024x1024 | 153.427 ms | 7.00 | 0.14 | 67% | ok |
| blas3 | trsm | f32 | 1024x1024x1024 | 70.131 ms | 15.31 | 0.15 | 73% | ok |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 129.771 ms | 0.13 | 1.03 | 10% | ok |
| factor | lu | f64 | 1024x1024 | 234.316 ms | 3.05 | 0.07 | 29% | ok |
| factor | lu | f32 | 1024x1024 | 123.736 ms | 5.79 | 0.07 | 28% | ok |
| factor | cholesky | f64 | 1024x1024 | 208.844 ms | 1.71 | 0.08 | 16% | ok |
| factor | cholesky | f32 | 1024x1024 | 202.236 ms | 1.77 | 0.04 | 8% | ok |
| factor | ldlt | f64 | 1024x1024 | 214.714 ms | 1.67 | 0.08 | 16% | ok |
| factor | ldlt_bk | f64 | 1024x1024 | 108.226 ms | 3.31 | 0.16 | 31% | ok |
| factor | qr | f64 | 1024x1024 | 6916.868 ms | 0.21 | 0.00 | 2% | ok |
| factor | lq | f64 | 1024x1024 | 574.572 ms | 2.49 | 0.03 | 24% | ok |
| factor | hessenberg | f64 | 1024x1024 | 7781.264 ms | 0.46 | 0.00 | 4% | ok |
| factor | householder | f64 | 1024x1024 | 20.421 ms | 0.21 | 0.82 | 8% | ok |
| factor | inv | f64 | 1024x1024 | 1193.836 ms | 1.80 | 0.01 | 17% | ok |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 140.286 ms | 5.10 | 0.12 | 49% | ok |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 643.073 ms | 0.28 | 0.01 | 3% | ok |
| eigen | eigen_symmetric | f64 | 512x512 | 2105.702 ms | 0.57 | 0.00 | 5% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 14926.777 ms | 0.08 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 512x512 | 1081.943 ms | 1.24 | 0.00 | 12% | ok |
| eigen | schur | f64 | 512x512 | 2972.156 ms | 1.13 | 0.00 | 11% | ok |
| eigen | svd | f64 | 512x512 | 2346.058 ms | 1.26 | 0.00 | 12% | ok |
| eigen | expm | f64 | 512x512 | 413.426 ms | 4.87 | 0.01 | 46% | ok |
| eigen | sylvester | f64 | 512x512 | 6751.524 ms | 1.19 | 0.00 | 11% | ok |
| elementwise | exp | f64 | n=33554432 | 606.301 ms | 0.06 | 0.89 | 8% | ok |
| elementwise | log | f64 | n=33554432 | 562.224 ms | 0.06 | 0.95 | 9% | ok |
| elementwise | sin | f64 | n=33554432 | 683.887 ms | 0.05 | 0.79 | 8% | ok |
| elementwise | sqrt | f64 | n=33554432 | 362.747 ms | 0.09 | 1.48 | 14% | ok |
| elementwise | erf | f64 | n=33554432 | 9962.988 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 10.526 ms | 1.00 | 10.35 | 99% | ok |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 12.975 ms | 0.81 | 8.40 | 81% | ok |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 61.607 ms | 2.72 | 5.72 | 55% | ok |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 99.246 ms | 0.13 | 0.46 | 4% | ok |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 38.764 ms | 0.26 | 2.16 | 21% | ok |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 307.801 ms | 0.35 | 0.01 | 3% | ok |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 561.830 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 158.502 ms | 0.67 | 0.02 | 6% | ok |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 904.442 ms | 0.24 | 0.00 | 2% | ok |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 931.293 ms | 0.23 | 0.00 | 2% | ok |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 371.070 ms | 0.58 | 0.01 | 5% | ok |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 1148.416 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 44.409 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 65.931 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 592.740 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 2.523 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 251.412 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 27.597 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 269.494 ms | 0.40 | 0.01 | 4% | ok |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 9.077 ms | 0.44 | 1.77 | 17% | ok |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 55.938 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 93.460 ms | 1.14 | 0.03 | 11% | ok |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 9.087 ms | 0.44 | 1.77 | 17% | ok |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 27.392 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 830.421 ms | 0.26 | 0.00 | 2% | ok |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.316 ms | 0.13 | 0.53 | 5% | ok |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 41.932 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 848.375 ms | 0.25 | 0.00 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.348 ms | 0.13 | 0.53 | 5% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 24.267 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1947.968 ms | 0.50 | 0.00 | 5% | ok |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 19.299 ms | 0.39 | 1.55 | 15% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 62.455 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 602.435 ms | 1.61 | 0.00 | 15% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 19.345 ms | 0.39 | 1.55 | 15% | ok |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 24.518 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 5722.516 ms | 0.34 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 87.805 ms | 0.09 | 0.34 | 3% | ok |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 34.326 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 6544.543 ms | 0.30 | 0.00 | 3% | ok |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 87.622 ms | 0.09 | 0.34 | 3% | ok |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 635.484 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 983.530 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 369.403 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 288.432 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 305.520 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 280.033 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 649.440 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 509.773 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 306.991 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 329.885 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 292.630 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 301.687 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 372.637 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.529 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 256^2 grid | 19.057 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 256^2 grid | 149.982 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 256^2 grid | 12.059 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 256^2 grid | 11.538 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 256^2 grid | 0.366 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 8.667 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 14.912 ms | 0.70 | 7.87 | 75% | ok |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 13.489 ms | 0.78 | 8.70 | 83% | ok |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 16.066 ms | 0.65 | 7.31 | 70% | ok |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 180.244 ms | 0.86 | 6.48 | 62% | ok |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 562.135 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 571.854 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 1452.593 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 69.290 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=33554432 | 28.868 ms | 2.32 | 9.30 | 89% | ok |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 497.445 ms | 4.32 | 0.03 | 41% | ok |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 9.550 ms | 7.03 | 7.47 | 72% | ok |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 24.491 ms | 2.74 | 10.96 | n/a | ok |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.079 ms | 5.56 | 11.11 | n/a | ok |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 8.819 ms | 7.61 | 7.61 | n/a | ok |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 7.405 ms | 9.06 | 9.06 | n/a | ok |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 150.919 ms | 14.23 | 0.08 | n/a | ok |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 176.491 ms | 12.17 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 175.623 ms | 12.23 | 0.04 | n/a | ok |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 128.375 ms | 16.73 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 97.446 ms | 22.04 | 0.06 | n/a | ok |

