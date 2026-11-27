## arduino-ventuno-q -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON: -march=armv8.2-a+dotprod+fp16+rcpc -mtune=cortex-a78c]`, SIMD NEON_WITHOUT_AES |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.6 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 16.8, f32 33.6 GFLOP/s |
| bandwidth | read 22.3, update 24.7, triad 23.1 GB/s (ceiling 24.7) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 94% | 104% | 0 |
| blas2 | 10 | 65% | 72% | 0 |
| blas3 | 10 | 48% | 58% | 0 |
| factor | 12 | 12% | 31% | 0 |
| eigen | 8 | 15% | 32% | 0 |
| elementwise | 5 | 5% | 7% | 0 |
| sparse | 5 | 30% | 67% | 0 |
| direct | 19 | 2% | 7% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 3% | 10% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 58% | 0 |
| mg | 1 | 36% | 36% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 67% | 133% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.729 ms | 2.93 | 23.43 | 95% | ok |
| blas1 | dot | f32 | n=8388608 | 2.884 ms | 5.82 | 23.27 | 94% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.832 ms | 2.14 | 25.71 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.960 ms | 4.24 | 25.42 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 5.315 ms | 1.58 | 25.25 | 102% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.156 ms | 5.32 | 21.27 | 86% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.693 ms | 0.96 | 7.72 | 31% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.642 ms | 0.48 | 3.80 | 15% | ok |
| blas1 | sum | f64 | n=8388608 | 8.520 ms | 0.98 | 7.88 | 32% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 200.676 ms | 0.92 | 0.67 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 8.336 ms | 4.03 | 16.11 | 65% | ok |
| blas2 | gemv | f32 | n=4096 | 4.081 ms | 8.22 | 16.45 | 67% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.820 ms | 4.29 | 17.17 | 70% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.023 ms | 8.34 | 16.69 | 68% | ok |
| blas2 | symv | f64 | n=4096 | 11.651 ms | 2.88 | 5.77 | 23% | ok |
| blas2 | trmv | f64 | n=4096 | 4.947 ms | 3.39 | 13.58 | 55% | ok |
| blas2 | trmv | f32 | n=4096 | 2.701 ms | 6.21 | 12.44 | 50% | ok |
| blas2 | trsv | f64 | n=4096 | 4.767 ms | 3.52 | 14.09 | 57% | ok |
| blas2 | trsv | f32 | n=4096 | 2.741 ms | 6.12 | 12.25 | 50% | ok |
| blas2 | ger | f64 | n=4096 | 15.173 ms | 2.21 | 17.70 | 72% | ok |
| blas3 | gemm | f64 | 512x512x512 | 28.428 ms | 9.44 | 0.22 | 56% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.787 ms | 19.47 | 0.23 | 58% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 18.050 ms | 7.44 | 0.35 | 44% | ok |
| blas3 | symm | f64 | 512x512x512 | 29.199 ms | 9.19 | 0.22 | 55% | ok |
| blas3 | syrk | f64 | 512x512x512 | 16.754 ms | 8.01 | 0.25 | 48% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 32.987 ms | 8.14 | 0.19 | 48% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.623 ms | 7.62 | 0.30 | 45% | ok |
| blas3 | trsm | f64 | 512x512x512 | 17.978 ms | 7.47 | 0.29 | 44% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.506 ms | 15.78 | 0.31 | 47% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 32.400 ms | 0.16 | 1.31 | 5% | ok |
| factor | lu | f64 | 512x512 | 26.429 ms | 3.39 | 0.16 | 20% | ok |
| factor | lu | f32 | 512x512 | 13.440 ms | 6.66 | 0.16 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 22.472 ms | 1.99 | 0.19 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 21.010 ms | 2.13 | 0.10 | 6% | ok |
| factor | ldlt | f64 | 512x512 | 21.646 ms | 2.07 | 0.19 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 11.212 ms | 3.99 | 0.37 | 24% | ok |
| factor | qr | f64 | 512x512 | 246.672 ms | 0.73 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 61.103 ms | 2.93 | 0.07 | 17% | ok |
| factor | hessenberg | f64 | 512x512 | 342.069 ms | 1.31 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.534 ms | 0.68 | 2.73 | 11% | ok |
| factor | inv | f64 | 512x512 | 145.011 ms | 1.85 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.974 ms | 5.27 | 0.25 | 31% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.918 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.261 ms | 1.55 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 605.460 ms | 0.25 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.977 ms | 2.47 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 151.401 ms | 2.77 | 0.01 | 16% | ok |
| eigen | svd | f64 | 256x256 | 230.204 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 47.480 ms | 5.30 | 0.02 | 32% | ok |
| eigen | sylvester | f64 | 256x256 | 358.914 ms | 2.80 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 90.251 ms | 0.09 | 1.49 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 111.208 ms | 0.08 | 1.21 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 150.918 ms | 0.06 | 0.89 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 73.338 ms | 0.11 | 1.83 | 7% | ok |
| elementwise | erf | f64 | n=8388608 | 119.184 ms | 0.07 | 1.13 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.637 ms | 1.60 | 16.63 | 67% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.111 ms | 1.24 | 12.90 | 52% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.975 ms | 3.50 | 7.35 | 30% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.209 ms | 0.25 | 0.87 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.746 ms | 0.23 | 1.90 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 37.280 ms | 0.30 | 0.02 | 2% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 66.207 ms | 0.17 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.388 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 103.420 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 122.294 ms | 0.18 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 42.682 ms | 0.52 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 79.393 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.404 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.424 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 125.841 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.568 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.129 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.622 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 29.737 ms | 0.38 | 0.02 | 2% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.840 ms | 0.41 | 1.64 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.301 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.998 ms | 0.93 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.771 ms | 0.43 | 1.71 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.487 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 88.051 ms | 0.25 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.736 ms | 0.13 | 0.53 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 8.848 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 103.739 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.851 ms | 0.13 | 0.52 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 12.232 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 666.020 ms | 0.46 | 0.00 | 3% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.415 ms | 0.45 | 1.82 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.212 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 187.096 ms | 1.65 | 0.00 | 10% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.353 ms | 0.46 | 1.83 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.705 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1768.966 ms | 0.35 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.165 ms | 0.12 | 0.46 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.900 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1235.380 ms | 0.50 | 0.00 | 3% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 30.112 ms | 0.11 | 0.45 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.920 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 162.378 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 54.174 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 29.224 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.696 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 34.211 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 62.440 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.724 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.705 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 39.409 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.823 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 34.713 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 40.814 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.110 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.961 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 31.213 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.114 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 3.013 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.066 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.742 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.064 ms | 1.27 | 14.21 | 58% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.481 ms | 1.05 | 11.82 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.933 ms | 0.89 | 10.00 | 41% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.334 ms | 1.19 | 8.99 | 36% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 235.461 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 230.707 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 146.331 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.158 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.057 ms | 4.14 | 16.54 | 67% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 41.753 ms | 6.43 | 0.10 | 38% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.749 ms | 22.40 | 23.80 | 133% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.857 ms | 5.87 | 23.49 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.416 ms | 11.85 | 23.70 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.699 ms | 24.01 | 24.01 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.543 ms | 10.87 | 10.87 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.420 ms | 15.41 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 16.688 ms | 16.09 | 0.13 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 16.576 ms | 16.19 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.678 ms | 72.99 | 0.43 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.139 ms | 20.43 | 0.12 | n/a | ok |

