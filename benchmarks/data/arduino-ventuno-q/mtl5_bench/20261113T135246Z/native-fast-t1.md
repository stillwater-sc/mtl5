## arduino-ventuno-q -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON_WITHOUT_AES |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.6 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 16.8, f32 33.6 GFLOP/s |
| bandwidth | read 22.5, update 24.5, triad 23.0 GB/s (ceiling 24.5) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 95% | 104% | 0 |
| blas2 | 10 | 68% | 71% | 0 |
| blas3 | 10 | 45% | 58% | 0 |
| factor | 12 | 12% | 32% | 0 |
| eigen | 8 | 15% | 31% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 30% | 67% | 0 |
| direct | 19 | 2% | 7% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 9% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 40% | 57% | 0 |
| mg | 1 | 37% | 37% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 39% | 67% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.745 ms | 2.92 | 23.36 | 95% | ok |
| blas1 | dot | f32 | n=8388608 | 2.886 ms | 5.81 | 23.25 | 95% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.854 ms | 2.14 | 25.63 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.969 ms | 4.23 | 25.36 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 5.352 ms | 1.57 | 25.08 | 102% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.194 ms | 5.25 | 21.01 | 86% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.346 ms | 1.01 | 8.04 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.247 ms | 0.49 | 3.89 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.402 ms | 1.00 | 7.99 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 201.741 ms | 0.91 | 0.67 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 7.981 ms | 4.20 | 16.83 | 69% | ok |
| blas2 | gemv | f32 | n=4096 | 3.995 ms | 8.40 | 16.81 | 68% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.727 ms | 4.34 | 17.38 | 71% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.009 ms | 8.37 | 16.75 | 68% | ok |
| blas2 | symv | f64 | n=4096 | 17.149 ms | 1.96 | 3.92 | 16% | ok |
| blas2 | trmv | f64 | n=4096 | 5.001 ms | 3.35 | 13.43 | 55% | ok |
| blas2 | trmv | f32 | n=4096 | 2.626 ms | 6.39 | 12.79 | 52% | ok |
| blas2 | trsv | f64 | n=4096 | 4.785 ms | 3.51 | 14.04 | 57% | ok |
| blas2 | trsv | f32 | n=4096 | 2.640 ms | 6.35 | 12.72 | 52% | ok |
| blas2 | ger | f64 | n=4096 | 15.711 ms | 2.14 | 17.09 | 70% | ok |
| blas3 | gemm | f64 | 512x512x512 | 28.097 ms | 9.55 | 0.22 | 57% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.830 ms | 19.41 | 0.23 | 58% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 17.902 ms | 7.50 | 0.35 | 45% | ok |
| blas3 | symm | f64 | 512x512x512 | 29.071 ms | 9.23 | 0.22 | 55% | ok |
| blas3 | syrk | f64 | 512x512x512 | 40.878 ms | 3.28 | 0.10 | 20% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 82.410 ms | 3.26 | 0.08 | 19% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.909 ms | 7.49 | 0.29 | 45% | ok |
| blas3 | trsm | f64 | 512x512x512 | 18.002 ms | 7.46 | 0.29 | 44% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.678 ms | 15.47 | 0.30 | 46% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 30.579 ms | 0.17 | 1.39 | 6% | ok |
| factor | lu | f64 | 512x512 | 24.651 ms | 3.63 | 0.17 | 22% | ok |
| factor | lu | f32 | 512x512 | 14.369 ms | 6.23 | 0.15 | 19% | ok |
| factor | cholesky | f64 | 512x512 | 22.357 ms | 2.00 | 0.19 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 19.661 ms | 2.28 | 0.11 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 22.783 ms | 1.96 | 0.18 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 68.602 ms | 0.65 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 251.011 ms | 0.71 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 60.809 ms | 2.94 | 0.07 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 350.209 ms | 1.28 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.559 ms | 0.67 | 2.69 | 11% | ok |
| factor | inv | f64 | 512x512 | 146.684 ms | 1.83 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.711 ms | 5.35 | 0.25 | 32% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.947 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.062 ms | 1.56 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 601.567 ms | 0.25 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.094 ms | 2.50 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 150.044 ms | 2.80 | 0.01 | 17% | ok |
| eigen | svd | f64 | 256x256 | 230.816 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 47.768 ms | 5.27 | 0.02 | 31% | ok |
| eigen | sylvester | f64 | 256x256 | 356.972 ms | 2.82 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 88.870 ms | 0.09 | 1.51 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 107.038 ms | 0.08 | 1.25 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 147.995 ms | 0.06 | 0.91 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 71.034 ms | 0.12 | 1.89 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 117.656 ms | 0.07 | 1.14 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.652 ms | 1.58 | 16.48 | 67% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.187 ms | 1.20 | 12.45 | 51% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.088 ms | 3.46 | 7.28 | 30% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.181 ms | 0.25 | 0.87 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.498 ms | 0.24 | 1.96 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 185.667 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 61.311 ms | 0.18 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.458 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 102.692 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 139.760 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 42.975 ms | 0.52 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 208.197 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.748 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.253 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.696 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.656 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.884 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 178.049 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.880 ms | 0.40 | 1.61 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.960 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.627 ms | 0.96 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.744 ms | 0.43 | 1.73 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.517 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 86.350 ms | 0.26 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.726 ms | 0.13 | 0.53 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.990 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 121.169 ms | 0.18 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.756 ms | 0.13 | 0.53 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 21.793 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 16923.537 ms | 0.02 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.490 ms | 0.45 | 1.80 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 48.297 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 211.238 ms | 1.46 | 0.00 | 9% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.647 ms | 0.44 | 1.76 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.603 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1716.112 ms | 0.36 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.766 ms | 0.12 | 0.47 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 29.791 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2515.802 ms | 0.25 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.923 ms | 0.12 | 0.48 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.458 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 103.320 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 53.681 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 28.898 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.413 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 34.067 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 63.230 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.953 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.663 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 37.458 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.533 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.284 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 46.109 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.112 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.969 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 30.945 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.130 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.984 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.063 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.723 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.085 ms | 1.26 | 14.06 | 57% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 4.673 ms | 0.56 | 6.28 | 26% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.963 ms | 0.88 | 9.90 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.081 ms | 1.20 | 9.06 | 37% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 234.038 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 269.083 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 145.266 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.324 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.103 ms | 4.09 | 16.35 | 67% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 41.467 ms | 6.47 | 0.10 | 39% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.650 ms | 6.33 | 6.73 | 38% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.824 ms | 5.94 | 23.76 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.398 ms | 12.00 | 24.01 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.330 ms | 7.20 | 7.20 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.257 ms | 3.94 | 3.94 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.566 ms | 15.28 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 16.744 ms | 16.03 | 0.13 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 16.597 ms | 16.17 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 23.666 ms | 11.34 | 0.07 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 50.930 ms | 5.27 | 0.03 | n/a | ok |

