## jetson-orin -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 10.7 GFLOP/s, f32 21.5 GFLOP/s |
| ... hardware, NEON | f64 10.7, f32 21.5 GFLOP/s |
| ... this build, NEON | f64 10.7, f32 21.5 GFLOP/s |
| bandwidth | read 11.5, update 16.2, triad 14.8 GB/s (ceiling 16.2) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 74% | 131% | 0 |
| blas2 | 10 | 63% | 124% | 0 |
| blas3 | 10 | 50% | 61% | 0 |
| factor | 12 | 13% | 33% | 0 |
| eigen | 8 | 16% | 33% | 0 |
| elementwise | 5 | 8% | 15% | 0 |
| sparse | 5 | 31% | 73% | 0 |
| direct | 19 | 2% | 9% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 12 | 3% | 10% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 40% | 48% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 66% | 114% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.568 ms | 1.45 | 11.60 | 72% | ok |
| blas1 | dot | f32 | n=8388608 | 5.618 ms | 2.99 | 11.95 | 74% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.290 ms | 1.37 | 16.38 | 101% | ok |
| blas1 | axpy | f32 | n=8388608 | 6.139 ms | 2.73 | 16.40 | 101% | ok |
| blas1 | scale | f64 | n=8388608 | 6.328 ms | 1.33 | 21.21 | 131% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.587 ms | 3.00 | 12.01 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.750 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.501 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 192.234 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 13.186 ms | 2.54 | 10.18 | 63% | ok |
| blas2 | gemv | f32 | n=4096 | 6.262 ms | 5.36 | 10.72 | 66% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 12.922 ms | 2.60 | 10.39 | 64% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.106 ms | 5.49 | 11.00 | 68% | ok |
| blas2 | symv | f64 | n=4096 | 21.322 ms | 1.57 | 3.15 | 19% | ok |
| blas2 | trmv | f64 | n=4096 | 9.212 ms | 1.82 | 7.29 | 45% | ok |
| blas2 | trmv | f32 | n=4096 | 4.238 ms | 3.96 | 7.92 | 49% | ok |
| blas2 | trsv | f64 | n=4096 | 9.200 ms | 1.82 | 7.30 | 45% | ok |
| blas2 | trsv | f32 | n=4096 | 4.167 ms | 4.03 | 8.06 | 50% | ok |
| blas2 | ger | f64 | n=4096 | 13.424 ms | 2.50 | 20.00 | 124% | ok |
| blas3 | gemm | f64 | 512x512x512 | 41.421 ms | 6.48 | 0.15 | 60% | ok |
| blas3 | gemm | f32 | 512x512x512 | 20.478 ms | 13.11 | 0.15 | 61% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 25.765 ms | 5.21 | 0.24 | 49% | ok |
| blas3 | symm | f64 | 512x512x512 | 42.700 ms | 6.29 | 0.15 | 59% | ok |
| blas3 | syrk | f64 | 512x512x512 | 49.565 ms | 2.71 | 0.08 | 25% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 92.383 ms | 2.91 | 0.07 | 27% | ok |
| blas3 | trmm | f64 | 512x512x512 | 24.853 ms | 5.40 | 0.21 | 50% | ok |
| blas3 | trsm | f64 | 512x512x512 | 25.696 ms | 5.22 | 0.20 | 49% | ok |
| blas3 | trsm | f32 | 512x512x512 | 12.511 ms | 10.73 | 0.21 | 50% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.400 ms | 0.56 | 4.52 | 28% | ok |
| factor | lu | f64 | 512x512 | 35.889 ms | 2.49 | 0.12 | 23% | ok |
| factor | lu | f32 | 512x512 | 20.313 ms | 4.40 | 0.10 | 21% | ok |
| factor | cholesky | f64 | 512x512 | 34.130 ms | 1.31 | 0.12 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 30.165 ms | 1.48 | 0.07 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 34.640 ms | 1.29 | 0.12 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 86.881 ms | 0.51 | 0.05 | 5% | ok |
| factor | qr | f64 | 512x512 | 345.483 ms | 0.52 | 0.01 | 5% | ok |
| factor | lq | f64 | 512x512 | 93.147 ms | 1.92 | 0.05 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 481.389 ms | 0.93 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 2.024 ms | 0.52 | 2.07 | 13% | ok |
| factor | inv | f64 | 512x512 | 193.081 ms | 1.39 | 0.02 | 13% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 25.358 ms | 3.53 | 0.17 | 33% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.644 ms | 0.36 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 147.215 ms | 1.03 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 972.344 ms | 0.16 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 99.764 ms | 1.68 | 0.01 | 16% | ok |
| eigen | schur | f64 | 256x256 | 214.338 ms | 1.96 | 0.00 | 18% | ok |
| eigen | svd | f64 | 256x256 | 359.344 ms | 1.03 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 72.083 ms | 3.49 | 0.01 | 33% | ok |
| eigen | sylvester | f64 | 256x256 | 518.701 ms | 1.94 | 0.00 | 18% | ok |
| elementwise | exp | f64 | n=8388608 | 75.997 ms | 0.11 | 1.77 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 107.889 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 178.778 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.182 ms | 0.15 | 2.43 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 126.912 ms | 0.07 | 1.06 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.302 ms | 1.14 | 11.83 | 73% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.049 ms | 0.86 | 8.93 | 55% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.366 ms | 2.41 | 5.07 | 31% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 22.110 ms | 0.15 | 0.52 | 3% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.648 ms | 0.17 | 1.43 | 9% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 283.370 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 90.206 ms | 0.12 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.379 ms | 0.30 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 135.427 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 193.648 ms | 0.12 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 62.594 ms | 0.36 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 290.388 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.033 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.103 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.134 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 66.323 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 8.779 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 272.060 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.211 ms | 0.34 | 1.37 | 8% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 18.206 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.582 ms | 0.67 | 0.04 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.108 ms | 0.36 | 1.43 | 9% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.653 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 115.678 ms | 0.19 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.717 ms | 0.10 | 0.39 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 14.751 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 166.216 ms | 0.13 | 0.00 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.687 ms | 0.10 | 0.39 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 31.218 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 25372.887 ms | 0.01 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 10.205 ms | 0.33 | 1.32 | 8% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 65.294 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 281.201 ms | 1.10 | 0.00 | 10% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.319 ms | 0.33 | 1.31 | 8% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.054 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1899.254 ms | 0.32 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 37.336 ms | 0.09 | 0.36 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 41.828 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 3187.065 ms | 0.19 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 37.382 ms | 0.09 | 0.36 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.463 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 149.144 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.299 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.616 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.856 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.778 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.252 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.156 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 50.096 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.885 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.703 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.567 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 68.479 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.166 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 7.021 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 45.535 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.480 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.276 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.090 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.395 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 7.528 ms | 0.35 | 3.90 | 24% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.740 ms | 0.70 | 7.84 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.518 ms | 0.58 | 6.49 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 44.895 ms | 0.86 | 6.47 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 358.994 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 419.220 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 193.903 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.773 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 6.260 ms | 2.68 | 10.72 | 66% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 63.958 ms | 4.20 | 0.07 | 39% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.370 ms | 12.25 | 13.01 | 114% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.625 ms | 2.98 | 11.93 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.699 ms | 6.22 | 12.43 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.229 ms | 13.65 | 13.65 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.347 ms | 7.15 | 7.15 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 25.968 ms | 10.34 | 0.12 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 27.830 ms | 9.65 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 33.382 ms | 8.04 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 8.565 ms | 31.34 | 0.18 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 23.390 ms | 11.48 | 0.07 | n/a | ok |

