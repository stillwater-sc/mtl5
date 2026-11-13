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
| blas1 | 10 | 73% | 131% | 0 |
| blas2 | 10 | 57% | 123% | 0 |
| blas3 | 10 | 50% | 61% | 0 |
| factor | 12 | 13% | 33% | 0 |
| eigen | 8 | 16% | 32% | 0 |
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
| mixed | 3 | 66% | 119% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.586 ms | 1.45 | 11.58 | 71% | ok |
| blas1 | dot | f32 | n=8388608 | 5.642 ms | 2.97 | 11.89 | 73% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.240 ms | 1.37 | 16.45 | 101% | ok |
| blas1 | axpy | f32 | n=8388608 | 5.974 ms | 2.81 | 16.85 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 6.329 ms | 1.33 | 21.21 | 131% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.590 ms | 3.00 | 12.01 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.502 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.751 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.501 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 192.103 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 14.478 ms | 2.32 | 9.27 | 57% | ok |
| blas2 | gemv | f32 | n=4096 | 6.275 ms | 5.35 | 10.70 | 66% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 14.015 ms | 2.39 | 9.58 | 59% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.115 ms | 5.49 | 10.98 | 68% | ok |
| blas2 | symv | f64 | n=4096 | 21.348 ms | 1.57 | 3.15 | 19% | ok |
| blas2 | trmv | f64 | n=4096 | 9.112 ms | 1.84 | 7.37 | 45% | ok |
| blas2 | trmv | f32 | n=4096 | 4.265 ms | 3.93 | 7.88 | 49% | ok |
| blas2 | trsv | f64 | n=4096 | 9.139 ms | 1.84 | 7.35 | 45% | ok |
| blas2 | trsv | f32 | n=4096 | 4.219 ms | 3.98 | 7.96 | 49% | ok |
| blas2 | ger | f64 | n=4096 | 13.403 ms | 2.50 | 20.03 | 123% | ok |
| blas3 | gemm | f64 | 512x512x512 | 41.437 ms | 6.48 | 0.15 | 60% | ok |
| blas3 | gemm | f32 | 512x512x512 | 20.480 ms | 13.11 | 0.15 | 61% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 25.787 ms | 5.20 | 0.24 | 48% | ok |
| blas3 | symm | f64 | 512x512x512 | 42.556 ms | 6.31 | 0.15 | 59% | ok |
| blas3 | syrk | f64 | 512x512x512 | 49.517 ms | 2.71 | 0.08 | 25% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 92.068 ms | 2.92 | 0.07 | 27% | ok |
| blas3 | trmm | f64 | 512x512x512 | 24.897 ms | 5.39 | 0.21 | 50% | ok |
| blas3 | trsm | f64 | 512x512x512 | 25.765 ms | 5.21 | 0.20 | 49% | ok |
| blas3 | trsm | f32 | 512x512x512 | 12.534 ms | 10.71 | 0.21 | 50% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.471 ms | 0.56 | 4.48 | 28% | ok |
| factor | lu | f64 | 512x512 | 35.774 ms | 2.50 | 0.12 | 23% | ok |
| factor | lu | f32 | 512x512 | 20.324 ms | 4.40 | 0.10 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 34.234 ms | 1.31 | 0.12 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 30.156 ms | 1.48 | 0.07 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 34.631 ms | 1.29 | 0.12 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 84.417 ms | 0.53 | 0.05 | 5% | ok |
| factor | qr | f64 | 512x512 | 333.546 ms | 0.54 | 0.01 | 5% | ok |
| factor | lq | f64 | 512x512 | 93.457 ms | 1.91 | 0.04 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 470.722 ms | 0.95 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 1.987 ms | 0.53 | 2.11 | 13% | ok |
| factor | inv | f64 | 512x512 | 193.277 ms | 1.39 | 0.02 | 13% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 25.348 ms | 3.53 | 0.17 | 33% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.584 ms | 0.36 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 145.543 ms | 1.04 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 959.519 ms | 0.16 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 98.704 ms | 1.70 | 0.01 | 16% | ok |
| eigen | schur | f64 | 256x256 | 214.856 ms | 1.95 | 0.00 | 18% | ok |
| eigen | svd | f64 | 256x256 | 359.355 ms | 1.03 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 72.312 ms | 3.48 | 0.01 | 32% | ok |
| eigen | sylvester | f64 | 256x256 | 514.319 ms | 1.96 | 0.00 | 18% | ok |
| elementwise | exp | f64 | n=8388608 | 76.097 ms | 0.11 | 1.76 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.284 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 178.826 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.223 ms | 0.15 | 2.43 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 126.726 ms | 0.07 | 1.06 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.302 ms | 1.14 | 11.83 | 73% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.056 ms | 0.86 | 8.91 | 55% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.492 ms | 2.39 | 5.03 | 31% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 22.503 ms | 0.14 | 0.51 | 3% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.634 ms | 0.17 | 1.43 | 9% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 281.322 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 90.360 ms | 0.12 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.298 ms | 0.30 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 135.444 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 191.855 ms | 0.12 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 61.964 ms | 0.36 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 291.542 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.051 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.046 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.125 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 65.968 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 8.808 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 270.330 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.187 ms | 0.35 | 1.38 | 9% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 18.227 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.586 ms | 0.67 | 0.04 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.110 ms | 0.36 | 1.43 | 9% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.663 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 115.466 ms | 0.19 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.607 ms | 0.10 | 0.40 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 14.810 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 165.970 ms | 0.13 | 0.00 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.720 ms | 0.10 | 0.39 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 31.364 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 25398.565 ms | 0.01 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 10.165 ms | 0.33 | 1.33 | 8% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 65.422 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 278.297 ms | 1.11 | 0.00 | 10% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.341 ms | 0.33 | 1.30 | 8% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.083 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1898.672 ms | 0.33 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 37.383 ms | 0.09 | 0.36 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 41.818 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 3180.967 ms | 0.19 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 37.497 ms | 0.09 | 0.36 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.245 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 148.413 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.176 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.614 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.752 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.640 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.227 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.127 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 50.113 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.689 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.627 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.574 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 68.169 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.166 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.993 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 45.377 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.491 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.245 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.090 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.370 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 7.633 ms | 0.34 | 3.84 | 24% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.744 ms | 0.70 | 7.83 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.519 ms | 0.58 | 6.49 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 44.919 ms | 0.86 | 6.47 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 353.131 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 414.149 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 198.071 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.863 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 6.263 ms | 2.68 | 10.72 | 66% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 64.160 ms | 4.18 | 0.07 | 39% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.310 ms | 12.80 | 13.60 | 119% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.620 ms | 2.99 | 11.94 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.713 ms | 6.18 | 12.37 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.227 ms | 13.67 | 13.67 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.348 ms | 7.15 | 7.15 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 26.048 ms | 10.31 | 0.12 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 27.843 ms | 9.64 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 33.419 ms | 8.03 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 8.584 ms | 31.27 | 0.18 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 23.388 ms | 11.48 | 0.07 | n/a | ok |

