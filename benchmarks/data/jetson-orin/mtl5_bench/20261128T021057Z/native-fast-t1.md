## jetson-orin -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 10.7 GFLOP/s, f32 21.5 GFLOP/s |
| ... hardware, NEON | f64 10.7, f32 21.5 GFLOP/s |
| ... this build, NEON | f64 10.7, f32 21.5 GFLOP/s |
| bandwidth | read 11.5, update 16.1, triad 14.7 GB/s (ceiling 16.1) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 74% | 131% | 0 |
| blas2 | 10 | 59% | 124% | 0 |
| blas3 | 10 | 51% | 61% | 0 |
| factor | 12 | 13% | 32% | 0 |
| eigen | 8 | 16% | 33% | 0 |
| elementwise | 5 | 8% | 15% | 0 |
| sparse | 5 | 31% | 74% | 0 |
| direct | 19 | 2% | 9% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 4% | 11% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 49% | 62% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 67% | 107% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.597 ms | 1.45 | 11.57 | 72% | ok |
| blas1 | dot | f32 | n=8388608 | 5.663 ms | 2.96 | 11.85 | 74% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.256 ms | 1.37 | 16.43 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 6.011 ms | 2.79 | 16.75 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 6.365 ms | 1.32 | 21.09 | 131% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.626 ms | 2.98 | 11.93 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.501 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.751 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.501 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 192.154 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 14.254 ms | 2.35 | 9.42 | 59% | ok |
| blas2 | gemv | f32 | n=4096 | 6.620 ms | 5.07 | 10.14 | 63% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 13.642 ms | 2.46 | 9.84 | 61% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.127 ms | 5.48 | 10.96 | 68% | ok |
| blas2 | symv | f64 | n=4096 | 23.977 ms | 1.40 | 2.80 | 17% | ok |
| blas2 | trmv | f64 | n=4096 | 8.935 ms | 1.88 | 7.52 | 47% | ok |
| blas2 | trmv | f32 | n=4096 | 4.199 ms | 4.00 | 8.00 | 50% | ok |
| blas2 | trsv | f64 | n=4096 | 9.013 ms | 1.86 | 7.45 | 46% | ok |
| blas2 | trsv | f32 | n=4096 | 4.192 ms | 4.00 | 8.01 | 50% | ok |
| blas2 | ger | f64 | n=4096 | 13.462 ms | 2.49 | 19.94 | 124% | ok |
| blas3 | gemm | f64 | 512x512x512 | 41.265 ms | 6.51 | 0.15 | 61% | ok |
| blas3 | gemm | f32 | 512x512x512 | 20.424 ms | 13.14 | 0.15 | 61% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 25.718 ms | 5.22 | 0.24 | 49% | ok |
| blas3 | symm | f64 | 512x512x512 | 42.108 ms | 6.37 | 0.15 | 59% | ok |
| blas3 | syrk | f64 | 512x512x512 | 24.502 ms | 5.48 | 0.17 | 51% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 49.105 ms | 5.47 | 0.13 | 51% | ok |
| blas3 | trmm | f64 | 512x512x512 | 24.777 ms | 5.42 | 0.21 | 50% | ok |
| blas3 | trsm | f64 | 512x512x512 | 25.781 ms | 5.21 | 0.20 | 49% | ok |
| blas3 | trsm | f32 | 512x512x512 | 12.499 ms | 10.74 | 0.21 | 50% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.720 ms | 0.55 | 4.37 | 27% | ok |
| factor | lu | f64 | 512x512 | 35.474 ms | 2.52 | 0.12 | 24% | ok |
| factor | lu | f32 | 512x512 | 20.603 ms | 4.34 | 0.10 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 31.863 ms | 1.40 | 0.13 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 30.141 ms | 1.48 | 0.07 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 32.400 ms | 1.38 | 0.13 | 13% | ok |
| factor | ldlt_bk | f64 | 512x512 | 16.302 ms | 2.74 | 0.26 | 26% | ok |
| factor | qr | f64 | 512x512 | 332.536 ms | 0.54 | 0.01 | 5% | ok |
| factor | lq | f64 | 512x512 | 93.150 ms | 1.92 | 0.05 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 477.072 ms | 0.94 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 2.084 ms | 0.50 | 2.01 | 12% | ok |
| factor | inv | f64 | 512x512 | 192.689 ms | 1.39 | 0.02 | 13% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 25.653 ms | 3.49 | 0.16 | 32% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.567 ms | 0.36 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 145.073 ms | 1.04 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 900.521 ms | 0.17 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 99.142 ms | 1.69 | 0.01 | 16% | ok |
| eigen | schur | f64 | 256x256 | 215.088 ms | 1.95 | 0.00 | 18% | ok |
| eigen | svd | f64 | 256x256 | 358.878 ms | 1.03 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 71.358 ms | 3.53 | 0.01 | 33% | ok |
| eigen | sylvester | f64 | 256x256 | 518.668 ms | 1.94 | 0.00 | 18% | ok |
| elementwise | exp | f64 | n=8388608 | 75.996 ms | 0.11 | 1.77 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.258 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 178.265 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.270 ms | 0.15 | 2.43 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 126.623 ms | 0.07 | 1.06 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.280 ms | 1.15 | 11.95 | 74% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.068 ms | 0.85 | 8.88 | 55% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.605 ms | 2.38 | 5.00 | 31% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.235 ms | 0.16 | 0.57 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.707 ms | 0.17 | 1.42 | 9% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 54.448 ms | 0.21 | 0.01 | 2% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 97.764 ms | 0.11 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 35.936 ms | 0.31 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 135.958 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 156.662 ms | 0.14 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 72.369 ms | 0.31 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 106.164 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 11.739 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 16.654 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 178.682 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.839 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 64.442 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 7.662 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 43.982 ms | 0.25 | 0.02 | 2% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.196 ms | 0.34 | 1.38 | 9% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 16.378 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.650 ms | 0.67 | 0.04 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.093 ms | 0.36 | 1.45 | 9% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.479 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 115.546 ms | 0.19 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.570 ms | 0.10 | 0.40 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 12.137 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 132.033 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.574 ms | 0.10 | 0.40 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 16.741 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 845.861 ms | 0.36 | 0.00 | 3% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 9.842 ms | 0.34 | 1.37 | 9% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 36.590 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 257.386 ms | 1.20 | 0.00 | 11% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.118 ms | 0.33 | 1.33 | 8% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 15.889 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1887.827 ms | 0.33 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.598 ms | 0.09 | 0.37 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 23.040 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1539.188 ms | 0.40 | 0.00 | 4% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.481 ms | 0.09 | 0.37 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.377 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 167.675 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.017 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.615 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 51.033 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.874 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 88.188 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.446 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.886 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 57.485 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.519 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 50.798 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 59.282 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.166 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.855 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 45.752 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.492 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.309 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.112 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.382 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.937 ms | 0.89 | 9.98 | 62% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.739 ms | 0.70 | 7.84 | 49% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.525 ms | 0.58 | 6.48 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 44.839 ms | 0.86 | 6.48 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 351.977 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 354.950 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 190.831 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.688 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 6.260 ms | 2.68 | 10.72 | 67% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 64.101 ms | 4.19 | 0.07 | 39% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.468 ms | 11.43 | 12.14 | 107% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.621 ms | 2.98 | 11.94 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.724 ms | 6.16 | 12.32 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.241 ms | 13.52 | 13.52 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.347 ms | 7.15 | 7.15 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 26.151 ms | 10.26 | 0.12 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 27.159 ms | 9.88 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 33.210 ms | 8.08 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 8.554 ms | 31.38 | 0.18 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 23.388 ms | 11.48 | 0.07 | n/a | ok |

