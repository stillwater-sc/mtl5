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
| bandwidth | read 11.5, update 16.1, triad 14.8 GB/s (ceiling 16.1) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 74% | 131% | 0 |
| blas2 | 10 | 60% | 124% | 0 |
| blas3 | 10 | 51% | 61% | 0 |
| factor | 12 | 13% | 33% | 0 |
| eigen | 8 | 16% | 33% | 0 |
| elementwise | 5 | 8% | 15% | 0 |
| sparse | 5 | 32% | 74% | 0 |
| direct | 19 | 2% | 9% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 4% | 11% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 63% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 66% | 117% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.608 ms | 1.45 | 11.56 | 72% | ok |
| blas1 | dot | f32 | n=8388608 | 5.656 ms | 2.97 | 11.87 | 74% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.248 ms | 1.37 | 16.44 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 5.989 ms | 2.80 | 16.81 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 6.356 ms | 1.32 | 21.12 | 131% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.607 ms | 2.99 | 11.97 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.499 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.751 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.502 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 192.412 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 13.689 ms | 2.45 | 9.81 | 61% | ok |
| blas2 | gemv | f32 | n=4096 | 6.151 ms | 5.46 | 10.92 | 68% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 13.891 ms | 2.42 | 9.67 | 60% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.218 ms | 5.40 | 10.80 | 67% | ok |
| blas2 | symv | f64 | n=4096 | 24.667 ms | 1.36 | 2.72 | 17% | ok |
| blas2 | trmv | f64 | n=4096 | 9.371 ms | 1.79 | 7.17 | 44% | ok |
| blas2 | trmv | f32 | n=4096 | 4.256 ms | 3.94 | 7.89 | 49% | ok |
| blas2 | trsv | f64 | n=4096 | 9.292 ms | 1.81 | 7.23 | 45% | ok |
| blas2 | trsv | f32 | n=4096 | 4.154 ms | 4.04 | 8.08 | 50% | ok |
| blas2 | ger | f64 | n=4096 | 13.425 ms | 2.50 | 20.00 | 124% | ok |
| blas3 | gemm | f64 | 512x512x512 | 41.367 ms | 6.49 | 0.15 | 60% | ok |
| blas3 | gemm | f32 | 512x512x512 | 20.465 ms | 13.12 | 0.15 | 61% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 25.782 ms | 5.21 | 0.24 | 49% | ok |
| blas3 | symm | f64 | 512x512x512 | 42.254 ms | 6.35 | 0.15 | 59% | ok |
| blas3 | syrk | f64 | 512x512x512 | 24.534 ms | 5.47 | 0.17 | 51% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 49.172 ms | 5.46 | 0.13 | 51% | ok |
| blas3 | trmm | f64 | 512x512x512 | 24.877 ms | 5.40 | 0.21 | 50% | ok |
| blas3 | trsm | f64 | 512x512x512 | 25.848 ms | 5.19 | 0.20 | 48% | ok |
| blas3 | trsm | f32 | 512x512x512 | 12.521 ms | 10.72 | 0.21 | 50% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.673 ms | 0.55 | 4.39 | 27% | ok |
| factor | lu | f64 | 512x512 | 35.479 ms | 2.52 | 0.12 | 23% | ok |
| factor | lu | f32 | 512x512 | 20.593 ms | 4.35 | 0.10 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 31.897 ms | 1.40 | 0.13 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 30.109 ms | 1.49 | 0.07 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 32.568 ms | 1.37 | 0.13 | 13% | ok |
| factor | ldlt_bk | f64 | 512x512 | 16.265 ms | 2.75 | 0.26 | 26% | ok |
| factor | qr | f64 | 512x512 | 336.687 ms | 0.53 | 0.01 | 5% | ok |
| factor | lq | f64 | 512x512 | 93.377 ms | 1.92 | 0.04 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 474.138 ms | 0.94 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 2.039 ms | 0.52 | 2.06 | 13% | ok |
| factor | inv | f64 | 512x512 | 192.099 ms | 1.40 | 0.02 | 13% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 25.620 ms | 3.49 | 0.16 | 33% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.631 ms | 0.36 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 145.491 ms | 1.04 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 900.422 ms | 0.17 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 99.290 ms | 1.69 | 0.01 | 16% | ok |
| eigen | schur | f64 | 256x256 | 215.670 ms | 1.94 | 0.00 | 18% | ok |
| eigen | svd | f64 | 256x256 | 358.949 ms | 1.03 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 71.463 ms | 3.52 | 0.01 | 33% | ok |
| eigen | sylvester | f64 | 256x256 | 520.950 ms | 1.93 | 0.00 | 18% | ok |
| elementwise | exp | f64 | n=8388608 | 75.963 ms | 0.11 | 1.77 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.127 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 179.199 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.182 ms | 0.15 | 2.43 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 127.395 ms | 0.07 | 1.05 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.295 ms | 1.14 | 11.87 | 74% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.064 ms | 0.85 | 8.89 | 55% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.075 ms | 2.45 | 5.16 | 32% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.240 ms | 0.16 | 0.57 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.703 ms | 0.17 | 1.42 | 9% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 54.440 ms | 0.21 | 0.01 | 2% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 97.843 ms | 0.11 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 35.965 ms | 0.31 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 136.383 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 157.206 ms | 0.14 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 72.111 ms | 0.31 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 106.114 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.087 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.110 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 179.520 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.892 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 67.130 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 7.750 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 44.191 ms | 0.25 | 0.02 | 2% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.135 ms | 0.35 | 1.42 | 9% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 16.474 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.763 ms | 0.67 | 0.04 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.086 ms | 0.36 | 1.45 | 9% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.573 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 116.440 ms | 0.19 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.610 ms | 0.10 | 0.40 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 12.296 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 131.081 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.575 ms | 0.10 | 0.40 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 16.828 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 846.907 ms | 0.36 | 0.00 | 3% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 9.876 ms | 0.34 | 1.36 | 8% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 35.899 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 256.229 ms | 1.20 | 0.00 | 11% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.065 ms | 0.33 | 1.34 | 8% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 15.949 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1914.167 ms | 0.32 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.387 ms | 0.09 | 0.37 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 22.761 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1501.403 ms | 0.41 | 0.00 | 4% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.410 ms | 0.09 | 0.37 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.236 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 167.422 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.021 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.465 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.874 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.606 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 87.486 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.386 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.677 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 57.362 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.206 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 50.303 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 58.676 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.167 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.798 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 45.539 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.509 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.310 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.112 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.353 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.878 ms | 0.91 | 10.19 | 63% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.748 ms | 0.70 | 7.82 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.521 ms | 0.58 | 6.49 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 44.677 ms | 0.86 | 6.51 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 359.373 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 363.098 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 189.779 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.547 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 6.261 ms | 2.68 | 10.72 | 66% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 64.078 ms | 4.19 | 0.07 | 39% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.338 ms | 12.54 | 13.33 | 117% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.616 ms | 2.99 | 11.95 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.719 ms | 6.17 | 12.34 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.220 ms | 13.75 | 13.75 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.347 ms | 7.15 | 7.15 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 26.092 ms | 10.29 | 0.12 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 27.143 ms | 9.89 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 33.376 ms | 8.04 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 8.554 ms | 31.38 | 0.18 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 23.374 ms | 11.48 | 0.07 | n/a | ok |

