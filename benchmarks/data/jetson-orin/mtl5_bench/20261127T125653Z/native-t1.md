## jetson-orin -- native, 1 thread, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 10.7 GFLOP/s, f32 21.5 GFLOP/s |
| ... hardware, NEON | f64 10.7, f32 21.5 GFLOP/s |
| ... this build, scalar | f64 5.4, f32 5.4 GFLOP/s |
| bandwidth | read 11.5, update 16.1, triad 14.8 GB/s (ceiling 16.1) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 72% | 131% | 0 |
| blas2 | 10 | 50% | 124% | 0 |
| blas3 | 10 | 33% | 34% | 0 |
| factor | 12 | 13% | 33% | 0 |
| eigen | 8 | 16% | 23% | 0 |
| elementwise | 5 | 8% | 15% | 0 |
| sparse | 5 | 31% | 73% | 0 |
| direct | 19 | 2% | 9% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 4% | 11% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 49% | 62% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 33% | 33% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.545 ms | 1.45 | 11.63 | 72% | ok |
| blas1 | dot | f32 | n=8388608 | 6.002 ms | 2.80 | 11.18 | 69% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.233 ms | 1.37 | 16.46 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 6.018 ms | 2.79 | 16.73 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 6.359 ms | 1.32 | 21.11 | 131% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.611 ms | 2.99 | 11.96 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.748 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 192.369 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 12.931 ms | 2.59 | 10.38 | 64% | ok |
| blas2 | gemv | f32 | n=4096 | 7.720 ms | 4.35 | 8.70 | 54% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 16.676 ms | 2.01 | 8.05 | 50% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.401 ms | 5.24 | 10.49 | 65% | ok |
| blas2 | symv | f64 | n=4096 | 25.820 ms | 1.30 | 2.60 | 16% | ok |
| blas2 | trmv | f64 | n=4096 | 9.017 ms | 1.86 | 7.45 | 46% | ok |
| blas2 | trmv | f32 | n=4096 | 5.377 ms | 3.12 | 6.25 | 39% | ok |
| blas2 | trsv | f64 | n=4096 | 8.894 ms | 1.89 | 7.55 | 47% | ok |
| blas2 | trsv | f32 | n=4096 | 4.885 ms | 3.43 | 6.88 | 43% | ok |
| blas2 | ger | f64 | n=4096 | 13.435 ms | 2.50 | 19.99 | 124% | ok |
| blas3 | gemm | f64 | 512x512x512 | 75.199 ms | 3.57 | 0.08 | 33% | ok |
| blas3 | gemm | f32 | 512x512x512 | 37.319 ms | 7.19 | 0.08 | 34% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 49.406 ms | 2.72 | 0.13 | 25% | ok |
| blas3 | symm | f64 | 512x512x512 | 73.413 ms | 3.66 | 0.09 | 34% | ok |
| blas3 | syrk | f64 | 512x512x512 | 51.325 ms | 2.62 | 0.08 | 24% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 92.858 ms | 2.89 | 0.07 | 27% | ok |
| blas3 | trmm | f64 | 512x512x512 | 37.183 ms | 3.61 | 0.14 | 34% | ok |
| blas3 | trsm | f64 | 512x512x512 | 37.878 ms | 3.54 | 0.14 | 33% | ok |
| blas3 | trsm | f32 | 512x512x512 | 18.740 ms | 7.16 | 0.14 | 33% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.901 ms | 0.54 | 4.29 | 27% | ok |
| factor | lu | f64 | 512x512 | 35.286 ms | 2.54 | 0.12 | 24% | ok |
| factor | lu | f32 | 512x512 | 20.535 ms | 4.36 | 0.10 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 31.927 ms | 1.40 | 0.13 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 30.105 ms | 1.49 | 0.07 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 32.359 ms | 1.38 | 0.13 | 13% | ok |
| factor | ldlt_bk | f64 | 512x512 | 16.247 ms | 2.75 | 0.26 | 26% | ok |
| factor | qr | f64 | 512x512 | 332.121 ms | 0.54 | 0.01 | 5% | ok |
| factor | lq | f64 | 512x512 | 93.360 ms | 1.92 | 0.04 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 471.804 ms | 0.95 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 2.003 ms | 0.52 | 2.09 | 13% | ok |
| factor | inv | f64 | 512x512 | 190.710 ms | 1.41 | 0.02 | 13% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 25.216 ms | 3.55 | 0.17 | 33% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.619 ms | 0.36 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 145.175 ms | 1.04 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 898.606 ms | 0.17 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 99.208 ms | 1.69 | 0.01 | 16% | ok |
| eigen | schur | f64 | 256x256 | 214.493 ms | 1.96 | 0.00 | 18% | ok |
| eigen | svd | f64 | 256x256 | 358.662 ms | 1.03 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 101.706 ms | 2.47 | 0.01 | 23% | ok |
| eigen | sylvester | f64 | 256x256 | 536.272 ms | 1.88 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 76.032 ms | 0.11 | 1.77 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.275 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 178.582 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.250 ms | 0.15 | 2.43 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 127.086 ms | 0.07 | 1.06 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.311 ms | 1.13 | 11.78 | 73% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.026 ms | 0.86 | 9.00 | 56% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.500 ms | 2.39 | 5.03 | 31% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.978 ms | 0.16 | 0.55 | 3% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 12.186 ms | 0.16 | 1.37 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 54.044 ms | 0.21 | 0.01 | 2% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 89.868 ms | 0.12 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 36.038 ms | 0.31 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 135.070 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 154.063 ms | 0.15 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 61.734 ms | 0.36 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 106.121 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.166 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.215 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 180.784 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.897 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 66.723 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 7.812 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 43.706 ms | 0.26 | 0.02 | 2% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.194 ms | 0.34 | 1.38 | 9% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 16.688 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.780 ms | 0.67 | 0.04 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.111 ms | 0.36 | 1.43 | 9% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.619 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 116.159 ms | 0.19 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.887 ms | 0.10 | 0.38 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 12.324 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 129.683 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.834 ms | 0.10 | 0.39 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 17.051 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 848.604 ms | 0.36 | 0.00 | 3% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 10.483 ms | 0.32 | 1.29 | 8% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 37.069 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 256.598 ms | 1.20 | 0.00 | 11% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.310 ms | 0.33 | 1.31 | 8% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.202 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1934.214 ms | 0.32 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 38.105 ms | 0.09 | 0.35 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 23.424 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1516.540 ms | 0.41 | 0.00 | 4% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 38.214 ms | 0.09 | 0.35 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.314 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 230.058 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.242 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.981 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 51.197 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.977 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.249 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.969 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 50.026 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 57.871 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.668 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 50.713 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 59.504 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.167 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.873 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 44.669 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.505 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.405 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.097 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.364 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.927 ms | 0.89 | 10.02 | 62% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.734 ms | 0.70 | 7.85 | 49% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 8.179 ms | 0.32 | 3.59 | 22% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 45.041 ms | 0.85 | 6.45 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 361.163 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 365.102 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 191.517 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.976 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 12.499 ms | 1.34 | 5.37 | 33% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 75.452 ms | 3.56 | 0.06 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 5.580 ms | 3.01 | 3.19 | 28% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.630 ms | 2.98 | 11.92 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.718 ms | 6.17 | 12.35 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.672 ms | 2.96 | 2.96 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.757 ms | 9.55 | 9.55 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 38.084 ms | 7.05 | 0.08 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 41.155 ms | 6.52 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 31.860 ms | 8.43 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 31.880 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 60.097 ms | 4.47 | 0.03 | n/a | ok |

