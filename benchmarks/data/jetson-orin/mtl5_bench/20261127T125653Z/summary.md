# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,6, CPUs 0,1,2,3,4,5, clock policy nvpmodel 25W (mode 1), jetson_clocks; schedutil 1344-1344 MHz on cpus 0,1,2,3,4,5, 2026-11-27T13:22:07Z.

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

## jetson-orin -- native, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 64.2 GFLOP/s, f32 128.5 GFLOP/s |
| ... hardware, NEON | f64 64.2, f32 128.5 GFLOP/s |
| ... this build, scalar | f64 32.1, f32 32.1 GFLOP/s |
| bandwidth | read 28.0, update 36.6, triad 32.7 GB/s (ceiling 36.6) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 83% | 129% | 0 | 37% |
| blas2 | 10 | 55% | 81% | 0 | 19% |
| blas3 | 10 | 6% | 19% | 0 | 17% |
| factor | 12 | 4% | 21% | 0 | 22% |
| eigen | 8 | 3% | 8% | 0 | 17% |
| elementwise | 5 | 3% | 7% | 0 | 17% |
| sparse | 5 | 24% | 88% | 0 | 17% |
| direct | 19 | 0% | 4% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 1% | 4% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 21% | 27% | 0 | 17% |
| mg | 1 | 19% | 19% | 0 | 18% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 15% | 32% | 0 | 20% |
| integer | 9 | n/a | n/a | 0 | 71% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.573 ms | 3.67 | 29.35 | 80% | ok | 2.5x (42%) |
| blas1 | dot | f32 | n=8388608 | 2.196 ms | 7.64 | 30.55 | 83% | ok | 2.7x (46%) |
| blas1 | axpy | f64 | n=8388608 | 5.526 ms | 3.04 | 36.43 | 99% | ok | 2.2x (37%) |
| blas1 | axpy | f32 | n=8388608 | 2.464 ms | 6.81 | 40.86 | 112% | ok | 2.4x (41%) |
| blas1 | scale | f64 | n=8388608 | 2.845 ms | 2.95 | 47.18 | 129% | ok | 2.2x (37%) |
| blas1 | two_norm | f64 | n=8388608 | 2.184 ms | 7.68 | 30.72 | 84% | ok | 2.6x (43%) |
| blas1 | one_norm | f64 | n=8388608 | 12.508 ms | 0.67 | 5.37 | 15% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.764 ms | 0.45 | 3.58 | 10% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.510 ms | 0.67 | 5.36 | 15% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 192.019 ms | 0.96 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 5.217 ms | 6.43 | 25.74 | 70% | ok | 2.5x (41%) |
| blas2 | gemv | f32 | n=4096 | 2.277 ms | 14.74 | 29.49 | 81% | ok | 3.4x (57%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 6.477 ms | 5.18 | 20.73 | 57% | ok | 2.6x (43%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.269 ms | 10.27 | 20.54 | 56% | ok | 2.0x (33%) |
| blas2 | symv | f64 | n=4096 | 22.963 ms | 1.46 | 2.93 | 8% | ok | 1.1x (19%) |
| blas2 | trmv | f64 | n=4096 | 8.883 ms | 1.89 | 7.56 | 21% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 5.187 ms | 3.23 | 6.48 | 18% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 8.649 ms | 1.94 | 7.77 | 21% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 4.760 ms | 3.52 | 7.06 | 19% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 13.397 ms | 2.50 | 20.04 | 55% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 21.692 ms | 12.37 | 0.29 | 19% | ok | 3.5x (58%) |
| blas3 | gemm | f32 | 512x512x512 | 12.347 ms | 21.74 | 0.25 | 17% | ok | 3.0x (50%) |
| blas3 | gemmt | f64 | 512x512x512 | 48.709 ms | 2.76 | 0.13 | 4% | ok | 1.0x (17%) |
| blas3 | symm | f64 | 512x512x512 | 75.095 ms | 3.57 | 0.08 | 6% | ok | 1.0x (16%) |
| blas3 | syrk | f64 | 512x512x512 | 50.497 ms | 2.66 | 0.08 | 4% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 512x512x512 | 89.628 ms | 3.00 | 0.07 | 5% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 512x512x512 | 37.917 ms | 3.54 | 0.14 | 6% | ok | 1.0x (16%) |
| blas3 | trsm | f64 | 512x512x512 | 38.687 ms | 3.47 | 0.14 | 5% | ok | 1.0x (16%) |
| blas3 | trsm | f32 | 512x512x512 | 19.177 ms | 7.00 | 0.14 | 5% | ok | 1.0x (16%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.347 ms | 0.57 | 4.54 | 12% | ok | 1.1x (18%) |
| factor | lu | f64 | 512x512 | 24.712 ms | 3.62 | 0.17 | 6% | ok | 1.4x (24%) |
| factor | lu | f32 | 512x512 | 14.423 ms | 6.20 | 0.15 | 5% | ok | 1.4x (24%) |
| factor | cholesky | f64 | 512x512 | 31.679 ms | 1.41 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 30.141 ms | 1.48 | 0.07 | 1% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 32.152 ms | 1.39 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 16.144 ms | 2.77 | 0.26 | 4% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 208.991 ms | 0.86 | 0.02 | 1% | ok | 1.6x (26%) |
| factor | lq | f64 | 512x512 | 69.639 ms | 2.57 | 0.06 | 4% | ok | 1.3x (22%) |
| factor | hessenberg | f64 | 512x512 | 302.294 ms | 1.48 | 0.01 | 2% | ok | 1.6x (26%) |
| factor | householder | f64 | 512x512 | 0.548 ms | 1.92 | 7.66 | 21% | ok | 3.7x (61%) |
| factor | inv | f64 | 512x512 | 181.520 ms | 1.48 | 0.02 | 2% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 19.359 ms | 4.62 | 0.22 | 7% | ok | 1.3x (22%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.981 ms | 0.36 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 145.442 ms | 1.04 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 900.055 ms | 0.17 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 99.734 ms | 1.68 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 215.530 ms | 1.95 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 358.628 ms | 1.03 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 48.682 ms | 5.17 | 0.02 | 8% | ok | 2.1x (35%) |
| eigen | sylvester | f64 | 256x256 | 504.577 ms | 2.00 | 0.00 | 3% | ok | 1.1x (18%) |
| elementwise | exp | f64 | n=8388608 | 75.923 ms | 0.11 | 1.77 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 108.014 ms | 0.08 | 1.24 | 3% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 177.911 ms | 0.05 | 0.75 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 55.188 ms | 0.15 | 2.43 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 127.016 ms | 0.07 | 1.06 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.844 ms | 3.10 | 32.24 | 88% | ok | 2.7x (46%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.039 ms | 0.86 | 8.96 | 24% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 5.915 ms | 7.08 | 14.89 | 41% | ok | 3.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.970 ms | 0.16 | 0.55 | 1% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.804 ms | 0.17 | 1.41 | 4% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 53.793 ms | 0.21 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 90.079 ms | 0.12 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 36.205 ms | 0.31 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 135.245 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 155.324 ms | 0.14 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 60.960 ms | 0.37 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 106.088 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.145 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.165 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 180.802 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.896 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 65.925 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 7.861 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 43.746 ms | 0.26 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.349 ms | 0.32 | 1.29 | 4% | ok | 0.9x (16%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 16.677 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.798 ms | 0.67 | 0.04 | 1% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.134 ms | 0.35 | 1.42 | 4% | ok | 1.0x (16%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.629 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 116.046 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.718 ms | 0.10 | 0.39 | 1% | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 12.317 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 132.761 ms | 0.17 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.844 ms | 0.10 | 0.39 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 17.030 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 847.535 ms | 0.36 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 10.393 ms | 0.32 | 1.30 | 4% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 37.060 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 256.956 ms | 1.20 | 0.00 | 2% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.298 ms | 0.33 | 1.31 | 4% | ok | 1.0x (17%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.241 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1941.512 ms | 0.32 | 0.00 | 0% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 37.849 ms | 0.09 | 0.36 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 23.458 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1504.730 ms | 0.41 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 38.022 ms | 0.09 | 0.35 | 1% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.147 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 229.342 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 77.990 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.791 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.925 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.847 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 88.846 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.715 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.708 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 57.609 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.508 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 50.513 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 59.625 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.168 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.811 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 44.711 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.494 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.386 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.097 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.349 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.994 ms | 0.87 | 9.80 | 27% | ok | 1.0x (16%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.736 ms | 0.70 | 7.85 | 21% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 8.143 ms | 0.32 | 3.60 | 10% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 42.021 ms | 0.92 | 6.92 | 19% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 350.011 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 353.801 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 188.813 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.906 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 12.505 ms | 1.34 | 5.37 | 15% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.894 ms | 20.82 | 0.33 | 32% | ok | 5.9x (98%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 4.567 ms | 3.67 | 3.90 | 11% | ok | 1.2x (20%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.611 ms | 2.99 | 11.96 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.712 ms | 6.19 | 12.37 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.701 ms | 2.94 | 2.94 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.761 ms | 9.53 | 9.53 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 8.985 ms | 29.88 | 0.35 | n/a | ok | 4.2x (71%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 7.081 ms | 37.91 | 0.30 | n/a | ok | 5.8x (97%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.490 ms | 48.90 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.490 ms | 48.90 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 10.299 ms | 26.06 | 0.15 | n/a | ok | 5.8x (97%) |

## jetson-orin -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 10.7 GFLOP/s, f32 21.5 GFLOP/s |
| ... hardware, NEON | f64 10.7, f32 21.5 GFLOP/s |
| ... this build, NEON | f64 10.7, f32 21.5 GFLOP/s |
| bandwidth | read 11.5, update 16.2, triad 14.8 GB/s (ceiling 16.2) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 74% | 130% | 0 |
| blas2 | 10 | 63% | 124% | 0 |
| blas3 | 10 | 51% | 61% | 0 |
| factor | 12 | 13% | 33% | 0 |
| eigen | 8 | 16% | 33% | 0 |
| elementwise | 5 | 8% | 15% | 0 |
| sparse | 5 | 31% | 72% | 0 |
| direct | 19 | 2% | 9% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 4% | 11% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 62% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 66% | 117% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.597 ms | 1.45 | 11.57 | 71% | ok |
| blas1 | dot | f32 | n=8388608 | 5.628 ms | 2.98 | 11.92 | 74% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.250 ms | 1.37 | 16.43 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 5.992 ms | 2.80 | 16.80 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 6.363 ms | 1.32 | 21.09 | 130% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.600 ms | 3.00 | 11.98 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.503 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.754 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.502 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 192.361 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 13.130 ms | 2.56 | 10.23 | 63% | ok |
| blas2 | gemv | f32 | n=4096 | 6.261 ms | 5.36 | 10.72 | 66% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 13.157 ms | 2.55 | 10.21 | 63% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.115 ms | 5.49 | 10.98 | 68% | ok |
| blas2 | symv | f64 | n=4096 | 25.973 ms | 1.29 | 2.59 | 16% | ok |
| blas2 | trmv | f64 | n=4096 | 9.100 ms | 1.84 | 7.38 | 46% | ok |
| blas2 | trmv | f32 | n=4096 | 4.367 ms | 3.84 | 7.69 | 48% | ok |
| blas2 | trsv | f64 | n=4096 | 9.266 ms | 1.81 | 7.25 | 45% | ok |
| blas2 | trsv | f32 | n=4096 | 4.265 ms | 3.93 | 7.87 | 49% | ok |
| blas2 | ger | f64 | n=4096 | 13.430 ms | 2.50 | 19.99 | 124% | ok |
| blas3 | gemm | f64 | 512x512x512 | 41.264 ms | 6.51 | 0.15 | 61% | ok |
| blas3 | gemm | f32 | 512x512x512 | 20.715 ms | 12.96 | 0.15 | 60% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 25.793 ms | 5.20 | 0.24 | 48% | ok |
| blas3 | symm | f64 | 512x512x512 | 42.183 ms | 6.36 | 0.15 | 59% | ok |
| blas3 | syrk | f64 | 512x512x512 | 24.547 ms | 5.47 | 0.17 | 51% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 49.251 ms | 5.45 | 0.13 | 51% | ok |
| blas3 | trmm | f64 | 512x512x512 | 24.904 ms | 5.39 | 0.21 | 50% | ok |
| blas3 | trsm | f64 | 512x512x512 | 25.626 ms | 5.24 | 0.20 | 49% | ok |
| blas3 | trsm | f32 | 512x512x512 | 12.434 ms | 10.79 | 0.21 | 50% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.908 ms | 0.54 | 4.29 | 26% | ok |
| factor | lu | f64 | 512x512 | 35.288 ms | 2.54 | 0.12 | 24% | ok |
| factor | lu | f32 | 512x512 | 20.613 ms | 4.34 | 0.10 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 34.110 ms | 1.31 | 0.12 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 30.115 ms | 1.49 | 0.07 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 34.677 ms | 1.29 | 0.12 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 16.297 ms | 2.75 | 0.26 | 26% | ok |
| factor | qr | f64 | 512x512 | 350.040 ms | 0.51 | 0.01 | 5% | ok |
| factor | lq | f64 | 512x512 | 93.326 ms | 1.92 | 0.04 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 489.637 ms | 0.91 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 2.011 ms | 0.52 | 2.09 | 13% | ok |
| factor | inv | f64 | 512x512 | 190.589 ms | 1.41 | 0.02 | 13% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 25.227 ms | 3.55 | 0.17 | 33% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.552 ms | 0.36 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 145.446 ms | 1.04 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 962.623 ms | 0.16 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 98.819 ms | 1.70 | 0.01 | 16% | ok |
| eigen | schur | f64 | 256x256 | 214.719 ms | 1.95 | 0.00 | 18% | ok |
| eigen | svd | f64 | 256x256 | 358.942 ms | 1.03 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 71.608 ms | 3.51 | 0.01 | 33% | ok |
| eigen | sylvester | f64 | 256x256 | 518.514 ms | 1.94 | 0.00 | 18% | ok |
| elementwise | exp | f64 | n=8388608 | 75.994 ms | 0.11 | 1.77 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.224 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 179.041 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.257 ms | 0.15 | 2.43 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 127.363 ms | 0.07 | 1.05 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.326 ms | 1.13 | 11.71 | 72% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.056 ms | 0.86 | 8.91 | 55% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.369 ms | 2.41 | 5.07 | 31% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.047 ms | 0.16 | 0.57 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 12.209 ms | 0.16 | 1.36 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 53.240 ms | 0.21 | 0.01 | 2% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 97.573 ms | 0.11 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 36.228 ms | 0.31 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 134.543 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 155.151 ms | 0.14 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 60.717 ms | 0.37 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 112.911 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.113 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 16.967 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 181.287 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.893 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 65.656 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 7.798 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 43.133 ms | 0.26 | 0.02 | 2% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.163 ms | 0.35 | 1.40 | 9% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 16.578 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.799 ms | 0.67 | 0.04 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.104 ms | 0.36 | 1.44 | 9% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.615 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 114.674 ms | 0.20 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.718 ms | 0.10 | 0.39 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 12.356 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 132.866 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.960 ms | 0.09 | 0.38 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 16.910 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 841.530 ms | 0.37 | 0.00 | 3% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 10.460 ms | 0.32 | 1.29 | 8% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 36.976 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 257.588 ms | 1.20 | 0.00 | 11% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.400 ms | 0.32 | 1.30 | 8% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.215 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1887.998 ms | 0.33 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 38.316 ms | 0.09 | 0.35 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 23.477 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1534.386 ms | 0.40 | 0.00 | 4% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 38.182 ms | 0.09 | 0.35 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.152 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 229.851 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.124 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.718 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 51.036 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.917 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 88.517 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.084 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.717 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 57.681 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.415 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 50.649 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 59.218 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.168 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.891 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 45.054 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.522 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.322 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.097 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.388 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.941 ms | 0.89 | 9.97 | 62% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.740 ms | 0.70 | 7.84 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.516 ms | 0.58 | 6.49 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 45.012 ms | 0.85 | 6.46 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 358.295 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 362.065 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 191.835 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.681 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 6.261 ms | 2.68 | 10.72 | 66% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 64.057 ms | 4.19 | 0.07 | 39% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.338 ms | 12.54 | 13.33 | 117% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.648 ms | 2.97 | 11.88 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.719 ms | 6.17 | 12.34 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.242 ms | 13.50 | 13.50 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.347 ms | 7.15 | 7.15 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 26.004 ms | 10.32 | 0.12 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 27.842 ms | 9.64 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 33.358 ms | 8.05 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 8.573 ms | 31.31 | 0.18 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 23.393 ms | 11.47 | 0.07 | n/a | ok |

## jetson-orin -- native-fast, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 64.2 GFLOP/s, f32 128.6 GFLOP/s |
| ... hardware, NEON | f64 64.0, f32 128.6 GFLOP/s |
| ... this build, NEON | f64 64.2, f32 128.4 GFLOP/s |
| bandwidth | read 27.8, update 34.7, triad 33.0 GB/s (ceiling 34.7) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 87% | 136% | 0 | 37% |
| blas2 | 10 | 50% | 85% | 0 | 19% |
| blas3 | 10 | 30% | 51% | 0 | 59% |
| factor | 12 | 4% | 23% | 0 | 24% |
| eigen | 8 | 3% | 9% | 0 | 17% |
| elementwise | 5 | 4% | 7% | 0 | 17% |
| sparse | 5 | 26% | 91% | 0 | 18% |
| direct | 19 | 0% | 4% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 1% | 4% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 23% | 29% | 0 | 17% |
| mg | 1 | 20% | 20% | 0 | 18% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 34% | 39% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 80% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.578 ms | 3.66 | 29.31 | 85% | ok | 2.5x (42%) |
| blas1 | dot | f32 | n=8388608 | 2.224 ms | 7.54 | 30.17 | 87% | ok | 2.5x (42%) |
| blas1 | axpy | f64 | n=8388608 | 5.523 ms | 3.04 | 36.45 | 105% | ok | 2.2x (37%) |
| blas1 | axpy | f32 | n=8388608 | 2.464 ms | 6.81 | 40.86 | 118% | ok | 2.4x (41%) |
| blas1 | scale | f64 | n=8388608 | 2.846 ms | 2.95 | 47.16 | 136% | ok | 2.2x (37%) |
| blas1 | two_norm | f64 | n=8388608 | 2.181 ms | 7.69 | 30.77 | 89% | ok | 2.6x (43%) |
| blas1 | one_norm | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 15% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.750 ms | 0.45 | 3.58 | 10% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 15% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 191.819 ms | 0.96 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 5.202 ms | 6.45 | 25.81 | 74% | ok | 2.5x (42%) |
| blas2 | gemv | f32 | n=4096 | 2.290 ms | 14.65 | 29.32 | 85% | ok | 2.7x (46%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.765 ms | 4.32 | 17.29 | 50% | ok | 1.7x (28%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.566 ms | 9.41 | 18.83 | 54% | ok | 1.7x (29%) |
| blas2 | symv | f64 | n=4096 | 23.305 ms | 1.44 | 2.88 | 8% | ok | 1.1x (19%) |
| blas2 | trmv | f64 | n=4096 | 9.138 ms | 1.84 | 7.35 | 21% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 4.281 ms | 3.92 | 7.85 | 23% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 9.091 ms | 1.85 | 7.39 | 21% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 4.275 ms | 3.92 | 7.86 | 23% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 13.389 ms | 2.51 | 20.05 | 58% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 8.152 ms | 32.93 | 0.77 | 51% | ok | 5.1x (84%) |
| blas3 | gemm | f32 | 512x512x512 | 4.084 ms | 65.73 | 0.77 | 51% | ok | 5.1x (85%) |
| blas3 | gemmt | f64 | 512x512x512 | 7.415 ms | 18.10 | 0.85 | 28% | ok | 3.5x (58%) |
| blas3 | symm | f64 | 512x512x512 | 9.131 ms | 29.40 | 0.69 | 46% | ok | 4.6x (77%) |
| blas3 | syrk | f64 | 512x512x512 | 6.989 ms | 19.20 | 0.60 | 30% | ok | 3.5x (59%) |
| blas3 | syr2k | f64 | 512x512x512 | 13.962 ms | 19.23 | 0.45 | 30% | ok | 3.5x (59%) |
| blas3 | trmm | f64 | 512x512x512 | 10.844 ms | 12.38 | 0.48 | 19% | ok | 2.3x (38%) |
| blas3 | trsm | f64 | 512x512x512 | 16.130 ms | 8.32 | 0.33 | 13% | ok | 1.6x (26%) |
| blas3 | trsm | f32 | 512x512x512 | 5.553 ms | 24.17 | 0.47 | 19% | ok | 2.2x (37%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.922 ms | 0.53 | 4.28 | 12% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 24.180 ms | 3.70 | 0.17 | 6% | ok | 1.5x (24%) |
| factor | lu | f32 | 512x512 | 14.574 ms | 6.14 | 0.14 | 5% | ok | 1.4x (24%) |
| factor | cholesky | f64 | 512x512 | 33.831 ms | 1.32 | 0.12 | 2% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 30.104 ms | 1.49 | 0.07 | 1% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 34.469 ms | 1.30 | 0.12 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 15.962 ms | 2.80 | 0.26 | 4% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 206.783 ms | 0.87 | 0.02 | 1% | ok | 1.7x (28%) |
| factor | lq | f64 | 512x512 | 59.544 ms | 3.01 | 0.07 | 5% | ok | 1.6x (26%) |
| factor | hessenberg | f64 | 512x512 | 319.578 ms | 1.40 | 0.01 | 2% | ok | 1.5x (26%) |
| factor | householder | f64 | 512x512 | 0.530 ms | 1.98 | 7.91 | 23% | ok | 3.8x (63%) |
| factor | inv | f64 | 512x512 | 180.948 ms | 1.48 | 0.02 | 2% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 19.624 ms | 4.56 | 0.21 | 7% | ok | 1.3x (21%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.423 ms | 0.36 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 145.165 ms | 1.04 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 897.214 ms | 0.17 | 0.00 | 0% | ok | 1.1x (18%) |
| eigen | eigenvalue | f64 | 256x256 | 99.001 ms | 1.69 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 214.213 ms | 1.96 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 358.365 ms | 1.03 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 43.006 ms | 5.85 | 0.02 | 9% | ok | 1.7x (28%) |
| eigen | sylvester | f64 | 256x256 | 504.839 ms | 1.99 | 0.00 | 3% | ok | 1.0x (17%) |
| elementwise | exp | f64 | n=8388608 | 75.883 ms | 0.11 | 1.77 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 108.081 ms | 0.08 | 1.24 | 4% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 178.084 ms | 0.05 | 0.75 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 55.156 ms | 0.15 | 2.43 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 127.127 ms | 0.07 | 1.06 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.861 ms | 3.04 | 31.63 | 91% | ok | 2.7x (45%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.011 ms | 0.87 | 9.04 | 26% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 5.417 ms | 7.73 | 16.25 | 47% | ok | 3.2x (53%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.062 ms | 0.16 | 0.57 | 2% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.512 ms | 0.17 | 1.45 | 4% | ok | 1.1x (18%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 53.139 ms | 0.21 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 97.796 ms | 0.11 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 35.987 ms | 0.31 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 136.596 ms | 0.16 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 157.511 ms | 0.14 | 0.00 | 0% | ok | 1.0x (16%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 60.521 ms | 0.37 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 112.405 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.118 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 16.961 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 180.388 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.897 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 65.557 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 7.747 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 43.054 ms | 0.26 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.109 ms | 0.36 | 1.43 | 4% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 16.384 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.799 ms | 0.67 | 0.04 | 1% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.111 ms | 0.36 | 1.43 | 4% | ok | 1.0x (17%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.558 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 116.121 ms | 0.19 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.761 ms | 0.10 | 0.39 | 1% | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 12.218 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 132.026 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.679 ms | 0.10 | 0.39 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 16.920 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 813.474 ms | 0.38 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 10.465 ms | 0.32 | 1.29 | 4% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 36.986 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 258.000 ms | 1.20 | 0.00 | 2% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.298 ms | 0.33 | 1.31 | 4% | ok | 1.0x (17%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.326 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1892.008 ms | 0.33 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 37.978 ms | 0.09 | 0.35 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 23.558 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1522.588 ms | 0.41 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 37.962 ms | 0.09 | 0.35 | 1% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.192 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 229.104 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.150 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.685 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 51.053 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.835 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 88.393 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.901 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.818 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 57.566 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.338 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 50.543 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 59.180 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.168 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.905 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 44.875 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.522 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.337 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.097 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.401 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.886 ms | 0.91 | 10.16 | 29% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.739 ms | 0.70 | 7.84 | 23% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.519 ms | 0.58 | 6.49 | 19% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 42.058 ms | 0.91 | 6.91 | 20% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 360.685 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 364.373 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 188.474 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.848 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 6.269 ms | 2.68 | 10.71 | 31% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.297 ms | 21.83 | 0.34 | 34% | ok | 5.2x (87%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.319 ms | 12.72 | 13.51 | 39% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.648 ms | 2.97 | 11.88 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.722 ms | 6.16 | 12.33 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.243 ms | 13.50 | 13.50 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.350 ms | 7.14 | 7.14 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 5.137 ms | 52.26 | 0.61 | n/a | ok | 5.1x (84%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.809 ms | 46.21 | 0.36 | n/a | ok | 4.8x (80%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 6.628 ms | 40.50 | 0.24 | n/a | ok | 5.0x (84%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.761 ms | 152.47 | 0.89 | n/a | ok | 4.9x (81%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.535 ms | 59.20 | 0.35 | n/a | ok | 5.2x (86%) |

## jetson-orin -- blas, 1 thread, class medium

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
| bandwidth | read 11.5, update 16.1, triad 14.7 GB/s (ceiling 16.1) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 72% | 131% | 0 |
| blas2 | 10 | 64% | 125% | 0 |
| blas3 | 10 | 89% | 95% | 0 |
| factor | 12 | 42% | 61% | 0 |
| eigen | 8 | 37% | 80% | 0 |
| elementwise | 5 | 8% | 15% | 0 |
| sparse | 5 | 31% | 73% | 0 |
| direct | 23 | 2% | 14% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 15 | 8% | 36% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 49% | 62% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 33% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.555 ms | 1.45 | 11.62 | 72% | ok |
| blas1 | dot | f32 | n=8388608 | 5.997 ms | 2.80 | 11.19 | 69% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.246 ms | 1.37 | 16.44 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 5.977 ms | 2.81 | 16.84 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 6.362 ms | 1.32 | 21.10 | 131% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.618 ms | 2.99 | 11.95 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.501 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.752 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.501 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 191.890 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 12.615 ms | 2.66 | 10.64 | 66% | ok |
| blas2 | gemv | f32 | n=4096 | 5.893 ms | 5.69 | 11.39 | 71% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 13.007 ms | 2.58 | 10.32 | 64% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.803 ms | 5.78 | 11.57 | 72% | ok |
| blas2 | symv | f64 | n=4096 | 15.938 ms | 2.11 | 4.21 | 26% | ok |
| blas2 | trmv | f64 | n=4096 | 8.593 ms | 1.95 | 7.82 | 48% | ok |
| blas2 | trmv | f32 | n=4096 | 4.889 ms | 3.43 | 6.87 | 43% | ok |
| blas2 | trsv | f64 | n=4096 | 8.562 ms | 1.96 | 7.85 | 49% | ok |
| blas2 | trsv | f32 | n=4096 | 4.777 ms | 3.51 | 7.03 | 44% | ok |
| blas2 | ger | f64 | n=4096 | 13.341 ms | 2.52 | 20.13 | 125% | ok |
| blas3 | gemm | f64 | 512x512x512 | 26.654 ms | 10.07 | 0.24 | 94% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.110 ms | 20.48 | 0.24 | 95% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 16.602 ms | 8.08 | 0.38 | 75% | ok |
| blas3 | symm | f64 | 512x512x512 | 26.718 ms | 10.05 | 0.24 | 94% | ok |
| blas3 | syrk | f64 | 512x512x512 | 14.446 ms | 9.29 | 0.29 | 87% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 28.236 ms | 9.51 | 0.22 | 89% | ok |
| blas3 | trmm | f64 | 512x512x512 | 13.651 ms | 9.83 | 0.38 | 92% | ok |
| blas3 | trsm | f64 | 512x512x512 | 14.976 ms | 8.96 | 0.35 | 84% | ok |
| blas3 | trsm | f32 | 512x512x512 | 7.564 ms | 17.74 | 0.35 | 83% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.417 ms | 0.56 | 4.51 | 28% | ok |
| factor | lu | f64 | 512x512 | 13.681 ms | 6.54 | 0.31 | 61% | ok |
| factor | lu | f32 | 512x512 | 8.383 ms | 10.67 | 0.25 | 50% | ok |
| factor | cholesky | f64 | 512x512 | 7.904 ms | 5.66 | 0.53 | 53% | ok |
| factor | cholesky | f32 | 512x512 | 4.938 ms | 9.06 | 0.42 | 42% | ok |
| factor | ldlt | f64 | 512x512 | 32.274 ms | 1.39 | 0.13 | 13% | ok |
| factor | ldlt_bk | f64 | 512x512 | 16.160 ms | 2.77 | 0.26 | 26% | ok |
| factor | qr | f64 | 512x512 | 28.454 ms | 6.29 | 0.15 | 59% | ok |
| factor | lq | f64 | 512x512 | 91.904 ms | 1.95 | 0.05 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 476.664 ms | 0.94 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 2.066 ms | 0.51 | 2.03 | 13% | ok |
| factor | inv | f64 | 512x512 | 173.559 ms | 1.55 | 0.02 | 14% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 14.805 ms | 6.04 | 0.28 | 56% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 10.405 ms | 2.15 | 0.10 | 20% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 144.749 ms | 1.04 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 897.347 ms | 0.17 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 62.932 ms | 2.67 | 0.02 | 25% | ok |
| eigen | schur | f64 | 256x256 | 90.060 ms | 4.66 | 0.01 | 43% | ok |
| eigen | svd | f64 | 256x256 | 43.264 ms | 8.53 | 0.02 | 80% | ok |
| eigen | expm | f64 | 256x256 | 55.997 ms | 4.49 | 0.02 | 42% | ok |
| eigen | sylvester | f64 | 256x256 | 253.190 ms | 3.98 | 0.01 | 37% | ok |
| elementwise | exp | f64 | n=8388608 | 75.988 ms | 0.11 | 1.77 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.073 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 178.366 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.170 ms | 0.15 | 2.43 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 126.498 ms | 0.07 | 1.06 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.309 ms | 1.13 | 11.80 | 73% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.007 ms | 0.87 | 9.06 | 56% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.395 ms | 2.41 | 5.06 | 31% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.412 ms | 0.16 | 0.56 | 3% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.730 ms | 0.17 | 1.42 | 9% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 53.965 ms | 0.21 | 0.01 | 2% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 87.947 ms | 0.13 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 35.997 ms | 0.31 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 130.849 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 150.912 ms | 0.15 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 60.743 ms | 0.37 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 105.826 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.083 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 16.990 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 181.486 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.891 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 65.607 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 7.807 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 43.663 ms | 0.26 | 0.02 | 2% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.154 ms | 0.35 | 1.40 | 9% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 16.560 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.762 ms | 0.67 | 0.04 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.085 ms | 0.36 | 1.45 | 9% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.608 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 113.397 ms | 0.20 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.546 ms | 0.10 | 0.40 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 12.296 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 127.436 ms | 0.18 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.605 ms | 0.10 | 0.40 | 2% | ok |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 28.165 ms | 0.40 | 0.03 | 4% | ok |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 5.955 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 20.450 ms | 0.55 | 0.04 | 5% | ok |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.378 ms | 0.55 | 2.20 | 14% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 16.760 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 845.097 ms | 0.37 | 0.00 | 3% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 9.972 ms | 0.34 | 1.35 | 8% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 36.836 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 256.518 ms | 1.20 | 0.00 | 11% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.089 ms | 0.33 | 1.34 | 8% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.215 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1907.064 ms | 0.32 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.674 ms | 0.09 | 0.37 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 23.448 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1529.938 ms | 0.40 | 0.00 | 4% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.701 ms | 0.09 | 0.37 | 2% | ok |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 11.479 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 79.179 ms | 3.90 | 0.01 | 36% | ok |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 3.145 ms | 1.07 | 4.28 | 27% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.387 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 230.015 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.351 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.957 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 51.230 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 50.068 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 88.892 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.160 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 50.116 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 57.844 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.587 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 50.722 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 59.709 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.167 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.750 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 44.607 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.491 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.293 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.097 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 4.083 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.918 ms | 0.90 | 10.05 | 62% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.737 ms | 0.70 | 7.85 | 49% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.517 ms | 0.58 | 6.49 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 44.976 ms | 0.85 | 6.46 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 351.320 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 356.834 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 196.283 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 21.260 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 12.503 ms | 1.34 | 5.37 | 33% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 75.430 ms | 3.56 | 0.06 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 4.569 ms | 3.67 | 3.90 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.612 ms | 2.99 | 11.96 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.748 ms | 6.11 | 12.21 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.672 ms | 2.96 | 2.96 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.758 ms | 9.55 | 9.55 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 38.116 ms | 7.04 | 0.08 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 41.106 ms | 6.53 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 31.915 ms | 8.41 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 31.891 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 60.149 ms | 4.46 | 0.03 | n/a | ok |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

## jetson-orin -- blas, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 64.3 GFLOP/s, f32 128.5 GFLOP/s |
| ... hardware, NEON | f64 64.3, f32 128.5 GFLOP/s |
| ... this build, scalar | f64 32.1, f32 32.1 GFLOP/s |
| bandwidth | read 27.2, update 33.9, triad 32.1 GB/s (ceiling 33.9) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 90% | 124% | 0 | 37% |
| blas2 | 10 | 69% | 124% | 0 | 43% |
| blas3 | 10 | 71% | 87% | 0 | 85% |
| factor | 12 | 13% | 24% | 0 | 26% |
| eigen | 8 | 7% | 19% | 0 | 20% |
| elementwise | 5 | 4% | 7% | 0 | 17% |
| sparse | 5 | 27% | 96% | 0 | 17% |
| direct | 23 | 1% | 6% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 15 | 1% | 13% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 23% | 30% | 0 | 17% |
| mg | 1 | 20% | 20% | 0 | 18% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 16% | 32% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 70% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.580 ms | 3.66 | 29.31 | 87% | ok | 2.5x (42%) |
| blas1 | dot | f32 | n=8388608 | 2.206 ms | 7.60 | 30.42 | 90% | ok | 2.7x (45%) |
| blas1 | axpy | f64 | n=8388608 | 5.561 ms | 3.02 | 36.20 | 107% | ok | 2.2x (37%) |
| blas1 | axpy | f32 | n=8388608 | 2.718 ms | 6.17 | 37.04 | 109% | ok | 2.2x (37%) |
| blas1 | scale | f64 | n=8388608 | 3.186 ms | 2.63 | 42.12 | 124% | ok | 2.0x (33%) |
| blas1 | two_norm | f64 | n=8388608 | 2.200 ms | 7.62 | 30.50 | 90% | ok | 2.6x (43%) |
| blas1 | one_norm | f64 | n=8388608 | 12.502 ms | 0.67 | 5.37 | 16% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.752 ms | 0.45 | 3.58 | 11% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.502 ms | 0.67 | 5.37 | 16% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 190.938 ms | 0.97 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 4.620 ms | 7.26 | 29.06 | 86% | ok | 2.7x (46%) |
| blas2 | gemv | f32 | n=4096 | 2.279 ms | 14.72 | 29.46 | 87% | ok | 2.6x (43%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 6.456 ms | 5.20 | 20.80 | 61% | ok | 2.0x (34%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.828 ms | 8.76 | 17.54 | 52% | ok | 1.5x (25%) |
| blas2 | symv | f64 | n=4096 | 5.238 ms | 6.41 | 12.82 | 38% | ok | 3.0x (51%) |
| blas2 | trmv | f64 | n=4096 | 2.842 ms | 5.90 | 23.64 | 70% | ok | 3.0x (50%) |
| blas2 | trmv | f32 | n=4096 | 1.435 ms | 11.69 | 23.41 | 69% | ok | 3.4x (57%) |
| blas2 | trsv | f64 | n=4096 | 8.476 ms | 1.98 | 7.93 | 23% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 4.746 ms | 3.53 | 7.08 | 21% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 6.401 ms | 5.24 | 41.95 | 124% | ok | 2.1x (35%) |
| blas3 | gemm | f64 | 512x512x512 | 4.826 ms | 55.62 | 1.30 | 87% | ok | 5.5x (92%) |
| blas3 | gemm | f32 | 512x512x512 | 2.474 ms | 108.52 | 1.27 | 84% | ok | 5.3x (88%) |
| blas3 | gemmt | f64 | 512x512x512 | 4.082 ms | 32.88 | 1.54 | 51% | ok | 4.1x (68%) |
| blas3 | symm | f64 | 512x512x512 | 4.778 ms | 56.18 | 1.32 | 87% | ok | 5.6x (93%) |
| blas3 | syrk | f64 | 512x512x512 | 3.620 ms | 37.08 | 1.16 | 58% | ok | 4.0x (67%) |
| blas3 | syr2k | f64 | 512x512x512 | 5.966 ms | 44.99 | 1.05 | 70% | ok | 4.7x (79%) |
| blas3 | trmm | f64 | 512x512x512 | 2.593 ms | 51.75 | 2.02 | 81% | ok | 5.3x (88%) |
| blas3 | trsm | f64 | 512x512x512 | 2.922 ms | 45.93 | 1.79 | 71% | ok | 5.1x (85%) |
| blas3 | trsm | f32 | 512x512x512 | 1.570 ms | 85.48 | 1.67 | 67% | ok | 4.8x (80%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.431 ms | 0.56 | 4.50 | 13% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 5.693 ms | 15.72 | 0.74 | 24% | ok | 2.4x (40%) |
| factor | lu | f32 | 512x512 | 4.600 ms | 19.45 | 0.46 | 15% | ok | 1.8x (30%) |
| factor | cholesky | f64 | 512x512 | 4.156 ms | 10.77 | 1.01 | 17% | ok | 1.9x (32%) |
| factor | cholesky | f32 | 512x512 | 3.225 ms | 13.87 | 0.65 | 11% | ok | 1.5x (26%) |
| factor | ldlt | f64 | 512x512 | 31.992 ms | 1.40 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 15.950 ms | 2.81 | 0.26 | 4% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 13.720 ms | 13.04 | 0.31 | 20% | ok | 2.1x (35%) |
| factor | lq | f64 | 512x512 | 59.345 ms | 3.02 | 0.07 | 5% | ok | 1.5x (26%) |
| factor | hessenberg | f64 | 512x512 | 317.461 ms | 1.41 | 0.01 | 2% | ok | 1.5x (25%) |
| factor | householder | f64 | 512x512 | 0.535 ms | 1.96 | 7.84 | 23% | ok | 3.9x (64%) |
| factor | inv | f64 | 512x512 | 163.252 ms | 1.64 | 0.03 | 3% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 10.930 ms | 8.19 | 0.38 | 13% | ok | 1.4x (23%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 7.076 ms | 3.16 | 0.15 | 5% | ok | 1.5x (25%) |
| eigen | eigen_symmetric | f64 | 256x256 | 144.142 ms | 1.05 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 938.183 ms | 0.16 | 0.00 | 0% | ok | 1.0x (16%) |
| eigen | eigenvalue | f64 | 256x256 | 56.554 ms | 2.97 | 0.02 | 5% | ok | 1.1x (19%) |
| eigen | schur | f64 | 256x256 | 74.261 ms | 5.65 | 0.01 | 9% | ok | 1.2x (20%) |
| eigen | svd | f64 | 256x256 | 29.690 ms | 12.43 | 0.04 | 19% | ok | 1.5x (24%) |
| eigen | expm | f64 | 256x256 | 36.302 ms | 6.93 | 0.03 | 11% | ok | 1.5x (26%) |
| eigen | sylvester | f64 | 256x256 | 210.190 ms | 4.79 | 0.01 | 7% | ok | 1.2x (20%) |
| elementwise | exp | f64 | n=8388608 | 75.654 ms | 0.11 | 1.77 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 108.025 ms | 0.08 | 1.24 | 4% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 177.688 ms | 0.05 | 0.76 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 55.138 ms | 0.15 | 2.43 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 126.345 ms | 0.07 | 1.06 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.838 ms | 3.12 | 32.50 | 96% | ok | 2.8x (46%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.022 ms | 0.87 | 9.01 | 27% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.714 ms | 8.88 | 18.68 | 55% | ok | 3.7x (61%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.422 ms | 0.16 | 0.56 | 2% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.193 ms | 0.18 | 1.49 | 4% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 53.805 ms | 0.21 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 88.416 ms | 0.13 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 36.003 ms | 0.31 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 135.088 ms | 0.17 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 153.820 ms | 0.15 | 0.00 | 0% | ok | 1.0x (16%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 60.707 ms | 0.37 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 106.012 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.116 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.005 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 181.427 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.894 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 66.443 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 7.767 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 43.378 ms | 0.26 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.124 ms | 0.36 | 1.42 | 4% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 16.380 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.821 ms | 0.67 | 0.04 | 1% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.092 ms | 0.36 | 1.45 | 4% | ok | 1.0x (17%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.612 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 116.521 ms | 0.19 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.584 ms | 0.10 | 0.40 | 1% | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 12.485 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 129.871 ms | 0.17 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.499 ms | 0.10 | 0.40 | 1% | ok | 1.0x (17%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 30.043 ms | 0.37 | 0.02 | 1% | ok | 0.9x (16%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 6.004 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 22.400 ms | 0.50 | 0.03 | 1% | ok | 0.9x (15%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.458 ms | 0.52 | 2.07 | 6% | ok | 0.9x (16%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 16.843 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 845.413 ms | 0.37 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 9.840 ms | 0.34 | 1.37 | 4% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 36.857 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 257.557 ms | 1.20 | 0.00 | 2% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.011 ms | 0.34 | 1.35 | 4% | ok | 1.0x (17%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.208 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1912.444 ms | 0.32 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.620 ms | 0.09 | 0.37 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 23.757 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1522.761 ms | 0.41 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.499 ms | 0.09 | 0.37 | 1% | ok | 1.0x (17%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 11.450 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 406.474 ms | 0.76 | 0.00 | 1% | ok | 0.2x (3%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 3.131 ms | 1.08 | 4.30 | 13% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.015 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 229.242 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 77.790 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.755 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.906 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.808 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 88.994 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.866 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.748 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 57.585 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.438 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 50.623 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 59.239 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.168 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.676 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 44.451 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.499 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.303 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.096 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 4.204 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.925 ms | 0.89 | 10.03 | 30% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.738 ms | 0.70 | 7.84 | 23% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.515 ms | 0.58 | 6.50 | 19% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 42.072 ms | 0.91 | 6.91 | 20% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 352.069 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 357.331 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 193.946 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 21.030 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 12.504 ms | 1.34 | 5.37 | 16% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.907 ms | 20.80 | 0.32 | 32% | ok | 5.8x (97%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 5.607 ms | 2.99 | 3.18 | 9% | ok | 0.8x (14%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.693 ms | 2.95 | 11.79 | n/a | ok | 1.0x (16%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.715 ms | 6.18 | 12.36 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.673 ms | 2.96 | 2.96 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.757 ms | 9.55 | 9.55 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 9.093 ms | 29.52 | 0.35 | n/a | ok | 4.2x (70%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 7.067 ms | 37.98 | 0.30 | n/a | ok | 5.8x (97%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.483 ms | 48.96 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.481 ms | 48.98 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 10.302 ms | 26.06 | 0.15 | n/a | ok | 5.8x (97%) |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

