## jetson-orin -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 10.7 GFLOP/s, f32 21.4 GFLOP/s |
| ... hardware, NEON | f64 10.7, f32 21.4 GFLOP/s |
| ... this build, scalar | f64 5.4, f32 5.4 GFLOP/s |
| bandwidth | read 11.4, update 16.1, triad 14.8 GB/s (ceiling 16.1) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 72% | 131% | 0 |
| blas2 | 10 | 64% | 124% | 0 |
| blas3 | 10 | 89% | 96% | 0 |
| factor | 12 | 42% | 61% | 0 |
| eigen | 8 | 37% | 79% | 0 |
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
| blas1 | dot | f64 | n=8388608 | 11.591 ms | 1.45 | 11.58 | 72% | ok |
| blas1 | dot | f32 | n=8388608 | 6.594 ms | 2.54 | 10.18 | 63% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.275 ms | 1.37 | 16.40 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 5.965 ms | 2.81 | 16.87 | 105% | ok |
| blas1 | scale | f64 | n=8388608 | 6.369 ms | 1.32 | 21.07 | 131% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.617 ms | 2.99 | 11.95 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.750 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 192.322 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 12.625 ms | 2.66 | 10.64 | 66% | ok |
| blas2 | gemv | f32 | n=4096 | 5.888 ms | 5.70 | 11.40 | 71% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 12.997 ms | 2.58 | 10.33 | 64% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.788 ms | 5.80 | 11.60 | 72% | ok |
| blas2 | symv | f64 | n=4096 | 15.933 ms | 2.11 | 4.22 | 26% | ok |
| blas2 | trmv | f64 | n=4096 | 8.592 ms | 1.95 | 7.82 | 48% | ok |
| blas2 | trmv | f32 | n=4096 | 4.829 ms | 3.47 | 6.96 | 43% | ok |
| blas2 | trsv | f64 | n=4096 | 8.675 ms | 1.93 | 7.74 | 48% | ok |
| blas2 | trsv | f32 | n=4096 | 4.809 ms | 3.49 | 6.98 | 43% | ok |
| blas2 | ger | f64 | n=4096 | 13.366 ms | 2.51 | 20.09 | 124% | ok |
| blas3 | gemm | f64 | 512x512x512 | 26.626 ms | 10.08 | 0.24 | 94% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.095 ms | 20.50 | 0.24 | 96% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 16.569 ms | 8.10 | 0.38 | 75% | ok |
| blas3 | symm | f64 | 512x512x512 | 26.673 ms | 10.06 | 0.24 | 94% | ok |
| blas3 | syrk | f64 | 512x512x512 | 14.488 ms | 9.26 | 0.29 | 86% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 28.029 ms | 9.58 | 0.22 | 89% | ok |
| blas3 | trmm | f64 | 512x512x512 | 13.600 ms | 9.87 | 0.39 | 92% | ok |
| blas3 | trsm | f64 | 512x512x512 | 14.936 ms | 8.99 | 0.35 | 84% | ok |
| blas3 | trsm | f32 | 512x512x512 | 7.561 ms | 17.75 | 0.35 | 83% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.805 ms | 0.54 | 4.33 | 27% | ok |
| factor | lu | f64 | 512x512 | 13.579 ms | 6.59 | 0.31 | 61% | ok |
| factor | lu | f32 | 512x512 | 8.483 ms | 10.55 | 0.25 | 49% | ok |
| factor | cholesky | f64 | 512x512 | 7.869 ms | 5.69 | 0.53 | 53% | ok |
| factor | cholesky | f32 | 512x512 | 5.022 ms | 8.91 | 0.42 | 42% | ok |
| factor | ldlt | f64 | 512x512 | 32.330 ms | 1.38 | 0.13 | 13% | ok |
| factor | ldlt_bk | f64 | 512x512 | 16.198 ms | 2.76 | 0.26 | 26% | ok |
| factor | qr | f64 | 512x512 | 28.398 ms | 6.30 | 0.15 | 59% | ok |
| factor | lq | f64 | 512x512 | 91.988 ms | 1.95 | 0.05 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 466.888 ms | 0.96 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 2.075 ms | 0.51 | 2.02 | 13% | ok |
| factor | inv | f64 | 512x512 | 173.762 ms | 1.54 | 0.02 | 14% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 14.900 ms | 6.01 | 0.28 | 56% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 10.442 ms | 2.14 | 0.10 | 20% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 145.900 ms | 1.03 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 906.800 ms | 0.17 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 63.008 ms | 2.66 | 0.02 | 25% | ok |
| eigen | schur | f64 | 256x256 | 90.221 ms | 4.65 | 0.01 | 43% | ok |
| eigen | svd | f64 | 256x256 | 43.291 ms | 8.53 | 0.02 | 79% | ok |
| eigen | expm | f64 | 256x256 | 56.515 ms | 4.45 | 0.02 | 41% | ok |
| eigen | sylvester | f64 | 256x256 | 254.718 ms | 3.95 | 0.01 | 37% | ok |
| elementwise | exp | f64 | n=8388608 | 76.034 ms | 0.11 | 1.77 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.109 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 178.515 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.237 ms | 0.15 | 2.43 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 127.352 ms | 0.07 | 1.05 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.310 ms | 1.13 | 11.79 | 73% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.023 ms | 0.87 | 9.01 | 56% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.362 ms | 2.41 | 5.07 | 31% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.139 ms | 0.16 | 0.57 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 12.229 ms | 0.16 | 1.36 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 54.721 ms | 0.20 | 0.01 | 2% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 101.376 ms | 0.11 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 36.268 ms | 0.31 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 132.023 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 151.659 ms | 0.15 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 60.737 ms | 0.37 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 105.931 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.045 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 16.888 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 182.114 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.901 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 67.452 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 7.830 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 44.511 ms | 0.25 | 0.02 | 2% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.270 ms | 0.33 | 1.33 | 8% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 16.538 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.838 ms | 0.66 | 0.04 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.163 ms | 0.35 | 1.40 | 9% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.637 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 113.184 ms | 0.20 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.711 ms | 0.10 | 0.39 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 12.299 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 128.118 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.763 ms | 0.10 | 0.39 | 2% | ok |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 28.008 ms | 0.40 | 0.03 | 4% | ok |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 5.953 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 20.387 ms | 0.55 | 0.04 | 5% | ok |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.368 ms | 0.55 | 2.21 | 14% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 16.903 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 832.420 ms | 0.37 | 0.00 | 3% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 10.506 ms | 0.32 | 1.28 | 8% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 36.943 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 259.107 ms | 1.19 | 0.00 | 11% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.379 ms | 0.32 | 1.30 | 8% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.083 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1891.821 ms | 0.33 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 38.175 ms | 0.09 | 0.35 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 23.287 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1536.867 ms | 0.40 | 0.00 | 4% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 38.067 ms | 0.09 | 0.35 | 2% | ok |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 11.433 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 79.373 ms | 3.89 | 0.01 | 36% | ok |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 3.160 ms | 1.07 | 4.26 | 26% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.106 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 168.237 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.451 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.905 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 51.374 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.952 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.103 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.238 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 50.020 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 57.678 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.607 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 50.732 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 59.394 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.170 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.892 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 45.006 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.536 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.380 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.110 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 4.032 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.916 ms | 0.90 | 10.06 | 62% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.740 ms | 0.70 | 7.84 | 49% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.527 ms | 0.58 | 6.48 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 45.049 ms | 0.85 | 6.45 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 350.067 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 357.328 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 191.303 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 21.037 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 12.503 ms | 1.34 | 5.37 | 33% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 75.387 ms | 3.56 | 0.06 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 4.573 ms | 3.67 | 3.90 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.623 ms | 2.98 | 11.93 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.706 ms | 6.20 | 12.40 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.675 ms | 2.96 | 2.96 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.758 ms | 9.54 | 9.54 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 38.146 ms | 7.04 | 0.08 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 41.160 ms | 6.52 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 31.866 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 31.867 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 60.128 ms | 4.46 | 0.03 | n/a | ok |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

