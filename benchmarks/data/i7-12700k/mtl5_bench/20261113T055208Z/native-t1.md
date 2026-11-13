## i7-12700k -- native, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `8867b44b3f52` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.4 GFLOP/s |
| bandwidth | read 33.3, update 38.7, triad 32.6 GB/s (ceiling 38.7) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 44% | 97% | 0 |
| blas2 | 10 | 79% | 96% | 0 |
| blas3 | 10 | 17% | 18% | 0 |
| factor | 12 | 8% | 19% | 0 |
| eigen | 8 | 6% | 12% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 49% | 95% | 0 |
| direct | 19 | 1% | 31% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 24% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 62% | 77% | 0 |
| mg | 1 | 57% | 57% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 35% | 85% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.610 ms | 2.99 | 23.92 | 62% | ok |
| blas1 | dot | f32 | n=8388608 | 4.212 ms | 3.98 | 15.93 | 41% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.300 ms | 2.66 | 31.95 | 83% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.683 ms | 3.58 | 21.49 | 56% | ok |
| blas1 | scale | f64 | n=8388608 | 3.570 ms | 2.35 | 37.60 | 97% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.845 ms | 3.46 | 13.85 | 36% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.007 ms | 2.09 | 16.75 | 43% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.033 ms | 1.19 | 9.54 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 3.979 ms | 2.11 | 16.87 | 44% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 76.337 ms | 2.42 | 1.76 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.625 ms | 9.26 | 37.04 | 96% | ok |
| blas2 | gemv | f32 | n=4096 | 2.756 ms | 12.17 | 24.36 | 63% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 4.703 ms | 7.13 | 28.55 | 74% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.203 ms | 15.23 | 30.48 | 79% | ok |
| blas2 | symv | f64 | n=4096 | 6.600 ms | 5.08 | 10.18 | 26% | ok |
| blas2 | trmv | f64 | n=4096 | 2.017 ms | 8.32 | 33.30 | 86% | ok |
| blas2 | trmv | f32 | n=4096 | 1.423 ms | 11.79 | 23.60 | 61% | ok |
| blas2 | trsv | f64 | n=4096 | 1.970 ms | 8.52 | 34.11 | 88% | ok |
| blas2 | trsv | f32 | n=4096 | 1.387 ms | 12.10 | 24.22 | 63% | ok |
| blas2 | ger | f64 | n=4096 | 7.666 ms | 4.38 | 35.03 | 90% | ok |
| blas3 | gemm | f64 | 512x512x512 | 20.043 ms | 13.39 | 0.31 | 17% | ok |
| blas3 | gemm | f32 | 512x512x512 | 10.617 ms | 25.28 | 0.30 | 16% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 11.040 ms | 12.16 | 0.57 | 16% | ok |
| blas3 | symm | f64 | 512x512x512 | 18.549 ms | 14.47 | 0.34 | 18% | ok |
| blas3 | syrk | f64 | 512x512x512 | 13.518 ms | 9.93 | 0.31 | 13% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 24.546 ms | 10.94 | 0.26 | 14% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.948 ms | 13.49 | 0.53 | 17% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.754 ms | 13.76 | 0.54 | 18% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.137 ms | 26.13 | 0.51 | 17% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.102 ms | 0.41 | 3.24 | 8% | ok |
| factor | lu | f64 | 512x512 | 11.572 ms | 7.73 | 0.36 | 10% | ok |
| factor | lu | f32 | 512x512 | 4.527 ms | 19.77 | 0.46 | 13% | ok |
| factor | cholesky | f64 | 512x512 | 7.332 ms | 6.10 | 0.57 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.605 ms | 6.77 | 0.32 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.466 ms | 5.99 | 0.56 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 47.408 ms | 0.94 | 0.09 | 1% | ok |
| factor | qr | f64 | 512x512 | 229.944 ms | 0.78 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 25.829 ms | 6.93 | 0.16 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 270.702 ms | 1.65 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.622 ms | 0.65 | 2.59 | 7%\* | ok |
| factor | inv | f64 | 512x512 | 54.056 ms | 4.97 | 0.08 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.175 ms | 14.49 | 0.68 | 19% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 18.339 ms | 1.22 | 0.06 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 59.157 ms | 2.55 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 752.217 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 34.236 ms | 4.90 | 0.03 | 6% | ok |
| eigen | schur | f64 | 256x256 | 86.008 ms | 4.88 | 0.01 | 6% | ok |
| eigen | svd | f64 | 256x256 | 112.238 ms | 3.29 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 26.494 ms | 9.50 | 0.04 | 12% | ok |
| eigen | sylvester | f64 | 256x256 | 190.677 ms | 5.28 | 0.01 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 39.358 ms | 0.21 | 3.41 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 45.822 ms | 0.18 | 2.93 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 60.010 ms | 0.14 | 2.24 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 28.988 ms | 0.29 | 4.63 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 201.496 ms | 0.04 | 0.67 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.743 ms | 3.52 | 36.67 | 95% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.929 ms | 2.82 | 29.31 | 76% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.669 ms | 8.97 | 18.86 | 49% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.853 ms | 0.67 | 2.37 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.248 ms | 1.61 | 13.34 | 34%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 91.297 ms | 0.12 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.489 ms | 0.44 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.890 ms | 1.26 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.994 ms | 0.56 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 53.545 ms | 0.42 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 17.422 ms | 1.28 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 99.476 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.389 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.021 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.226 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.546 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.362 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 88.520 ms | 0.13 | 0.01 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.248 ms | 3.04 | 12.17 | 31%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.463 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.183 ms | 2.67 | 0.17 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.248 ms | 3.05 | 12.19 | 31%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.096 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.766 ms | 0.63 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.629 ms | 1.20 | 4.81 | 12%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.908 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.716 ms | 0.47 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.637 ms | 1.19 | 4.75 | 12%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 9.548 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 7435.889 ms | 0.04 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.427 ms | 2.36 | 9.44 | 24%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 18.809 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 67.616 ms | 4.56 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.448 ms | 2.33 | 9.31 | 24%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.864 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 555.493 ms | 1.11 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.524 ms | 0.45 | 1.79 | 5%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.653 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 642.588 ms | 0.96 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.516 ms | 0.45 | 1.79 | 5%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 31.250 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 33.290 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 23.032 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.514 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 15.132 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 14.026 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.817 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.460 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.372 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 17.299 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.531 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.965 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.295 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.035 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.673 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.364 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.072 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.005 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.707 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.980 ms | 2.67 | 29.94 | 77% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.217 ms | 2.15 | 24.10 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.562 ms | 1.68 | 18.78 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.248 ms | 2.90 | 21.94 | 57% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 30.185 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 45.271 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 34.900 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 5.019 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.898 ms | 3.43 | 13.70 | 35% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 38.984 ms | 6.89 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.544 ms | 30.86 | 32.79 | 85%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.301 ms | 7.29 | 29.17 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.992 ms | 16.91 | 33.81 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.720 ms | 9.75 | 9.75 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.720 ms | 9.75 | 9.75 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.428 ms | 15.40 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.822 ms | 12.89 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.239 ms | 26.22 | 0.15 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.240 ms | 26.21 | 0.15 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 26.855 ms | 10.00 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

