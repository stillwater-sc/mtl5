## arduino-ventuno-q -- native, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.6 GFLOP/s |
| ... this build, scalar | f64 8.4, f32 8.4 GFLOP/s |
| bandwidth | read 22.6, update 24.9, triad 23.1 GB/s (ceiling 24.9) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 84% | 104% | 0 |
| blas2 | 10 | 60% | 71% | 0 |
| blas3 | 10 | 30% | 33% | 0 |
| factor | 12 | 12% | 32% | 0 |
| eigen | 8 | 15% | 22% | 0 |
| elementwise | 5 | 5% | 7% | 0 |
| sparse | 5 | 30% | 66% | 0 |
| direct | 19 | 2% | 7% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 3% | 10% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 47% | 56% | 0 |
| mg | 1 | 36% | 36% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 34% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.668 ms | 2.96 | 23.68 | 95% | ok |
| blas1 | dot | f32 | n=8388608 | 4.551 ms | 3.69 | 14.75 | 59% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.799 ms | 2.15 | 25.81 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.948 ms | 4.25 | 25.49 | 102% | ok |
| blas1 | scale | f64 | n=8388608 | 5.323 ms | 1.58 | 25.22 | 101% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.200 ms | 5.24 | 20.97 | 84% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.562 ms | 0.98 | 7.84 | 31% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.425 ms | 0.48 | 3.85 | 15% | ok |
| blas1 | sum | f64 | n=8388608 | 8.398 ms | 1.00 | 7.99 | 32% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 200.180 ms | 0.92 | 0.67 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 7.984 ms | 4.20 | 16.82 | 68% | ok |
| blas2 | gemv | f32 | n=4096 | 5.757 ms | 5.83 | 11.66 | 47% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 9.908 ms | 3.39 | 13.55 | 54% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.512 ms | 7.44 | 14.88 | 60% | ok |
| blas2 | symv | f64 | n=4096 | 12.230 ms | 2.74 | 5.49 | 22% | ok |
| blas2 | trmv | f64 | n=4096 | 4.449 ms | 3.77 | 15.10 | 61% | ok |
| blas2 | trmv | f32 | n=4096 | 3.326 ms | 5.04 | 10.10 | 41% | ok |
| blas2 | trsv | f64 | n=4096 | 4.484 ms | 3.74 | 14.98 | 60% | ok |
| blas2 | trsv | f32 | n=4096 | 3.035 ms | 5.53 | 11.07 | 44% | ok |
| blas2 | ger | f64 | n=4096 | 15.249 ms | 2.20 | 17.61 | 71% | ok |
| blas3 | gemm | f64 | 512x512x512 | 51.829 ms | 5.18 | 0.12 | 31% | ok |
| blas3 | gemm | f32 | 512x512x512 | 24.067 ms | 11.15 | 0.13 | 33% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 40.457 ms | 3.32 | 0.16 | 20% | ok |
| blas3 | symm | f64 | 512x512x512 | 52.422 ms | 5.12 | 0.12 | 30% | ok |
| blas3 | syrk | f64 | 512x512x512 | 39.831 ms | 3.37 | 0.11 | 20% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 81.504 ms | 3.29 | 0.08 | 20% | ok |
| blas3 | trmm | f64 | 512x512x512 | 25.821 ms | 5.20 | 0.20 | 31% | ok |
| blas3 | trsm | f64 | 512x512x512 | 26.373 ms | 5.09 | 0.20 | 30% | ok |
| blas3 | trsm | f32 | 512x512x512 | 11.934 ms | 11.25 | 0.22 | 33% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 31.250 ms | 0.17 | 1.36 | 5% | ok |
| factor | lu | f64 | 512x512 | 24.106 ms | 3.71 | 0.17 | 22% | ok |
| factor | lu | f32 | 512x512 | 13.001 ms | 6.88 | 0.16 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 22.280 ms | 2.01 | 0.19 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 19.632 ms | 2.28 | 0.11 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 21.509 ms | 2.08 | 0.20 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 11.079 ms | 4.04 | 0.38 | 24% | ok |
| factor | qr | f64 | 512x512 | 250.582 ms | 0.71 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 60.680 ms | 2.95 | 0.07 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 348.934 ms | 1.28 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.560 ms | 0.67 | 2.69 | 11% | ok |
| factor | inv | f64 | 512x512 | 145.533 ms | 1.84 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.525 ms | 5.41 | 0.25 | 32% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 40.000 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.252 ms | 1.55 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 590.249 ms | 0.26 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.291 ms | 2.49 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 149.835 ms | 2.80 | 0.01 | 17% | ok |
| eigen | svd | f64 | 256x256 | 230.110 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 66.905 ms | 3.76 | 0.02 | 22% | ok |
| eigen | sylvester | f64 | 256x256 | 368.935 ms | 2.73 | 0.00 | 16% | ok |
| elementwise | exp | f64 | n=8388608 | 89.478 ms | 0.09 | 1.50 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 111.117 ms | 0.08 | 1.21 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 150.579 ms | 0.06 | 0.89 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 73.496 ms | 0.11 | 1.83 | 7% | ok |
| elementwise | erf | f64 | n=8388608 | 118.923 ms | 0.07 | 1.13 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.666 ms | 1.57 | 16.34 | 66% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.136 ms | 1.23 | 12.75 | 51% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.978 ms | 3.50 | 7.35 | 30% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.015 ms | 0.25 | 0.88 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 9.070 ms | 0.22 | 1.84 | 7% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 36.850 ms | 0.30 | 0.02 | 2% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 58.758 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.310 ms | 0.43 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 104.823 ms | 0.21 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 121.608 ms | 0.18 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.592 ms | 0.51 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 74.056 ms | 0.03 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.688 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.327 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 124.896 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.583 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 43.666 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.512 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 29.341 ms | 0.38 | 0.02 | 2% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.828 ms | 0.41 | 1.65 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.240 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.937 ms | 0.94 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.782 ms | 0.42 | 1.70 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.405 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 89.851 ms | 0.25 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.933 ms | 0.13 | 0.51 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 8.656 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 101.856 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.944 ms | 0.13 | 0.51 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 11.817 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 666.065 ms | 0.46 | 0.00 | 3% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.704 ms | 0.44 | 1.75 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 27.711 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 185.373 ms | 1.66 | 0.00 | 10% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.643 ms | 0.44 | 1.76 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.693 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1683.838 ms | 0.37 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 30.170 ms | 0.11 | 0.45 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.572 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1231.413 ms | 0.50 | 0.00 | 3% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 30.417 ms | 0.11 | 0.44 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 71.431 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 161.640 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 54.677 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 29.010 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.876 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 34.341 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 62.898 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 64.151 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 35.106 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 39.523 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.959 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.011 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 41.165 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.112 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 5.015 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 30.690 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.141 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 3.019 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.066 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.787 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.121 ms | 1.23 | 13.83 | 56% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.482 ms | 1.05 | 11.81 | 47% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 5.025 ms | 0.52 | 5.84 | 23% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.301 ms | 1.19 | 9.00 | 36% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 231.898 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 235.815 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 144.754 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.309 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 8.033 ms | 2.09 | 8.35 | 34% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 48.585 ms | 5.53 | 0.09 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.942 ms | 5.70 | 6.06 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.851 ms | 5.88 | 23.54 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.408 ms | 11.92 | 23.83 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.653 ms | 4.59 | 4.59 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.167 ms | 14.38 | 14.38 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 24.749 ms | 10.85 | 0.13 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 26.376 ms | 10.18 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 20.312 ms | 13.22 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 20.317 ms | 13.21 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 38.469 ms | 6.98 | 0.04 | n/a | ok |

