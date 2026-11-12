## arduino-ventuno-q -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON_WITHOUT_AES |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.5 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 16.8, f32 33.6 GFLOP/s |
| bandwidth | read 22.6, update 24.9, triad 23.2 GB/s (ceiling 24.9) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 94% | 104% | 0 |
| blas2 | 10 | 54% | 69% | 0 |
| blas3 | 10 | 45% | 58% | 0 |
| factor | 12 | 12% | 25% | 0 |
| eigen | 8 | 15% | 31% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 29% | 67% | 0 |
| direct | 7 | 1% | 3% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 47% | 56% | 0 |
| mg | 1 | 37% | 37% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 38% | 66% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.676 ms | 2.96 | 23.65 | 95% | ok |
| blas1 | dot | f32 | n=8388608 | 2.874 ms | 5.84 | 23.35 | 94% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.802 ms | 2.15 | 25.80 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.944 ms | 4.25 | 25.52 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 5.252 ms | 1.60 | 25.56 | 103% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.078 ms | 5.45 | 21.80 | 88% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.376 ms | 1.00 | 8.01 | 32% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.558 ms | 0.48 | 3.82 | 15% | ok |
| blas1 | sum | f64 | n=8388608 | 8.148 ms | 1.03 | 8.24 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 198.924 ms | 0.93 | 0.67 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 8.371 ms | 4.01 | 16.04 | 65% | ok |
| blas2 | gemv | f32 | n=4096 | 4.064 ms | 8.26 | 16.52 | 66% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 49.616 ms | 0.68 | 2.71 | 11% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 24.640 ms | 1.36 | 2.72 | 11% | ok |
| blas2 | symv | f64 | n=4096 | 17.224 ms | 1.95 | 3.90 | 16% | ok |
| blas2 | trmv | f64 | n=4096 | 5.033 ms | 3.33 | 13.35 | 54% | ok |
| blas2 | trmv | f32 | n=4096 | 2.617 ms | 6.41 | 12.83 | 52% | ok |
| blas2 | trsv | f64 | n=4096 | 4.868 ms | 3.45 | 13.80 | 56% | ok |
| blas2 | trsv | f32 | n=4096 | 2.653 ms | 6.32 | 12.66 | 51% | ok |
| blas2 | ger | f64 | n=4096 | 15.595 ms | 2.15 | 17.22 | 69% | ok |
| blas3 | gemm | f64 | 512x512x512 | 28.161 ms | 9.53 | 0.22 | 57% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.831 ms | 19.41 | 0.23 | 58% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 17.801 ms | 7.54 | 0.35 | 45% | ok |
| blas3 | symm | f64 | 512x512x512 | 28.985 ms | 9.26 | 0.22 | 55% | ok |
| blas3 | syrk | f64 | 512x512x512 | 42.563 ms | 3.15 | 0.10 | 19% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 81.656 ms | 3.29 | 0.08 | 20% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.715 ms | 7.58 | 0.30 | 45% | ok |
| blas3 | trsm | f64 | 512x512x512 | 17.905 ms | 7.50 | 0.29 | 45% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.532 ms | 15.73 | 0.31 | 47% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 30.647 ms | 0.17 | 1.39 | 6% | ok |
| factor | lu | f64 | 512x512 | 24.687 ms | 3.62 | 0.17 | 22% | ok |
| factor | lu | f32 | 512x512 | 13.224 ms | 6.77 | 0.16 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 22.256 ms | 2.01 | 0.19 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 19.635 ms | 2.28 | 0.11 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 22.613 ms | 1.98 | 0.19 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 68.770 ms | 0.65 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 252.740 ms | 0.71 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 60.617 ms | 2.95 | 0.07 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 352.106 ms | 1.27 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.578 ms | 0.67 | 2.66 | 11% | ok |
| factor | inv | f64 | 512x512 | 145.579 ms | 1.84 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 20.939 ms | 4.27 | 0.20 | 25% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.872 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.007 ms | 1.56 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 612.887 ms | 0.25 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.202 ms | 2.50 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 149.842 ms | 2.80 | 0.01 | 17% | ok |
| eigen | svd | f64 | 256x256 | 230.314 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 47.953 ms | 5.25 | 0.02 | 31% | ok |
| eigen | sylvester | f64 | 256x256 | 356.540 ms | 2.82 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 87.845 ms | 0.10 | 1.53 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 109.993 ms | 0.08 | 1.22 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 147.246 ms | 0.06 | 0.91 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 71.384 ms | 0.12 | 1.88 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 116.753 ms | 0.07 | 1.15 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.639 ms | 1.60 | 16.61 | 67% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.138 ms | 1.22 | 12.73 | 51% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.018 ms | 3.48 | 7.33 | 29% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.391 ms | 0.24 | 0.86 | 3% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.409 ms | 0.24 | 1.98 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 182.608 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 61.413 ms | 0.18 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.042 ms | 0.43 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 102.862 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 139.992 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.281 ms | 0.52 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 204.914 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.357 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.072 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.695 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.222 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 71.016 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 106.816 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 54.363 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 28.813 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.541 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 33.785 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 63.297 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.395 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.466 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 37.371 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.568 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.672 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 47.214 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.111 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.841 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 30.723 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.057 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.941 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.064 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.736 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.090 ms | 1.25 | 14.03 | 56% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.498 ms | 1.05 | 11.74 | 47% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.948 ms | 0.89 | 9.95 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 31.860 ms | 1.21 | 9.12 | 37% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 235.103 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 273.155 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 160.436 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.199 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.110 ms | 4.08 | 16.33 | 66% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 41.736 ms | 6.43 | 0.10 | 38% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.656 ms | 6.32 | 6.71 | 38% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.842 ms | 5.90 | 23.62 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.388 ms | 12.09 | 24.18 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.331 ms | 7.20 | 7.20 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.256 ms | 3.94 | 3.94 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.322 ms | 15.50 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 16.664 ms | 16.11 | 0.13 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 16.562 ms | 16.21 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 23.628 ms | 11.36 | 0.07 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 50.783 ms | 5.29 | 0.03 | n/a | ok |

