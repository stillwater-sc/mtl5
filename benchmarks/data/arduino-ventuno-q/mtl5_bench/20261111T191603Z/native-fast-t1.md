## arduino-ventuno-q -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON_WITHOUT_AES |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.6 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 16.8, f32 33.6 GFLOP/s |
| bandwidth | read 22.7, update 24.9, triad 23.3 GB/s (ceiling 24.9) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 94% | 104% | 0 |
| blas2 | 10 | 55% | 71% | 0 |
| blas3 | 10 | 45% | 58% | 0 |
| factor | 12 | 12% | 26% | 0 |
| eigen | 8 | 15% | 31% | 0 |
| elementwise | 5 | 5% | 7% | 0 |
| sparse | 5 | 29% | 68% | 0 |
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
| blas1 | dot | f64 | n=8388608 | 5.701 ms | 2.94 | 23.54 | 95% | ok |
| blas1 | dot | f32 | n=8388608 | 2.878 ms | 5.83 | 23.31 | 94% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.801 ms | 2.15 | 25.81 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.941 ms | 4.26 | 25.54 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 5.332 ms | 1.57 | 25.17 | 101% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.161 ms | 5.31 | 21.23 | 85% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.590 ms | 0.98 | 7.81 | 31% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.349 ms | 0.48 | 3.87 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.755 ms | 0.96 | 7.67 | 31% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 198.557 ms | 0.93 | 0.68 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 8.251 ms | 4.07 | 16.28 | 65% | ok |
| blas2 | gemv | f32 | n=4096 | 4.052 ms | 8.28 | 16.57 | 67% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 49.182 ms | 0.68 | 2.73 | 11% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 25.699 ms | 1.31 | 2.61 | 11% | ok |
| blas2 | symv | f64 | n=4096 | 16.930 ms | 1.98 | 3.97 | 16% | ok |
| blas2 | trmv | f64 | n=4096 | 4.952 ms | 3.39 | 13.56 | 55% | ok |
| blas2 | trmv | f32 | n=4096 | 2.592 ms | 6.47 | 12.96 | 52% | ok |
| blas2 | trsv | f64 | n=4096 | 4.788 ms | 3.50 | 14.03 | 56% | ok |
| blas2 | trsv | f32 | n=4096 | 2.593 ms | 6.47 | 12.95 | 52% | ok |
| blas2 | ger | f64 | n=4096 | 15.218 ms | 2.20 | 17.64 | 71% | ok |
| blas3 | gemm | f64 | 512x512x512 | 28.189 ms | 9.52 | 0.22 | 57% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.842 ms | 19.39 | 0.23 | 58% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 17.957 ms | 7.47 | 0.35 | 44% | ok |
| blas3 | symm | f64 | 512x512x512 | 29.177 ms | 9.20 | 0.22 | 55% | ok |
| blas3 | syrk | f64 | 512x512x512 | 41.220 ms | 3.26 | 0.10 | 19% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 79.548 ms | 3.37 | 0.08 | 20% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.649 ms | 7.60 | 0.30 | 45% | ok |
| blas3 | trsm | f64 | 512x512x512 | 18.067 ms | 7.43 | 0.29 | 44% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.526 ms | 15.74 | 0.31 | 47% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 31.380 ms | 0.17 | 1.35 | 5% | ok |
| factor | lu | f64 | 512x512 | 24.602 ms | 3.64 | 0.17 | 22% | ok |
| factor | lu | f32 | 512x512 | 13.196 ms | 6.78 | 0.16 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 22.260 ms | 2.01 | 0.19 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 19.666 ms | 2.27 | 0.11 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 22.616 ms | 1.98 | 0.19 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 67.975 ms | 0.66 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 249.647 ms | 0.72 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 60.589 ms | 2.95 | 0.07 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 347.760 ms | 1.29 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.553 ms | 0.68 | 2.70 | 11% | ok |
| factor | inv | f64 | 512x512 | 143.459 ms | 1.87 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 20.862 ms | 4.29 | 0.20 | 26% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.882 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 96.701 ms | 1.56 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 604.340 ms | 0.25 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.220 ms | 2.50 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 149.906 ms | 2.80 | 0.01 | 17% | ok |
| eigen | svd | f64 | 256x256 | 230.064 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 48.004 ms | 5.24 | 0.02 | 31% | ok |
| eigen | sylvester | f64 | 256x256 | 355.257 ms | 2.83 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 89.862 ms | 0.09 | 1.49 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 110.900 ms | 0.08 | 1.21 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 149.958 ms | 0.06 | 0.90 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 71.961 ms | 0.12 | 1.87 | 7% | ok |
| elementwise | erf | f64 | n=8388608 | 117.191 ms | 0.07 | 1.15 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.620 ms | 1.62 | 16.81 | 68% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.156 ms | 1.21 | 12.63 | 51% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.029 ms | 3.48 | 7.32 | 29% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.383 ms | 0.24 | 0.86 | 3% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.460 ms | 0.24 | 1.97 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 181.889 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 61.650 ms | 0.18 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.561 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 103.552 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 141.067 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.271 ms | 0.52 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 203.997 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.748 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 14.105 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.712 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 44.046 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.892 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 104.593 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 53.819 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 28.929 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.449 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 33.866 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 63.542 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.128 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.728 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 37.230 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.682 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.597 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 47.039 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.114 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.882 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 31.046 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.058 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.905 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.064 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.722 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.102 ms | 1.25 | 13.95 | 56% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.489 ms | 1.05 | 11.78 | 47% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.941 ms | 0.89 | 9.97 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.006 ms | 1.20 | 9.08 | 37% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 232.202 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 270.564 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 149.555 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.228 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.079 ms | 4.11 | 16.45 | 66% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 41.762 ms | 6.43 | 0.10 | 38% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.652 ms | 6.33 | 6.72 | 38% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.825 ms | 5.94 | 23.76 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.388 ms | 12.08 | 24.17 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.320 ms | 7.23 | 7.23 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.255 ms | 3.94 | 3.94 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.316 ms | 15.50 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 16.639 ms | 16.13 | 0.13 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 16.573 ms | 16.20 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 23.623 ms | 11.36 | 0.07 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 50.869 ms | 5.28 | 0.03 | n/a | ok |

