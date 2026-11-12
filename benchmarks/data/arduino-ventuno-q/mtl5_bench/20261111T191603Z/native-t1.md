## arduino-ventuno-q -- native, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.6 GFLOP/s |
| ... this build, scalar | f64 8.4, f32 8.4 GFLOP/s |
| bandwidth | read 23.1, update 24.8, triad 23.4 GB/s (ceiling 24.8) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 83% | 104% | 0 |
| blas2 | 10 | 55% | 71% | 0 |
| blas3 | 10 | 30% | 33% | 0 |
| factor | 12 | 12% | 25% | 0 |
| eigen | 8 | 15% | 22% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 30% | 66% | 0 |
| direct | 7 | 1% | 3% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 47% | 56% | 0 |
| mg | 1 | 36% | 36% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 33% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.648 ms | 2.97 | 23.77 | 96% | ok |
| blas1 | dot | f32 | n=8388608 | 4.517 ms | 3.71 | 14.86 | 60% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.810 ms | 2.15 | 25.78 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.952 ms | 4.25 | 25.47 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 5.350 ms | 1.57 | 25.09 | 101% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.237 ms | 5.18 | 20.73 | 83% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.187 ms | 1.02 | 8.20 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.287 ms | 0.49 | 3.88 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.362 ms | 1.00 | 8.03 | 32% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 196.176 ms | 0.94 | 0.68 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 9.257 ms | 3.62 | 14.51 | 58% | ok |
| blas2 | gemv | f32 | n=4096 | 6.915 ms | 4.85 | 9.71 | 39% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 9.817 ms | 3.42 | 13.68 | 55% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.329 ms | 7.75 | 15.51 | 62% | ok |
| blas2 | symv | f64 | n=4096 | 16.936 ms | 1.98 | 3.97 | 16% | ok |
| blas2 | trmv | f64 | n=4096 | 4.955 ms | 3.39 | 13.56 | 55% | ok |
| blas2 | trmv | f32 | n=4096 | 6.805 ms | 2.47 | 4.94 | 20% | ok |
| blas2 | trsv | f64 | n=4096 | 4.954 ms | 3.39 | 13.56 | 55% | ok |
| blas2 | trsv | f32 | n=4096 | 20.718 ms | 0.81 | 1.62 | 7% | ok |
| blas2 | ger | f64 | n=4096 | 15.256 ms | 2.20 | 17.60 | 71% | ok |
| blas3 | gemm | f64 | 512x512x512 | 53.873 ms | 4.98 | 0.12 | 30% | ok |
| blas3 | gemm | f32 | 512x512x512 | 25.232 ms | 10.64 | 0.12 | 32% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 41.049 ms | 3.27 | 0.15 | 20% | ok |
| blas3 | symm | f64 | 512x512x512 | 52.486 ms | 5.11 | 0.12 | 31% | ok |
| blas3 | syrk | f64 | 512x512x512 | 43.366 ms | 3.10 | 0.10 | 18% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 79.913 ms | 3.36 | 0.08 | 20% | ok |
| blas3 | trmm | f64 | 512x512x512 | 25.870 ms | 5.19 | 0.20 | 31% | ok |
| blas3 | trsm | f64 | 512x512x512 | 26.327 ms | 5.10 | 0.20 | 30% | ok |
| blas3 | trsm | f32 | 512x512x512 | 12.115 ms | 11.08 | 0.22 | 33% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 32.062 ms | 0.17 | 1.32 | 5% | ok |
| factor | lu | f64 | 512x512 | 26.158 ms | 3.42 | 0.16 | 20% | ok |
| factor | lu | f32 | 512x512 | 13.186 ms | 6.79 | 0.16 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 22.459 ms | 1.99 | 0.19 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 19.631 ms | 2.28 | 0.11 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 21.788 ms | 2.05 | 0.19 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 69.248 ms | 0.65 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 249.655 ms | 0.72 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 60.831 ms | 2.94 | 0.07 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 347.591 ms | 1.29 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.545 ms | 0.68 | 2.71 | 11% | ok |
| factor | inv | f64 | 512x512 | 145.030 ms | 1.85 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 21.397 ms | 4.18 | 0.20 | 25% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.945 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.285 ms | 1.55 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 592.811 ms | 0.25 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.319 ms | 2.49 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 151.903 ms | 2.76 | 0.01 | 16% | ok |
| eigen | svd | f64 | 256x256 | 231.130 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 67.411 ms | 3.73 | 0.02 | 22% | ok |
| eigen | sylvester | f64 | 256x256 | 379.393 ms | 2.65 | 0.00 | 16% | ok |
| elementwise | exp | f64 | n=8388608 | 89.134 ms | 0.09 | 1.51 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 110.964 ms | 0.08 | 1.21 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 149.831 ms | 0.06 | 0.90 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 71.381 ms | 0.12 | 1.88 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 117.077 ms | 0.07 | 1.15 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.669 ms | 1.57 | 16.32 | 66% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.179 ms | 1.20 | 12.50 | 50% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.007 ms | 3.49 | 7.33 | 30% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.369 ms | 0.24 | 0.86 | 3% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.765 ms | 0.23 | 1.90 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 181.876 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 58.962 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.333 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 103.207 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 139.816 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.235 ms | 0.52 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 207.110 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.643 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.782 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.705 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.875 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.463 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 103.856 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 53.914 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 29.059 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.443 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 33.705 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 63.674 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.987 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.765 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 36.992 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.997 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.313 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 47.442 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.112 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 5.231 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 30.495 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.058 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.952 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.063 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.731 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.109 ms | 1.24 | 13.91 | 56% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.496 ms | 1.05 | 11.75 | 47% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 5.035 ms | 0.52 | 5.82 | 23% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.432 ms | 1.19 | 8.96 | 36% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 231.698 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 270.586 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 164.754 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.426 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 8.135 ms | 2.06 | 8.25 | 33% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 48.743 ms | 5.51 | 0.09 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.942 ms | 5.70 | 6.06 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.846 ms | 5.89 | 23.58 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.399 ms | 11.99 | 23.99 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.654 ms | 4.59 | 4.59 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.150 ms | 14.58 | 14.58 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 26.726 ms | 10.04 | 0.12 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 26.354 ms | 10.19 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 20.313 ms | 13.22 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 20.297 ms | 13.23 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 38.404 ms | 6.99 | 0.04 | n/a | ok |

