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
| bandwidth | read 22.8, update 24.8, triad 23.5 GB/s (ceiling 24.8) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 86% | 104% | 0 |
| blas2 | 10 | 53% | 70% | 0 |
| blas3 | 10 | 31% | 33% | 0 |
| factor | 12 | 12% | 25% | 0 |
| eigen | 8 | 15% | 22% | 0 |
| elementwise | 5 | 5% | 7% | 0 |
| sparse | 5 | 29% | 67% | 0 |
| direct | 7 | 1% | 3% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 57% | 0 |
| mg | 1 | 36% | 36% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 34% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.645 ms | 2.97 | 23.78 | 96% | ok |
| blas1 | dot | f32 | n=8388608 | 4.301 ms | 3.90 | 15.60 | 63% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.825 ms | 2.14 | 25.73 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.959 ms | 4.24 | 25.42 | 102% | ok |
| blas1 | scale | f64 | n=8388608 | 5.317 ms | 1.58 | 25.24 | 102% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.156 ms | 5.32 | 21.26 | 86% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.303 ms | 1.01 | 8.08 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.389 ms | 0.48 | 3.86 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.437 ms | 0.99 | 7.95 | 32% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 197.318 ms | 0.94 | 0.68 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 9.490 ms | 3.54 | 14.15 | 57% | ok |
| blas2 | gemv | f32 | n=4096 | 7.043 ms | 4.76 | 9.53 | 38% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 9.812 ms | 3.42 | 13.69 | 55% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.578 ms | 7.33 | 14.67 | 59% | ok |
| blas2 | symv | f64 | n=4096 | 17.375 ms | 1.93 | 3.87 | 16% | ok |
| blas2 | trmv | f64 | n=4096 | 5.128 ms | 3.27 | 13.10 | 53% | ok |
| blas2 | trmv | f32 | n=4096 | 6.863 ms | 2.44 | 4.89 | 20% | ok |
| blas2 | trsv | f64 | n=4096 | 5.160 ms | 3.25 | 13.02 | 52% | ok |
| blas2 | trsv | f32 | n=4096 | 20.807 ms | 0.81 | 1.61 | 7% | ok |
| blas2 | ger | f64 | n=4096 | 15.354 ms | 2.19 | 17.49 | 70% | ok |
| blas3 | gemm | f64 | 512x512x512 | 51.977 ms | 5.16 | 0.12 | 31% | ok |
| blas3 | gemm | f32 | 512x512x512 | 24.970 ms | 10.75 | 0.13 | 32% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 38.900 ms | 3.45 | 0.16 | 21% | ok |
| blas3 | symm | f64 | 512x512x512 | 53.379 ms | 5.03 | 0.12 | 30% | ok |
| blas3 | syrk | f64 | 512x512x512 | 40.900 ms | 3.28 | 0.10 | 20% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 79.004 ms | 3.40 | 0.08 | 20% | ok |
| blas3 | trmm | f64 | 512x512x512 | 25.960 ms | 5.17 | 0.20 | 31% | ok |
| blas3 | trsm | f64 | 512x512x512 | 26.104 ms | 5.14 | 0.20 | 31% | ok |
| blas3 | trsm | f32 | 512x512x512 | 12.129 ms | 11.07 | 0.22 | 33% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 30.489 ms | 0.17 | 1.39 | 6% | ok |
| factor | lu | f64 | 512x512 | 25.441 ms | 3.52 | 0.16 | 21% | ok |
| factor | lu | f32 | 512x512 | 13.191 ms | 6.78 | 0.16 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 22.212 ms | 2.01 | 0.19 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 19.621 ms | 2.28 | 0.11 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 21.430 ms | 2.09 | 0.20 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 68.692 ms | 0.65 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 249.357 ms | 0.72 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 60.909 ms | 2.94 | 0.07 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 347.275 ms | 1.29 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.542 ms | 0.68 | 2.72 | 11% | ok |
| factor | inv | f64 | 512x512 | 147.412 ms | 1.82 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 21.071 ms | 4.25 | 0.20 | 25% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.815 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.044 ms | 1.56 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 591.530 ms | 0.26 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.208 ms | 2.50 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 150.393 ms | 2.79 | 0.01 | 17% | ok |
| eigen | svd | f64 | 256x256 | 230.525 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 67.537 ms | 3.73 | 0.02 | 22% | ok |
| eigen | sylvester | f64 | 256x256 | 370.132 ms | 2.72 | 0.00 | 16% | ok |
| elementwise | exp | f64 | n=8388608 | 87.629 ms | 0.10 | 1.53 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 111.960 ms | 0.07 | 1.20 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 148.325 ms | 0.06 | 0.90 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 72.164 ms | 0.12 | 1.86 | 7% | ok |
| elementwise | erf | f64 | n=8388608 | 117.684 ms | 0.07 | 1.14 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.640 ms | 1.60 | 16.61 | 67% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.157 ms | 1.21 | 12.63 | 51% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.101 ms | 3.46 | 7.28 | 29% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.314 ms | 0.24 | 0.86 | 3% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.783 ms | 0.23 | 1.90 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 181.733 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 59.014 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.603 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 103.876 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 140.222 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.142 ms | 0.52 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 200.762 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.602 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.216 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.696 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.376 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.548 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 104.223 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 53.959 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 28.889 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.305 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 33.932 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 63.901 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.711 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.359 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 37.477 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 36.213 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.707 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 47.575 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.113 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.833 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 30.575 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.053 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.917 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.064 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.743 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.077 ms | 1.26 | 14.12 | 57% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.482 ms | 1.05 | 11.82 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 5.024 ms | 0.52 | 5.84 | 24% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.454 ms | 1.18 | 8.96 | 36% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 231.574 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 270.304 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 161.327 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.405 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 8.066 ms | 2.08 | 8.32 | 34% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 48.644 ms | 5.52 | 0.09 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.939 ms | 5.71 | 6.06 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.853 ms | 5.88 | 23.52 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.384 ms | 12.12 | 24.25 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.649 ms | 4.60 | 4.60 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.181 ms | 14.20 | 14.20 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 24.668 ms | 10.88 | 0.13 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 26.359 ms | 10.18 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 20.297 ms | 13.23 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 20.294 ms | 13.23 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 38.432 ms | 6.98 | 0.04 | n/a | ok |

