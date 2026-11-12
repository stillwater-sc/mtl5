## jetson-orin -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 10.7 GFLOP/s, f32 21.5 GFLOP/s |
| ... hardware, NEON | f64 10.7, f32 21.5 GFLOP/s |
| ... this build, scalar | f64 5.4, f32 5.4 GFLOP/s |
| bandwidth | read 11.5, update 16.2, triad 14.8 GB/s (ceiling 16.2) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 72% | 131% | 0 |
| blas2 | 10 | 64% | 123% | 0 |
| blas3 | 10 | 89% | 95% | 0 |
| factor | 12 | 42% | 61% | 0 |
| eigen | 8 | 37% | 80% | 0 |
| elementwise | 5 | 8% | 15% | 0 |
| sparse | 5 | 32% | 74% | 0 |
| direct | 7 | 1% | 3% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 49% | 60% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 33% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.577 ms | 1.45 | 11.59 | 72% | ok |
| blas1 | dot | f32 | n=8388608 | 6.595 ms | 2.54 | 10.18 | 63% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.253 ms | 1.37 | 16.43 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 5.962 ms | 2.81 | 16.88 | 105% | ok |
| blas1 | scale | f64 | n=8388608 | 6.352 ms | 1.32 | 21.13 | 131% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.602 ms | 2.99 | 11.98 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.503 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.755 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.504 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 192.923 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 12.616 ms | 2.66 | 10.64 | 66% | ok |
| blas2 | gemv | f32 | n=4096 | 5.872 ms | 5.71 | 11.44 | 71% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 12.930 ms | 2.60 | 10.39 | 64% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.844 ms | 5.74 | 11.49 | 71% | ok |
| blas2 | symv | f64 | n=4096 | 15.946 ms | 2.10 | 4.21 | 26% | ok |
| blas2 | trmv | f64 | n=4096 | 8.644 ms | 1.94 | 7.77 | 48% | ok |
| blas2 | trmv | f32 | n=4096 | 4.867 ms | 3.45 | 6.90 | 43% | ok |
| blas2 | trsv | f64 | n=4096 | 8.626 ms | 1.95 | 7.79 | 48% | ok |
| blas2 | trsv | f32 | n=4096 | 4.849 ms | 3.46 | 6.93 | 43% | ok |
| blas2 | ger | f64 | n=4096 | 13.527 ms | 2.48 | 19.85 | 123% | ok |
| blas3 | gemm | f64 | 512x512x512 | 26.634 ms | 10.08 | 0.24 | 94% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.109 ms | 20.48 | 0.24 | 95% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 16.563 ms | 8.10 | 0.38 | 76% | ok |
| blas3 | symm | f64 | 512x512x512 | 26.622 ms | 10.08 | 0.24 | 94% | ok |
| blas3 | syrk | f64 | 512x512x512 | 14.417 ms | 9.31 | 0.29 | 87% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 28.107 ms | 9.55 | 0.22 | 89% | ok |
| blas3 | trmm | f64 | 512x512x512 | 13.614 ms | 9.86 | 0.39 | 92% | ok |
| blas3 | trsm | f64 | 512x512x512 | 14.962 ms | 8.97 | 0.35 | 84% | ok |
| blas3 | trsm | f32 | 512x512x512 | 7.557 ms | 17.76 | 0.35 | 83% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.903 ms | 0.54 | 4.29 | 27% | ok |
| factor | lu | f64 | 512x512 | 13.670 ms | 6.55 | 0.31 | 61% | ok |
| factor | lu | f32 | 512x512 | 8.450 ms | 10.59 | 0.25 | 49% | ok |
| factor | cholesky | f64 | 512x512 | 7.972 ms | 5.61 | 0.53 | 52% | ok |
| factor | cholesky | f32 | 512x512 | 4.990 ms | 8.97 | 0.42 | 42% | ok |
| factor | ldlt | f64 | 512x512 | 32.420 ms | 1.38 | 0.13 | 13% | ok |
| factor | ldlt_bk | f64 | 512x512 | 85.291 ms | 0.52 | 0.05 | 5% | ok |
| factor | qr | f64 | 512x512 | 28.443 ms | 6.29 | 0.15 | 59% | ok |
| factor | lq | f64 | 512x512 | 93.142 ms | 1.92 | 0.05 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 474.391 ms | 0.94 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 1.984 ms | 0.53 | 2.11 | 13% | ok |
| factor | inv | f64 | 512x512 | 176.173 ms | 1.52 | 0.02 | 14% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 17.681 ms | 5.06 | 0.24 | 47% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 10.359 ms | 2.16 | 0.10 | 20% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 145.389 ms | 1.04 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 898.794 ms | 0.17 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 62.924 ms | 2.67 | 0.02 | 25% | ok |
| eigen | schur | f64 | 256x256 | 90.236 ms | 4.65 | 0.01 | 43% | ok |
| eigen | svd | f64 | 256x256 | 43.237 ms | 8.54 | 0.02 | 80% | ok |
| eigen | expm | f64 | 256x256 | 57.040 ms | 4.41 | 0.02 | 41% | ok |
| eigen | sylvester | f64 | 256x256 | 256.259 ms | 3.93 | 0.01 | 37% | ok |
| elementwise | exp | f64 | n=8388608 | 75.919 ms | 0.11 | 1.77 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.365 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 178.943 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.172 ms | 0.15 | 2.43 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 127.298 ms | 0.07 | 1.05 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.290 ms | 1.14 | 11.89 | 74% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.080 ms | 0.85 | 8.84 | 55% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.129 ms | 2.44 | 5.14 | 32% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.531 ms | 0.16 | 0.56 | 3% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.866 ms | 0.17 | 1.40 | 9% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 280.839 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 101.986 ms | 0.11 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.520 ms | 0.30 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 132.606 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 192.011 ms | 0.12 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 62.012 ms | 0.36 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 291.412 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.161 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.319 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.131 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 66.860 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 105.781 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 148.911 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.205 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.608 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.727 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.667 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.845 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.446 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.993 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.923 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.652 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.386 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 68.989 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.167 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.690 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 45.041 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.473 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.220 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.092 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 3.925 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.025 ms | 0.87 | 9.70 | 60% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.739 ms | 0.70 | 7.84 | 49% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.506 ms | 0.58 | 6.51 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 45.091 ms | 0.85 | 6.45 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 350.560 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 361.523 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 195.933 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 21.109 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 12.504 ms | 1.34 | 5.37 | 33% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 75.660 ms | 3.55 | 0.06 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 4.574 ms | 3.67 | 3.90 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.617 ms | 2.99 | 11.95 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.707 ms | 6.20 | 12.39 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.687 ms | 2.95 | 2.95 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.758 ms | 9.54 | 9.54 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 38.042 ms | 7.06 | 0.08 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 41.158 ms | 6.52 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 31.879 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 31.880 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 60.137 ms | 4.46 | 0.03 | n/a | ok |

