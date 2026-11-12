## jetson-orin -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 10.7 GFLOP/s, f32 21.5 GFLOP/s |
| ... hardware, NEON | f64 10.7, f32 21.5 GFLOP/s |
| ... this build, NEON | f64 10.7, f32 21.5 GFLOP/s |
| bandwidth | read 11.5, update 16.1, triad 14.8 GB/s (ceiling 16.1) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 74% | 131% | 0 |
| blas2 | 10 | 50% | 124% | 0 |
| blas3 | 10 | 50% | 61% | 0 |
| factor | 12 | 13% | 27% | 0 |
| eigen | 8 | 16% | 33% | 0 |
| elementwise | 5 | 8% | 15% | 0 |
| sparse | 5 | 32% | 74% | 0 |
| direct | 7 | 1% | 3% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 49% | 56% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 66% | 118% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.600 ms | 1.45 | 11.57 | 72% | ok |
| blas1 | dot | f32 | n=8388608 | 5.644 ms | 2.97 | 11.89 | 74% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.234 ms | 1.37 | 16.46 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 6.005 ms | 2.79 | 16.76 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 6.333 ms | 1.32 | 21.19 | 131% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.590 ms | 3.00 | 12.01 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.503 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.754 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.503 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 192.376 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 14.061 ms | 2.39 | 9.55 | 59% | ok |
| blas2 | gemv | f32 | n=4096 | 6.629 ms | 5.06 | 10.13 | 63% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 50.140 ms | 0.67 | 2.68 | 17% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 24.683 ms | 1.36 | 2.72 | 17% | ok |
| blas2 | symv | f64 | n=4096 | 21.349 ms | 1.57 | 3.15 | 19% | ok |
| blas2 | trmv | f64 | n=4096 | 9.059 ms | 1.85 | 7.41 | 46% | ok |
| blas2 | trmv | f32 | n=4096 | 4.189 ms | 4.00 | 8.02 | 50% | ok |
| blas2 | trsv | f64 | n=4096 | 9.080 ms | 1.85 | 7.40 | 46% | ok |
| blas2 | trsv | f32 | n=4096 | 4.137 ms | 4.06 | 8.12 | 50% | ok |
| blas2 | ger | f64 | n=4096 | 13.434 ms | 2.50 | 19.99 | 124% | ok |
| blas3 | gemm | f64 | 512x512x512 | 41.579 ms | 6.46 | 0.15 | 60% | ok |
| blas3 | gemm | f32 | 512x512x512 | 20.561 ms | 13.06 | 0.15 | 61% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 25.796 ms | 5.20 | 0.24 | 48% | ok |
| blas3 | symm | f64 | 512x512x512 | 42.326 ms | 6.34 | 0.15 | 59% | ok |
| blas3 | syrk | f64 | 512x512x512 | 49.973 ms | 2.69 | 0.08 | 25% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 92.608 ms | 2.90 | 0.07 | 27% | ok |
| blas3 | trmm | f64 | 512x512x512 | 25.001 ms | 5.37 | 0.21 | 50% | ok |
| blas3 | trsm | f64 | 512x512x512 | 25.888 ms | 5.18 | 0.20 | 48% | ok |
| blas3 | trsm | f32 | 512x512x512 | 12.608 ms | 10.65 | 0.21 | 50% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.748 ms | 0.54 | 4.36 | 27% | ok |
| factor | lu | f64 | 512x512 | 35.275 ms | 2.54 | 0.12 | 24% | ok |
| factor | lu | f32 | 512x512 | 20.300 ms | 4.41 | 0.10 | 21% | ok |
| factor | cholesky | f64 | 512x512 | 33.852 ms | 1.32 | 0.12 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 30.085 ms | 1.49 | 0.07 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 32.164 ms | 1.39 | 0.13 | 13% | ok |
| factor | ldlt_bk | f64 | 512x512 | 83.081 ms | 0.54 | 0.05 | 5% | ok |
| factor | qr | f64 | 512x512 | 322.564 ms | 0.55 | 0.01 | 5% | ok |
| factor | lq | f64 | 512x512 | 91.908 ms | 1.95 | 0.05 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 462.338 ms | 0.97 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 1.932 ms | 0.54 | 2.17 | 13% | ok |
| factor | inv | f64 | 512x512 | 190.447 ms | 1.41 | 0.02 | 13% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 30.478 ms | 2.94 | 0.14 | 27% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.609 ms | 0.36 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 145.078 ms | 1.04 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 903.792 ms | 0.17 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 98.263 ms | 1.71 | 0.01 | 16% | ok |
| eigen | schur | f64 | 256x256 | 215.559 ms | 1.95 | 0.00 | 18% | ok |
| eigen | svd | f64 | 256x256 | 358.715 ms | 1.03 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 71.895 ms | 3.50 | 0.01 | 33% | ok |
| eigen | sylvester | f64 | 256x256 | 516.291 ms | 1.95 | 0.00 | 18% | ok |
| elementwise | exp | f64 | n=8388608 | 76.110 ms | 0.11 | 1.76 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.240 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 178.312 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.349 ms | 0.15 | 2.42 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 126.552 ms | 0.07 | 1.06 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.289 ms | 1.14 | 11.90 | 74% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.074 ms | 0.85 | 8.86 | 55% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.090 ms | 2.45 | 5.15 | 32% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.212 ms | 0.16 | 0.57 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.854 ms | 0.17 | 1.40 | 9% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 278.649 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 98.639 ms | 0.11 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.186 ms | 0.30 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 135.623 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 190.641 ms | 0.12 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 62.104 ms | 0.36 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 288.777 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.030 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 16.667 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.139 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 66.111 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.193 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 148.377 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.182 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.525 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.791 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.700 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 88.810 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.016 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.893 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.995 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.567 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.265 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 68.600 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.166 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.762 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 45.025 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.511 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.242 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.089 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.252 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.223 ms | 0.81 | 9.10 | 56% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.742 ms | 0.70 | 7.84 | 49% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.522 ms | 0.58 | 6.48 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 45.151 ms | 0.85 | 6.44 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 363.716 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 422.873 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 201.373 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 21.231 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 6.261 ms | 2.68 | 10.72 | 66% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 64.009 ms | 4.19 | 0.07 | 39% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.320 ms | 12.71 | 13.50 | 118% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.624 ms | 2.98 | 11.93 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.696 ms | 6.22 | 12.45 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.218 ms | 13.78 | 13.78 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.352 ms | 7.13 | 7.13 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 26.078 ms | 10.29 | 0.12 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 27.846 ms | 9.64 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 33.459 ms | 8.02 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 8.560 ms | 31.36 | 0.18 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 23.397 ms | 11.47 | 0.07 | n/a | ok |

