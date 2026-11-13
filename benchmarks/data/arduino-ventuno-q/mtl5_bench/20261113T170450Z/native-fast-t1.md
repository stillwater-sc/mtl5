## arduino-ventuno-q -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON: -march=armv8.2-a+dotprod+fp16+rcpc -mtune=cortex-a78c]`, SIMD NEON_WITHOUT_AES |
| commit | `bf315e4604cf` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.6 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 16.8, f32 33.6 GFLOP/s |
| bandwidth | read 22.1, update 24.3, triad 22.9 GB/s (ceiling 24.3) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 96% | 105% | 0 |
| blas2 | 10 | 66% | 73% | 0 |
| blas3 | 10 | 45% | 58% | 0 |
| factor | 12 | 12% | 31% | 0 |
| eigen | 8 | 15% | 31% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 30% | 68% | 0 |
| direct | 19 | 2% | 7% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 9% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 56% | 0 |
| mg | 1 | 37% | 37% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 67% | 132% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.741 ms | 2.92 | 23.38 | 96% | ok |
| blas1 | dot | f32 | n=8388608 | 2.888 ms | 5.81 | 23.24 | 96% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.917 ms | 2.12 | 25.43 | 105% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.983 ms | 4.21 | 25.27 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 5.244 ms | 1.60 | 25.60 | 105% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.173 ms | 5.29 | 21.15 | 87% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.661 ms | 0.97 | 7.75 | 32% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.655 ms | 0.48 | 3.80 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.072 ms | 1.04 | 8.31 | 34% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 195.766 ms | 0.94 | 0.69 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 8.355 ms | 4.02 | 16.07 | 66% | ok |
| blas2 | gemv | f32 | n=4096 | 4.017 ms | 8.35 | 16.71 | 69% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.743 ms | 4.33 | 17.34 | 71% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.967 ms | 8.46 | 16.93 | 70% | ok |
| blas2 | symv | f64 | n=4096 | 17.009 ms | 1.97 | 3.95 | 16% | ok |
| blas2 | trmv | f64 | n=4096 | 4.919 ms | 3.41 | 13.66 | 56% | ok |
| blas2 | trmv | f32 | n=4096 | 2.749 ms | 6.10 | 12.22 | 50% | ok |
| blas2 | trsv | f64 | n=4096 | 4.761 ms | 3.52 | 14.11 | 58% | ok |
| blas2 | trsv | f32 | n=4096 | 2.806 ms | 5.98 | 11.97 | 49% | ok |
| blas2 | ger | f64 | n=4096 | 15.222 ms | 2.20 | 17.64 | 73% | ok |
| blas3 | gemm | f64 | 512x512x512 | 28.080 ms | 9.56 | 0.22 | 57% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.795 ms | 19.46 | 0.23 | 58% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 18.064 ms | 7.43 | 0.35 | 44% | ok |
| blas3 | symm | f64 | 512x512x512 | 29.081 ms | 9.23 | 0.22 | 55% | ok |
| blas3 | syrk | f64 | 512x512x512 | 40.520 ms | 3.31 | 0.10 | 20% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 81.501 ms | 3.29 | 0.08 | 20% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.616 ms | 7.62 | 0.30 | 45% | ok |
| blas3 | trsm | f64 | 512x512x512 | 18.102 ms | 7.41 | 0.29 | 44% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.510 ms | 15.77 | 0.31 | 47% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 28.571 ms | 0.19 | 1.49 | 6% | ok |
| factor | lu | f64 | 512x512 | 24.420 ms | 3.66 | 0.17 | 22% | ok |
| factor | lu | f32 | 512x512 | 13.050 ms | 6.86 | 0.16 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 22.404 ms | 2.00 | 0.19 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 21.026 ms | 2.13 | 0.10 | 6% | ok |
| factor | ldlt | f64 | 512x512 | 21.646 ms | 2.07 | 0.19 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 69.563 ms | 0.64 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 245.986 ms | 0.73 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 60.859 ms | 2.94 | 0.07 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 343.146 ms | 1.30 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.527 ms | 0.69 | 2.75 | 11% | ok |
| factor | inv | f64 | 512x512 | 145.791 ms | 1.84 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.951 ms | 5.28 | 0.25 | 31% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.782 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.494 ms | 1.55 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 608.132 ms | 0.25 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.614 ms | 2.48 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 151.337 ms | 2.77 | 0.01 | 17% | ok |
| eigen | svd | f64 | 256x256 | 230.004 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 47.851 ms | 5.26 | 0.02 | 31% | ok |
| eigen | sylvester | f64 | 256x256 | 359.454 ms | 2.80 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 85.810 ms | 0.10 | 1.56 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 108.576 ms | 0.08 | 1.24 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 150.238 ms | 0.06 | 0.89 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 72.595 ms | 0.12 | 1.85 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 108.822 ms | 0.08 | 1.23 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.651 ms | 1.59 | 16.49 | 68% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.179 ms | 1.20 | 12.49 | 51% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.256 ms | 3.42 | 7.18 | 30% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.465 ms | 0.24 | 0.85 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.776 ms | 0.23 | 1.90 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 183.514 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 59.518 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.522 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 102.640 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 138.525 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.602 ms | 0.51 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 201.098 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.472 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.491 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.702 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.325 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.999 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 176.350 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.854 ms | 0.41 | 1.63 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 13.501 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.525 ms | 0.97 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.717 ms | 0.44 | 1.76 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.575 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 87.024 ms | 0.26 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.791 ms | 0.13 | 0.52 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 10.197 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 118.892 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.858 ms | 0.13 | 0.52 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 21.687 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 16855.901 ms | 0.02 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.683 ms | 0.44 | 1.75 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 47.998 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 210.803 ms | 1.46 | 0.00 | 9% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.398 ms | 0.46 | 1.82 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.927 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1761.450 ms | 0.35 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.895 ms | 0.12 | 0.47 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 29.682 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2449.155 ms | 0.25 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.522 ms | 0.12 | 0.47 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.992 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 105.297 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 54.330 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 29.086 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.764 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 34.101 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 63.838 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.901 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.789 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 37.873 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 36.005 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.911 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 46.654 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.110 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.916 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 31.449 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.076 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.956 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.064 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.773 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.153 ms | 1.22 | 13.62 | 56% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.512 ms | 1.04 | 11.68 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.951 ms | 0.89 | 9.94 | 41% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.284 ms | 1.19 | 9.00 | 37% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 233.085 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 269.736 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 147.567 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.251 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.113 ms | 4.08 | 16.32 | 67% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 41.658 ms | 6.44 | 0.10 | 38% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.758 ms | 22.14 | 23.52 | 132% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.834 ms | 5.92 | 23.68 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.572 ms | 10.67 | 21.34 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.736 ms | 22.79 | 22.79 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.521 ms | 11.03 | 11.03 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.374 ms | 15.45 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 16.993 ms | 15.80 | 0.12 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 16.749 ms | 16.03 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.697 ms | 72.61 | 0.43 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.160 ms | 20.40 | 0.12 | n/a | ok |

