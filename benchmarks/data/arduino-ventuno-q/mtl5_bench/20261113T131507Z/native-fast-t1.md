## arduino-ventuno-q -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON_WITHOUT_AES |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.5 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 16.8, f32 33.6 GFLOP/s |
| bandwidth | read 22.2, update 24.6, triad 23.2 GB/s (ceiling 24.6) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 95% | 104% | 0 |
| blas2 | 10 | 67% | 70% | 0 |
| blas3 | 10 | 45% | 58% | 0 |
| factor | 12 | 12% | 32% | 0 |
| eigen | 8 | 15% | 31% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 30% | 66% | 0 |
| direct | 19 | 2% | 7% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 9% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 40% | 57% | 0 |
| mg | 1 | 37% | 37% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 38% | 67% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.754 ms | 2.92 | 23.33 | 95% | ok |
| blas1 | dot | f32 | n=8388608 | 2.880 ms | 5.83 | 23.30 | 95% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.880 ms | 2.13 | 25.55 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.990 ms | 4.20 | 25.23 | 102% | ok |
| blas1 | scale | f64 | n=8388608 | 5.334 ms | 1.57 | 25.16 | 102% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.083 ms | 5.44 | 21.77 | 88% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.722 ms | 0.96 | 7.69 | 31% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.781 ms | 0.47 | 3.77 | 15% | ok |
| blas1 | sum | f64 | n=8388608 | 8.803 ms | 0.95 | 7.62 | 31% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 199.420 ms | 0.93 | 0.67 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 8.071 ms | 4.16 | 16.64 | 68% | ok |
| blas2 | gemv | f32 | n=4096 | 4.091 ms | 8.20 | 16.41 | 67% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.802 ms | 4.30 | 17.21 | 70% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.981 ms | 8.43 | 16.87 | 68% | ok |
| blas2 | symv | f64 | n=4096 | 17.261 ms | 1.94 | 3.89 | 16% | ok |
| blas2 | trmv | f64 | n=4096 | 4.981 ms | 3.37 | 13.49 | 55% | ok |
| blas2 | trmv | f32 | n=4096 | 2.622 ms | 6.40 | 12.81 | 52% | ok |
| blas2 | trsv | f64 | n=4096 | 4.783 ms | 3.51 | 14.04 | 57% | ok |
| blas2 | trsv | f32 | n=4096 | 2.658 ms | 6.31 | 12.63 | 51% | ok |
| blas2 | ger | f64 | n=4096 | 15.473 ms | 2.17 | 17.35 | 70% | ok |
| blas3 | gemm | f64 | 512x512x512 | 28.205 ms | 9.52 | 0.22 | 57% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.815 ms | 19.43 | 0.23 | 58% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 17.912 ms | 7.49 | 0.35 | 45% | ok |
| blas3 | symm | f64 | 512x512x512 | 29.084 ms | 9.23 | 0.22 | 55% | ok |
| blas3 | syrk | f64 | 512x512x512 | 41.928 ms | 3.20 | 0.10 | 19% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 81.040 ms | 3.31 | 0.08 | 20% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.798 ms | 7.54 | 0.29 | 45% | ok |
| blas3 | trsm | f64 | 512x512x512 | 17.831 ms | 7.53 | 0.29 | 45% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.532 ms | 15.73 | 0.31 | 47% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 26.268 ms | 0.20 | 1.62 | 7% | ok |
| factor | lu | f64 | 512x512 | 24.320 ms | 3.68 | 0.17 | 22% | ok |
| factor | lu | f32 | 512x512 | 12.810 ms | 6.99 | 0.16 | 21% | ok |
| factor | cholesky | f64 | 512x512 | 22.376 ms | 2.00 | 0.19 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 19.660 ms | 2.28 | 0.11 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 22.946 ms | 1.95 | 0.18 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 68.380 ms | 0.65 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 251.855 ms | 0.71 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 61.018 ms | 2.93 | 0.07 | 17% | ok |
| factor | hessenberg | f64 | 512x512 | 349.998 ms | 1.28 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.537 ms | 0.68 | 2.73 | 11% | ok |
| factor | inv | f64 | 512x512 | 144.373 ms | 1.86 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.665 ms | 5.37 | 0.25 | 32% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.982 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.279 ms | 1.55 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 615.256 ms | 0.25 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.250 ms | 2.49 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 150.625 ms | 2.78 | 0.01 | 17% | ok |
| eigen | svd | f64 | 256x256 | 230.727 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 47.810 ms | 5.26 | 0.02 | 31% | ok |
| eigen | sylvester | f64 | 256x256 | 357.299 ms | 2.82 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 84.549 ms | 0.10 | 1.59 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 103.500 ms | 0.08 | 1.30 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 148.552 ms | 0.06 | 0.90 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 65.731 ms | 0.13 | 2.04 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 110.356 ms | 0.08 | 1.22 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.687 ms | 1.55 | 16.14 | 66% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.135 ms | 1.23 | 12.76 | 52% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.991 ms | 3.49 | 7.34 | 30% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.490 ms | 0.24 | 0.85 | 3% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.511 ms | 0.24 | 1.96 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 184.091 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 60.897 ms | 0.18 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.435 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 101.989 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 139.046 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.130 ms | 0.52 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 207.623 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.829 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.334 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.701 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 43.515 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.890 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 175.000 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.794 ms | 0.42 | 1.69 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.801 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.458 ms | 0.98 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.753 ms | 0.43 | 1.73 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.416 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 87.228 ms | 0.26 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.806 ms | 0.13 | 0.52 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 10.015 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 118.398 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.829 ms | 0.13 | 0.52 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 21.543 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 16892.197 ms | 0.02 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.348 ms | 0.46 | 1.83 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 47.087 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 210.469 ms | 1.47 | 0.00 | 9% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.589 ms | 0.44 | 1.78 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.529 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1697.655 ms | 0.36 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.334 ms | 0.12 | 0.49 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 29.087 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2536.966 ms | 0.24 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.744 ms | 0.12 | 0.49 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.786 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 104.352 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 53.872 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 28.879 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.469 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 33.859 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 63.330 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.760 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.721 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 37.534 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 36.169 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.367 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 46.708 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.113 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.995 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 30.963 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.138 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.982 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.065 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.744 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.087 ms | 1.25 | 14.05 | 57% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 4.752 ms | 0.55 | 6.17 | 25% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.965 ms | 0.88 | 9.89 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.191 ms | 1.19 | 9.03 | 37% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 234.177 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 271.098 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 148.139 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.172 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.084 ms | 4.11 | 16.43 | 67% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 41.611 ms | 6.45 | 0.10 | 38% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.654 ms | 6.32 | 6.72 | 38% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.822 ms | 5.94 | 23.78 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.394 ms | 12.04 | 24.07 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.314 ms | 7.25 | 7.25 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.254 ms | 3.94 | 3.94 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.397 ms | 15.43 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 16.952 ms | 15.83 | 0.12 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 16.616 ms | 16.16 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 23.646 ms | 11.35 | 0.07 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 50.977 ms | 5.27 | 0.03 | n/a | ok |

