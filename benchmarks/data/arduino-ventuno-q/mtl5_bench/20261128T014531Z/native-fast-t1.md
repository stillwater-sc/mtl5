## arduino-ventuno-q -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON: -march=armv8.2-a+dotprod+fp16+rcpc -mtune=cortex-a78c]`, SIMD NEON_WITHOUT_AES |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.6 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 16.8, f32 33.6 GFLOP/s |
| bandwidth | read 21.9, update 24.6, triad 22.9 GB/s (ceiling 24.6) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 94% | 104% | 0 |
| blas2 | 10 | 65% | 72% | 0 |
| blas3 | 10 | 49% | 58% | 0 |
| factor | 12 | 12% | 32% | 0 |
| eigen | 8 | 15% | 32% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 29% | 66% | 0 |
| direct | 19 | 2% | 7% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 3% | 10% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 56% | 0 |
| mg | 1 | 36% | 36% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 67% | 133% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.768 ms | 2.91 | 23.27 | 94% | ok |
| blas1 | dot | f32 | n=8388608 | 2.887 ms | 5.81 | 23.25 | 94% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.822 ms | 2.14 | 25.74 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.981 ms | 4.21 | 25.29 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 5.325 ms | 1.58 | 25.21 | 102% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.148 ms | 5.33 | 21.32 | 87% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.342 ms | 1.01 | 8.04 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.688 ms | 0.47 | 3.79 | 15% | ok |
| blas1 | sum | f64 | n=8388608 | 8.360 ms | 1.00 | 8.03 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 198.566 ms | 0.93 | 0.68 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 8.337 ms | 4.02 | 16.11 | 65% | ok |
| blas2 | gemv | f32 | n=4096 | 4.037 ms | 8.31 | 16.63 | 68% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.657 ms | 4.38 | 17.54 | 71% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.946 ms | 8.50 | 17.02 | 69% | ok |
| blas2 | symv | f64 | n=4096 | 12.781 ms | 2.63 | 5.26 | 21% | ok |
| blas2 | trmv | f64 | n=4096 | 4.637 ms | 3.62 | 14.49 | 59% | ok |
| blas2 | trmv | f32 | n=4096 | 2.709 ms | 6.19 | 12.40 | 50% | ok |
| blas2 | trsv | f64 | n=4096 | 4.988 ms | 3.36 | 13.47 | 55% | ok |
| blas2 | trsv | f32 | n=4096 | 2.781 ms | 6.03 | 12.08 | 49% | ok |
| blas2 | ger | f64 | n=4096 | 15.192 ms | 2.21 | 17.67 | 72% | ok |
| blas3 | gemm | f64 | 512x512x512 | 28.214 ms | 9.51 | 0.22 | 57% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.784 ms | 19.47 | 0.23 | 58% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 17.792 ms | 7.54 | 0.35 | 45% | ok |
| blas3 | symm | f64 | 512x512x512 | 28.978 ms | 9.26 | 0.22 | 55% | ok |
| blas3 | syrk | f64 | 512x512x512 | 16.448 ms | 8.16 | 0.25 | 49% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 32.565 ms | 8.24 | 0.19 | 49% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.845 ms | 7.52 | 0.29 | 45% | ok |
| blas3 | trsm | f64 | 512x512x512 | 17.859 ms | 7.52 | 0.29 | 45% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.511 ms | 15.77 | 0.31 | 47% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 26.559 ms | 0.20 | 1.60 | 6% | ok |
| factor | lu | f64 | 512x512 | 24.268 ms | 3.69 | 0.17 | 22% | ok |
| factor | lu | f32 | 512x512 | 13.027 ms | 6.87 | 0.16 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 22.518 ms | 1.99 | 0.19 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 20.995 ms | 2.13 | 0.10 | 6% | ok |
| factor | ldlt | f64 | 512x512 | 21.346 ms | 2.10 | 0.20 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 11.218 ms | 3.99 | 0.37 | 24% | ok |
| factor | qr | f64 | 512x512 | 245.601 ms | 0.73 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 61.009 ms | 2.93 | 0.07 | 17% | ok |
| factor | hessenberg | f64 | 512x512 | 339.084 ms | 1.32 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.503 ms | 0.70 | 2.79 | 11% | ok |
| factor | inv | f64 | 512x512 | 145.137 ms | 1.85 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.721 ms | 5.35 | 0.25 | 32% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 40.080 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 99.948 ms | 1.51 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 592.748 ms | 0.25 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.539 ms | 2.48 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 151.822 ms | 2.76 | 0.01 | 16% | ok |
| eigen | svd | f64 | 256x256 | 230.148 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 47.449 ms | 5.30 | 0.02 | 32% | ok |
| eigen | sylvester | f64 | 256x256 | 362.468 ms | 2.78 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 85.933 ms | 0.10 | 1.56 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 106.521 ms | 0.08 | 1.26 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 147.828 ms | 0.06 | 0.91 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 72.154 ms | 0.12 | 1.86 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 113.827 ms | 0.07 | 1.18 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.664 ms | 1.57 | 16.36 | 66% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.133 ms | 1.23 | 12.77 | 52% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.170 ms | 3.44 | 7.23 | 29% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.022 ms | 0.25 | 0.88 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.786 ms | 0.23 | 1.89 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 38.233 ms | 0.29 | 0.02 | 2% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 58.497 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.282 ms | 0.43 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 104.224 ms | 0.21 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 122.165 ms | 0.18 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 42.441 ms | 0.53 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 78.265 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.409 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.427 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 125.477 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.570 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.130 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.602 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 30.390 ms | 0.37 | 0.02 | 2% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.811 ms | 0.42 | 1.67 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.085 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.913 ms | 0.94 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.728 ms | 0.44 | 1.75 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.571 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 88.226 ms | 0.25 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.836 ms | 0.13 | 0.52 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 8.762 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 102.838 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.836 ms | 0.13 | 0.52 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 12.336 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 671.465 ms | 0.46 | 0.00 | 3% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.418 ms | 0.45 | 1.82 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.065 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 187.508 ms | 1.65 | 0.00 | 10% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.338 ms | 0.46 | 1.84 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.588 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1764.353 ms | 0.35 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.755 ms | 0.11 | 0.45 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.750 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1205.520 ms | 0.51 | 0.00 | 3% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.205 ms | 0.12 | 0.46 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 71.219 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 117.239 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 54.550 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 28.972 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.719 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 34.136 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 62.431 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.570 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.755 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 39.294 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.788 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 34.588 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 40.873 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.109 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 5.069 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 31.106 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.113 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.996 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.076 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.762 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.108 ms | 1.24 | 13.91 | 56% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.495 ms | 1.05 | 11.76 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.972 ms | 0.88 | 9.87 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.404 ms | 1.19 | 8.97 | 36% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 236.557 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 234.194 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 144.755 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.256 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.045 ms | 4.15 | 16.59 | 67% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 41.560 ms | 6.46 | 0.10 | 38% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.751 ms | 22.33 | 23.72 | 133% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.829 ms | 5.93 | 23.73 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.402 ms | 11.97 | 23.94 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.699 ms | 24.00 | 24.00 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.534 ms | 10.94 | 10.94 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.374 ms | 15.45 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 16.675 ms | 16.10 | 0.13 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 16.605 ms | 16.17 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.608 ms | 74.40 | 0.44 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.182 ms | 20.36 | 0.12 | n/a | ok |

