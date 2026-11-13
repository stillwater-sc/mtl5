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
| bandwidth | read 22.3, update 24.7, triad 23.1 GB/s (ceiling 24.7) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 94% | 105% | 0 |
| blas2 | 10 | 66% | 72% | 0 |
| blas3 | 10 | 45% | 58% | 0 |
| factor | 12 | 12% | 31% | 0 |
| eigen | 8 | 15% | 31% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 29% | 66% | 0 |
| direct | 19 | 1% | 7% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 9% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 57% | 0 |
| mg | 1 | 36% | 36% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 67% | 132% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.736 ms | 2.92 | 23.40 | 95% | ok |
| blas1 | dot | f32 | n=8388608 | 2.886 ms | 5.81 | 23.25 | 94% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.752 ms | 2.16 | 25.97 | 105% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.917 ms | 4.28 | 25.70 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 5.314 ms | 1.58 | 25.26 | 102% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.141 ms | 5.34 | 21.37 | 87% | ok |
| blas1 | one_norm | f64 | n=8388608 | 9.014 ms | 0.93 | 7.44 | 30% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.631 ms | 0.48 | 3.81 | 15% | ok |
| blas1 | sum | f64 | n=8388608 | 8.317 ms | 1.01 | 8.07 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 199.963 ms | 0.92 | 0.67 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 8.195 ms | 4.09 | 16.39 | 66% | ok |
| blas2 | gemv | f32 | n=4096 | 4.073 ms | 8.24 | 16.48 | 67% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.866 ms | 4.27 | 17.07 | 69% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.971 ms | 8.45 | 16.91 | 68% | ok |
| blas2 | symv | f64 | n=4096 | 16.071 ms | 2.09 | 4.18 | 17% | ok |
| blas2 | trmv | f64 | n=4096 | 4.888 ms | 3.43 | 13.74 | 56% | ok |
| blas2 | trmv | f32 | n=4096 | 2.688 ms | 6.24 | 12.49 | 51% | ok |
| blas2 | trsv | f64 | n=4096 | 4.768 ms | 3.52 | 14.09 | 57% | ok |
| blas2 | trsv | f32 | n=4096 | 2.750 ms | 6.10 | 12.21 | 49% | ok |
| blas2 | ger | f64 | n=4096 | 15.196 ms | 2.21 | 17.67 | 72% | ok |
| blas3 | gemm | f64 | 512x512x512 | 28.268 ms | 9.50 | 0.22 | 57% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.743 ms | 19.53 | 0.23 | 58% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 18.060 ms | 7.43 | 0.35 | 44% | ok |
| blas3 | symm | f64 | 512x512x512 | 29.148 ms | 9.21 | 0.22 | 55% | ok |
| blas3 | syrk | f64 | 512x512x512 | 41.873 ms | 3.21 | 0.10 | 19% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 84.832 ms | 3.16 | 0.07 | 19% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.708 ms | 7.58 | 0.30 | 45% | ok |
| blas3 | trsm | f64 | 512x512x512 | 17.964 ms | 7.47 | 0.29 | 44% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.493 ms | 15.80 | 0.31 | 47% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 26.493 ms | 0.20 | 1.60 | 6% | ok |
| factor | lu | f64 | 512x512 | 24.351 ms | 3.67 | 0.17 | 22% | ok |
| factor | lu | f32 | 512x512 | 13.046 ms | 6.86 | 0.16 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 22.522 ms | 1.99 | 0.19 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 21.018 ms | 2.13 | 0.10 | 6% | ok |
| factor | ldlt | f64 | 512x512 | 21.556 ms | 2.08 | 0.19 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 68.900 ms | 0.65 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 246.684 ms | 0.73 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 60.865 ms | 2.94 | 0.07 | 17% | ok |
| factor | hessenberg | f64 | 512x512 | 346.236 ms | 1.29 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.488 ms | 0.71 | 2.82 | 11% | ok |
| factor | inv | f64 | 512x512 | 145.303 ms | 1.85 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.922 ms | 5.29 | 0.25 | 31% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 40.346 ms | 0.55 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 99.011 ms | 1.53 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 629.162 ms | 0.24 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 68.115 ms | 2.46 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 152.798 ms | 2.74 | 0.01 | 16% | ok |
| eigen | svd | f64 | 256x256 | 230.284 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 48.838 ms | 5.15 | 0.02 | 31% | ok |
| eigen | sylvester | f64 | 256x256 | 360.924 ms | 2.79 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 86.781 ms | 0.10 | 1.55 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 103.559 ms | 0.08 | 1.30 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 151.687 ms | 0.06 | 0.88 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 70.501 ms | 0.12 | 1.90 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 116.501 ms | 0.07 | 1.15 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.667 ms | 1.57 | 16.34 | 66% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.157 ms | 1.21 | 12.62 | 51% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.131 ms | 3.45 | 7.26 | 29% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.231 ms | 0.25 | 0.87 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.721 ms | 0.23 | 1.91 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 183.979 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 59.821 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.753 ms | 0.42 | 0.03 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 103.895 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 140.800 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.534 ms | 0.51 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 202.838 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.878 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 14.506 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.703 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.572 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.101 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 175.307 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.828 ms | 0.41 | 1.65 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.943 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.638 ms | 0.96 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.693 ms | 0.45 | 1.79 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.566 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 89.222 ms | 0.25 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.936 ms | 0.13 | 0.51 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 10.251 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 124.352 ms | 0.18 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.884 ms | 0.13 | 0.51 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 22.505 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 16814.876 ms | 0.02 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.613 ms | 0.44 | 1.77 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 46.962 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 215.839 ms | 1.43 | 0.00 | 9% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.752 ms | 0.43 | 1.74 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.924 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1706.270 ms | 0.36 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.645 ms | 0.12 | 0.47 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 30.158 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2437.504 ms | 0.25 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.273 ms | 0.12 | 0.48 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.626 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 104.119 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 54.675 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 29.033 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.641 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 33.872 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 63.811 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.627 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.522 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 37.471 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.609 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.596 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 46.938 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.109 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.932 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 31.260 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.083 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.957 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.064 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.744 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.086 ms | 1.25 | 14.06 | 57% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.500 ms | 1.05 | 11.73 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.944 ms | 0.89 | 9.96 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.277 ms | 1.19 | 9.01 | 36% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 233.946 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 270.470 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 145.570 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.177 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.086 ms | 4.11 | 16.43 | 67% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 41.577 ms | 6.46 | 0.10 | 38% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.755 ms | 22.21 | 23.60 | 132% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.838 ms | 5.91 | 23.65 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.395 ms | 12.03 | 24.06 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.695 ms | 24.13 | 24.13 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.525 ms | 11.00 | 11.00 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.356 ms | 15.47 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 16.687 ms | 16.09 | 0.13 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 16.591 ms | 16.18 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.678 ms | 72.99 | 0.43 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.558 ms | 19.80 | 0.12 | n/a | ok |

