## jetson-orin -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 10.7 GFLOP/s, f32 21.5 GFLOP/s |
| ... hardware, NEON | f64 10.7, f32 21.5 GFLOP/s |
| ... this build, scalar | f64 5.4, f32 5.4 GFLOP/s |
| bandwidth | read 11.5, update 16.1, triad 14.7 GB/s (ceiling 16.1) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 72% | 131% | 0 |
| blas2 | 10 | 64% | 125% | 0 |
| blas3 | 10 | 89% | 95% | 0 |
| factor | 12 | 42% | 61% | 0 |
| eigen | 8 | 37% | 80% | 0 |
| elementwise | 5 | 8% | 15% | 0 |
| sparse | 5 | 31% | 73% | 0 |
| direct | 23 | 2% | 14% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 15 | 8% | 36% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 49% | 62% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 33% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.555 ms | 1.45 | 11.62 | 72% | ok |
| blas1 | dot | f32 | n=8388608 | 5.997 ms | 2.80 | 11.19 | 69% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.246 ms | 1.37 | 16.44 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 5.977 ms | 2.81 | 16.84 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 6.362 ms | 1.32 | 21.10 | 131% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.618 ms | 2.99 | 11.95 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.501 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.752 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.501 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 191.890 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 12.615 ms | 2.66 | 10.64 | 66% | ok |
| blas2 | gemv | f32 | n=4096 | 5.893 ms | 5.69 | 11.39 | 71% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 13.007 ms | 2.58 | 10.32 | 64% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.803 ms | 5.78 | 11.57 | 72% | ok |
| blas2 | symv | f64 | n=4096 | 15.938 ms | 2.11 | 4.21 | 26% | ok |
| blas2 | trmv | f64 | n=4096 | 8.593 ms | 1.95 | 7.82 | 48% | ok |
| blas2 | trmv | f32 | n=4096 | 4.889 ms | 3.43 | 6.87 | 43% | ok |
| blas2 | trsv | f64 | n=4096 | 8.562 ms | 1.96 | 7.85 | 49% | ok |
| blas2 | trsv | f32 | n=4096 | 4.777 ms | 3.51 | 7.03 | 44% | ok |
| blas2 | ger | f64 | n=4096 | 13.341 ms | 2.52 | 20.13 | 125% | ok |
| blas3 | gemm | f64 | 512x512x512 | 26.654 ms | 10.07 | 0.24 | 94% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.110 ms | 20.48 | 0.24 | 95% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 16.602 ms | 8.08 | 0.38 | 75% | ok |
| blas3 | symm | f64 | 512x512x512 | 26.718 ms | 10.05 | 0.24 | 94% | ok |
| blas3 | syrk | f64 | 512x512x512 | 14.446 ms | 9.29 | 0.29 | 87% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 28.236 ms | 9.51 | 0.22 | 89% | ok |
| blas3 | trmm | f64 | 512x512x512 | 13.651 ms | 9.83 | 0.38 | 92% | ok |
| blas3 | trsm | f64 | 512x512x512 | 14.976 ms | 8.96 | 0.35 | 84% | ok |
| blas3 | trsm | f32 | 512x512x512 | 7.564 ms | 17.74 | 0.35 | 83% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.417 ms | 0.56 | 4.51 | 28% | ok |
| factor | lu | f64 | 512x512 | 13.681 ms | 6.54 | 0.31 | 61% | ok |
| factor | lu | f32 | 512x512 | 8.383 ms | 10.67 | 0.25 | 50% | ok |
| factor | cholesky | f64 | 512x512 | 7.904 ms | 5.66 | 0.53 | 53% | ok |
| factor | cholesky | f32 | 512x512 | 4.938 ms | 9.06 | 0.42 | 42% | ok |
| factor | ldlt | f64 | 512x512 | 32.274 ms | 1.39 | 0.13 | 13% | ok |
| factor | ldlt_bk | f64 | 512x512 | 16.160 ms | 2.77 | 0.26 | 26% | ok |
| factor | qr | f64 | 512x512 | 28.454 ms | 6.29 | 0.15 | 59% | ok |
| factor | lq | f64 | 512x512 | 91.904 ms | 1.95 | 0.05 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 476.664 ms | 0.94 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 2.066 ms | 0.51 | 2.03 | 13% | ok |
| factor | inv | f64 | 512x512 | 173.559 ms | 1.55 | 0.02 | 14% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 14.805 ms | 6.04 | 0.28 | 56% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 10.405 ms | 2.15 | 0.10 | 20% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 144.749 ms | 1.04 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 897.347 ms | 0.17 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 62.932 ms | 2.67 | 0.02 | 25% | ok |
| eigen | schur | f64 | 256x256 | 90.060 ms | 4.66 | 0.01 | 43% | ok |
| eigen | svd | f64 | 256x256 | 43.264 ms | 8.53 | 0.02 | 80% | ok |
| eigen | expm | f64 | 256x256 | 55.997 ms | 4.49 | 0.02 | 42% | ok |
| eigen | sylvester | f64 | 256x256 | 253.190 ms | 3.98 | 0.01 | 37% | ok |
| elementwise | exp | f64 | n=8388608 | 75.988 ms | 0.11 | 1.77 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.073 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 178.366 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.170 ms | 0.15 | 2.43 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 126.498 ms | 0.07 | 1.06 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.309 ms | 1.13 | 11.80 | 73% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.007 ms | 0.87 | 9.06 | 56% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.395 ms | 2.41 | 5.06 | 31% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.412 ms | 0.16 | 0.56 | 3% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.730 ms | 0.17 | 1.42 | 9% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 53.965 ms | 0.21 | 0.01 | 2% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 87.947 ms | 0.13 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 35.997 ms | 0.31 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 130.849 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 150.912 ms | 0.15 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 60.743 ms | 0.37 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 105.826 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.083 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 16.990 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 181.486 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.891 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 65.607 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 7.807 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 43.663 ms | 0.26 | 0.02 | 2% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.154 ms | 0.35 | 1.40 | 9% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 16.560 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.762 ms | 0.67 | 0.04 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.085 ms | 0.36 | 1.45 | 9% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.608 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 113.397 ms | 0.20 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.546 ms | 0.10 | 0.40 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 12.296 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 127.436 ms | 0.18 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.605 ms | 0.10 | 0.40 | 2% | ok |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 28.165 ms | 0.40 | 0.03 | 4% | ok |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 5.955 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 20.450 ms | 0.55 | 0.04 | 5% | ok |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.378 ms | 0.55 | 2.20 | 14% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 16.760 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 845.097 ms | 0.37 | 0.00 | 3% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 9.972 ms | 0.34 | 1.35 | 8% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 36.836 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 256.518 ms | 1.20 | 0.00 | 11% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.089 ms | 0.33 | 1.34 | 8% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.215 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1907.064 ms | 0.32 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.674 ms | 0.09 | 0.37 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 23.448 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1529.938 ms | 0.40 | 0.00 | 4% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.701 ms | 0.09 | 0.37 | 2% | ok |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 11.479 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 79.179 ms | 3.90 | 0.01 | 36% | ok |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 3.145 ms | 1.07 | 4.28 | 27% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.387 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 230.015 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.351 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.957 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 51.230 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 50.068 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 88.892 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.160 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 50.116 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 57.844 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.587 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 50.722 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 59.709 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.167 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.750 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 44.607 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.491 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.293 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.097 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 4.083 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.918 ms | 0.90 | 10.05 | 62% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.737 ms | 0.70 | 7.85 | 49% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.517 ms | 0.58 | 6.49 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 44.976 ms | 0.85 | 6.46 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 351.320 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 356.834 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 196.283 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 21.260 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 12.503 ms | 1.34 | 5.37 | 33% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 75.430 ms | 3.56 | 0.06 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 4.569 ms | 3.67 | 3.90 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.612 ms | 2.99 | 11.96 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.748 ms | 6.11 | 12.21 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.672 ms | 2.96 | 2.96 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.758 ms | 9.55 | 9.55 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 38.116 ms | 7.04 | 0.08 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 41.106 ms | 6.53 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 31.915 ms | 8.41 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 31.891 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 60.149 ms | 4.46 | 0.03 | n/a | ok |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

