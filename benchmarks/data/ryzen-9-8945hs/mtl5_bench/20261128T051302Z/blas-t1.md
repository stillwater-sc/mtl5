## ryzen-9-8945hs -- blas, 1 thread, class large

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 78.6 GFLOP/s, f32 126.7 GFLOP/s |
| ... hardware, AVX-512F | f64 78.6, f32 126.7 GFLOP/s |
| ... this build, scalar | f64 1.9, f32 1.9 GFLOP/s |
| bandwidth | read 36.9, update 52.1, triad 37.5 GB/s (ceiling 52.1) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 20% | 97% | 0 |
| blas2 | 10 | 75% | 79% | 0 |
| blas3 | 10 | 73% | 89% | 0 |
| factor | 12 | 14% | 28% | 0 |
| eigen | 8 | 13% | 43% | 0 |
| elementwise | 5 | 6% | 10% | 0 |
| sparse | 5 | 21% | 46% | 0 |
| direct | 19 | 1% | 7% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 7% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 29% | 32% | 0 |
| mg | 1 | 22% | 22% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 11% | 16% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=33554432 | 47.029 ms | 1.43 | 11.42 | 22% | ok |
| blas1 | dot | f32 | n=33554432 | 49.010 ms | 1.37 | 5.48 | 11% | ok |
| blas1 | axpy | f64 | n=33554432 | 59.882 ms | 1.12 | 13.45 | 26% | ok |
| blas1 | axpy | f32 | n=33554432 | 59.651 ms | 1.13 | 6.75 | 13% | ok |
| blas1 | scale | f64 | n=33554432 | 10.638 ms | 3.15 | 50.47 | 97% | ok |
| blas1 | two_norm | f64 | n=33554432 | 49.045 ms | 1.37 | 5.47 | 11% | ok |
| blas1 | one_norm | f64 | n=33554432 | 25.672 ms | 1.31 | 10.46 | 20% | ok |
| blas1 | infinity_norm | f64 | n=33554432 | 25.714 ms | 1.30 | 10.44 | 20% | ok |
| blas1 | sum | f64 | n=33554432 | 25.663 ms | 1.31 | 10.46 | 20% | ok |
| blas1 | dot_k3 | f64 | n=33554432 | 996.267 ms | 0.74 | 0.54 | 1% | ok |
| blas2 | gemv | f64 | n=8192 | 13.405 ms | 10.01 | 40.06 | 77% | ok |
| blas2 | gemv | f32 | n=8192 | 6.560 ms | 20.46 | 40.93 | 79% | ok |
| blas2 | gemv_colmajor | f64 | n=8192 | 13.781 ms | 9.74 | 38.97 | 75% | ok |
| blas2 | gemv_colmajor | f32 | n=8192 | 6.656 ms | 20.16 | 40.34 | 77% | ok |
| blas2 | symv | f64 | n=8192 | 7.513 ms | 17.87 | 35.75 | 69% | ok |
| blas2 | trmv | f64 | n=8192 | 7.163 ms | 9.37 | 37.50 | 72% | ok |
| blas2 | trmv | f32 | n=8192 | 3.925 ms | 17.10 | 34.21 | 66% | ok |
| blas2 | trsv | f64 | n=8192 | 6.662 ms | 10.07 | 40.31 | 77% | ok |
| blas2 | trsv | f32 | n=8192 | 4.022 ms | 16.69 | 33.39 | 64% | ok |
| blas2 | ger | f64 | n=8192 | 28.819 ms | 4.66 | 37.26 | 72% | ok |
| blas3 | gemm | f64 | 1024x1024x1024 | 36.276 ms | 59.20 | 0.69 | 75% | ok |
| blas3 | gemm | f32 | 1024x1024x1024 | 19.076 ms | 112.57 | 0.66 | 89% | ok |
| blas3 | gemmt | f64 | 1024x1024x1024 | 24.266 ms | 44.25 | 1.04 | 56% | ok |
| blas3 | symm | f64 | 1024x1024x1024 | 37.249 ms | 57.65 | 0.68 | 73% | ok |
| blas3 | syrk | f64 | 1024x1024x1024 | 24.354 ms | 44.09 | 0.69 | 56% | ok |
| blas3 | syr2k | f64 | 1024x1024x1024 | 42.712 ms | 50.28 | 0.59 | 64% | ok |
| blas3 | trmm | f64 | 1024x1024x1024 | 18.273 ms | 58.76 | 1.15 | 75% | ok |
| blas3 | trsm | f64 | 1024x1024x1024 | 23.215 ms | 46.25 | 0.90 | 59% | ok |
| blas3 | trsm | f32 | 1024x1024x1024 | 11.340 ms | 94.69 | 0.92 | 75% | ok |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 34.803 ms | 0.48 | 3.86 | 7% | ok |
| factor | lu | f64 | 1024x1024 | 32.157 ms | 22.26 | 0.52 | 28% | ok |
| factor | lu | f32 | 1024x1024 | 25.268 ms | 28.33 | 0.33 | 22% | ok |
| factor | cholesky | f64 | 1024x1024 | 26.942 ms | 13.28 | 0.62 | 17% | ok |
| factor | cholesky | f32 | 1024x1024 | 20.885 ms | 17.14 | 0.40 | 14% | ok |
| factor | ldlt | f64 | 1024x1024 | 127.961 ms | 2.80 | 0.13 | 4% | ok |
| factor | ldlt_bk | f64 | 1024x1024 | 92.407 ms | 3.87 | 0.18 | 5% | ok |
| factor | qr | f64 | 1024x1024 | 69.619 ms | 20.56 | 0.24 | 26% | ok |
| factor | lq | f64 | 1024x1024 | 653.793 ms | 2.19 | 0.03 | 3% | ok |
| factor | hessenberg | f64 | 1024x1024 | 6909.171 ms | 0.52 | 0.00 | 1% | ok |
| factor | householder | f64 | 1024x1024 | 17.588 ms | 0.24 | 0.95 | 2%\* | ok |
| factor | inv | f64 | 1024x1024 | 569.318 ms | 3.77 | 0.03 | 5% | ok |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 36.989 ms | 19.35 | 0.45 | 25% | ok |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 13.055 ms | 13.71 | 0.32 | 17% | ok |
| eigen | eigen_symmetric | f64 | 512x512 | 3656.342 ms | 0.33 | 0.00 | 0% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 18281.568 ms | 0.07 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 512x512 | 300.541 ms | 4.47 | 0.01 | 6% | ok |
| eigen | schur | f64 | 512x512 | 340.360 ms | 9.86 | 0.01 | 13% | ok |
| eigen | svd | f64 | 512x512 | 86.546 ms | 34.12 | 0.05 | 43% | ok |
| eigen | expm | f64 | 512x512 | 172.116 ms | 11.70 | 0.02 | 15% | ok |
| eigen | sylvester | f64 | 512x512 | 1532.577 ms | 5.25 | 0.00 | 7% | ok |
| elementwise | exp | f64 | n=33554432 | 133.612 ms | 0.25 | 4.02 | 8% | ok |
| elementwise | log | f64 | n=33554432 | 215.356 ms | 0.16 | 2.49 | 5% | ok |
| elementwise | sin | f64 | n=33554432 | 191.995 ms | 0.17 | 2.80 | 5% | ok |
| elementwise | sqrt | f64 | n=33554432 | 105.431 ms | 0.32 | 5.09 | 10% | ok |
| elementwise | erf | f64 | n=33554432 | 170.524 ms | 0.20 | 3.15 | 6% | ok |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 4.514 ms | 2.32 | 24.14 | 46% | ok |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 6.418 ms | 1.63 | 16.98 | 33% | ok |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 31.883 ms | 5.26 | 11.05 | 21% | ok |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 98.252 ms | 0.13 | 0.47 | 1% | ok |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 19.531 ms | 0.52 | 4.29 | 8% | ok |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 169.353 ms | 0.63 | 0.02 | 1% | ok |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 300.278 ms | 0.36 | 0.01 | 0% | ok |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 107.512 ms | 1.00 | 0.03 | 1% | ok |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 536.298 ms | 0.40 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 459.061 ms | 0.47 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 215.435 ms | 0.99 | 0.01 | 1% | ok |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 623.115 ms | 0.04 | 0.00 | 0% | ok |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 22.961 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 32.175 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 348.187 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 1.378 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 105.674 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 18.428 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 142.434 ms | 0.75 | 0.02 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 4.181 ms | 0.96 | 3.84 | 7%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 40.885 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 49.592 ms | 2.16 | 0.06 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 4.130 ms | 0.97 | 3.89 | 7%\* | ok |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 18.008 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 487.253 ms | 0.44 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.270 ms | 0.26 | 1.05 | 2%\* | ok |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 27.774 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 412.736 ms | 0.52 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.374 ms | 0.26 | 1.04 | 2%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 12.756 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1058.218 ms | 0.92 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 7.821 ms | 0.96 | 3.84 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 35.348 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 252.548 ms | 3.84 | 0.01 | 5% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 8.102 ms | 0.93 | 3.70 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 14.026 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 3796.882 ms | 0.51 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.500 ms | 0.25 | 0.98 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 19.333 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1958.906 ms | 0.99 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.379 ms | 0.25 | 0.99 | 2% | ok |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 388.061 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 582.061 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 243.020 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 189.214 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 213.673 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 169.227 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 614.227 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 305.666 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 202.365 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 237.615 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 215.931 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 232.972 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 291.035 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.213 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 256^2 grid | 13.725 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 256^2 grid | 126.366 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 256^2 grid | 10.348 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 256^2 grid | 9.710 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 256^2 grid | 0.146 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 7.268 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 7.028 ms | 1.49 | 16.70 | 32% | ok |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 7.911 ms | 1.32 | 14.84 | 29% | ok |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 9.083 ms | 1.15 | 12.92 | 25% | ok |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 100.071 ms | 1.55 | 11.68 | 22% | ok |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 187.553 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 187.904 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 823.621 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 39.103 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=33554432 | 49.081 ms | 1.37 | 5.47 | 11% | ok |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 298.505 ms | 7.19 | 0.06 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 8.335 ms | 8.05 | 8.55 | 16% | ok |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 16.334 ms | 4.11 | 16.43 | n/a | ok |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 13.024 ms | 5.15 | 10.31 | n/a | ok |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 8.550 ms | 7.85 | 7.85 | n/a | ok |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 8.546 ms | 7.85 | 7.85 | n/a | ok |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 79.081 ms | 27.16 | 0.16 | n/a | ok |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 305.779 ms | 7.02 | 0.03 | n/a | ok |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 274.896 ms | 7.81 | 0.02 | n/a | ok |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 274.736 ms | 7.82 | 0.02 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 276.018 ms | 7.78 | 0.02 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

