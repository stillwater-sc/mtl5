## ryzen-9-8945hs -- blas, 1 thread, class large

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 78.4 GFLOP/s, f32 126.6 GFLOP/s |
| ... hardware, AVX-512F | f64 78.4, f32 126.6 GFLOP/s |
| ... this build, scalar | f64 1.9, f32 1.9 GFLOP/s |
| bandwidth | read 36.9, update 52.0, triad 37.8 GB/s (ceiling 52.0) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 20% | 96% | 0 |
| blas2 | 10 | 70% | 77% | 0 |
| blas3 | 10 | 74% | 91% | 0 |
| factor | 12 | 13% | 28% | 0 |
| eigen | 8 | 13% | 43% | 0 |
| elementwise | 5 | 6% | 10% | 0 |
| sparse | 5 | 22% | 50% | 0 |
| direct | 19 | 1% | 8% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 7% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 29% | 32% | 0 |
| mg | 1 | 22% | 22% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 11% | 19% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=33554432 | 47.218 ms | 1.42 | 11.37 | 22% | ok |
| blas1 | dot | f32 | n=33554432 | 49.018 ms | 1.37 | 5.48 | 11% | ok |
| blas1 | axpy | f64 | n=33554432 | 59.909 ms | 1.12 | 13.44 | 26% | ok |
| blas1 | axpy | f32 | n=33554432 | 59.660 ms | 1.12 | 6.75 | 13% | ok |
| blas1 | scale | f64 | n=33554432 | 10.765 ms | 3.12 | 49.87 | 96% | ok |
| blas1 | two_norm | f64 | n=33554432 | 49.045 ms | 1.37 | 5.47 | 11% | ok |
| blas1 | one_norm | f64 | n=33554432 | 25.680 ms | 1.31 | 10.45 | 20% | ok |
| blas1 | infinity_norm | f64 | n=33554432 | 25.712 ms | 1.31 | 10.44 | 20% | ok |
| blas1 | sum | f64 | n=33554432 | 25.663 ms | 1.31 | 10.46 | 20% | ok |
| blas1 | dot_k3 | f64 | n=33554432 | 996.202 ms | 0.74 | 0.54 | 1% | ok |
| blas2 | gemv | f64 | n=8192 | 14.879 ms | 9.02 | 36.09 | 69% | ok |
| blas2 | gemv | f32 | n=8192 | 7.546 ms | 17.79 | 35.58 | 68% | ok |
| blas2 | gemv_colmajor | f64 | n=8192 | 16.059 ms | 8.36 | 33.44 | 64% | ok |
| blas2 | gemv_colmajor | f32 | n=8192 | 6.734 ms | 19.93 | 39.87 | 77% | ok |
| blas2 | symv | f64 | n=8192 | 7.674 ms | 17.49 | 35.00 | 67% | ok |
| blas2 | trmv | f64 | n=8192 | 7.346 ms | 9.14 | 36.56 | 70% | ok |
| blas2 | trmv | f32 | n=8192 | 4.095 ms | 16.39 | 32.79 | 63% | ok |
| blas2 | trsv | f64 | n=8192 | 7.032 ms | 9.54 | 38.19 | 73% | ok |
| blas2 | trsv | f32 | n=8192 | 3.566 ms | 18.82 | 37.65 | 72% | ok |
| blas2 | ger | f64 | n=8192 | 29.107 ms | 4.61 | 36.89 | 71% | ok |
| blas3 | gemm | f64 | 1024x1024x1024 | 36.248 ms | 59.24 | 0.69 | 76% | ok |
| blas3 | gemm | f32 | 1024x1024x1024 | 18.690 ms | 114.90 | 0.67 | 91% | ok |
| blas3 | gemmt | f64 | 1024x1024x1024 | 23.984 ms | 44.77 | 1.05 | 57% | ok |
| blas3 | symm | f64 | 1024x1024x1024 | 37.174 ms | 57.77 | 0.68 | 74% | ok |
| blas3 | syrk | f64 | 1024x1024x1024 | 24.379 ms | 44.04 | 0.69 | 56% | ok |
| blas3 | syr2k | f64 | 1024x1024x1024 | 42.688 ms | 50.31 | 0.59 | 64% | ok |
| blas3 | trmm | f64 | 1024x1024x1024 | 18.226 ms | 58.91 | 1.15 | 75% | ok |
| blas3 | trsm | f64 | 1024x1024x1024 | 23.254 ms | 46.18 | 0.90 | 59% | ok |
| blas3 | trsm | f32 | 1024x1024x1024 | 11.377 ms | 94.38 | 0.92 | 75% | ok |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 34.860 ms | 0.48 | 3.85 | 7% | ok |
| factor | lu | f64 | 1024x1024 | 32.164 ms | 22.26 | 0.52 | 28% | ok |
| factor | lu | f32 | 1024x1024 | 25.143 ms | 28.47 | 0.33 | 22% | ok |
| factor | cholesky | f64 | 1024x1024 | 27.110 ms | 13.20 | 0.62 | 17% | ok |
| factor | cholesky | f32 | 1024x1024 | 20.947 ms | 17.09 | 0.40 | 13% | ok |
| factor | ldlt | f64 | 1024x1024 | 127.872 ms | 2.80 | 0.13 | 4% | ok |
| factor | ldlt_bk | f64 | 1024x1024 | 91.949 ms | 3.89 | 0.18 | 5% | ok |
| factor | qr | f64 | 1024x1024 | 69.734 ms | 20.53 | 0.24 | 26% | ok |
| factor | lq | f64 | 1024x1024 | 657.763 ms | 2.18 | 0.03 | 3% | ok |
| factor | hessenberg | f64 | 1024x1024 | 6925.015 ms | 0.52 | 0.00 | 1% | ok |
| factor | householder | f64 | 1024x1024 | 17.568 ms | 0.24 | 0.96 | 2%\* | ok |
| factor | inv | f64 | 1024x1024 | 568.188 ms | 3.78 | 0.03 | 5% | ok |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 36.989 ms | 19.35 | 0.45 | 25% | ok |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 13.176 ms | 13.58 | 0.32 | 17% | ok |
| eigen | eigen_symmetric | f64 | 512x512 | 3696.633 ms | 0.33 | 0.00 | 0% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 18106.514 ms | 0.07 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 512x512 | 298.294 ms | 4.50 | 0.01 | 6% | ok |
| eigen | schur | f64 | 512x512 | 337.197 ms | 9.95 | 0.01 | 13% | ok |
| eigen | svd | f64 | 512x512 | 86.778 ms | 34.03 | 0.05 | 43% | ok |
| eigen | expm | f64 | 512x512 | 172.943 ms | 11.64 | 0.02 | 15% | ok |
| eigen | sylvester | f64 | 512x512 | 1553.904 ms | 5.18 | 0.00 | 7% | ok |
| elementwise | exp | f64 | n=33554432 | 133.379 ms | 0.25 | 4.03 | 8% | ok |
| elementwise | log | f64 | n=33554432 | 215.842 ms | 0.16 | 2.49 | 5% | ok |
| elementwise | sin | f64 | n=33554432 | 192.122 ms | 0.17 | 2.79 | 5% | ok |
| elementwise | sqrt | f64 | n=33554432 | 105.048 ms | 0.32 | 5.11 | 10% | ok |
| elementwise | erf | f64 | n=33554432 | 169.972 ms | 0.20 | 3.16 | 6% | ok |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 4.168 ms | 2.51 | 26.15 | 50% | ok |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 6.457 ms | 1.62 | 16.88 | 32% | ok |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 30.703 ms | 5.46 | 11.47 | 22% | ok |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 100.251 ms | 0.13 | 0.46 | 1% | ok |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 19.583 ms | 0.52 | 4.28 | 8% | ok |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 171.115 ms | 0.63 | 0.02 | 1% | ok |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 303.339 ms | 0.35 | 0.01 | 0% | ok |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 104.551 ms | 1.02 | 0.03 | 1% | ok |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 522.022 ms | 0.41 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 454.568 ms | 0.47 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 215.712 ms | 0.99 | 0.01 | 1% | ok |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 623.124 ms | 0.04 | 0.00 | 0% | ok |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 21.754 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 32.423 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 341.277 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 1.368 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 108.392 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 17.826 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 142.644 ms | 0.75 | 0.02 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 4.093 ms | 0.98 | 3.92 | 8%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 41.034 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 49.843 ms | 2.15 | 0.06 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 4.148 ms | 0.97 | 3.87 | 7%\* | ok |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 18.763 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 488.982 ms | 0.44 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.314 ms | 0.26 | 1.05 | 2%\* | ok |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 28.107 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 409.527 ms | 0.52 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.404 ms | 0.26 | 1.04 | 2%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 12.709 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1130.666 ms | 0.86 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 7.934 ms | 0.95 | 3.78 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 35.129 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 253.163 ms | 3.83 | 0.01 | 5% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 7.857 ms | 0.95 | 3.82 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 14.115 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 3645.126 ms | 0.53 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.511 ms | 0.25 | 0.98 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 19.387 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1950.247 ms | 0.99 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.360 ms | 0.25 | 0.99 | 2% | ok |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 388.392 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 574.978 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 240.858 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 189.210 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 214.234 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 167.103 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 616.813 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 306.172 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 201.467 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 239.536 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 216.702 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 232.920 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 289.753 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.214 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 256^2 grid | 13.478 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 256^2 grid | 124.763 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 256^2 grid | 10.244 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 256^2 grid | 9.633 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 256^2 grid | 0.155 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 7.346 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 7.058 ms | 1.48 | 16.63 | 32% | ok |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 7.693 ms | 1.36 | 15.26 | 29% | ok |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 9.115 ms | 1.15 | 12.88 | 25% | ok |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 100.645 ms | 1.55 | 11.61 | 22% | ok |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 187.958 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 189.142 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 820.716 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 41.788 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=33554432 | 49.020 ms | 1.37 | 5.48 | 11% | ok |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 298.905 ms | 7.18 | 0.06 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 7.408 ms | 9.06 | 9.63 | 19% | ok |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 16.251 ms | 4.13 | 16.52 | n/a | ok |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 13.351 ms | 5.03 | 10.05 | n/a | ok |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 8.540 ms | 7.86 | 7.86 | n/a | ok |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 8.540 ms | 7.86 | 7.86 | n/a | ok |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 78.772 ms | 27.26 | 0.16 | n/a | ok |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 305.520 ms | 7.03 | 0.03 | n/a | ok |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 274.692 ms | 7.82 | 0.02 | n/a | ok |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 274.764 ms | 7.82 | 0.02 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 275.789 ms | 7.79 | 0.02 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

