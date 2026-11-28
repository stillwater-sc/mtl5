## xeon-e5-2420v2 -- native-fast, 1 thread, class large

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 21.0 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 21.0 GFLOP/s |
| ... this build, SSE4 | f64 10.5, f32 21.0 GFLOP/s |
| bandwidth | read 10.5, update 10.4, triad 8.2 GB/s (ceiling 10.5) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 98% | 108% | 0 |
| blas2 | 10 | 89% | 97% | 0 |
| blas3 | 10 | 82% | 95% | 0 |
| factor | 12 | 17% | 48% | 0 |
| eigen | 8 | 11% | 47% | 0 |
| elementwise | 5 | 8% | 14% | 0 |
| sparse | 5 | 55% | 100% | 0 |
| direct | 19 | 4% | 17% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 15% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 75% | 83% | 0 |
| mg | 1 | 62% | 62% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 72% | 90% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=33554432 | 47.934 ms | 1.40 | 11.20 | 107% | ok |
| blas1 | dot | f32 | n=33554432 | 23.950 ms | 2.80 | 11.21 | 107% | ok |
| blas1 | axpy | f64 | n=33554432 | 78.283 ms | 0.86 | 10.29 | 98% | ok |
| blas1 | axpy | f32 | n=33554432 | 38.593 ms | 1.74 | 10.43 | 100% | ok |
| blas1 | scale | f64 | n=33554432 | 56.530 ms | 0.59 | 9.50 | 91% | ok |
| blas1 | two_norm | f64 | n=33554432 | 23.649 ms | 2.84 | 11.35 | 108% | ok |
| blas1 | one_norm | f64 | n=33554432 | 39.366 ms | 0.85 | 6.82 | 65% | ok |
| blas1 | infinity_norm | f64 | n=33554432 | 52.704 ms | 0.64 | 5.09 | 49% | ok |
| blas1 | sum | f64 | n=33554432 | 39.079 ms | 0.86 | 6.87 | 66% | ok |
| blas1 | dot_k3 | f64 | n=33554432 | 1085.469 ms | 0.68 | 0.49 | 6% | ok |
| blas2 | gemv | f64 | n=8192 | 57.092 ms | 2.35 | 9.41 | 90% | ok |
| blas2 | gemv | f32 | n=8192 | 28.850 ms | 4.65 | 9.31 | 89% | ok |
| blas2 | gemv_colmajor | f64 | n=8192 | 52.857 ms | 2.54 | 10.16 | 97% | ok |
| blas2 | gemv_colmajor | f32 | n=8192 | 26.598 ms | 5.05 | 10.09 | 96% | ok |
| blas2 | symv | f64 | n=8192 | 54.634 ms | 2.46 | 4.92 | 47% | ok |
| blas2 | trmv | f64 | n=8192 | 29.212 ms | 2.30 | 9.19 | 88% | ok |
| blas2 | trmv | f32 | n=8192 | 14.615 ms | 4.59 | 9.19 | 88% | ok |
| blas2 | trsv | f64 | n=8192 | 29.312 ms | 2.29 | 9.16 | 87% | ok |
| blas2 | trsv | f32 | n=8192 | 14.391 ms | 4.66 | 9.33 | 89% | ok |
| blas2 | ger | f64 | n=8192 | 119.334 ms | 1.12 | 9.00 | 86% | ok |
| blas3 | gemm | f64 | 1024x1024x1024 | 224.993 ms | 9.54 | 0.11 | 91% | ok |
| blas3 | gemm | f32 | 1024x1024x1024 | 107.785 ms | 19.92 | 0.12 | 95% | ok |
| blas3 | gemmt | f64 | 1024x1024x1024 | 131.433 ms | 8.17 | 0.19 | 78% | ok |
| blas3 | symm | f64 | 1024x1024x1024 | 231.160 ms | 9.29 | 0.11 | 88% | ok |
| blas3 | syrk | f64 | 1024x1024x1024 | 122.881 ms | 8.74 | 0.14 | 83% | ok |
| blas3 | syr2k | f64 | 1024x1024x1024 | 247.870 ms | 8.66 | 0.10 | 82% | ok |
| blas3 | trmm | f64 | 1024x1024x1024 | 148.680 ms | 7.22 | 0.14 | 69% | ok |
| blas3 | trsm | f64 | 1024x1024x1024 | 151.748 ms | 7.08 | 0.14 | 67% | ok |
| blas3 | trsm | f32 | 1024x1024x1024 | 69.297 ms | 15.49 | 0.15 | 74% | ok |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 132.156 ms | 0.13 | 1.02 | 10% | ok |
| factor | lu | f64 | 1024x1024 | 232.048 ms | 3.08 | 0.07 | 29% | ok |
| factor | lu | f32 | 1024x1024 | 123.399 ms | 5.80 | 0.07 | 28% | ok |
| factor | cholesky | f64 | 1024x1024 | 206.755 ms | 1.73 | 0.08 | 16% | ok |
| factor | cholesky | f32 | 1024x1024 | 200.110 ms | 1.79 | 0.04 | 9% | ok |
| factor | ldlt | f64 | 1024x1024 | 213.468 ms | 1.68 | 0.08 | 16% | ok |
| factor | ldlt_bk | f64 | 1024x1024 | 107.846 ms | 3.32 | 0.16 | 32% | ok |
| factor | qr | f64 | 1024x1024 | 6918.841 ms | 0.21 | 0.00 | 2% | ok |
| factor | lq | f64 | 1024x1024 | 575.271 ms | 2.49 | 0.03 | 24% | ok |
| factor | hessenberg | f64 | 1024x1024 | 7787.259 ms | 0.46 | 0.00 | 4% | ok |
| factor | householder | f64 | 1024x1024 | 20.478 ms | 0.20 | 0.82 | 8% | ok |
| factor | inv | f64 | 1024x1024 | 1179.453 ms | 1.82 | 0.01 | 17% | ok |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 140.426 ms | 5.10 | 0.12 | 48% | ok |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 635.961 ms | 0.28 | 0.01 | 3% | ok |
| eigen | eigen_symmetric | f64 | 512x512 | 2114.544 ms | 0.57 | 0.00 | 5% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 14729.421 ms | 0.08 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 512x512 | 1077.245 ms | 1.25 | 0.00 | 12% | ok |
| eigen | schur | f64 | 512x512 | 2976.590 ms | 1.13 | 0.00 | 11% | ok |
| eigen | svd | f64 | 512x512 | 2335.613 ms | 1.26 | 0.00 | 12% | ok |
| eigen | expm | f64 | 512x512 | 409.950 ms | 4.91 | 0.01 | 47% | ok |
| eigen | sylvester | f64 | 512x512 | 6736.834 ms | 1.20 | 0.00 | 11% | ok |
| elementwise | exp | f64 | n=33554432 | 605.205 ms | 0.06 | 0.89 | 8% | ok |
| elementwise | log | f64 | n=33554432 | 561.804 ms | 0.06 | 0.96 | 9% | ok |
| elementwise | sin | f64 | n=33554432 | 688.823 ms | 0.05 | 0.78 | 7% | ok |
| elementwise | sqrt | f64 | n=33554432 | 361.719 ms | 0.09 | 1.48 | 14% | ok |
| elementwise | erf | f64 | n=33554432 | 9973.321 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 10.452 ms | 1.00 | 10.43 | 100% | ok |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 12.996 ms | 0.81 | 8.39 | 80% | ok |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 61.505 ms | 2.73 | 5.73 | 55% | ok |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 99.073 ms | 0.13 | 0.47 | 4% | ok |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 38.700 ms | 0.26 | 2.17 | 21% | ok |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 306.151 ms | 0.35 | 0.01 | 3% | ok |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 558.863 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 158.958 ms | 0.67 | 0.02 | 6% | ok |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 903.709 ms | 0.24 | 0.00 | 2% | ok |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 926.489 ms | 0.23 | 0.00 | 2% | ok |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 372.646 ms | 0.57 | 0.01 | 5% | ok |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 1090.627 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 44.510 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 65.617 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 597.738 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 2.532 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 251.798 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 27.445 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 270.540 ms | 0.40 | 0.01 | 4% | ok |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 9.093 ms | 0.44 | 1.77 | 17% | ok |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 56.179 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 93.276 ms | 1.15 | 0.03 | 11% | ok |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 9.090 ms | 0.44 | 1.77 | 17% | ok |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 27.434 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 833.960 ms | 0.26 | 0.00 | 2% | ok |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.277 ms | 0.13 | 0.53 | 5% | ok |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 41.675 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 850.439 ms | 0.25 | 0.00 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.249 ms | 0.13 | 0.53 | 5% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 24.215 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1952.941 ms | 0.50 | 0.00 | 5% | ok |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 19.327 ms | 0.39 | 1.55 | 15% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 62.206 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 609.497 ms | 1.59 | 0.00 | 15% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 19.287 ms | 0.39 | 1.56 | 15% | ok |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 24.682 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 5856.982 ms | 0.33 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 88.052 ms | 0.09 | 0.34 | 3% | ok |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 34.749 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 6526.814 ms | 0.30 | 0.00 | 3% | ok |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 87.361 ms | 0.09 | 0.34 | 3% | ok |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 637.325 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 1006.862 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 365.863 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 291.238 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 306.373 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 283.365 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 655.415 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 513.758 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 309.686 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 331.967 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 288.955 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 304.799 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 371.483 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.513 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 256^2 grid | 19.061 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 256^2 grid | 149.929 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 256^2 grid | 12.001 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 256^2 grid | 11.513 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 256^2 grid | 0.365 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 8.536 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 14.887 ms | 0.70 | 7.88 | 75% | ok |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 13.471 ms | 0.78 | 8.71 | 83% | ok |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 16.037 ms | 0.65 | 7.32 | 70% | ok |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 179.728 ms | 0.87 | 6.50 | 62% | ok |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 559.322 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 592.827 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 1431.146 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 64.686 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=33554432 | 28.604 ms | 2.35 | 9.38 | 90% | ok |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 496.767 ms | 4.32 | 0.03 | 41% | ok |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 9.453 ms | 7.10 | 7.54 | 72% | ok |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 24.368 ms | 2.75 | 11.02 | n/a | ok |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.046 ms | 5.57 | 11.14 | n/a | ok |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 8.770 ms | 7.65 | 7.65 | n/a | ok |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 7.407 ms | 9.06 | 9.06 | n/a | ok |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 150.613 ms | 14.26 | 0.08 | n/a | ok |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 175.863 ms | 12.21 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 175.042 ms | 12.27 | 0.04 | n/a | ok |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 127.558 ms | 16.84 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 96.720 ms | 22.20 | 0.07 | n/a | ok |

