## i7-12700k -- native, 1 thread, class large

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.3 GFLOP/s |
| bandwidth | read 33.1, update 38.6, triad 32.4 GB/s (ceiling 38.6) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 43% | 96% | 0 |
| blas2 | 10 | 71% | 93% | 0 |
| blas3 | 10 | 15% | 15% | 0 |
| factor | 12 | 7% | 14% | 0 |
| eigen | 8 | 3% | 10% | 0 |
| elementwise | 5 | 7% | 11% | 0 |
| sparse | 5 | 45% | 79% | 0 |
| direct | 19 | 1% | 18% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 4% | 16% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 61% | 68% | 0 |
| mg | 1 | 44% | 44% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 35% | 69% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=33554432 | 22.636 ms | 2.96 | 23.72 | 61% | ok |
| blas1 | dot | f32 | n=33554432 | 17.280 ms | 3.88 | 15.53 | 40% | ok |
| blas1 | axpy | f64 | n=33554432 | 25.860 ms | 2.60 | 31.14 | 81% | ok |
| blas1 | axpy | f32 | n=33554432 | 18.855 ms | 3.56 | 21.36 | 55% | ok |
| blas1 | scale | f64 | n=33554432 | 14.442 ms | 2.32 | 37.17 | 96% | ok |
| blas1 | two_norm | f64 | n=33554432 | 19.848 ms | 3.38 | 13.52 | 35% | ok |
| blas1 | one_norm | f64 | n=33554432 | 16.475 ms | 2.04 | 16.29 | 42% | ok |
| blas1 | infinity_norm | f64 | n=33554432 | 28.280 ms | 1.19 | 9.49 | 25% | ok |
| blas1 | sum | f64 | n=33554432 | 16.285 ms | 2.06 | 16.48 | 43% | ok |
| blas1 | dot_k3 | f64 | n=33554432 | 326.550 ms | 2.26 | 1.64 | 4% | ok |
| blas2 | gemv | f64 | n=8192 | 15.035 ms | 8.93 | 35.72 | 93% | ok |
| blas2 | gemv | f32 | n=8192 | 11.358 ms | 11.82 | 23.64 | 61% | ok |
| blas2 | gemv_colmajor | f64 | n=8192 | 19.999 ms | 6.71 | 26.85 | 70% | ok |
| blas2 | gemv_colmajor | f32 | n=8192 | 9.838 ms | 13.64 | 27.29 | 71% | ok |
| blas2 | symv | f64 | n=8192 | 14.720 ms | 9.12 | 18.25 | 47% | ok |
| blas2 | trmv | f64 | n=8192 | 8.093 ms | 8.29 | 33.18 | 86% | ok |
| blas2 | trmv | f32 | n=8192 | 5.913 ms | 11.35 | 22.71 | 59% | ok |
| blas2 | trsv | f64 | n=8192 | 8.013 ms | 8.37 | 33.52 | 87% | ok |
| blas2 | trsv | f32 | n=8192 | 5.890 ms | 11.39 | 22.80 | 59% | ok |
| blas2 | ger | f64 | n=8192 | 31.330 ms | 4.28 | 34.28 | 89% | ok |
| blas3 | gemm | f64 | 1024x1024x1024 | 181.695 ms | 11.82 | 0.14 | 15% | ok |
| blas3 | gemm | f32 | 1024x1024x1024 | 91.378 ms | 23.50 | 0.14 | 15% | ok |
| blas3 | gemmt | f64 | 1024x1024x1024 | 103.439 ms | 10.38 | 0.24 | 13% | ok |
| blas3 | symm | f64 | 1024x1024x1024 | 180.308 ms | 11.91 | 0.14 | 15% | ok |
| blas3 | syrk | f64 | 1024x1024x1024 | 109.646 ms | 9.79 | 0.15 | 13% | ok |
| blas3 | syr2k | f64 | 1024x1024x1024 | 197.245 ms | 10.89 | 0.13 | 14% | ok |
| blas3 | trmm | f64 | 1024x1024x1024 | 89.866 ms | 11.95 | 0.23 | 15% | ok |
| blas3 | trsm | f64 | 1024x1024x1024 | 89.463 ms | 12.00 | 0.23 | 15% | ok |
| blas3 | trsm | f32 | 1024x1024x1024 | 44.662 ms | 24.04 | 0.23 | 15% | ok |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 44.528 ms | 0.38 | 3.01 | 8% | ok |
| factor | lu | f64 | 1024x1024 | 108.667 ms | 6.59 | 0.15 | 8% | ok |
| factor | lu | f32 | 1024x1024 | 58.089 ms | 12.32 | 0.14 | 8% | ok |
| factor | cholesky | f64 | 1024x1024 | 69.239 ms | 5.17 | 0.24 | 7% | ok |
| factor | cholesky | f32 | 1024x1024 | 62.689 ms | 5.71 | 0.13 | 4% | ok |
| factor | ldlt | f64 | 1024x1024 | 69.734 ms | 5.13 | 0.24 | 7% | ok |
| factor | ldlt_bk | f64 | 1024x1024 | 43.328 ms | 8.26 | 0.39 | 11% | ok |
| factor | qr | f64 | 1024x1024 | 2274.358 ms | 0.63 | 0.01 | 1% | ok |
| factor | lq | f64 | 1024x1024 | 225.643 ms | 6.34 | 0.07 | 8% | ok |
| factor | hessenberg | f64 | 1024x1024 | 2624.018 ms | 1.36 | 0.01 | 2% | ok |
| factor | householder | f64 | 1024x1024 | 6.852 ms | 0.61 | 2.45 | 6%\* | ok |
| factor | inv | f64 | 1024x1024 | 438.413 ms | 4.90 | 0.04 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 63.621 ms | 11.25 | 0.26 | 14% | ok |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 278.091 ms | 0.64 | 0.02 | 1% | ok |
| eigen | eigen_symmetric | f64 | 512x512 | 977.245 ms | 1.24 | 0.00 | 2% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 12871.926 ms | 0.09 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 512x512 | 452.958 ms | 2.96 | 0.01 | 4% | ok |
| eigen | schur | f64 | 512x512 | 1429.416 ms | 2.35 | 0.00 | 3% | ok |
| eigen | svd | f64 | 512x512 | 1298.810 ms | 2.27 | 0.00 | 3% | ok |
| eigen | expm | f64 | 512x512 | 254.295 ms | 7.92 | 0.02 | 10% | ok |
| eigen | sylvester | f64 | 512x512 | 3006.886 ms | 2.68 | 0.00 | 3% | ok |
| elementwise | exp | f64 | n=33554432 | 163.474 ms | 0.21 | 3.28 | 9% | ok |
| elementwise | log | f64 | n=33554432 | 190.367 ms | 0.18 | 2.82 | 7% | ok |
| elementwise | sin | f64 | n=33554432 | 246.807 ms | 0.14 | 2.18 | 6% | ok |
| elementwise | sqrt | f64 | n=33554432 | 122.527 ms | 0.27 | 4.38 | 11% | ok |
| elementwise | erf | f64 | n=33554432 | 819.584 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 3.573 ms | 2.93 | 30.50 | 79% | ok |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 4.136 ms | 2.53 | 26.35 | 68% | ok |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 20.432 ms | 8.20 | 17.24 | 45% | ok |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 31.603 ms | 0.41 | 1.46 | 4% | ok |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 14.459 ms | 0.70 | 5.80 | 15% | ok |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 104.228 ms | 1.03 | 0.03 | 1% | ok |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 188.525 ms | 0.57 | 0.02 | 1% | ok |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 51.587 ms | 2.07 | 0.06 | 3% | ok |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 333.227 ms | 0.64 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 311.005 ms | 0.69 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 136.602 ms | 1.57 | 0.02 | 2% | ok |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 368.798 ms | 0.06 | 0.00 | 0% | ok |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 17.141 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 23.847 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 264.794 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 0.910 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 93.556 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 8.580 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 93.916 ms | 1.14 | 0.03 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 2.424 ms | 1.66 | 6.62 | 17%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 18.238 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 30.953 ms | 3.46 | 0.09 | 4% | ok |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 2.283 ms | 1.76 | 7.03 | 18%\* | ok |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 8.141 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 309.588 ms | 0.69 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.758 ms | 0.46 | 1.83 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 13.365 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 279.587 ms | 0.77 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.768 ms | 0.46 | 1.83 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 8.915 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 627.630 ms | 1.54 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 4.848 ms | 1.55 | 6.19 | 16% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 19.131 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 200.570 ms | 4.83 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 4.817 ms | 1.56 | 6.23 | 16% | ok |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 8.856 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 2096.471 ms | 0.92 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.486 ms | 0.38 | 1.54 | 4% | ok |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 13.048 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1428.373 ms | 1.36 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.480 ms | 0.38 | 1.54 | 4% | ok |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 254.144 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 288.793 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 144.352 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 116.707 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 123.446 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 108.988 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 272.914 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 180.538 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 121.000 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 134.873 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 119.577 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 125.249 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 152.877 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.171 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 256^2 grid | 6.185 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 256^2 grid | 49.502 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 256^2 grid | 4.369 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 256^2 grid | 4.093 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 256^2 grid | 0.118 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 2.861 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 4.466 ms | 2.35 | 26.28 | 68% | ok |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 4.998 ms | 2.10 | 23.49 | 61% | ok |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 6.350 ms | 1.65 | 18.48 | 48% | ok |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 68.376 ms | 2.28 | 17.09 | 44% | ok |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 296.193 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 292.241 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 521.007 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 20.766 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=33554432 | 19.702 ms | 3.41 | 13.62 | 35% | ok |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 312.322 ms | 6.88 | 0.05 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 2.672 ms | 25.11 | 26.68 | 69% | ok |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 9.498 ms | 7.07 | 28.26 | n/a | ok |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 4.692 ms | 14.30 | 28.60 | n/a | ok |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 6.998 ms | 9.59 | 9.59 | n/a | ok |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 6.998 ms | 9.59 | 9.59 | n/a | ok |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 137.260 ms | 15.65 | 0.09 | n/a | ok |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 166.381 ms | 12.91 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 81.583 ms | 26.32 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 81.520 ms | 26.34 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 214.841 ms | 10.00 | 0.03 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

