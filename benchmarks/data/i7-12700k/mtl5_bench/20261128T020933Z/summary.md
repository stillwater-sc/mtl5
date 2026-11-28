# mtl5_bench platform characterization

Class `large`, variants native native-fast blas, threads 1,8, CPUs 0,2,4,6,8,10,12,14, clock policy governor performance on the P-cores, turbo on; performance 800-4900 MHz on cpus 0,2,8,10,12,14; performance 800-5000 MHz on cpus 4,6, 2026-11-28T02:57:18Z.

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

## i7-12700k -- native, 8 threads, class large

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 600.7 GFLOP/s, f32 1201.5 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.7, f32 1201.5 GFLOP/s |
| ... this build, scalar | f64 9.9, f32 10.0 GFLOP/s |
| bandwidth | read 83.5, update 78.7, triad 77.1 GB/s (ceiling 83.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 86% | 105% | 0 | 29% |
| blas2 | 10 | 42% | 103% | 0 | 14% |
| blas3 | 10 | 2% | 16% | 0 | 14% |
| factor | 12 | 2% | 22% | 0 | 56% |
| eigen | 8 | 0% | 4% | 0 | 16% |
| elementwise | 5 | 3% | 5% | 0 | 13% |
| sparse | 5 | 32% | 112% | 0 | 14% |
| direct | 19 | 0% | 8% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 7% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 14% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 32% | 0 | 13% |
| mg | 1 | 23% | 23% | 0 | 14% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 32% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 89% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 6.285 ms | 10.68 | 85.42 | 102% | ok | 3.6x (45%) |
| blas1 | dot | f32 | n=33554432 | 3.070 ms | 21.86 | 87.45 | 105% | ok | 5.6x (70%) |
| blas1 | axpy | f64 | n=33554432 | 11.167 ms | 6.01 | 72.11 | 86% | ok | 2.3x (29%) |
| blas1 | axpy | f32 | n=33554432 | 5.603 ms | 11.98 | 71.86 | 86% | ok | 3.4x (42%) |
| blas1 | scale | f64 | n=33554432 | 7.467 ms | 4.49 | 71.90 | 86% | ok | 1.9x (24%) |
| blas1 | two_norm | f64 | n=33554432 | 3.274 ms | 20.50 | 82.00 | 98% | ok | 6.1x (76%) |
| blas1 | one_norm | f64 | n=33554432 | 16.305 ms | 2.06 | 16.46 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=33554432 | 27.875 ms | 1.20 | 9.63 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=33554432 | 16.142 ms | 2.08 | 16.63 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=33554432 | 318.510 ms | 2.32 | 1.69 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=8192 | 6.530 ms | 20.55 | 82.24 | 98% | ok | 2.3x (29%) |
| blas2 | gemv | f32 | n=8192 | 3.132 ms | 42.86 | 85.74 | 103% | ok | 3.6x (45%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 6.489 ms | 20.68 | 82.75 | 99% | ok | 3.1x (39%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 3.290 ms | 40.80 | 81.62 | 98% | ok | 3.0x (37%) |
| blas2 | symv | f64 | n=8192 | 13.086 ms | 10.26 | 20.52 | 25% | ok | 1.1x (14%) |
| blas2 | trmv | f64 | n=8192 | 8.032 ms | 8.36 | 33.44 | 40% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=8192 | 5.685 ms | 11.81 | 23.62 | 28% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=8192 | 7.896 ms | 8.50 | 34.01 | 41% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=8192 | 5.554 ms | 12.08 | 24.18 | 29% | ok | 1.1x (13%) |
| blas2 | ger | f64 | n=8192 | 30.747 ms | 4.37 | 34.93 | 42% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 22.875 ms | 93.88 | 1.10 | 16% | ok | 7.9x (99%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 11.662 ms | 184.14 | 1.08 | 15% | ok | 7.8x (98%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 93.839 ms | 11.44 | 0.27 | 2% | ok | 1.1x (14%) |
| blas3 | symm | f64 | 1024x1024x1024 | 174.027 ms | 12.34 | 0.14 | 2% | ok | 1.0x (13%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 98.811 ms | 10.87 | 0.17 | 2% | ok | 1.1x (14%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 174.431 ms | 12.31 | 0.14 | 2% | ok | 1.1x (14%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 85.785 ms | 12.52 | 0.24 | 2% | ok | 1.0x (13%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 86.094 ms | 12.47 | 0.24 | 2% | ok | 1.0x (13%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 43.491 ms | 24.69 | 0.24 | 2% | ok | 1.0x (13%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 43.714 ms | 0.38 | 3.07 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 1024x1024 | 19.456 ms | 36.79 | 0.86 | 6% | ok | 5.6x (70%) |
| factor | lu | f32 | 1024x1024 | 12.977 ms | 55.16 | 0.65 | 5% | ok | 4.5x (56%) |
| factor | cholesky | f64 | 1024x1024 | 31.532 ms | 11.35 | 0.53 | 2% | ok | 2.2x (27%) |
| factor | cholesky | f32 | 1024x1024 | 29.376 ms | 12.18 | 0.29 | 1% | ok | 2.1x (27%) |
| factor | ldlt | f64 | 1024x1024 | 66.799 ms | 5.36 | 0.25 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 1024x1024 | 41.321 ms | 8.66 | 0.41 | 1% | ok | 1.0x (13%) |
| factor | qr | f64 | 1024x1024 | 434.070 ms | 3.30 | 0.04 | 1% | ok | 5.2x (65%) |
| factor | lq | f64 | 1024x1024 | 45.737 ms | 31.30 | 0.37 | 5% | ok | 4.9x (62%) |
| factor | hessenberg | f64 | 1024x1024 | 507.526 ms | 7.05 | 0.03 | 1% | ok | 5.2x (65%) |
| factor | householder | f64 | 1024x1024 | 0.898 ms | 4.67 | 18.68 | 22%\* | ok | 7.6x (95%) |
| factor | inv | f64 | 1024x1024 | 346.703 ms | 6.19 | 0.05 | 1% | ok | 1.3x (16%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 18.883 ms | 37.91 | 0.89 | 6% | ok | 3.4x (42%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 165.402 ms | 1.08 | 0.03 | 0% | ok | 1.7x (21%) |
| eigen | eigen_symmetric | f64 | 512x512 | 824.853 ms | 1.46 | 0.01 | 0% | ok | 1.2x (15%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 12136.407 ms | 0.10 | 0.00 | 0% | ok | 1.1x (13%) |
| eigen | eigenvalue | f64 | 512x512 | 329.157 ms | 4.08 | 0.01 | 1% | ok | 1.4x (17%) |
| eigen | schur | f64 | 512x512 | 1170.941 ms | 2.87 | 0.00 | 0% | ok | 1.2x (15%) |
| eigen | svd | f64 | 512x512 | 1226.651 ms | 2.41 | 0.00 | 0% | ok | 1.1x (13%) |
| eigen | expm | f64 | 512x512 | 87.718 ms | 22.95 | 0.05 | 4% | ok | 2.9x (36%) |
| eigen | sylvester | f64 | 512x512 | 2411.511 ms | 3.34 | 0.00 | 1% | ok | 1.2x (16%) |
| elementwise | exp | f64 | n=33554432 | 160.675 ms | 0.21 | 3.34 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=33554432 | 185.010 ms | 0.18 | 2.90 | 3% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=33554432 | 241.771 ms | 0.14 | 2.22 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=33554432 | 119.513 ms | 0.28 | 4.49 | 5% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=33554432 | 811.040 ms | 0.04 | 0.66 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 1.162 ms | 9.02 | 93.80 | 112% | ok | 3.1x (38%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 4.118 ms | 2.54 | 26.46 | 32% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 6.837 ms | 24.52 | 51.52 | 62% | ok | 3.0x (37%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 43.508 ms | 0.30 | 1.06 | 1% | ok | 0.7x (9%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 13.241 ms | 0.77 | 6.33 | 8% | ok | 1.1x (14%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 100.622 ms | 1.06 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 184.931 ms | 0.58 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 49.323 ms | 2.17 | 0.06 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 314.167 ms | 0.68 | 0.01 | 0% | ok | 1.1x (13%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 299.220 ms | 0.72 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 128.402 ms | 1.67 | 0.02 | 0% | ok | 1.1x (13%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 359.570 ms | 0.06 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 16.414 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 22.873 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 260.145 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 0.895 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 91.564 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 8.354 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 92.810 ms | 1.15 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 2.615 ms | 1.54 | 6.14 | 7%\* | ok | 0.9x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 17.817 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 28.626 ms | 3.74 | 0.10 | 1% | ok | 1.1x (14%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 2.493 ms | 1.61 | 6.44 | 8%\* | ok | 0.9x (11%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 7.913 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 286.762 ms | 0.75 | 0.01 | 0% | ok | 1.1x (13%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.713 ms | 0.46 | 1.84 | 2%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 12.916 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 268.450 ms | 0.80 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.722 ms | 0.46 | 1.84 | 2%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 8.782 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 593.698 ms | 1.63 | 0.00 | 0% | ok | 1.1x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 5.048 ms | 1.49 | 5.94 | 7% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 18.911 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 177.602 ms | 5.46 | 0.01 | 1% | ok | 1.1x (14%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 5.015 ms | 1.50 | 5.98 | 7% | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 8.646 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 1882.633 ms | 1.03 | 0.00 | 0% | ok | 1.1x (14%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.639 ms | 0.38 | 1.53 | 2% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 12.709 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1360.162 ms | 1.43 | 0.00 | 0% | ok | 1.1x (13%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.611 ms | 0.38 | 1.53 | 2% | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 236.935 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 226.110 ms | 0.00 | 0.00 | n/a | ok | 1.3x (16%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 148.458 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 105.830 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 112.343 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 98.039 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 273.346 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 184.498 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 108.793 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 122.585 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 109.555 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 115.299 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 154.172 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.156 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | ilu_0 | f64 | 256^2 grid | 6.001 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 256^2 grid | 48.625 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 256^2 grid | 4.290 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 256^2 grid | 4.018 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 256^2 grid | 0.109 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 2.808 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 4.384 ms | 2.39 | 26.78 | 32% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 4.921 ms | 2.13 | 23.85 | 29% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 6.251 ms | 1.68 | 18.78 | 22% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 61.466 ms | 2.53 | 19.01 | 23% | ok | 1.1x (14%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 290.541 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 291.465 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 505.930 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 20.296 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=33554432 | 19.281 ms | 3.48 | 13.92 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 41.040 ms | 52.33 | 0.41 | 9% | ok | 7.6x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 2.628 ms | 25.53 | 27.13 | 32% | ok | 1.0x (13%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 9.387 ms | 7.15 | 28.60 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 4.622 ms | 14.52 | 29.04 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 6.882 ms | 9.75 | 9.75 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 6.854 ms | 9.79 | 9.79 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 19.292 ms | 111.32 | 0.65 | n/a | ok | 7.1x (89%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 21.871 ms | 98.19 | 0.38 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 10.743 ms | 199.90 | 0.59 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 10.740 ms | 199.94 | 0.59 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 28.123 ms | 76.36 | 0.22 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## i7-12700k -- native-fast, 1 thread, class large

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.5 GFLOP/s |
| ... this build, AVX2 | f64 78.3, f32 156.6 GFLOP/s |
| bandwidth | read 37.6, update 41.2, triad 31.3 GB/s (ceiling 41.2) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 86% | 101% | 0 |
| blas2 | 10 | 91% | 94% | 0 |
| blas3 | 10 | 77% | 93% | 0 |
| factor | 12 | 7% | 15% | 0 |
| eigen | 8 | 3% | 25% | 0 |
| elementwise | 5 | 7% | 11% | 0 |
| sparse | 5 | 51% | 75% | 0 |
| direct | 19 | 1% | 16% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 4% | 15% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 57% | 65% | 0 |
| mg | 1 | 42% | 42% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 79% | 81% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=33554432 | 15.179 ms | 4.42 | 35.37 | 86% | ok |
| blas1 | dot | f32 | n=33554432 | 7.504 ms | 8.94 | 35.77 | 87% | ok |
| blas1 | axpy | f64 | n=33554432 | 19.452 ms | 3.45 | 41.40 | 100% | ok |
| blas1 | axpy | f32 | n=33554432 | 9.670 ms | 6.94 | 41.64 | 101% | ok |
| blas1 | scale | f64 | n=33554432 | 13.936 ms | 2.41 | 38.52 | 93% | ok |
| blas1 | two_norm | f64 | n=33554432 | 9.587 ms | 7.00 | 28.00 | 68% | ok |
| blas1 | one_norm | f64 | n=33554432 | 16.475 ms | 2.04 | 16.29 | 40% | ok |
| blas1 | infinity_norm | f64 | n=33554432 | 28.258 ms | 1.19 | 9.50 | 23% | ok |
| blas1 | sum | f64 | n=33554432 | 16.238 ms | 2.07 | 16.53 | 40% | ok |
| blas1 | dot_k3 | f64 | n=33554432 | 300.098 ms | 2.46 | 1.79 | 4% | ok |
| blas2 | gemv | f64 | n=8192 | 13.853 ms | 9.69 | 38.77 | 94% | ok |
| blas2 | gemv | f32 | n=8192 | 6.919 ms | 19.40 | 38.80 | 94% | ok |
| blas2 | gemv_colmajor | f64 | n=8192 | 14.233 ms | 9.43 | 37.73 | 91% | ok |
| blas2 | gemv_colmajor | f32 | n=8192 | 7.083 ms | 18.95 | 37.91 | 92% | ok |
| blas2 | symv | f64 | n=8192 | 12.371 ms | 10.85 | 21.71 | 53% | ok |
| blas2 | trmv | f64 | n=8192 | 7.221 ms | 9.29 | 37.19 | 90% | ok |
| blas2 | trmv | f32 | n=8192 | 3.666 ms | 18.31 | 36.63 | 89% | ok |
| blas2 | trsv | f64 | n=8192 | 7.238 ms | 9.27 | 37.11 | 90% | ok |
| blas2 | trsv | f32 | n=8192 | 3.684 ms | 18.22 | 36.45 | 88% | ok |
| blas2 | ger | f64 | n=8192 | 28.616 ms | 4.69 | 37.53 | 91% | ok |
| blas3 | gemm | f64 | 1024x1024x1024 | 30.187 ms | 71.14 | 0.83 | 91% | ok |
| blas3 | gemm | f32 | 1024x1024x1024 | 14.784 ms | 145.25 | 0.85 | 93% | ok |
| blas3 | gemmt | f64 | 1024x1024x1024 | 18.962 ms | 56.63 | 1.33 | 72% | ok |
| blas3 | symm | f64 | 1024x1024x1024 | 32.722 ms | 65.63 | 0.77 | 84% | ok |
| blas3 | syrk | f64 | 1024x1024x1024 | 17.611 ms | 60.97 | 0.95 | 78% | ok |
| blas3 | syr2k | f64 | 1024x1024x1024 | 35.836 ms | 59.92 | 0.70 | 77% | ok |
| blas3 | trmm | f64 | 1024x1024x1024 | 21.872 ms | 49.09 | 0.96 | 63% | ok |
| blas3 | trsm | f64 | 1024x1024x1024 | 22.469 ms | 47.79 | 0.93 | 61% | ok |
| blas3 | trsm | f32 | 1024x1024x1024 | 10.556 ms | 101.72 | 0.99 | 65% | ok |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 44.062 ms | 0.38 | 3.05 | 7% | ok |
| factor | lu | f64 | 1024x1024 | 100.981 ms | 7.09 | 0.17 | 9% | ok |
| factor | lu | f32 | 1024x1024 | 56.055 ms | 12.77 | 0.15 | 8% | ok |
| factor | cholesky | f64 | 1024x1024 | 68.725 ms | 5.21 | 0.24 | 7% | ok |
| factor | cholesky | f32 | 1024x1024 | 62.246 ms | 5.75 | 0.13 | 4% | ok |
| factor | ldlt | f64 | 1024x1024 | 69.084 ms | 5.18 | 0.24 | 7% | ok |
| factor | ldlt_bk | f64 | 1024x1024 | 40.661 ms | 8.80 | 0.41 | 11% | ok |
| factor | qr | f64 | 1024x1024 | 2257.084 ms | 0.63 | 0.01 | 1% | ok |
| factor | lq | f64 | 1024x1024 | 212.880 ms | 6.73 | 0.08 | 9% | ok |
| factor | hessenberg | f64 | 1024x1024 | 2563.502 ms | 1.40 | 0.01 | 2% | ok |
| factor | householder | f64 | 1024x1024 | 6.767 ms | 0.62 | 2.48 | 6%\* | ok |
| factor | inv | f64 | 1024x1024 | 443.943 ms | 4.84 | 0.04 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 61.666 ms | 11.61 | 0.27 | 15% | ok |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 270.060 ms | 0.66 | 0.02 | 1% | ok |
| eigen | eigen_symmetric | f64 | 512x512 | 949.429 ms | 1.27 | 0.00 | 2% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 12249.069 ms | 0.10 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 512x512 | 440.368 ms | 3.05 | 0.01 | 4% | ok |
| eigen | schur | f64 | 512x512 | 1384.994 ms | 2.42 | 0.00 | 3% | ok |
| eigen | svd | f64 | 512x512 | 1091.818 ms | 2.70 | 0.00 | 3% | ok |
| eigen | expm | f64 | 512x512 | 103.623 ms | 19.43 | 0.04 | 25% | ok |
| eigen | sylvester | f64 | 512x512 | 2863.032 ms | 2.81 | 0.00 | 4% | ok |
| elementwise | exp | f64 | n=33554432 | 164.428 ms | 0.20 | 3.27 | 8% | ok |
| elementwise | log | f64 | n=33554432 | 189.045 ms | 0.18 | 2.84 | 7% | ok |
| elementwise | sin | f64 | n=33554432 | 246.677 ms | 0.14 | 2.18 | 5% | ok |
| elementwise | sqrt | f64 | n=33554432 | 121.972 ms | 0.28 | 4.40 | 11% | ok |
| elementwise | erf | f64 | n=33554432 | 814.944 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 3.531 ms | 2.97 | 30.86 | 75% | ok |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 4.161 ms | 2.52 | 26.19 | 64% | ok |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 16.837 ms | 9.96 | 20.92 | 51% | ok |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 29.848 ms | 0.44 | 1.54 | 4% | ok |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 14.570 ms | 0.70 | 5.75 | 14% | ok |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 104.648 ms | 1.02 | 0.03 | 1% | ok |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 196.812 ms | 0.54 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 53.525 ms | 2.00 | 0.05 | 3% | ok |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 303.558 ms | 0.70 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 328.090 ms | 0.65 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 129.289 ms | 1.66 | 0.02 | 2% | ok |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 379.909 ms | 0.06 | 0.00 | 0% | ok |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 17.144 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 24.033 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 266.752 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 0.975 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 92.525 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 8.628 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 97.259 ms | 1.10 | 0.03 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 2.458 ms | 1.63 | 6.53 | 16%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 18.350 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 32.341 ms | 3.31 | 0.09 | 4% | ok |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 2.449 ms | 1.64 | 6.56 | 16%\* | ok |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 8.252 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 279.193 ms | 0.77 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.806 ms | 0.46 | 1.82 | 4%\* | ok |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 13.503 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 288.542 ms | 0.74 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.783 ms | 0.46 | 1.83 | 4%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 9.005 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 619.941 ms | 1.56 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 4.846 ms | 1.55 | 6.19 | 15% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 19.392 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 207.903 ms | 4.66 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 5.244 ms | 1.43 | 5.72 | 14% | ok |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 9.008 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 1781.876 ms | 1.09 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.512 ms | 0.38 | 1.54 | 4% | ok |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 13.207 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 2214.990 ms | 0.88 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.473 ms | 0.39 | 1.54 | 4% | ok |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 230.277 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 253.264 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 131.516 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 106.506 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 113.154 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 103.577 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 208.839 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 163.192 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 113.227 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 116.802 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 103.002 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 107.140 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 124.791 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.145 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 256^2 grid | 6.341 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 256^2 grid | 51.283 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 256^2 grid | 4.385 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 256^2 grid | 4.116 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 256^2 grid | 0.122 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 2.869 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 4.410 ms | 2.38 | 26.62 | 65% | ok |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 4.994 ms | 2.10 | 23.50 | 57% | ok |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 5.803 ms | 1.81 | 20.23 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 68.002 ms | 2.29 | 17.18 | 42% | ok |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 295.222 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 296.065 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 510.910 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 19.243 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=33554432 | 8.235 ms | 8.15 | 32.60 | 79% | ok |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 47.454 ms | 45.25 | 0.35 | 58% | ok |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 2.126 ms | 31.57 | 33.55 | 81% | ok |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 7.994 ms | 8.39 | 33.58 | n/a | ok |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 3.908 ms | 17.17 | 34.34 | n/a | ok |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 2.098 ms | 31.99 | 31.99 | n/a | ok |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 2.095 ms | 32.04 | 32.04 | n/a | ok |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 31.357 ms | 68.49 | 0.40 | n/a | ok |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 44.057 ms | 48.74 | 0.19 | n/a | ok |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 44.006 ms | 48.80 | 0.14 | n/a | ok |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 16.361 ms | 131.26 | 0.38 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 14.645 ms | 146.63 | 0.43 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## i7-12700k -- native-fast, 8 threads, class large

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 600.8 GFLOP/s, f32 1201.8 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.8, f32 1201.7 GFLOP/s |
| ... this build, AVX2 | f64 600.8, f32 1201.8 GFLOP/s |
| bandwidth | read 81.9, update 78.9, triad 76.9 GB/s (ceiling 81.9) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 89% | 108% | 0 | 22% |
| blas2 | 10 | 46% | 108% | 0 | 13% |
| blas3 | 10 | 36% | 80% | 0 | 47% |
| factor | 12 | 2% | 23% | 0 | 61% |
| eigen | 8 | 0% | 5% | 0 | 15% |
| elementwise | 5 | 4% | 5% | 0 | 13% |
| sparse | 5 | 32% | 113% | 0 | 14% |
| direct | 19 | 0% | 8% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 7% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 14% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 33% | 0 | 13% |
| mg | 1 | 23% | 23% | 0 | 14% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 41% | 54% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 88% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 6.380 ms | 10.52 | 84.15 | 103% | ok | 2.4x (30%) |
| blas1 | dot | f32 | n=33554432 | 3.040 ms | 22.07 | 88.30 | 108% | ok | 2.5x (31%) |
| blas1 | axpy | f64 | n=33554432 | 11.049 ms | 6.07 | 72.88 | 89% | ok | 1.8x (22%) |
| blas1 | axpy | f32 | n=33554432 | 5.455 ms | 12.30 | 73.82 | 90% | ok | 1.8x (22%) |
| blas1 | scale | f64 | n=33554432 | 7.408 ms | 4.53 | 72.47 | 88% | ok | 1.9x (24%) |
| blas1 | two_norm | f64 | n=33554432 | 3.053 ms | 21.98 | 87.93 | 107% | ok | 3.1x (39%) |
| blas1 | one_norm | f64 | n=33554432 | 16.338 ms | 2.05 | 16.43 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=33554432 | 27.801 ms | 1.21 | 9.66 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=33554432 | 16.084 ms | 2.09 | 16.69 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=33554432 | 294.866 ms | 2.50 | 1.82 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=8192 | 6.359 ms | 21.11 | 84.45 | 103% | ok | 2.2x (27%) |
| blas2 | gemv | f32 | n=8192 | 3.040 ms | 44.16 | 88.34 | 108% | ok | 2.3x (28%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 6.416 ms | 20.92 | 83.69 | 102% | ok | 2.2x (28%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 3.128 ms | 42.90 | 85.83 | 105% | ok | 2.3x (28%) |
| blas2 | symv | f64 | n=8192 | 12.124 ms | 11.07 | 22.15 | 27% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=8192 | 7.106 ms | 9.44 | 37.79 | 46% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=8192 | 3.606 ms | 18.61 | 37.24 | 45% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=8192 | 7.114 ms | 9.43 | 37.75 | 46% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=8192 | 3.639 ms | 18.44 | 36.90 | 45% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=8192 | 28.466 ms | 4.72 | 37.72 | 46% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 4.457 ms | 481.79 | 5.65 | 80% | ok | 6.8x (85%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 2.225 ms | 965.09 | 5.65 | 80% | ok | 6.6x (83%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 4.993 ms | 215.03 | 5.04 | 36% | ok | 3.8x (47%) |
| blas3 | symm | f64 | 1024x1024x1024 | 6.603 ms | 325.25 | 3.81 | 54% | ok | 5.0x (62%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 5.106 ms | 210.29 | 3.29 | 35% | ok | 3.4x (43%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 9.600 ms | 223.68 | 2.62 | 37% | ok | 3.7x (47%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 9.327 ms | 115.13 | 2.25 | 19% | ok | 2.3x (29%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 9.617 ms | 111.65 | 2.18 | 19% | ok | 2.3x (29%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 4.928 ms | 217.87 | 2.13 | 18% | ok | 2.1x (27%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 43.345 ms | 0.39 | 3.10 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 1024x1024 | 16.424 ms | 43.58 | 1.02 | 7% | ok | 6.1x (77%) |
| factor | lu | f32 | 1024x1024 | 11.543 ms | 62.01 | 0.73 | 5% | ok | 4.9x (61%) |
| factor | cholesky | f64 | 1024x1024 | 30.976 ms | 11.55 | 0.54 | 2% | ok | 2.2x (28%) |
| factor | cholesky | f32 | 1024x1024 | 29.392 ms | 12.18 | 0.29 | 1% | ok | 2.1x (26%) |
| factor | ldlt | f64 | 1024x1024 | 66.375 ms | 5.39 | 0.25 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 1024x1024 | 38.704 ms | 9.25 | 0.43 | 2% | ok | 1.1x (13%) |
| factor | qr | f64 | 1024x1024 | 433.847 ms | 3.30 | 0.04 | 1% | ok | 5.2x (65%) |
| factor | lq | f64 | 1024x1024 | 40.662 ms | 35.21 | 0.41 | 6% | ok | 5.2x (65%) |
| factor | hessenberg | f64 | 1024x1024 | 498.497 ms | 7.18 | 0.03 | 1% | ok | 5.1x (64%) |
| factor | householder | f64 | 1024x1024 | 0.893 ms | 4.70 | 18.79 | 23%\* | ok | 7.6x (95%) |
| factor | inv | f64 | 1024x1024 | 356.455 ms | 6.02 | 0.05 | 1% | ok | 1.2x (16%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 17.985 ms | 39.80 | 0.93 | 7% | ok | 3.4x (43%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 156.722 ms | 1.14 | 0.03 | 0% | ok | 1.7x (22%) |
| eigen | eigen_symmetric | f64 | 512x512 | 790.947 ms | 1.53 | 0.01 | 0% | ok | 1.2x (15%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 11838.384 ms | 0.10 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 512x512 | 324.653 ms | 4.13 | 0.01 | 1% | ok | 1.4x (17%) |
| eigen | schur | f64 | 512x512 | 1120.761 ms | 2.99 | 0.00 | 0% | ok | 1.2x (15%) |
| eigen | svd | f64 | 512x512 | 1153.151 ms | 2.56 | 0.00 | 0% | ok | 0.9x (12%) |
| eigen | expm | f64 | 512x512 | 72.063 ms | 27.94 | 0.06 | 5% | ok | 1.4x (18%) |
| eigen | sylvester | f64 | 512x512 | 2324.628 ms | 3.46 | 0.00 | 1% | ok | 1.2x (15%) |
| elementwise | exp | f64 | n=33554432 | 161.856 ms | 0.21 | 3.32 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=33554432 | 186.318 ms | 0.18 | 2.88 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=33554432 | 243.760 ms | 0.14 | 2.20 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=33554432 | 121.413 ms | 0.28 | 4.42 | 5% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=33554432 | 814.947 ms | 0.04 | 0.66 | 1% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 1.176 ms | 8.91 | 92.68 | 113% | ok | 3.0x (38%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 4.215 ms | 2.49 | 25.86 | 32% | ok | 1.0x (12%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 6.843 ms | 24.50 | 51.47 | 63% | ok | 2.5x (31%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 42.065 ms | 0.31 | 1.10 | 1% | ok | 0.7x (9%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 13.206 ms | 0.77 | 6.35 | 8% | ok | 1.1x (14%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 102.654 ms | 1.04 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 193.110 ms | 0.55 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 53.188 ms | 2.01 | 0.06 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 297.757 ms | 0.72 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 326.500 ms | 0.66 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 125.324 ms | 1.71 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 374.840 ms | 0.06 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 16.472 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 23.223 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 262.464 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 0.916 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 90.583 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 8.392 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 94.790 ms | 1.13 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 2.601 ms | 1.54 | 6.17 | 8%\* | ok | 0.9x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 17.938 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 32.319 ms | 3.31 | 0.09 | 1% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 2.658 ms | 1.51 | 6.04 | 7%\* | ok | 0.9x (12%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 8.098 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 273.105 ms | 0.78 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.750 ms | 0.46 | 1.84 | 2%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 13.066 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 301.279 ms | 0.71 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.780 ms | 0.46 | 1.83 | 2%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 8.789 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 584.498 ms | 1.66 | 0.00 | 0% | ok | 1.1x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 5.072 ms | 1.48 | 5.91 | 7% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 18.989 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 205.238 ms | 4.72 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 5.459 ms | 1.37 | 5.49 | 7% | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 8.765 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 1762.700 ms | 1.10 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.556 ms | 0.38 | 1.53 | 2% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 12.906 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 2217.056 ms | 0.87 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.562 ms | 0.38 | 1.53 | 2% | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 213.888 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 195.408 ms | 0.00 | 0.00 | n/a | ok | 1.3x (16%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 133.595 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 96.044 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 101.908 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 92.749 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 193.616 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 166.461 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 101.903 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 103.542 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 93.194 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 96.710 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 114.692 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.132 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | ilu_0 | f64 | 256^2 grid | 6.198 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 256^2 grid | 50.461 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 256^2 grid | 4.274 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 256^2 grid | 4.007 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 256^2 grid | 0.113 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 2.813 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 4.350 ms | 2.41 | 26.98 | 33% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 4.929 ms | 2.13 | 23.81 | 29% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 5.728 ms | 1.83 | 20.49 | 25% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 61.580 ms | 2.53 | 18.98 | 23% | ok | 1.1x (14%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 288.921 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 290.717 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 489.930 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 18.831 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=33554432 | 8.143 ms | 8.24 | 32.97 | 40% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 6.646 ms | 323.12 | 2.52 | 54% | ok | 7.1x (89%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 2.114 ms | 31.74 | 33.72 | 41% | ok | 1.0x (13%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 7.898 ms | 8.50 | 33.99 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 3.830 ms | 17.52 | 35.05 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 2.089 ms | 32.12 | 32.12 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 2.097 ms | 32.00 | 32.00 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 4.411 ms | 486.86 | 2.85 | n/a | ok | 7.1x (89%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 6.166 ms | 348.26 | 1.36 | n/a | ok | 7.1x (89%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 6.180 ms | 347.50 | 1.02 | n/a | ok | 7.1x (89%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 2.333 ms | 920.62 | 2.70 | n/a | ok | 7.0x (88%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 2.084 ms | 1030.46 | 3.02 | n/a | ok | 7.0x (88%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## i7-12700k -- blas, 1 thread, class large

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.5 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.5 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.3 GFLOP/s |
| bandwidth | read 33.1, update 38.7, triad 32.5 GB/s (ceiling 38.7) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 43% | 96% | 0 |
| blas2 | 10 | 95% | 101% | 0 |
| blas3 | 10 | 88% | 96% | 0 |
| factor | 12 | 29% | 51% | 0 |
| eigen | 8 | 25% | 85% | 0 |
| elementwise | 5 | 7% | 11% | 0 |
| sparse | 5 | 47% | 79% | 0 |
| direct | 31 | 5% | 28% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 21 | 16% | 67% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 61% | 69% | 0 |
| mg | 1 | 44% | 44% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 35% | 69% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=33554432 | 23.295 ms | 2.88 | 23.05 | 60% | ok |
| blas1 | dot | f32 | n=33554432 | 17.752 ms | 3.78 | 15.12 | 39% | ok |
| blas1 | axpy | f64 | n=33554432 | 25.972 ms | 2.58 | 31.01 | 80% | ok |
| blas1 | axpy | f32 | n=33554432 | 18.855 ms | 3.56 | 21.36 | 55% | ok |
| blas1 | scale | f64 | n=33554432 | 14.457 ms | 2.32 | 37.14 | 96% | ok |
| blas1 | two_norm | f64 | n=33554432 | 20.317 ms | 3.30 | 13.21 | 34% | ok |
| blas1 | one_norm | f64 | n=33554432 | 16.476 ms | 2.04 | 16.29 | 42% | ok |
| blas1 | infinity_norm | f64 | n=33554432 | 28.288 ms | 1.19 | 9.49 | 25% | ok |
| blas1 | sum | f64 | n=33554432 | 16.327 ms | 2.06 | 16.44 | 43% | ok |
| blas1 | dot_k3 | f64 | n=33554432 | 322.552 ms | 2.29 | 1.66 | 4% | ok |
| blas2 | gemv | f64 | n=8192 | 14.362 ms | 9.35 | 37.39 | 97% | ok |
| blas2 | gemv | f32 | n=8192 | 7.195 ms | 18.65 | 37.32 | 97% | ok |
| blas2 | gemv_colmajor | f64 | n=8192 | 14.967 ms | 8.97 | 35.88 | 93% | ok |
| blas2 | gemv_colmajor | f32 | n=8192 | 7.098 ms | 18.91 | 37.83 | 98% | ok |
| blas2 | symv | f64 | n=8192 | 7.688 ms | 17.46 | 34.93 | 90% | ok |
| blas2 | trmv | f64 | n=8192 | 7.516 ms | 8.93 | 35.73 | 92% | ok |
| blas2 | trmv | f32 | n=8192 | 3.952 ms | 16.98 | 33.98 | 88% | ok |
| blas2 | trsv | f64 | n=8192 | 7.320 ms | 9.17 | 36.69 | 95% | ok |
| blas2 | trsv | f32 | n=8192 | 3.648 ms | 18.39 | 36.81 | 95% | ok |
| blas2 | ger | f64 | n=8192 | 27.436 ms | 4.89 | 39.14 | 101% | ok |
| blas3 | gemm | f64 | 1024x1024x1024 | 29.159 ms | 73.65 | 0.86 | 94% | ok |
| blas3 | gemm | f32 | 1024x1024x1024 | 14.346 ms | 149.69 | 0.88 | 96% | ok |
| blas3 | gemmt | f64 | 1024x1024x1024 | 17.777 ms | 60.40 | 1.42 | 77% | ok |
| blas3 | symm | f64 | 1024x1024x1024 | 29.303 ms | 73.29 | 0.86 | 94% | ok |
| blas3 | syrk | f64 | 1024x1024x1024 | 18.600 ms | 57.73 | 0.90 | 74% | ok |
| blas3 | syr2k | f64 | 1024x1024x1024 | 35.891 ms | 59.83 | 0.70 | 76% | ok |
| blas3 | trmm | f64 | 1024x1024x1024 | 15.510 ms | 69.23 | 1.35 | 88% | ok |
| blas3 | trsm | f64 | 1024x1024x1024 | 15.637 ms | 68.67 | 1.34 | 88% | ok |
| blas3 | trsm | f32 | 1024x1024x1024 | 7.378 ms | 145.54 | 1.42 | 93% | ok |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 44.394 ms | 0.38 | 3.02 | 8% | ok |
| factor | lu | f64 | 1024x1024 | 18.592 ms | 38.50 | 0.90 | 49% | ok |
| factor | lu | f32 | 1024x1024 | 12.264 ms | 58.37 | 0.68 | 37% | ok |
| factor | cholesky | f64 | 1024x1024 | 11.392 ms | 31.42 | 1.47 | 40% | ok |
| factor | cholesky | f32 | 1024x1024 | 7.992 ms | 44.78 | 1.05 | 29% | ok |
| factor | ldlt | f64 | 1024x1024 | 69.347 ms | 5.16 | 0.24 | 7% | ok |
| factor | ldlt_bk | f64 | 1024x1024 | 42.486 ms | 8.42 | 0.39 | 11% | ok |
| factor | qr | f64 | 1024x1024 | 36.106 ms | 39.65 | 0.46 | 51% | ok |
| factor | lq | f64 | 1024x1024 | 218.406 ms | 6.56 | 0.08 | 8% | ok |
| factor | hessenberg | f64 | 1024x1024 | 2615.883 ms | 1.37 | 0.01 | 2% | ok |
| factor | householder | f64 | 1024x1024 | 6.833 ms | 0.61 | 2.46 | 6%\* | ok |
| factor | inv | f64 | 1024x1024 | 351.581 ms | 6.11 | 0.05 | 8% | ok |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 21.477 ms | 33.33 | 0.78 | 43% | ok |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 9.093 ms | 19.68 | 0.46 | 25% | ok |
| eigen | eigen_symmetric | f64 | 512x512 | 972.787 ms | 1.24 | 0.00 | 2% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 13090.976 ms | 0.09 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 512x512 | 117.687 ms | 11.40 | 0.04 | 15% | ok |
| eigen | schur | f64 | 512x512 | 140.828 ms | 23.83 | 0.03 | 30% | ok |
| eigen | svd | f64 | 512x512 | 44.229 ms | 66.76 | 0.09 | 85% | ok |
| eigen | expm | f64 | 512x512 | 93.421 ms | 21.55 | 0.04 | 28% | ok |
| eigen | sylvester | f64 | 512x512 | 453.770 ms | 17.75 | 0.01 | 23% | ok |
| elementwise | exp | f64 | n=33554432 | 163.655 ms | 0.21 | 3.28 | 8% | ok |
| elementwise | log | f64 | n=33554432 | 190.661 ms | 0.18 | 2.82 | 7% | ok |
| elementwise | sin | f64 | n=33554432 | 248.279 ms | 0.14 | 2.16 | 6% | ok |
| elementwise | sqrt | f64 | n=33554432 | 121.927 ms | 0.28 | 4.40 | 11% | ok |
| elementwise | erf | f64 | n=33554432 | 817.840 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 3.572 ms | 2.93 | 30.51 | 79% | ok |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 4.157 ms | 2.52 | 26.22 | 68% | ok |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 19.292 ms | 8.69 | 18.26 | 47% | ok |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 29.800 ms | 0.44 | 1.55 | 4% | ok |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 14.443 ms | 0.70 | 5.80 | 15% | ok |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 103.470 ms | 1.03 | 0.03 | 1% | ok |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 188.094 ms | 0.57 | 0.02 | 1% | ok |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 50.161 ms | 2.13 | 0.06 | 3% | ok |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 318.478 ms | 0.67 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 307.096 ms | 0.70 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 130.947 ms | 1.63 | 0.02 | 2% | ok |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 369.986 ms | 0.06 | 0.00 | 0% | ok |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 17.118 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 23.969 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 265.016 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 0.913 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 91.934 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 8.516 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 93.733 ms | 1.14 | 0.03 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 2.443 ms | 1.64 | 6.57 | 17%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 18.157 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 29.103 ms | 3.68 | 0.10 | 5% | ok |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 2.294 ms | 1.75 | 7.00 | 18%\* | ok |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 8.191 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 293.405 ms | 0.73 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.782 ms | 0.46 | 1.83 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 13.381 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 273.942 ms | 0.78 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.749 ms | 0.46 | 1.84 | 5%\* | ok |
| direct | ref_cholmod | f64 | 192^2 grid, nnz 183552 | 24.236 ms | 4.41 | 0.12 | 6% | ok |
| direct | ref_cholmod.analyze | f64 | 192^2 grid, nnz 183552 | 6.873 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 192^2 grid, nnz 183552 | 15.355 ms | 6.97 | 0.19 | 9% | ok |
| direct | ref_cholmod.solve | f64 | 192^2 grid, nnz 183552 | 1.479 ms | 2.71 | 10.86 | 28%\* | ok |
| direct | ref_pardiso_spd | f64 | 192^2 grid, nnz 183552 | 18.932 ms | 5.65 | 0.16 | 7% | ok |
| direct | ref_pardiso_spd.analyze | f64 | 192^2 grid, nnz 183552 | 7.849 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_pardiso_spd.factor | f64 | 192^2 grid, nnz 183552 | 8.729 ms | 12.26 | 0.34 | 16% | ok |
| direct | ref_pardiso_spd.solve | f64 | 192^2 grid, nnz 183552 | 2.066 ms | 1.94 | 7.77 | 20%\* | ok |
| direct | ref_pardiso_lu | f64 | 192^2 grid, nnz 183552 | 44.269 ms | 4.83 | 0.07 | 6% | ok |
| direct | ref_pardiso_lu.analyze | f64 | 192^2 grid, nnz 183552 | 25.931 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_pardiso_lu.factor | f64 | 192^2 grid, nnz 183552 | 14.758 ms | 14.50 | 0.20 | 19% | ok |
| direct | ref_pardiso_lu.solve | f64 | 192^2 grid, nnz 183552 | 2.813 ms | 1.43 | 5.71 | 15%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 8.981 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 619.948 ms | 1.56 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 4.839 ms | 1.55 | 6.20 | 16% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 19.235 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 182.898 ms | 5.30 | 0.01 | 7% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 4.792 ms | 1.56 | 6.26 | 16% | ok |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 8.897 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 1937.229 ms | 1.00 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.478 ms | 0.38 | 1.54 | 4% | ok |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 13.111 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1382.638 ms | 1.40 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.461 ms | 0.39 | 1.54 | 4% | ok |
| direct3d | ref_cholmod.analyze | f64 | 24^3 grid, nnz 93312 | 6.593 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 24^3 grid, nnz 93312 | 29.117 ms | 33.29 | 0.05 | 43% | ok |
| direct3d | ref_cholmod.solve | f64 | 24^3 grid, nnz 93312 | 1.152 ms | 6.51 | 26.05 | 67% | ok |
| direct3d | ref_pardiso_spd.analyze | f64 | 24^3 grid, nnz 93312 | 14.987 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_pardiso_spd.factor | f64 | 24^3 grid, nnz 93312 | 29.246 ms | 33.14 | 0.05 | 42% | ok |
| direct3d | ref_pardiso_spd.solve | f64 | 24^3 grid, nnz 93312 | 2.060 ms | 3.64 | 14.56 | 38% | ok |
| direct3d | ref_pardiso_lu.analyze | f64 | 24^3 grid, nnz 93312 | 17.159 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_pardiso_lu.factor | f64 | 24^3 grid, nnz 93312 | 58.848 ms | 32.94 | 0.03 | 42% | ok |
| direct3d | ref_pardiso_lu.solve | f64 | 24^3 grid, nnz 93312 | 2.936 ms | 2.55 | 10.22 | 26% | ok |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 252.217 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 285.018 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 144.278 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 116.373 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 122.800 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 108.332 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 269.420 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 180.331 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 120.331 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 134.279 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 119.325 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 125.755 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 152.401 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.171 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 256^2 grid | 6.388 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 256^2 grid | 50.081 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 256^2 grid | 4.400 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 256^2 grid | 4.116 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 256^2 grid | 0.117 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 4.502 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 4.429 ms | 2.37 | 26.50 | 69% | ok |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 4.992 ms | 2.10 | 23.51 | 61% | ok |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 6.345 ms | 1.65 | 18.50 | 48% | ok |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 68.275 ms | 2.28 | 17.11 | 44% | ok |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 296.777 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 296.665 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 508.871 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 20.774 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=33554432 | 19.577 ms | 3.43 | 13.71 | 35% | ok |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 312.128 ms | 6.88 | 0.05 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 2.672 ms | 25.12 | 26.69 | 69% | ok |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 9.519 ms | 7.05 | 28.20 | n/a | ok |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 4.667 ms | 14.38 | 28.76 | n/a | ok |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 6.999 ms | 9.59 | 9.59 | n/a | ok |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 6.995 ms | 9.59 | 9.59 | n/a | ok |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 138.310 ms | 15.53 | 0.09 | n/a | ok |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 166.255 ms | 12.92 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 81.573 ms | 26.33 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 81.555 ms | 26.33 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 214.752 ms | 10.00 | 0.03 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## i7-12700k -- blas, 8 threads, class large

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 600.7 GFLOP/s, f32 1201.5 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.7, f32 1201.5 GFLOP/s |
| ... this build, scalar | f64 10.0, f32 9.9 GFLOP/s |
| bandwidth | read 81.7, update 77.2, triad 75.4 GB/s (ceiling 81.7) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 88% | 107% | 0 | 29% |
| blas2 | 10 | 103% | 107% | 0 | 28% |
| blas3 | 10 | 75% | 92% | 0 | 82% |
| factor | 12 | 8% | 23% | 0 | 22% |
| eigen | 8 | 4% | 15% | 0 | 15% |
| elementwise | 5 | 3% | 5% | 0 | 13% |
| sparse | 5 | 32% | 115% | 0 | 14% |
| direct | 31 | 1% | 13% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 21 | 7% | 31% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 14% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 33% | 0 | 13% |
| mg | 1 | 23% | 23% | 0 | 14% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 33% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 90% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 6.288 ms | 10.67 | 85.39 | 105% | ok | 3.7x (46%) |
| blas1 | dot | f32 | n=33554432 | 3.074 ms | 21.83 | 87.32 | 107% | ok | 5.8x (72%) |
| blas1 | axpy | f64 | n=33554432 | 11.154 ms | 6.02 | 72.20 | 88% | ok | 2.3x (29%) |
| blas1 | axpy | f32 | n=33554432 | 5.602 ms | 11.98 | 71.88 | 88% | ok | 3.4x (42%) |
| blas1 | scale | f64 | n=33554432 | 7.460 ms | 4.50 | 71.96 | 88% | ok | 1.9x (24%) |
| blas1 | two_norm | f64 | n=33554432 | 3.279 ms | 20.47 | 81.87 | 100% | ok | 6.2x (77%) |
| blas1 | one_norm | f64 | n=33554432 | 16.324 ms | 2.06 | 16.44 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=33554432 | 27.898 ms | 1.20 | 9.62 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=33554432 | 16.160 ms | 2.08 | 16.61 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=33554432 | 318.015 ms | 2.32 | 1.69 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=8192 | 6.556 ms | 20.47 | 81.91 | 100% | ok | 2.2x (27%) |
| blas2 | gemv | f32 | n=8192 | 3.205 ms | 41.88 | 83.77 | 103% | ok | 2.2x (28%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 6.409 ms | 20.94 | 83.78 | 103% | ok | 2.3x (29%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 3.137 ms | 42.79 | 85.60 | 105% | ok | 2.3x (28%) |
| blas2 | symv | f64 | n=8192 | 3.190 ms | 42.07 | 84.18 | 103% | ok | 2.4x (30%) |
| blas2 | trmv | f64 | n=8192 | 3.171 ms | 21.16 | 84.69 | 104% | ok | 2.4x (30%) |
| blas2 | trmv | f32 | n=8192 | 1.542 ms | 43.52 | 87.09 | 107% | ok | 2.6x (32%) |
| blas2 | trsv | f64 | n=8192 | 7.224 ms | 9.29 | 37.18 | 46% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=8192 | 3.567 ms | 18.81 | 37.65 | 46% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=8192 | 15.166 ms | 8.85 | 70.81 | 87% | ok | 1.8x (23%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 4.059 ms | 529.10 | 6.20 | 88% | ok | 7.2x (90%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 1.945 ms | 1104.15 | 6.47 | 92% | ok | 7.4x (92%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 3.648 ms | 294.35 | 6.90 | 49% | ok | 4.9x (61%) |
| blas3 | symm | f64 | 1024x1024x1024 | 4.070 ms | 527.62 | 6.18 | 88% | ok | 7.2x (90%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 4.283 ms | 250.69 | 3.92 | 42% | ok | 4.3x (54%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 6.732 ms | 318.99 | 3.74 | 53% | ok | 5.3x (67%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 2.374 ms | 452.26 | 8.83 | 75% | ok | 6.5x (82%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 2.382 ms | 450.83 | 8.81 | 75% | ok | 6.6x (82%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 1.087 ms | 987.39 | 9.64 | 82% | ok | 6.8x (85%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 43.657 ms | 0.38 | 3.07 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 1024x1024 | 8.319 ms | 86.05 | 2.02 | 14% | ok | 2.2x (28%) |
| factor | lu | f32 | 1024x1024 | 6.962 ms | 102.82 | 1.20 | 9% | ok | 1.8x (22%) |
| factor | cholesky | f64 | 1024x1024 | 7.718 ms | 46.37 | 2.17 | 8% | ok | 1.5x (18%) |
| factor | cholesky | f32 | 1024x1024 | 6.149 ms | 58.21 | 1.36 | 5% | ok | 1.3x (16%) |
| factor | ldlt | f64 | 1024x1024 | 66.767 ms | 5.36 | 0.25 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 1024x1024 | 39.235 ms | 9.12 | 0.43 | 2% | ok | 1.1x (14%) |
| factor | qr | f64 | 1024x1024 | 20.147 ms | 71.06 | 0.83 | 12% | ok | 1.8x (22%) |
| factor | lq | f64 | 1024x1024 | 45.467 ms | 31.49 | 0.37 | 5% | ok | 4.8x (60%) |
| factor | hessenberg | f64 | 1024x1024 | 511.182 ms | 7.00 | 0.03 | 1% | ok | 5.1x (64%) |
| factor | householder | f64 | 1024x1024 | 0.894 ms | 4.70 | 18.77 | 23%\* | ok | 7.6x (96%) |
| factor | inv | f64 | 1024x1024 | 334.851 ms | 6.41 | 0.05 | 1% | ok | 1.0x (13%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 13.440 ms | 53.26 | 1.25 | 9% | ok | 1.6x (20%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 6.708 ms | 26.68 | 0.63 | 4% | ok | 1.4x (17%) |
| eigen | eigen_symmetric | f64 | 512x512 | 815.986 ms | 1.48 | 0.01 | 0% | ok | 1.2x (15%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 12501.523 ms | 0.10 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 512x512 | 118.675 ms | 11.31 | 0.04 | 2% | ok | 1.0x (12%) |
| eigen | schur | f64 | 512x512 | 128.612 ms | 26.09 | 0.03 | 4% | ok | 1.1x (14%) |
| eigen | svd | f64 | 512x512 | 33.685 ms | 87.66 | 0.12 | 15% | ok | 1.3x (16%) |
| eigen | expm | f64 | 512x512 | 68.088 ms | 29.57 | 0.06 | 5% | ok | 1.4x (17%) |
| eigen | sylvester | f64 | 512x512 | 420.506 ms | 19.15 | 0.01 | 3% | ok | 1.1x (13%) |
| elementwise | exp | f64 | n=33554432 | 160.953 ms | 0.21 | 3.34 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=33554432 | 188.456 ms | 0.18 | 2.85 | 3% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=33554432 | 242.340 ms | 0.14 | 2.22 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=33554432 | 120.632 ms | 0.28 | 4.45 | 5% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=33554432 | 810.645 ms | 0.04 | 0.66 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 1.160 ms | 9.03 | 93.94 | 115% | ok | 3.1x (38%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 4.187 ms | 2.50 | 26.03 | 32% | ok | 1.0x (12%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 6.842 ms | 24.50 | 51.48 | 63% | ok | 2.8x (35%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 29.368 ms | 0.45 | 1.57 | 2% | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 13.231 ms | 0.77 | 6.33 | 8% | ok | 1.1x (14%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 101.098 ms | 1.06 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 184.471 ms | 0.58 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 49.371 ms | 2.17 | 0.06 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 316.236 ms | 0.68 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 299.610 ms | 0.71 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 135.085 ms | 1.58 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 360.035 ms | 0.06 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 16.204 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 22.791 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 259.801 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 0.882 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 91.321 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 8.263 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 92.430 ms | 1.16 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 2.581 ms | 1.56 | 6.22 | 8%\* | ok | 0.9x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 17.788 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 28.730 ms | 3.72 | 0.10 | 1% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 2.488 ms | 1.61 | 6.45 | 8%\* | ok | 0.9x (12%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 7.918 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 289.717 ms | 0.74 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.751 ms | 0.46 | 1.83 | 2%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 12.855 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 268.064 ms | 0.80 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.737 ms | 0.46 | 1.84 | 2%\* | ok | 1.0x (13%) |
| direct | ref_cholmod | f64 | 192^2 grid, nnz 183552 | 24.980 ms | 4.28 | 0.12 | 1% | ok | 1.0x (12%) |
| direct | ref_cholmod.analyze | f64 | 192^2 grid, nnz 183552 | 6.777 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | ref_cholmod.factor | f64 | 192^2 grid, nnz 183552 | 15.775 ms | 6.78 | 0.19 | 1% | ok | 1.0x (12%) |
| direct | ref_cholmod.solve | f64 | 192^2 grid, nnz 183552 | 1.458 ms | 2.75 | 11.01 | 13%\* | ok | 1.0x (13%) |
| direct | ref_pardiso_spd | f64 | 192^2 grid, nnz 183552 | 11.623 ms | 9.21 | 0.25 | 2% | ok | 1.6x (20%) |
| direct | ref_pardiso_spd.analyze | f64 | 192^2 grid, nnz 183552 | 6.489 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| direct | ref_pardiso_spd.factor | f64 | 192^2 grid, nnz 183552 | 2.824 ms | 37.89 | 1.04 | 6% | ok | 3.1x (39%) |
| direct | ref_pardiso_spd.solve | f64 | 192^2 grid, nnz 183552 | 2.152 ms | 1.87 | 7.46 | 9%\* | ok | 1.0x (12%) |
| direct | ref_pardiso_lu | f64 | 192^2 grid, nnz 183552 | 31.450 ms | 6.80 | 0.09 | 1% | ok | 1.4x (18%) |
| direct | ref_pardiso_lu.analyze | f64 | 192^2 grid, nnz 183552 | 24.871 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | ref_pardiso_lu.factor | f64 | 192^2 grid, nnz 183552 | 4.573 ms | 46.79 | 0.64 | 8% | ok | 3.2x (40%) |
| direct | ref_pardiso_lu.solve | f64 | 192^2 grid, nnz 183552 | 2.841 ms | 1.41 | 5.65 | 7%\* | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 8.683 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 594.423 ms | 1.63 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 5.005 ms | 1.50 | 5.99 | 7% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 18.753 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 177.654 ms | 5.46 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 4.991 ms | 1.50 | 6.01 | 7% | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 8.580 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 1895.462 ms | 1.02 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.629 ms | 0.38 | 1.53 | 2% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 12.679 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1358.744 ms | 1.43 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.611 ms | 0.38 | 1.53 | 2% | ok | 1.0x (12%) |
| direct3d | ref_cholmod.analyze | f64 | 24^3 grid, nnz 93312 | 6.476 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | ref_cholmod.factor | f64 | 24^3 grid, nnz 93312 | 17.934 ms | 54.04 | 0.08 | 9% | ok | 1.6x (20%) |
| direct3d | ref_cholmod.solve | f64 | 24^3 grid, nnz 93312 | 1.175 ms | 6.38 | 25.53 | 31% | ok | 1.0x (12%) |
| direct3d | ref_pardiso_spd.analyze | f64 | 24^3 grid, nnz 93312 | 12.636 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| direct3d | ref_pardiso_spd.factor | f64 | 24^3 grid, nnz 93312 | 5.929 ms | 163.47 | 0.25 | 27% | ok | 4.9x (62%) |
| direct3d | ref_pardiso_spd.solve | f64 | 24^3 grid, nnz 93312 | 2.166 ms | 3.46 | 13.85 | 17% | ok | 1.0x (12%) |
| direct3d | ref_pardiso_lu.analyze | f64 | 24^3 grid, nnz 93312 | 14.421 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| direct3d | ref_pardiso_lu.factor | f64 | 24^3 grid, nnz 93312 | 12.188 ms | 159.04 | 0.12 | 26% | ok | 4.8x (60%) |
| direct3d | ref_pardiso_lu.solve | f64 | 24^3 grid, nnz 93312 | 3.051 ms | 2.46 | 9.83 | 12% | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 237.004 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 226.537 ms | 0.00 | 0.00 | n/a | ok | 1.3x (16%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 149.043 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 106.315 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 112.813 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 98.212 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 271.614 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 184.905 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 109.319 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 123.436 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 110.715 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 116.269 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 153.586 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.155 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | ilu_0 | f64 | 256^2 grid | 5.964 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | ilut | f64 | 256^2 grid | 49.138 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 256^2 grid | 4.277 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 256^2 grid | 4.008 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 256^2 grid | 0.107 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 4.539 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 4.380 ms | 2.39 | 26.80 | 33% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 4.921 ms | 2.13 | 23.85 | 29% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 6.246 ms | 1.68 | 18.79 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 61.797 ms | 2.52 | 18.91 | 23% | ok | 1.1x (14%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 290.798 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 291.536 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 491.592 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 20.332 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=33554432 | 19.277 ms | 3.48 | 13.93 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 41.032 ms | 52.34 | 0.41 | 9% | ok | 7.6x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 2.623 ms | 25.58 | 27.18 | 33% | ok | 1.0x (13%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 9.369 ms | 7.16 | 28.65 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 4.627 ms | 14.50 | 29.01 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 6.853 ms | 9.79 | 9.79 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 6.856 ms | 9.79 | 9.79 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 19.222 ms | 111.72 | 0.65 | n/a | ok | 7.2x (90%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 21.876 ms | 98.17 | 0.38 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 10.738 ms | 200.00 | 0.59 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 10.739 ms | 199.97 | 0.59 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 28.112 ms | 76.39 | 0.22 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

