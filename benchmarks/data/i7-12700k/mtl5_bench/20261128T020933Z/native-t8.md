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

