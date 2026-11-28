## xeon-e5-2420v2 -- native-fast, 6 threads, class large

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 59.1 GFLOP/s, f32 118.2 GFLOP/s |
| ... hardware, SSE2 | f64 59.1, f32 118.2 GFLOP/s |
| ... this build, SSE4 | f64 58.9, f32 118.0 GFLOP/s |
| bandwidth | read 11.3, update 10.5, triad 8.3 GB/s (ceiling 11.3) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 95% | 110% | 0 | 17% |
| blas2 | 10 | 84% | 109% | 0 | 18% |
| blas3 | 10 | 61% | 86% | 0 | 70% |
| factor | 12 | 6% | 40% | 0 | 51% |
| eigen | 8 | 2% | 15% | 0 | 19% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 69% | 98% | 0 | 17% |
| direct | 19 | 1% | 16% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 3% | 14% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 16% |
| smoother | 3 | 70% | 77% | 0 | 17% |
| mg | 1 | 60% | 60% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 67% | 82% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 86% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 43.471 ms | 1.54 | 12.35 | 110% | ok | 1.1x (18%) |
| blas1 | dot | f32 | n=33554432 | 21.746 ms | 3.09 | 12.34 | 110% | ok | 1.1x (18%) |
| blas1 | axpy | f64 | n=33554432 | 75.218 ms | 0.89 | 10.71 | 95% | ok | 1.1x (18%) |
| blas1 | axpy | f32 | n=33554432 | 36.619 ms | 1.83 | 11.00 | 98% | ok | 1.0x (17%) |
| blas1 | scale | f64 | n=33554432 | 54.126 ms | 0.62 | 9.92 | 88% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=33554432 | 22.364 ms | 3.00 | 12.00 | 107% | ok | 1.1x (18%) |
| blas1 | one_norm | f64 | n=33554432 | 39.100 ms | 0.86 | 6.87 | 61% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=33554432 | 52.566 ms | 0.64 | 5.11 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=33554432 | 40.607 ms | 0.83 | 6.61 | 59% | ok | 1.0x (16%) |
| blas1 | dot_k3 | f64 | n=33554432 | 1095.084 ms | 0.67 | 0.49 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=8192 | 43.713 ms | 3.07 | 12.28 | 109% | ok | 1.3x (21%) |
| blas2 | gemv | f32 | n=8192 | 21.871 ms | 6.14 | 12.28 | 109% | ok | 1.2x (21%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 43.661 ms | 3.07 | 12.30 | 109% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 22.439 ms | 5.98 | 11.97 | 106% | ok | 1.2x (19%) |
| blas2 | symv | f64 | n=8192 | 55.213 ms | 2.43 | 4.86 | 43% | ok | 1.0x (16%) |
| blas2 | trmv | f64 | n=8192 | 28.876 ms | 2.32 | 9.30 | 83% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=8192 | 14.129 ms | 4.75 | 9.50 | 84% | ok | 1.1x (18%) |
| blas2 | trsv | f64 | n=8192 | 28.630 ms | 2.34 | 9.38 | 83% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=8192 | 14.136 ms | 4.75 | 9.50 | 84% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=8192 | 118.828 ms | 1.13 | 9.04 | 80% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 44.870 ms | 47.86 | 0.56 | 81% | ok | 5.0x (84%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 21.203 ms | 101.28 | 0.59 | 86% | ok | 5.1x (85%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 31.911 ms | 33.65 | 0.79 | 57% | ok | 4.1x (69%) |
| blas3 | symm | f64 | 1024x1024x1024 | 52.087 ms | 41.23 | 0.48 | 70% | ok | 4.5x (75%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 29.475 ms | 36.43 | 0.57 | 62% | ok | 4.2x (70%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 59.131 ms | 36.32 | 0.43 | 61% | ok | 4.2x (70%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 54.469 ms | 19.71 | 0.39 | 33% | ok | 2.8x (46%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 56.410 ms | 19.03 | 0.37 | 32% | ok | 2.7x (45%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 26.054 ms | 41.21 | 0.40 | 35% | ok | 2.7x (45%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 129.422 ms | 0.13 | 1.04 | 9% | ok | 1.0x (17%) |
| factor | lu | f64 | 1024x1024 | 67.998 ms | 10.53 | 0.25 | 18% | ok | 3.4x (57%) |
| factor | lu | f32 | 1024x1024 | 40.214 ms | 17.80 | 0.21 | 15% | ok | 3.1x (51%) |
| factor | cholesky | f64 | 1024x1024 | 97.321 ms | 3.68 | 0.17 | 6% | ok | 2.1x (36%) |
| factor | cholesky | f32 | 1024x1024 | 94.156 ms | 3.80 | 0.09 | 3% | ok | 2.1x (36%) |
| factor | ldlt | f64 | 1024x1024 | 213.328 ms | 1.68 | 0.08 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 1024x1024 | 100.848 ms | 3.55 | 0.17 | 6% | ok | 1.1x (18%) |
| factor | qr | f64 | 1024x1024 | 1630.123 ms | 0.88 | 0.01 | 1% | ok | 4.2x (71%) |
| factor | lq | f64 | 1024x1024 | 148.080 ms | 9.67 | 0.11 | 16% | ok | 3.9x (65%) |
| factor | hessenberg | f64 | 1024x1024 | 1868.861 ms | 1.92 | 0.01 | 3% | ok | 4.2x (69%) |
| factor | householder | f64 | 1024x1024 | 3.736 ms | 1.12 | 4.49 | 40% | ok | 5.5x (91%) |
| factor | inv | f64 | 1024x1024 | 1018.845 ms | 2.11 | 0.02 | 4% | ok | 1.2x (20%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 55.899 ms | 12.81 | 0.30 | 22% | ok | 2.5x (42%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 408.622 ms | 0.44 | 0.01 | 1% | ok | 1.6x (26%) |
| eigen | eigen_symmetric | f64 | 512x512 | 1918.594 ms | 0.63 | 0.00 | 1% | ok | 1.1x (18%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 14575.338 ms | 0.08 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 512x512 | 861.385 ms | 1.56 | 0.00 | 3% | ok | 1.3x (21%) |
| eigen | schur | f64 | 512x512 | 2574.638 ms | 1.30 | 0.00 | 2% | ok | 1.2x (19%) |
| eigen | svd | f64 | 512x512 | 2306.536 ms | 1.28 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 512x512 | 226.553 ms | 8.89 | 0.02 | 15% | ok | 1.8x (30%) |
| eigen | sylvester | f64 | 512x512 | 5854.309 ms | 1.38 | 0.00 | 2% | ok | 1.2x (19%) |
| elementwise | exp | f64 | n=33554432 | 615.379 ms | 0.05 | 0.87 | 8% | ok | 1.0x (16%) |
| elementwise | log | f64 | n=33554432 | 563.363 ms | 0.06 | 0.95 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=33554432 | 682.847 ms | 0.05 | 0.79 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=33554432 | 362.171 ms | 0.09 | 1.48 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=33554432 | 9990.773 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 9.843 ms | 1.06 | 11.07 | 98% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 12.619 ms | 0.83 | 8.64 | 77% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 45.542 ms | 3.68 | 7.73 | 69% | ok | 1.4x (23%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 96.326 ms | 0.14 | 0.48 | 4% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 37.511 ms | 0.27 | 2.23 | 20% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 305.232 ms | 0.35 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 556.160 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 156.767 ms | 0.68 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 900.838 ms | 0.24 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 933.162 ms | 0.23 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 366.652 ms | 0.58 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 1094.192 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 45.130 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 64.291 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 591.647 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 2.522 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 251.941 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 26.737 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 270.712 ms | 0.40 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 9.091 ms | 0.44 | 1.77 | 16% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 55.153 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 92.838 ms | 1.15 | 0.03 | 2% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 9.224 ms | 0.44 | 1.74 | 15% | ok | 1.0x (16%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 27.266 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 837.151 ms | 0.26 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.518 ms | 0.13 | 0.53 | 5% | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 41.974 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 838.004 ms | 0.26 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.308 ms | 0.13 | 0.53 | 5% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 24.166 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1924.249 ms | 0.50 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 19.414 ms | 0.39 | 1.54 | 14% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 62.029 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 569.319 ms | 1.70 | 0.00 | 3% | ok | 1.1x (18%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 19.454 ms | 0.39 | 1.54 | 14% | ok | 1.0x (17%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 24.484 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 5764.603 ms | 0.34 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 88.186 ms | 0.09 | 0.34 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 34.462 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 6510.583 ms | 0.30 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 87.760 ms | 0.09 | 0.34 | 3% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 615.361 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 835.602 ms | 0.00 | 0.00 | n/a | ok | 1.2x (20%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 367.998 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 283.014 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 299.383 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 275.489 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 659.158 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 528.518 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 310.444 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 349.577 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 288.148 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 312.887 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 382.171 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.518 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 256^2 grid | 19.507 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | ilut | f64 | 256^2 grid | 152.003 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | ic_0 | f64 | 256^2 grid | 13.269 ms | 0.00 | 0.00 | n/a | ok | 0.9x (15%) |
| pc | ildl | f64 | 256^2 grid | 11.427 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 256^2 grid | 0.366 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 8.986 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 14.993 ms | 0.70 | 7.83 | 70% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 13.510 ms | 0.78 | 8.69 | 77% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 16.002 ms | 0.65 | 7.33 | 65% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 174.449 ms | 0.89 | 6.70 | 60% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 563.170 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 580.499 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 1428.973 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 65.701 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| mixed | dot | f32>f64 | n=33554432 | 29.023 ms | 2.31 | 9.25 | 82% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 93.683 ms | 22.92 | 0.18 | 39% | ok | 5.3x (88%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 9.456 ms | 7.10 | 7.54 | 67% | ok | 1.0x (17%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 24.456 ms | 2.74 | 10.98 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.372 ms | 5.42 | 10.85 | n/a | ok | 1.0x (16%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 8.820 ms | 7.61 | 7.61 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 7.350 ms | 9.13 | 9.13 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 29.197 ms | 73.55 | 0.43 | n/a | ok | 5.2x (86%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 33.220 ms | 64.65 | 0.25 | n/a | ok | 5.3x (89%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 33.028 ms | 65.02 | 0.19 | n/a | ok | 5.3x (89%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 24.147 ms | 88.94 | 0.26 | n/a | ok | 5.3x (89%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 18.525 ms | 115.92 | 0.34 | n/a | ok | 5.3x (88%) |

