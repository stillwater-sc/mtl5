## xeon-e5-2420v2 -- native-fast, 6 threads, class large

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 59.0 GFLOP/s, f32 118.1 GFLOP/s |
| ... hardware, SSE2 | f64 59.0, f32 118.1 GFLOP/s |
| ... this build, SSE4 | f64 59.0, f32 117.9 GFLOP/s |
| bandwidth | read 11.4, update 10.7, triad 8.5 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 94% | 108% | 0 | 17% |
| blas2 | 10 | 84% | 108% | 0 | 17% |
| blas3 | 10 | 62% | 85% | 0 | 70% |
| factor | 12 | 6% | 39% | 0 | 51% |
| eigen | 8 | 2% | 15% | 0 | 19% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 68% | 97% | 0 | 17% |
| direct | 19 | 1% | 15% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 3% | 13% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 16% |
| smoother | 3 | 69% | 77% | 0 | 17% |
| mg | 1 | 60% | 60% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 66% | 82% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 86% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 43.563 ms | 1.54 | 12.32 | 108% | ok | 1.1x (18%) |
| blas1 | dot | f32 | n=33554432 | 21.767 ms | 3.08 | 12.33 | 108% | ok | 1.1x (18%) |
| blas1 | axpy | f64 | n=33554432 | 75.128 ms | 0.89 | 10.72 | 94% | ok | 1.0x (17%) |
| blas1 | axpy | f32 | n=33554432 | 36.902 ms | 1.82 | 10.91 | 96% | ok | 1.0x (17%) |
| blas1 | scale | f64 | n=33554432 | 53.820 ms | 0.62 | 9.98 | 87% | ok | 1.1x (18%) |
| blas1 | two_norm | f64 | n=33554432 | 21.694 ms | 3.09 | 12.37 | 108% | ok | 1.1x (18%) |
| blas1 | one_norm | f64 | n=33554432 | 39.070 ms | 0.86 | 6.87 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=33554432 | 52.426 ms | 0.64 | 5.12 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=33554432 | 39.072 ms | 0.86 | 6.87 | 60% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=33554432 | 1101.283 ms | 0.67 | 0.49 | 4% | ok | 1.0x (16%) |
| blas2 | gemv | f64 | n=8192 | 43.729 ms | 3.07 | 12.28 | 108% | ok | 1.3x (22%) |
| blas2 | gemv | f32 | n=8192 | 21.945 ms | 6.12 | 12.24 | 107% | ok | 1.3x (22%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 43.804 ms | 3.06 | 12.26 | 107% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 22.446 ms | 5.98 | 11.96 | 105% | ok | 1.2x (20%) |
| blas2 | symv | f64 | n=8192 | 55.322 ms | 2.43 | 4.85 | 43% | ok | 1.0x (16%) |
| blas2 | trmv | f64 | n=8192 | 28.634 ms | 2.34 | 9.38 | 82% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=8192 | 14.042 ms | 4.78 | 9.56 | 84% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=8192 | 28.737 ms | 2.34 | 9.35 | 82% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=8192 | 14.097 ms | 4.76 | 9.53 | 83% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=8192 | 119.428 ms | 1.12 | 8.99 | 79% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 44.664 ms | 48.08 | 0.56 | 81% | ok | 5.0x (84%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 21.326 ms | 100.70 | 0.59 | 85% | ok | 5.1x (84%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 31.597 ms | 33.98 | 0.80 | 58% | ok | 4.2x (69%) |
| blas3 | symm | f64 | 1024x1024x1024 | 52.416 ms | 40.97 | 0.48 | 69% | ok | 4.4x (74%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 29.392 ms | 36.53 | 0.57 | 62% | ok | 4.2x (70%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 59.108 ms | 36.33 | 0.43 | 62% | ok | 4.2x (70%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 54.697 ms | 19.63 | 0.38 | 33% | ok | 2.7x (45%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 56.343 ms | 19.06 | 0.37 | 32% | ok | 2.7x (45%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 26.119 ms | 41.11 | 0.40 | 35% | ok | 2.7x (44%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 131.100 ms | 0.13 | 1.02 | 9% | ok | 1.0x (17%) |
| factor | lu | f64 | 1024x1024 | 68.508 ms | 10.45 | 0.24 | 18% | ok | 3.4x (56%) |
| factor | lu | f32 | 1024x1024 | 40.112 ms | 17.85 | 0.21 | 15% | ok | 3.1x (51%) |
| factor | cholesky | f64 | 1024x1024 | 102.076 ms | 3.51 | 0.16 | 6% | ok | 2.0x (34%) |
| factor | cholesky | f32 | 1024x1024 | 100.104 ms | 3.58 | 0.08 | 3% | ok | 2.0x (33%) |
| factor | ldlt | f64 | 1024x1024 | 213.559 ms | 1.68 | 0.08 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 1024x1024 | 100.652 ms | 3.56 | 0.17 | 6% | ok | 1.1x (18%) |
| factor | qr | f64 | 1024x1024 | 1633.923 ms | 0.88 | 0.01 | 1% | ok | 4.2x (71%) |
| factor | lq | f64 | 1024x1024 | 148.231 ms | 9.66 | 0.11 | 16% | ok | 3.9x (65%) |
| factor | hessenberg | f64 | 1024x1024 | 1897.444 ms | 1.89 | 0.01 | 3% | ok | 4.1x (68%) |
| factor | householder | f64 | 1024x1024 | 3.768 ms | 1.11 | 4.45 | 39% | ok | 5.4x (91%) |
| factor | inv | f64 | 1024x1024 | 1094.530 ms | 1.96 | 0.02 | 3% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 56.051 ms | 12.77 | 0.30 | 22% | ok | 2.5x (42%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 413.229 ms | 0.43 | 0.01 | 1% | ok | 1.5x (26%) |
| eigen | eigen_symmetric | f64 | 512x512 | 1853.981 ms | 0.65 | 0.00 | 1% | ok | 1.1x (19%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 14428.361 ms | 0.08 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 512x512 | 863.384 ms | 1.55 | 0.00 | 3% | ok | 1.2x (21%) |
| eigen | schur | f64 | 512x512 | 2570.615 ms | 1.31 | 0.00 | 2% | ok | 1.2x (19%) |
| eigen | svd | f64 | 512x512 | 2313.357 ms | 1.28 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 512x512 | 226.471 ms | 8.89 | 0.02 | 15% | ok | 1.8x (30%) |
| eigen | sylvester | f64 | 512x512 | 5828.320 ms | 1.38 | 0.00 | 2% | ok | 1.2x (19%) |
| elementwise | exp | f64 | n=33554432 | 618.150 ms | 0.05 | 0.87 | 8% | ok | 1.0x (16%) |
| elementwise | log | f64 | n=33554432 | 559.886 ms | 0.06 | 0.96 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=33554432 | 685.153 ms | 0.05 | 0.78 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=33554432 | 360.958 ms | 0.09 | 1.49 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=33554432 | 9896.111 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 9.853 ms | 1.06 | 11.06 | 97% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 12.740 ms | 0.82 | 8.55 | 75% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 45.663 ms | 3.67 | 7.71 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 96.058 ms | 0.14 | 0.48 | 4% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 37.595 ms | 0.27 | 2.23 | 20% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 312.440 ms | 0.34 | 0.01 | 1% | ok | 1.0x (16%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 559.291 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 158.249 ms | 0.68 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 893.080 ms | 0.24 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 923.644 ms | 0.23 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 375.998 ms | 0.57 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 1101.398 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 44.648 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 65.344 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 595.150 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 2.521 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 250.465 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 26.850 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 269.403 ms | 0.40 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 9.180 ms | 0.44 | 1.75 | 15% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 55.277 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 93.107 ms | 1.15 | 0.03 | 2% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 9.265 ms | 0.43 | 1.73 | 15% | ok | 1.0x (16%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 28.272 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 840.816 ms | 0.25 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 31.024 ms | 0.13 | 0.52 | 5% | ok | 1.0x (16%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 42.439 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 836.860 ms | 0.26 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.444 ms | 0.13 | 0.53 | 5% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 24.053 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1917.779 ms | 0.51 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 19.523 ms | 0.38 | 1.54 | 13% | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 67.593 ms | 0.00 | 0.00 | n/a | ok | 0.9x (15%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 573.167 ms | 1.69 | 0.00 | 3% | ok | 1.1x (18%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 19.531 ms | 0.38 | 1.54 | 13% | ok | 1.0x (16%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 24.573 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 5791.476 ms | 0.33 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 87.781 ms | 0.09 | 0.34 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 34.565 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 6583.403 ms | 0.29 | 0.00 | 0% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 87.996 ms | 0.09 | 0.34 | 3% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 612.950 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 828.839 ms | 0.00 | 0.00 | n/a | ok | 1.2x (20%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 385.037 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 280.505 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 297.328 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 273.859 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 654.813 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 527.953 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 306.818 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 318.484 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 285.844 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 310.849 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 381.223 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.521 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | ilu_0 | f64 | 256^2 grid | 19.316 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | ilut | f64 | 256^2 grid | 148.588 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 256^2 grid | 11.986 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 256^2 grid | 11.409 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 256^2 grid | 0.374 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 9.021 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 14.916 ms | 0.70 | 7.87 | 69% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 13.443 ms | 0.78 | 8.73 | 77% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 15.939 ms | 0.66 | 7.36 | 65% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 171.535 ms | 0.91 | 6.81 | 60% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 553.879 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 573.111 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 1422.123 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 64.749 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=33554432 | 28.557 ms | 2.35 | 9.40 | 82% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 93.895 ms | 22.87 | 0.18 | 39% | ok | 5.3x (88%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 9.483 ms | 7.08 | 7.52 | 66% | ok | 1.0x (17%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 25.252 ms | 2.66 | 10.63 | n/a | ok | 1.0x (16%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.166 ms | 5.52 | 11.03 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 8.811 ms | 7.62 | 7.62 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 7.363 ms | 9.11 | 9.11 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 29.233 ms | 73.46 | 0.43 | n/a | ok | 5.2x (86%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 33.287 ms | 64.51 | 0.25 | n/a | ok | 5.3x (88%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 33.139 ms | 64.80 | 0.19 | n/a | ok | 5.3x (88%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 24.161 ms | 88.88 | 0.26 | n/a | ok | 5.3x (88%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 18.547 ms | 115.79 | 0.34 | n/a | ok | 5.2x (87%) |

