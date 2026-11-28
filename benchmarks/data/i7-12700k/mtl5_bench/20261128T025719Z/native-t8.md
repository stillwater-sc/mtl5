## i7-12700k -- native, 8 threads, class large

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 600.8 GFLOP/s, f32 1201.4 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.8, f32 1201.4 GFLOP/s |
| ... this build, scalar | f64 9.9, f32 10.0 GFLOP/s |
| bandwidth | read 83.1, update 78.1, triad 76.7 GB/s (ceiling 83.1) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 86% | 105% | 0 | 29% |
| blas2 | 10 | 42% | 103% | 0 | 13% |
| blas3 | 10 | 2% | 16% | 0 | 14% |
| factor | 12 | 2% | 22% | 0 | 55% |
| eigen | 8 | 0% | 4% | 0 | 15% |
| elementwise | 5 | 3% | 5% | 0 | 13% |
| sparse | 5 | 31% | 110% | 0 | 14% |
| direct | 19 | 0% | 8% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 7% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 14% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 28% | 32% | 0 | 13% |
| mg | 1 | 22% | 22% | 0 | 14% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 33% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 95% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 6.280 ms | 10.69 | 85.49 | 103% | ok | 3.6x (45%) |
| blas1 | dot | f32 | n=33554432 | 3.067 ms | 21.88 | 87.52 | 105% | ok | 5.6x (70%) |
| blas1 | axpy | f64 | n=33554432 | 11.190 ms | 6.00 | 71.96 | 87% | ok | 2.3x (29%) |
| blas1 | axpy | f32 | n=33554432 | 5.611 ms | 11.96 | 71.76 | 86% | ok | 3.4x (42%) |
| blas1 | scale | f64 | n=33554432 | 7.468 ms | 4.49 | 71.89 | 86% | ok | 1.9x (24%) |
| blas1 | two_norm | f64 | n=33554432 | 3.344 ms | 20.07 | 80.27 | 97% | ok | 5.9x (74%) |
| blas1 | one_norm | f64 | n=33554432 | 16.308 ms | 2.06 | 16.46 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=33554432 | 27.871 ms | 1.20 | 9.63 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=33554432 | 16.161 ms | 2.08 | 16.61 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=33554432 | 317.075 ms | 2.33 | 1.69 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=8192 | 6.533 ms | 20.54 | 82.20 | 99% | ok | 2.3x (29%) |
| blas2 | gemv | f32 | n=8192 | 3.135 ms | 42.81 | 85.65 | 103% | ok | 3.5x (44%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 6.490 ms | 20.68 | 82.75 | 100% | ok | 3.1x (39%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 3.278 ms | 40.95 | 81.91 | 99% | ok | 3.0x (38%) |
| blas2 | symv | f64 | n=8192 | 12.946 ms | 10.37 | 20.75 | 25% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=8192 | 8.014 ms | 8.37 | 33.51 | 40% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=8192 | 5.641 ms | 11.90 | 23.80 | 29% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=8192 | 7.898 ms | 8.50 | 34.00 | 41% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=8192 | 5.540 ms | 12.11 | 24.24 | 29% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=8192 | 30.700 ms | 4.37 | 34.98 | 42% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 22.744 ms | 94.42 | 1.11 | 16% | ok | 7.5x (94%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 11.304 ms | 189.98 | 1.11 | 16% | ok | 7.2x (90%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 84.568 ms | 12.70 | 0.30 | 2% | ok | 1.2x (15%) |
| blas3 | symm | f64 | 1024x1024x1024 | 150.881 ms | 14.23 | 0.17 | 2% | ok | 1.1x (14%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 89.852 ms | 11.95 | 0.19 | 2% | ok | 1.1x (14%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 174.682 ms | 12.29 | 0.14 | 2% | ok | 1.1x (14%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 73.977 ms | 14.51 | 0.28 | 2% | ok | 1.1x (13%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 74.268 ms | 14.46 | 0.28 | 2% | ok | 1.1x (13%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 37.853 ms | 28.37 | 0.28 | 2% | ok | 1.0x (13%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 43.491 ms | 0.39 | 3.09 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 1024x1024 | 19.554 ms | 36.61 | 0.86 | 6% | ok | 5.4x (68%) |
| factor | lu | f32 | 1024x1024 | 12.805 ms | 55.90 | 0.66 | 5% | ok | 4.4x (55%) |
| factor | cholesky | f64 | 1024x1024 | 31.296 ms | 11.44 | 0.54 | 2% | ok | 2.2x (28%) |
| factor | cholesky | f32 | 1024x1024 | 29.601 ms | 12.09 | 0.28 | 1% | ok | 2.1x (26%) |
| factor | ldlt | f64 | 1024x1024 | 66.931 ms | 5.35 | 0.25 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 1024x1024 | 40.371 ms | 8.87 | 0.42 | 1% | ok | 1.1x (13%) |
| factor | qr | f64 | 1024x1024 | 433.548 ms | 3.30 | 0.04 | 1% | ok | 5.2x (65%) |
| factor | lq | f64 | 1024x1024 | 45.543 ms | 31.44 | 0.37 | 5% | ok | 4.8x (60%) |
| factor | hessenberg | f64 | 1024x1024 | 506.850 ms | 7.06 | 0.03 | 1% | ok | 5.1x (64%) |
| factor | householder | f64 | 1024x1024 | 0.899 ms | 4.67 | 18.67 | 22%\* | ok | 7.6x (96%) |
| factor | inv | f64 | 1024x1024 | 345.314 ms | 6.22 | 0.05 | 1% | ok | 1.3x (16%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 18.741 ms | 38.19 | 0.90 | 6% | ok | 3.3x (42%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 161.780 ms | 1.11 | 0.03 | 0% | ok | 1.7x (21%) |
| eigen | eigen_symmetric | f64 | 512x512 | 811.280 ms | 1.49 | 0.01 | 0% | ok | 1.2x (15%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 11983.344 ms | 0.10 | 0.00 | 0% | ok | 1.1x (13%) |
| eigen | eigenvalue | f64 | 512x512 | 332.613 ms | 4.04 | 0.01 | 1% | ok | 1.4x (17%) |
| eigen | schur | f64 | 512x512 | 1163.383 ms | 2.88 | 0.00 | 0% | ok | 1.2x (15%) |
| eigen | svd | f64 | 512x512 | 1231.619 ms | 2.40 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | expm | f64 | 512x512 | 87.935 ms | 22.90 | 0.05 | 4% | ok | 2.7x (33%) |
| eigen | sylvester | f64 | 512x512 | 2463.256 ms | 3.27 | 0.00 | 1% | ok | 1.2x (15%) |
| elementwise | exp | f64 | n=33554432 | 162.522 ms | 0.21 | 3.30 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=33554432 | 187.112 ms | 0.18 | 2.87 | 3% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=33554432 | 242.893 ms | 0.14 | 2.21 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=33554432 | 121.512 ms | 0.28 | 4.42 | 5% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=33554432 | 813.483 ms | 0.04 | 0.66 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 1.187 ms | 8.83 | 91.84 | 110% | ok | 3.0x (38%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 4.209 ms | 2.49 | 25.89 | 31% | ok | 1.0x (12%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 6.863 ms | 24.43 | 51.32 | 62% | ok | 2.8x (35%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 44.089 ms | 0.30 | 1.04 | 1% | ok | 0.7x (9%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 13.318 ms | 0.76 | 6.29 | 8% | ok | 1.1x (14%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 103.898 ms | 1.03 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 184.678 ms | 0.58 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 49.509 ms | 2.16 | 0.06 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 313.910 ms | 0.68 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 299.801 ms | 0.71 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 129.093 ms | 1.66 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 362.315 ms | 0.06 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 16.383 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 23.046 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 263.856 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 0.899 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 91.695 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 8.349 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 93.128 ms | 1.15 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 2.641 ms | 1.52 | 6.08 | 7%\* | ok | 0.9x (11%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 17.942 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 28.675 ms | 3.73 | 0.10 | 1% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 2.523 ms | 1.59 | 6.36 | 8%\* | ok | 0.9x (11%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 7.924 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 285.860 ms | 0.75 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.791 ms | 0.46 | 1.83 | 2%\* | ok | 1.0x (12%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 12.888 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 273.150 ms | 0.78 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.856 ms | 0.45 | 1.81 | 2%\* | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 8.801 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 604.652 ms | 1.60 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 5.122 ms | 1.46 | 5.86 | 7% | ok | 0.9x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 19.323 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 180.005 ms | 5.38 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 5.097 ms | 1.47 | 5.88 | 7% | ok | 0.9x (12%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 8.787 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 1956.739 ms | 0.99 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.983 ms | 0.38 | 1.50 | 2% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 12.919 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1370.897 ms | 1.41 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.799 ms | 0.38 | 1.51 | 2% | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 238.068 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 227.584 ms | 0.00 | 0.00 | n/a | ok | 1.3x (16%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 149.843 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 106.960 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 113.741 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 99.898 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 279.403 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 186.396 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 110.275 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 124.037 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 110.727 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 116.626 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 155.757 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.159 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 256^2 grid | 6.059 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 256^2 grid | 48.945 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 256^2 grid | 4.311 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 256^2 grid | 4.024 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 256^2 grid | 0.108 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 2.816 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 4.451 ms | 2.35 | 26.37 | 32% | ok | 1.0x (12%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 4.986 ms | 2.10 | 23.54 | 28% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 6.315 ms | 1.66 | 18.59 | 22% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 62.907 ms | 2.47 | 18.58 | 22% | ok | 1.1x (14%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 300.281 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 299.030 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 492.572 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 20.483 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=33554432 | 19.301 ms | 3.48 | 13.91 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 41.085 ms | 52.27 | 0.41 | 9% | ok | 7.6x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 2.636 ms | 25.46 | 27.05 | 33% | ok | 1.0x (13%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 9.399 ms | 7.14 | 28.56 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 4.681 ms | 14.34 | 28.67 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 6.899 ms | 9.73 | 9.73 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 6.960 ms | 9.64 | 9.64 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 18.086 ms | 118.74 | 0.70 | n/a | ok | 7.6x (95%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 21.894 ms | 98.08 | 0.38 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 10.742 ms | 199.92 | 0.59 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 10.751 ms | 199.75 | 0.59 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 28.139 ms | 76.32 | 0.22 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

