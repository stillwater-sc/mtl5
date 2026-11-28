## ryzen-9-8945hs -- native-fast, 8 threads, class large

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 506.1 GFLOP/s, f32 1012.4 GFLOP/s |
| ... hardware, AVX-512F | f64 506.1, f32 1012.4 GFLOP/s |
| ... this build, SSE2 | f64 319.5, f32 636.3 GFLOP/s |
| bandwidth | read 52.5, update 53.1, triad 40.8 GB/s (ceiling 53.1) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 106% | 112% | 0 | 13% |
| blas2 | 10 | 68% | 110% | 0 | 13% |
| blas3 | 10 | 14% | 24% | 0 | 52% |
| factor | 12 | 1% | 13% | 0 | 55% |
| eigen | 8 | 0% | 2% | 0 | 16% |
| elementwise | 5 | 7% | 10% | 0 | 13% |
| sparse | 5 | 32% | 103% | 0 | 15% |
| direct | 19 | 0% | 7% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 7% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 14% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 28% | 31% | 0 | 13% |
| mg | 1 | 24% | 24% | 0 | 14% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 55% | 58% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 83% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 9.024 ms | 7.44 | 59.49 | 112% | ok | 1.1x (13%) |
| blas1 | dot | f32 | n=33554432 | 4.509 ms | 14.88 | 59.54 | 112% | ok | 1.1x (14%) |
| blas1 | axpy | f64 | n=33554432 | 14.441 ms | 4.65 | 55.77 | 105% | ok | 1.0x (13%) |
| blas1 | axpy | f32 | n=33554432 | 7.145 ms | 9.39 | 56.36 | 106% | ok | 1.0x (13%) |
| blas1 | scale | f64 | n=33554432 | 9.494 ms | 3.53 | 56.55 | 106% | ok | 1.0x (13%) |
| blas1 | two_norm | f64 | n=33554432 | 4.518 ms | 14.85 | 59.41 | 112% | ok | 1.0x (13%) |
| blas1 | one_norm | f64 | n=33554432 | 25.613 ms | 1.31 | 10.48 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=33554432 | 25.648 ms | 1.31 | 10.47 | 20% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=33554432 | 25.602 ms | 1.31 | 10.48 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=33554432 | 994.066 ms | 0.74 | 0.54 | 1% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=8192 | 9.177 ms | 14.63 | 58.51 | 110% | ok | 1.6x (20%) |
| blas2 | gemv | f32 | n=8192 | 4.619 ms | 29.06 | 58.13 | 109% | ok | 1.5x (19%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 10.544 ms | 12.73 | 50.93 | 96% | ok | 1.4x (18%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 4.677 ms | 28.70 | 57.41 | 108% | ok | 1.7x (22%) |
| blas2 | symv | f64 | n=8192 | 14.952 ms | 8.98 | 17.96 | 34% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=8192 | 7.487 ms | 8.96 | 35.87 | 68% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=8192 | 3.974 ms | 16.89 | 33.79 | 64% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=8192 | 7.463 ms | 8.99 | 35.99 | 68% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=8192 | 4.177 ms | 16.07 | 32.15 | 61% | ok | 0.9x (12%) |
| blas2 | ger | f64 | n=8192 | 65.318 ms | 2.05 | 16.44 | 31% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 18.675 ms | 114.99 | 1.35 | 23% | ok | 5.8x (72%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 8.812 ms | 243.71 | 1.43 | 24% | ok | 6.0x (75%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 16.357 ms | 65.64 | 1.54 | 13% | ok | 4.0x (50%) |
| blas3 | symm | f64 | 1024x1024x1024 | 23.442 ms | 91.61 | 1.07 | 18% | ok | 4.8x (60%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 15.156 ms | 70.85 | 1.11 | 14% | ok | 4.2x (52%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 27.269 ms | 78.75 | 0.92 | 16% | ok | 4.5x (57%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 36.356 ms | 29.53 | 0.58 | 6% | ok | 2.3x (28%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 37.328 ms | 28.76 | 0.56 | 6% | ok | 2.2x (28%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 27.212 ms | 39.46 | 0.39 | 4% | ok | 1.8x (23%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 34.880 ms | 0.48 | 3.85 | 7% | ok | 1.0x (12%) |
| factor | lu | f64 | 1024x1024 | 71.805 ms | 9.97 | 0.23 | 2% | ok | 4.4x (55%) |
| factor | lu | f32 | 1024x1024 | 67.113 ms | 10.67 | 0.12 | 1% | ok | 4.6x (58%) |
| factor | cholesky | f64 | 1024x1024 | 55.996 ms | 6.39 | 0.30 | 1% | ok | 2.2x (28%) |
| factor | cholesky | f32 | 1024x1024 | 54.855 ms | 6.52 | 0.15 | 1% | ok | 2.3x (28%) |
| factor | ldlt | f64 | 1024x1024 | 126.770 ms | 2.82 | 0.13 | 1% | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 1024x1024 | 91.957 ms | 3.89 | 0.18 | 1% | ok | 1.0x (13%) |
| factor | qr | f64 | 1024x1024 | 1208.693 ms | 1.18 | 0.01 | 0% | ok | 4.9x (61%) |
| factor | lq | f64 | 1024x1024 | 136.585 ms | 10.48 | 0.12 | 2% | ok | 4.8x (60%) |
| factor | hessenberg | f64 | 1024x1024 | 1438.979 ms | 2.49 | 0.01 | 0% | ok | 4.8x (60%) |
| factor | householder | f64 | 1024x1024 | 2.513 ms | 1.67 | 6.68 | 13%\* | ok | 7.0x (87%) |
| factor | inv | f64 | 1024x1024 | 605.486 ms | 3.55 | 0.03 | 1% | ok | 1.4x (18%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 77.103 ms | 9.28 | 0.22 | 2% | ok | 4.2x (52%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 581.976 ms | 0.31 | 0.01 | 0% | ok | 1.6x (20%) |
| eigen | eigen_symmetric | f64 | 512x512 | 3245.514 ms | 0.37 | 0.00 | 0% | ok | 1.1x (14%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 18277.480 ms | 0.07 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 512x512 | 1277.044 ms | 1.05 | 0.00 | 0% | ok | 1.3x (16%) |
| eigen | schur | f64 | 512x512 | 2360.691 ms | 1.42 | 0.00 | 0% | ok | 1.2x (15%) |
| eigen | svd | f64 | 512x512 | 1371.348 ms | 2.15 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | expm | f64 | 512x512 | 176.345 ms | 11.42 | 0.02 | 2% | ok | 1.6x (20%) |
| eigen | sylvester | f64 | 512x512 | 5440.634 ms | 1.48 | 0.00 | 0% | ok | 1.2x (16%) |
| elementwise | exp | f64 | n=33554432 | 133.137 ms | 0.25 | 4.03 | 8% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=33554432 | 206.267 ms | 0.16 | 2.60 | 5% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=33554432 | 194.450 ms | 0.17 | 2.76 | 5% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=33554432 | 104.409 ms | 0.32 | 5.14 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=33554432 | 154.979 ms | 0.22 | 3.46 | 7% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 1.997 ms | 5.25 | 54.58 | 103% | ok | 2.1x (26%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 6.397 ms | 1.64 | 17.04 | 32% | ok | 1.2x (15%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 9.108 ms | 18.41 | 38.68 | 73% | ok | 3.5x (44%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 99.179 ms | 0.13 | 0.46 | 1% | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 18.360 ms | 0.55 | 4.56 | 9% | ok | 1.1x (13%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 167.229 ms | 0.64 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 270.691 ms | 0.40 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 98.549 ms | 1.09 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 520.594 ms | 0.41 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 452.633 ms | 0.47 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 215.715 ms | 0.99 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 613.624 ms | 0.04 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 21.825 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 32.155 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 343.394 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 1.353 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 104.227 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 18.953 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 143.601 ms | 0.75 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 4.114 ms | 0.98 | 3.90 | 7%\* | ok | 1.0x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 41.702 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 49.712 ms | 2.15 | 0.06 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 4.182 ms | 0.96 | 3.84 | 7%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 18.427 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 483.640 ms | 0.44 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.322 ms | 0.26 | 1.05 | 2%\* | ok | 1.0x (12%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 28.016 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 402.589 ms | 0.53 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.355 ms | 0.26 | 1.05 | 2%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 12.614 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1067.468 ms | 0.91 | 0.00 | 0% | ok | 1.1x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 7.968 ms | 0.94 | 3.76 | 7% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 36.085 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 251.860 ms | 3.85 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 8.137 ms | 0.92 | 3.69 | 7% | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 14.287 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 3629.008 ms | 0.53 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.645 ms | 0.24 | 0.98 | 2% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 21.040 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1893.705 ms | 1.02 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.607 ms | 0.24 | 0.98 | 2% | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 299.033 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 411.280 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 212.623 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 136.061 ms | 0.00 | 0.00 | n/a | ok | 1.2x (14%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 161.096 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 131.905 ms | 0.00 | 0.00 | n/a | ok | 1.2x (14%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 395.366 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 254.782 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 159.284 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 156.777 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 144.703 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 157.419 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 193.599 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.226 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 256^2 grid | 13.734 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 256^2 grid | 127.824 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 256^2 grid | 10.565 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 256^2 grid | 10.084 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| pc | ssor | f64 | 256^2 grid | 0.146 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 3.619 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 7.031 ms | 1.49 | 16.69 | 31% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 7.843 ms | 1.34 | 14.97 | 28% | ok | 1.0x (12%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 9.124 ms | 1.15 | 12.86 | 24% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 90.232 ms | 1.72 | 12.95 | 24% | ok | 1.1x (14%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 188.448 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 188.301 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 827.093 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 32.162 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=33554432 | 8.710 ms | 7.71 | 30.82 | 58% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 23.095 ms | 92.99 | 0.73 | 18% | ok | 6.5x (82%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 2.420 ms | 27.73 | 29.47 | 55% | ok | 1.0x (13%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 5.836 ms | 11.50 | 46.00 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 2.406 ms | 27.90 | 55.79 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 1.810 ms | 37.07 | 37.07 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 1.617 ms | 41.51 | 41.51 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 20.499 ms | 104.76 | 0.61 | n/a | ok | 7.2x (90%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 23.136 ms | 92.82 | 0.36 | n/a | ok | 7.5x (94%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 23.396 ms | 91.79 | 0.27 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 5.269 ms | 407.58 | 1.19 | n/a | ok | 6.7x (84%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 5.281 ms | 406.61 | 1.19 | n/a | ok | 6.7x (83%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

