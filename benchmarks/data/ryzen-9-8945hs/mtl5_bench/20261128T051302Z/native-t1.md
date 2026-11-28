## ryzen-9-8945hs -- native, 1 thread, class large

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 68.0 GFLOP/s, f32 126.5 GFLOP/s |
| ... hardware, AVX-512F | f64 68.0, f32 126.5 GFLOP/s |
| ... this build, scalar | f64 1.6, f32 1.9 GFLOP/s |
| bandwidth | read 36.7, update 51.9, triad 38.1 GB/s (ceiling 51.9) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 20% | 97% | 0 |
| blas2 | 10 | 38% | 71% | 0 |
| blas3 | 10 | 14% | 22% | 0 |
| factor | 12 | 3% | 5% | 0 |
| eigen | 8 | 2% | 9% | 0 |
| elementwise | 5 | 7% | 10% | 0 |
| sparse | 5 | 22% | 51% | 0 |
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
| blas1 | dot | f64 | n=33554432 | 57.579 ms | 1.17 | 9.32 | 18% | ok |
| blas1 | dot | f32 | n=33554432 | 49.019 ms | 1.37 | 5.48 | 11% | ok |
| blas1 | axpy | f64 | n=33554432 | 59.869 ms | 1.12 | 13.45 | 26% | ok |
| blas1 | axpy | f32 | n=33554432 | 59.683 ms | 1.12 | 6.75 | 13% | ok |
| blas1 | scale | f64 | n=33554432 | 10.664 ms | 3.15 | 50.34 | 97% | ok |
| blas1 | two_norm | f64 | n=33554432 | 53.299 ms | 1.26 | 5.04 | 10% | ok |
| blas1 | one_norm | f64 | n=33554432 | 25.603 ms | 1.31 | 10.48 | 20% | ok |
| blas1 | infinity_norm | f64 | n=33554432 | 25.641 ms | 1.31 | 10.47 | 20% | ok |
| blas1 | sum | f64 | n=33554432 | 25.592 ms | 1.31 | 10.49 | 20% | ok |
| blas1 | dot_k3 | f64 | n=33554432 | 996.285 ms | 0.74 | 0.54 | 1% | ok |
| blas2 | gemv | f64 | n=8192 | 14.625 ms | 9.18 | 36.72 | 71% | ok |
| blas2 | gemv | f32 | n=8192 | 13.120 ms | 10.23 | 20.47 | 39% | ok |
| blas2 | gemv_colmajor | f64 | n=8192 | 31.192 ms | 4.30 | 17.22 | 33% | ok |
| blas2 | gemv_colmajor | f32 | n=8192 | 13.524 ms | 9.92 | 19.85 | 38% | ok |
| blas2 | symv | f64 | n=8192 | 15.448 ms | 8.69 | 17.38 | 33% | ok |
| blas2 | trmv | f64 | n=8192 | 8.108 ms | 8.28 | 33.12 | 64% | ok |
| blas2 | trmv | f32 | n=8192 | 7.215 ms | 9.30 | 18.61 | 36% | ok |
| blas2 | trsv | f64 | n=8192 | 7.822 ms | 8.58 | 34.33 | 66% | ok |
| blas2 | trsv | f32 | n=8192 | 7.121 ms | 9.42 | 18.86 | 36% | ok |
| blas2 | ger | f64 | n=8192 | 64.990 ms | 2.07 | 16.52 | 32% | ok |
| blas3 | gemm | f64 | 1024x1024x1024 | 151.529 ms | 14.17 | 0.17 | 21% | ok |
| blas3 | gemm | f32 | 1024x1024x1024 | 76.883 ms | 27.93 | 0.16 | 22% | ok |
| blas3 | gemmt | f64 | 1024x1024x1024 | 96.398 ms | 11.14 | 0.26 | 16% | ok |
| blas3 | symm | f64 | 1024x1024x1024 | 152.941 ms | 14.04 | 0.16 | 21% | ok |
| blas3 | syrk | f64 | 1024x1024x1024 | 109.945 ms | 9.77 | 0.15 | 14% | ok |
| blas3 | syr2k | f64 | 1024x1024x1024 | 263.332 ms | 8.16 | 0.10 | 12% | ok |
| blas3 | trmm | f64 | 1024x1024x1024 | 242.955 ms | 4.42 | 0.09 | 7% | ok |
| blas3 | trsm | f64 | 1024x1024x1024 | 278.017 ms | 3.86 | 0.08 | 6% | ok |
| blas3 | trsm | f32 | 1024x1024x1024 | 255.337 ms | 4.21 | 0.04 | 3% | ok |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 34.748 ms | 0.48 | 3.86 | 7% | ok |
| factor | lu | f64 | 1024x1024 | 317.788 ms | 2.25 | 0.05 | 3% | ok |
| factor | lu | f32 | 1024x1024 | 314.703 ms | 2.27 | 0.03 | 2% | ok |
| factor | cholesky | f64 | 1024x1024 | 128.626 ms | 2.78 | 0.13 | 4% | ok |
| factor | cholesky | f32 | 1024x1024 | 124.994 ms | 2.86 | 0.07 | 2% | ok |
| factor | ldlt | f64 | 1024x1024 | 126.283 ms | 2.83 | 0.13 | 4% | ok |
| factor | ldlt_bk | f64 | 1024x1024 | 96.458 ms | 3.71 | 0.17 | 5% | ok |
| factor | qr | f64 | 1024x1024 | 5931.945 ms | 0.24 | 0.00 | 0% | ok |
| factor | lq | f64 | 1024x1024 | 658.837 ms | 2.17 | 0.03 | 3% | ok |
| factor | hessenberg | f64 | 1024x1024 | 6918.137 ms | 0.52 | 0.00 | 1% | ok |
| factor | householder | f64 | 1024x1024 | 17.449 ms | 0.24 | 0.96 | 2%\* | ok |
| factor | inv | f64 | 1024x1024 | 856.040 ms | 2.51 | 0.02 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 328.068 ms | 2.18 | 0.05 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 940.350 ms | 0.19 | 0.00 | 0% | ok |
| eigen | eigen_symmetric | f64 | 512x512 | 3684.172 ms | 0.33 | 0.00 | 0% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 18349.404 ms | 0.07 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 512x512 | 1625.980 ms | 0.83 | 0.00 | 1% | ok |
| eigen | schur | f64 | 512x512 | 2991.198 ms | 1.12 | 0.00 | 2% | ok |
| eigen | svd | f64 | 512x512 | 1374.160 ms | 2.15 | 0.00 | 3% | ok |
| eigen | expm | f64 | 512x512 | 325.153 ms | 6.19 | 0.01 | 9% | ok |
| eigen | sylvester | f64 | 512x512 | 6766.104 ms | 1.19 | 0.00 | 2% | ok |
| elementwise | exp | f64 | n=33554432 | 133.372 ms | 0.25 | 4.03 | 8% | ok |
| elementwise | log | f64 | n=33554432 | 206.749 ms | 0.16 | 2.60 | 5% | ok |
| elementwise | sin | f64 | n=33554432 | 194.749 ms | 0.17 | 2.76 | 5% | ok |
| elementwise | sqrt | f64 | n=33554432 | 105.084 ms | 0.32 | 5.11 | 10% | ok |
| elementwise | erf | f64 | n=33554432 | 155.472 ms | 0.22 | 3.45 | 7% | ok |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 4.077 ms | 2.57 | 26.73 | 51% | ok |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 6.391 ms | 1.64 | 17.05 | 33% | ok |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 30.602 ms | 5.48 | 11.51 | 22% | ok |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 98.226 ms | 0.13 | 0.47 | 1% | ok |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 19.540 ms | 0.52 | 4.29 | 8% | ok |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 170.018 ms | 0.63 | 0.02 | 1% | ok |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 302.748 ms | 0.35 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 106.832 ms | 1.00 | 0.03 | 1% | ok |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 544.555 ms | 0.39 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 442.005 ms | 0.48 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 218.903 ms | 0.98 | 0.01 | 1% | ok |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 621.446 ms | 0.04 | 0.00 | 0% | ok |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 23.036 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 32.638 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 345.237 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 1.373 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 107.400 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 17.232 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 143.513 ms | 0.75 | 0.02 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 4.120 ms | 0.97 | 3.90 | 8%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 41.387 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 49.670 ms | 2.15 | 0.06 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 4.138 ms | 0.97 | 3.88 | 7%\* | ok |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 18.232 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 495.480 ms | 0.43 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.235 ms | 0.26 | 1.05 | 2%\* | ok |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 28.093 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 399.251 ms | 0.54 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.328 ms | 0.26 | 1.05 | 2%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 12.895 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1046.901 ms | 0.93 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 7.889 ms | 0.95 | 3.80 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 35.533 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 251.538 ms | 3.85 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 8.036 ms | 0.93 | 3.73 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 14.069 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 3669.824 ms | 0.53 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.391 ms | 0.25 | 0.99 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 19.548 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1885.471 ms | 1.03 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.386 ms | 0.25 | 0.99 | 2% | ok |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 401.117 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 599.105 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 250.094 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 197.129 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 222.137 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 171.521 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 668.760 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 312.748 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 206.935 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 247.969 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 224.759 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 246.132 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 310.358 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.226 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 256^2 grid | 13.769 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 256^2 grid | 126.735 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 256^2 grid | 10.440 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 256^2 grid | 9.734 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 256^2 grid | 0.147 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 3.562 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 7.057 ms | 1.48 | 16.63 | 32% | ok |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 7.816 ms | 1.34 | 15.02 | 29% | ok |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 9.137 ms | 1.15 | 12.85 | 25% | ok |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 101.447 ms | 1.53 | 11.52 | 22% | ok |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 188.175 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 187.984 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 816.980 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 39.569 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=33554432 | 57.737 ms | 1.16 | 4.65 | 9% | ok |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 293.169 ms | 7.33 | 0.06 | 11% | ok |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 7.390 ms | 9.08 | 9.65 | 19% | ok |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 16.273 ms | 4.12 | 16.50 | n/a | ok |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.970 ms | 5.17 | 10.35 | n/a | ok |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 6.941 ms | 9.67 | 9.67 | n/a | ok |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 6.960 ms | 9.64 | 9.64 | n/a | ok |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 76.829 ms | 27.95 | 0.16 | n/a | ok |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 306.158 ms | 7.01 | 0.03 | n/a | ok |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 275.104 ms | 7.81 | 0.02 | n/a | ok |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 274.692 ms | 7.82 | 0.02 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 275.428 ms | 7.80 | 0.02 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

