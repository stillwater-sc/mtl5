## ryzen-9-8945hs -- native-fast, 1 thread, class large

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 63.3 GFLOP/s, f32 126.4 GFLOP/s |
| ... hardware, AVX-512F | f64 63.3, f32 126.4 GFLOP/s |
| ... this build, SSE2 | f64 40.7, f32 81.3 GFLOP/s |
| bandwidth | read 45.7, update 51.8, triad 37.8 GB/s (ceiling 51.8) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 105% | 110% | 0 |
| blas2 | 10 | 67% | 73% | 0 |
| blas3 | 10 | 27% | 32% | 0 |
| factor | 12 | 4% | 6% | 0 |
| eigen | 8 | 2% | 11% | 0 |
| elementwise | 5 | 7% | 10% | 0 |
| sparse | 5 | 21% | 51% | 0 |
| direct | 19 | 1% | 8% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 7% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 30% | 32% | 0 |
| mg | 1 | 23% | 23% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 57% | 59% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=33554432 | 9.724 ms | 6.90 | 55.21 | 107% | ok |
| blas1 | dot | f32 | n=33554432 | 5.025 ms | 13.35 | 53.42 | 103% | ok |
| blas1 | axpy | f64 | n=33554432 | 14.648 ms | 4.58 | 54.98 | 106% | ok |
| blas1 | axpy | f32 | n=33554432 | 7.372 ms | 9.10 | 54.62 | 105% | ok |
| blas1 | scale | f64 | n=33554432 | 9.825 ms | 3.42 | 54.64 | 105% | ok |
| blas1 | two_norm | f64 | n=33554432 | 4.698 ms | 14.28 | 57.13 | 110% | ok |
| blas1 | one_norm | f64 | n=33554432 | 25.624 ms | 1.31 | 10.48 | 20% | ok |
| blas1 | infinity_norm | f64 | n=33554432 | 25.659 ms | 1.31 | 10.46 | 20% | ok |
| blas1 | sum | f64 | n=33554432 | 25.610 ms | 1.31 | 10.48 | 20% | ok |
| blas1 | dot_k3 | f64 | n=33554432 | 995.324 ms | 0.74 | 0.54 | 1% | ok |
| blas2 | gemv | f64 | n=8192 | 14.451 ms | 9.29 | 37.16 | 72% | ok |
| blas2 | gemv | f32 | n=8192 | 7.064 ms | 19.00 | 38.01 | 73% | ok |
| blas2 | gemv_colmajor | f64 | n=8192 | 15.038 ms | 8.93 | 35.71 | 69% | ok |
| blas2 | gemv_colmajor | f32 | n=8192 | 8.113 ms | 16.54 | 33.10 | 64% | ok |
| blas2 | symv | f64 | n=8192 | 15.269 ms | 8.79 | 17.59 | 34% | ok |
| blas2 | trmv | f64 | n=8192 | 7.470 ms | 8.98 | 35.95 | 69% | ok |
| blas2 | trmv | f32 | n=8192 | 4.132 ms | 16.24 | 32.50 | 63% | ok |
| blas2 | trsv | f64 | n=8192 | 7.698 ms | 8.72 | 34.89 | 67% | ok |
| blas2 | trsv | f32 | n=8192 | 3.934 ms | 17.06 | 34.13 | 66% | ok |
| blas2 | ger | f64 | n=8192 | 65.618 ms | 2.05 | 16.37 | 32% | ok |
| blas3 | gemm | f64 | 1024x1024x1024 | 107.472 ms | 19.98 | 0.23 | 32% | ok |
| blas3 | gemm | f32 | 1024x1024x1024 | 53.213 ms | 40.36 | 0.24 | 32% | ok |
| blas3 | gemmt | f64 | 1024x1024x1024 | 64.873 ms | 16.55 | 0.39 | 26% | ok |
| blas3 | symm | f64 | 1024x1024x1024 | 112.638 ms | 19.07 | 0.22 | 30% | ok |
| blas3 | syrk | f64 | 1024x1024x1024 | 62.921 ms | 17.07 | 0.27 | 27% | ok |
| blas3 | syr2k | f64 | 1024x1024x1024 | 123.543 ms | 17.38 | 0.20 | 27% | ok |
| blas3 | trmm | f64 | 1024x1024x1024 | 82.576 ms | 13.00 | 0.25 | 21% | ok |
| blas3 | trsm | f64 | 1024x1024x1024 | 82.927 ms | 12.95 | 0.25 | 20% | ok |
| blas3 | trsm | f32 | 1024x1024x1024 | 49.517 ms | 21.68 | 0.21 | 17% | ok |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 34.736 ms | 0.48 | 3.86 | 7% | ok |
| factor | lu | f64 | 1024x1024 | 315.996 ms | 2.27 | 0.05 | 4% | ok |
| factor | lu | f32 | 1024x1024 | 309.650 ms | 2.31 | 0.03 | 2% | ok |
| factor | cholesky | f64 | 1024x1024 | 125.630 ms | 2.85 | 0.13 | 5% | ok |
| factor | cholesky | f32 | 1024x1024 | 124.311 ms | 2.88 | 0.07 | 2% | ok |
| factor | ldlt | f64 | 1024x1024 | 125.691 ms | 2.85 | 0.13 | 4% | ok |
| factor | ldlt_bk | f64 | 1024x1024 | 92.871 ms | 3.85 | 0.18 | 6% | ok |
| factor | qr | f64 | 1024x1024 | 5899.118 ms | 0.24 | 0.00 | 0% | ok |
| factor | lq | f64 | 1024x1024 | 659.034 ms | 2.17 | 0.03 | 3% | ok |
| factor | hessenberg | f64 | 1024x1024 | 6870.642 ms | 0.52 | 0.00 | 1% | ok |
| factor | householder | f64 | 1024x1024 | 17.521 ms | 0.24 | 0.96 | 2%\* | ok |
| factor | inv | f64 | 1024x1024 | 851.274 ms | 2.52 | 0.02 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 320.694 ms | 2.23 | 0.05 | 4% | ok |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 949.497 ms | 0.19 | 0.00 | 0% | ok |
| eigen | eigen_symmetric | f64 | 512x512 | 3684.904 ms | 0.33 | 0.00 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 18384.800 ms | 0.07 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 512x512 | 1640.412 ms | 0.82 | 0.00 | 1% | ok |
| eigen | schur | f64 | 512x512 | 2913.401 ms | 1.15 | 0.00 | 2% | ok |
| eigen | svd | f64 | 512x512 | 1373.046 ms | 2.15 | 0.00 | 3% | ok |
| eigen | expm | f64 | 512x512 | 280.351 ms | 7.18 | 0.01 | 11% | ok |
| eigen | sylvester | f64 | 512x512 | 6792.448 ms | 1.19 | 0.00 | 2% | ok |
| elementwise | exp | f64 | n=33554432 | 133.809 ms | 0.25 | 4.01 | 8% | ok |
| elementwise | log | f64 | n=33554432 | 206.368 ms | 0.16 | 2.60 | 5% | ok |
| elementwise | sin | f64 | n=33554432 | 194.252 ms | 0.17 | 2.76 | 5% | ok |
| elementwise | sqrt | f64 | n=33554432 | 105.360 ms | 0.32 | 5.10 | 10% | ok |
| elementwise | erf | f64 | n=33554432 | 155.437 ms | 0.22 | 3.45 | 7% | ok |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 4.133 ms | 2.53 | 26.37 | 51% | ok |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 7.638 ms | 1.37 | 14.27 | 28% | ok |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 31.695 ms | 5.29 | 11.11 | 21% | ok |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 98.533 ms | 0.13 | 0.47 | 1% | ok |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 19.615 ms | 0.52 | 4.27 | 8% | ok |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 169.706 ms | 0.63 | 0.02 | 1% | ok |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 276.678 ms | 0.39 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 101.041 ms | 1.06 | 0.03 | 2% | ok |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 519.882 ms | 0.41 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 449.612 ms | 0.48 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 217.345 ms | 0.98 | 0.01 | 2% | ok |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 621.327 ms | 0.04 | 0.00 | 0% | ok |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 22.617 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 32.564 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 343.045 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 1.373 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 112.541 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 18.989 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 144.512 ms | 0.74 | 0.02 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 4.089 ms | 0.98 | 3.93 | 8%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 42.647 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 49.407 ms | 2.17 | 0.06 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 4.175 ms | 0.96 | 3.85 | 7%\* | ok |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 18.382 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 500.402 ms | 0.43 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.301 ms | 0.26 | 1.05 | 2%\* | ok |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 28.233 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 401.392 ms | 0.53 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.385 ms | 0.26 | 1.04 | 2%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 12.658 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1126.757 ms | 0.86 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 7.865 ms | 0.95 | 3.81 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 35.111 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 252.566 ms | 3.84 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 8.052 ms | 0.93 | 3.73 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 14.078 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 3650.561 ms | 0.53 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.521 ms | 0.25 | 0.98 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 19.422 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1880.386 ms | 1.03 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.372 ms | 0.25 | 0.99 | 2% | ok |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 324.913 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 471.945 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 220.696 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 156.551 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 181.350 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 151.716 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 418.765 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 263.596 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 180.118 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 177.785 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 158.178 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 167.155 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 201.615 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.224 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 256^2 grid | 13.555 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 256^2 grid | 126.017 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 256^2 grid | 10.169 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 256^2 grid | 9.578 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 256^2 grid | 0.145 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 3.671 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 7.090 ms | 1.48 | 16.56 | 32% | ok |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 7.675 ms | 1.37 | 15.29 | 30% | ok |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 9.243 ms | 1.13 | 12.70 | 25% | ok |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 99.224 ms | 1.57 | 11.78 | 23% | ok |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 187.131 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 187.530 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 820.289 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 32.131 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=33554432 | 8.767 ms | 7.65 | 30.62 | 59% | ok |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 150.750 ms | 14.25 | 0.11 | 23% | ok |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 2.426 ms | 27.67 | 29.39 | 57% | ok |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 5.636 ms | 11.91 | 47.63 | n/a | ok |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 2.462 ms | 27.26 | 54.53 | n/a | ok |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 1.788 ms | 37.52 | 37.52 | n/a | ok |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 1.610 ms | 41.69 | 41.69 | n/a | ok |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 148.123 ms | 14.50 | 0.08 | n/a | ok |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 173.519 ms | 12.38 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 177.214 ms | 12.12 | 0.04 | n/a | ok |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 35.442 ms | 60.59 | 0.18 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 35.161 ms | 61.08 | 0.18 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

