## ryzen-9-8945hs -- native, 1 thread, class large

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 67.2 GFLOP/s, f32 126.6 GFLOP/s |
| ... hardware, AVX-512F | f64 67.2, f32 126.6 GFLOP/s |
| ... this build, scalar | f64 1.6, f32 1.9 GFLOP/s |
| bandwidth | read 36.7, update 51.8, triad 37.8 GB/s (ceiling 51.8) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 20% | 97% | 0 |
| blas2 | 10 | 39% | 70% | 0 |
| blas3 | 10 | 15% | 22% | 0 |
| factor | 12 | 3% | 6% | 0 |
| eigen | 8 | 2% | 9% | 0 |
| elementwise | 5 | 7% | 10% | 0 |
| sparse | 5 | 22% | 53% | 0 |
| direct | 19 | 1% | 8% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 7% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 29% | 32% | 0 |
| mg | 1 | 23% | 23% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 11% | 19% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=33554432 | 57.575 ms | 1.17 | 9.32 | 18% | ok |
| blas1 | dot | f32 | n=33554432 | 49.099 ms | 1.37 | 5.47 | 11% | ok |
| blas1 | axpy | f64 | n=33554432 | 59.880 ms | 1.12 | 13.45 | 26% | ok |
| blas1 | axpy | f32 | n=33554432 | 59.752 ms | 1.12 | 6.74 | 13% | ok |
| blas1 | scale | f64 | n=33554432 | 10.674 ms | 3.14 | 50.30 | 97% | ok |
| blas1 | two_norm | f64 | n=33554432 | 55.524 ms | 1.21 | 4.83 | 9% | ok |
| blas1 | one_norm | f64 | n=33554432 | 25.761 ms | 1.30 | 10.42 | 20% | ok |
| blas1 | infinity_norm | f64 | n=33554432 | 25.663 ms | 1.31 | 10.46 | 20% | ok |
| blas1 | sum | f64 | n=33554432 | 25.607 ms | 1.31 | 10.48 | 20% | ok |
| blas1 | dot_k3 | f64 | n=33554432 | 989.939 ms | 0.75 | 0.54 | 1% | ok |
| blas2 | gemv | f64 | n=8192 | 14.728 ms | 9.11 | 36.46 | 70% | ok |
| blas2 | gemv | f32 | n=8192 | 13.045 ms | 10.29 | 20.58 | 40% | ok |
| blas2 | gemv_colmajor | f64 | n=8192 | 27.370 ms | 4.90 | 19.62 | 38% | ok |
| blas2 | gemv_colmajor | f32 | n=8192 | 13.568 ms | 9.89 | 19.79 | 38% | ok |
| blas2 | symv | f64 | n=8192 | 13.327 ms | 10.07 | 20.15 | 39% | ok |
| blas2 | trmv | f64 | n=8192 | 7.449 ms | 9.01 | 36.06 | 70% | ok |
| blas2 | trmv | f32 | n=8192 | 6.923 ms | 9.69 | 19.40 | 37% | ok |
| blas2 | trsv | f64 | n=8192 | 7.557 ms | 8.88 | 35.54 | 69% | ok |
| blas2 | trsv | f32 | n=8192 | 6.919 ms | 9.70 | 19.41 | 37% | ok |
| blas2 | ger | f64 | n=8192 | 65.691 ms | 2.04 | 16.35 | 32% | ok |
| blas3 | gemm | f64 | 1024x1024x1024 | 151.164 ms | 14.21 | 0.17 | 21% | ok |
| blas3 | gemm | f32 | 1024x1024x1024 | 77.075 ms | 27.86 | 0.16 | 22% | ok |
| blas3 | gemmt | f64 | 1024x1024x1024 | 98.057 ms | 10.95 | 0.26 | 16% | ok |
| blas3 | symm | f64 | 1024x1024x1024 | 153.117 ms | 14.03 | 0.16 | 21% | ok |
| blas3 | syrk | f64 | 1024x1024x1024 | 108.798 ms | 9.87 | 0.15 | 15% | ok |
| blas3 | syr2k | f64 | 1024x1024x1024 | 264.554 ms | 8.12 | 0.10 | 12% | ok |
| blas3 | trmm | f64 | 1024x1024x1024 | 242.418 ms | 4.43 | 0.09 | 7% | ok |
| blas3 | trsm | f64 | 1024x1024x1024 | 277.786 ms | 3.87 | 0.08 | 6% | ok |
| blas3 | trsm | f32 | 1024x1024x1024 | 254.396 ms | 4.22 | 0.04 | 3% | ok |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 34.824 ms | 0.48 | 3.85 | 7% | ok |
| factor | lu | f64 | 1024x1024 | 313.960 ms | 2.28 | 0.05 | 3% | ok |
| factor | lu | f32 | 1024x1024 | 307.437 ms | 2.33 | 0.03 | 2% | ok |
| factor | cholesky | f64 | 1024x1024 | 126.019 ms | 2.84 | 0.13 | 4% | ok |
| factor | cholesky | f32 | 1024x1024 | 124.382 ms | 2.88 | 0.07 | 2% | ok |
| factor | ldlt | f64 | 1024x1024 | 126.067 ms | 2.84 | 0.13 | 4% | ok |
| factor | ldlt_bk | f64 | 1024x1024 | 93.002 ms | 3.85 | 0.18 | 6% | ok |
| factor | qr | f64 | 1024x1024 | 5892.150 ms | 0.24 | 0.00 | 0% | ok |
| factor | lq | f64 | 1024x1024 | 655.789 ms | 2.18 | 0.03 | 3% | ok |
| factor | hessenberg | f64 | 1024x1024 | 6951.913 ms | 0.51 | 0.00 | 1% | ok |
| factor | householder | f64 | 1024x1024 | 17.420 ms | 0.24 | 0.96 | 2%\* | ok |
| factor | inv | f64 | 1024x1024 | 855.962 ms | 2.51 | 0.02 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 327.478 ms | 2.19 | 0.05 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 950.277 ms | 0.19 | 0.00 | 0% | ok |
| eigen | eigen_symmetric | f64 | 512x512 | 3703.416 ms | 0.33 | 0.00 | 0% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 18167.491 ms | 0.07 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 512x512 | 1633.915 ms | 0.82 | 0.00 | 1% | ok |
| eigen | schur | f64 | 512x512 | 2967.929 ms | 1.13 | 0.00 | 2% | ok |
| eigen | svd | f64 | 512x512 | 1370.959 ms | 2.15 | 0.00 | 3% | ok |
| eigen | expm | f64 | 512x512 | 325.448 ms | 6.19 | 0.01 | 9% | ok |
| eigen | sylvester | f64 | 512x512 | 6825.553 ms | 1.18 | 0.00 | 2% | ok |
| elementwise | exp | f64 | n=33554432 | 133.125 ms | 0.25 | 4.03 | 8% | ok |
| elementwise | log | f64 | n=33554432 | 205.901 ms | 0.16 | 2.61 | 5% | ok |
| elementwise | sin | f64 | n=33554432 | 195.471 ms | 0.17 | 2.75 | 5% | ok |
| elementwise | sqrt | f64 | n=33554432 | 105.325 ms | 0.32 | 5.10 | 10% | ok |
| elementwise | erf | f64 | n=33554432 | 155.630 ms | 0.22 | 3.45 | 7% | ok |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 4.004 ms | 2.62 | 27.22 | 53% | ok |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 6.716 ms | 1.56 | 16.23 | 31% | ok |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 31.621 ms | 5.30 | 11.14 | 22% | ok |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 100.557 ms | 0.13 | 0.46 | 1% | ok |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 19.579 ms | 0.52 | 4.28 | 8% | ok |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 169.571 ms | 0.63 | 0.02 | 1% | ok |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 300.580 ms | 0.36 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 107.297 ms | 1.00 | 0.03 | 1% | ok |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 525.473 ms | 0.41 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 449.355 ms | 0.48 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 218.013 ms | 0.98 | 0.01 | 1% | ok |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 622.428 ms | 0.04 | 0.00 | 0% | ok |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 22.781 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 32.619 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 348.607 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 1.377 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 109.835 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 19.203 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 149.825 ms | 0.71 | 0.02 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 4.093 ms | 0.98 | 3.92 | 8%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 42.400 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 49.674 ms | 2.15 | 0.06 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 4.128 ms | 0.97 | 3.89 | 8%\* | ok |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 18.431 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 486.501 ms | 0.44 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.297 ms | 0.26 | 1.05 | 2%\* | ok |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 29.252 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 398.649 ms | 0.54 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.280 ms | 0.26 | 1.05 | 2%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 15.656 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1057.642 ms | 0.92 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 7.979 ms | 0.94 | 3.76 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 38.357 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 251.805 ms | 3.85 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 8.065 ms | 0.93 | 3.72 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 14.146 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 3636.512 ms | 0.53 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.389 ms | 0.25 | 0.99 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 19.786 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1875.749 ms | 1.03 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.562 ms | 0.25 | 0.98 | 2% | ok |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 401.861 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 616.459 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 246.315 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 195.941 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 222.562 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 169.675 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 666.928 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 311.500 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 205.343 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 250.607 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 226.834 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 248.030 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 311.798 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.225 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 256^2 grid | 13.708 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 256^2 grid | 126.172 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 256^2 grid | 10.247 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 256^2 grid | 9.651 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 256^2 grid | 0.144 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 3.472 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 7.126 ms | 1.47 | 16.47 | 32% | ok |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 7.833 ms | 1.34 | 14.98 | 29% | ok |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 9.122 ms | 1.15 | 12.87 | 25% | ok |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 98.618 ms | 1.58 | 11.85 | 23% | ok |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 187.658 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 187.423 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 820.974 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 39.864 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=33554432 | 57.537 ms | 1.17 | 4.67 | 9% | ok |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 293.629 ms | 7.31 | 0.06 | 11% | ok |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 7.425 ms | 9.04 | 9.60 | 19% | ok |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 16.298 ms | 4.12 | 16.47 | n/a | ok |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.988 ms | 5.17 | 10.33 | n/a | ok |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 6.932 ms | 9.68 | 9.68 | n/a | ok |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 6.933 ms | 9.68 | 9.68 | n/a | ok |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 76.594 ms | 28.04 | 0.16 | n/a | ok |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 305.306 ms | 7.03 | 0.03 | n/a | ok |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 274.602 ms | 7.82 | 0.02 | n/a | ok |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 274.499 ms | 7.82 | 0.02 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 276.008 ms | 7.78 | 0.02 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

