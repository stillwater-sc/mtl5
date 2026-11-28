# mtl5_bench platform characterization

Class `large`, variants native native-fast blas, threads 1,8, CPUs 0,2,4,6,8,10,12,14, clock policy power scheme copy of Balanced with processor state 100-100% and EPP 0, boost aggressive; AC, 2026-11-28T04:27:06Z.

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

## ryzen-9-8945hs -- native, 8 threads, class large

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 505.9 GFLOP/s, f32 1012.4 GFLOP/s |
| ... hardware, AVX-512F | f64 505.9, f32 1012.4 GFLOP/s |
| ... this build, scalar | f64 12.5, f32 14.5 GFLOP/s |
| bandwidth | read 52.4, update 52.9, triad 40.9 GB/s (ceiling 52.9) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 82% | 109% | 0 | 51% |
| blas2 | 10 | 67% | 109% | 0 | 12% |
| blas3 | 10 | 2% | 21% | 0 | 13% |
| factor | 12 | 1% | 14% | 0 | 55% |
| eigen | 8 | 0% | 2% | 0 | 16% |
| elementwise | 5 | 7% | 10% | 0 | 13% |
| sparse | 5 | 29% | 102% | 0 | 13% |
| direct | 19 | 0% | 7% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 7% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 28% | 31% | 0 | 12% |
| mg | 1 | 24% | 24% | 0 | 14% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 11% | 18% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 98% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 9.273 ms | 7.24 | 57.90 | 109% | ok | 6.2x (78%) |
| blas1 | dot | f32 | n=33554432 | 6.163 ms | 10.89 | 43.56 | 82% | ok | 8.0x (100%) |
| blas1 | axpy | f64 | n=33554432 | 14.635 ms | 4.59 | 55.02 | 104% | ok | 4.1x (51%) |
| blas1 | axpy | f32 | n=33554432 | 7.610 ms | 8.82 | 52.91 | 100% | ok | 7.9x (98%) |
| blas1 | scale | f64 | n=33554432 | 9.464 ms | 3.55 | 56.73 | 107% | ok | 1.1x (14%) |
| blas1 | two_norm | f64 | n=33554432 | 6.967 ms | 9.63 | 38.53 | 73% | ok | 8.0x (100%) |
| blas1 | one_norm | f64 | n=33554432 | 25.598 ms | 1.31 | 10.49 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=33554432 | 25.656 ms | 1.31 | 10.46 | 20% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=33554432 | 25.602 ms | 1.31 | 10.49 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=33554432 | 994.795 ms | 0.74 | 0.54 | 1% | ok | 1.0x (12%) |
| blas2 | gemv | f64 | n=8192 | 9.800 ms | 13.69 | 54.79 | 104% | ok | 1.5x (19%) |
| blas2 | gemv | f32 | n=8192 | 4.756 ms | 28.22 | 56.45 | 107% | ok | 2.7x (34%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 9.274 ms | 14.47 | 57.90 | 109% | ok | 3.0x (37%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 4.661 ms | 28.80 | 57.61 | 109% | ok | 2.9x (36%) |
| blas2 | symv | f64 | n=8192 | 13.977 ms | 9.60 | 19.22 | 36% | ok | 1.0x (12%) |
| blas2 | trmv | f64 | n=8192 | 7.586 ms | 8.85 | 35.40 | 67% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=8192 | 7.179 ms | 9.35 | 18.71 | 35% | ok | 1.0x (12%) |
| blas2 | trsv | f64 | n=8192 | 7.644 ms | 8.78 | 35.13 | 66% | ok | 1.0x (12%) |
| blas2 | trsv | f32 | n=8192 | 7.201 ms | 9.32 | 18.65 | 35% | ok | 1.0x (12%) |
| blas2 | ger | f64 | n=8192 | 65.735 ms | 2.04 | 16.34 | 31% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 19.851 ms | 108.18 | 1.27 | 21% | ok | 7.6x (95%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 9.962 ms | 215.57 | 1.26 | 21% | ok | 7.7x (97%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 98.905 ms | 10.86 | 0.25 | 2% | ok | 1.0x (12%) |
| blas3 | symm | f64 | 1024x1024x1024 | 152.127 ms | 14.12 | 0.17 | 3% | ok | 1.0x (13%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 109.647 ms | 9.79 | 0.15 | 2% | ok | 1.0x (12%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 263.870 ms | 8.14 | 0.10 | 2% | ok | 1.0x (13%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 242.429 ms | 4.43 | 0.09 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 277.864 ms | 3.86 | 0.08 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 253.917 ms | 4.23 | 0.04 | 0% | ok | 1.0x (13%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 34.427 ms | 0.49 | 3.90 | 7% | ok | 1.0x (13%) |
| factor | lu | f64 | 1024x1024 | 71.781 ms | 9.97 | 0.23 | 2% | ok | 4.4x (55%) |
| factor | lu | f32 | 1024x1024 | 68.262 ms | 10.49 | 0.12 | 1% | ok | 4.5x (56%) |
| factor | cholesky | f64 | 1024x1024 | 55.862 ms | 6.41 | 0.30 | 1% | ok | 2.3x (28%) |
| factor | cholesky | f32 | 1024x1024 | 54.816 ms | 6.53 | 0.15 | 1% | ok | 2.3x (28%) |
| factor | ldlt | f64 | 1024x1024 | 126.400 ms | 2.83 | 0.13 | 1% | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 1024x1024 | 92.341 ms | 3.88 | 0.18 | 1% | ok | 1.0x (13%) |
| factor | qr | f64 | 1024x1024 | 1209.737 ms | 1.18 | 0.01 | 0% | ok | 4.9x (61%) |
| factor | lq | f64 | 1024x1024 | 137.039 ms | 10.45 | 0.12 | 2% | ok | 4.8x (60%) |
| factor | hessenberg | f64 | 1024x1024 | 1458.296 ms | 2.45 | 0.01 | 0% | ok | 4.8x (60%) |
| factor | householder | f64 | 1024x1024 | 2.302 ms | 1.82 | 7.29 | 14%\* | ok | 7.6x (95%) |
| factor | inv | f64 | 1024x1024 | 605.446 ms | 3.55 | 0.03 | 1% | ok | 1.4x (18%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 79.030 ms | 9.06 | 0.21 | 2% | ok | 4.1x (52%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 578.245 ms | 0.31 | 0.01 | 0% | ok | 1.6x (21%) |
| eigen | eigen_symmetric | f64 | 512x512 | 3342.855 ms | 0.36 | 0.00 | 0% | ok | 1.1x (14%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 18330.918 ms | 0.07 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 512x512 | 1269.731 ms | 1.06 | 0.00 | 0% | ok | 1.3x (16%) |
| eigen | schur | f64 | 512x512 | 2249.204 ms | 1.49 | 0.00 | 0% | ok | 1.3x (16%) |
| eigen | svd | f64 | 512x512 | 1370.421 ms | 2.15 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | expm | f64 | 512x512 | 177.151 ms | 11.36 | 0.02 | 2% | ok | 1.8x (23%) |
| eigen | sylvester | f64 | 512x512 | 5499.700 ms | 1.46 | 0.00 | 0% | ok | 1.2x (16%) |
| elementwise | exp | f64 | n=33554432 | 153.841 ms | 0.22 | 3.49 | 7% | ok | 0.9x (11%) |
| elementwise | log | f64 | n=33554432 | 206.553 ms | 0.16 | 2.60 | 5% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=33554432 | 194.711 ms | 0.17 | 2.76 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=33554432 | 104.593 ms | 0.32 | 5.13 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=33554432 | 154.882 ms | 0.22 | 3.47 | 7% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 2.014 ms | 5.20 | 54.10 | 102% | ok | 2.0x (25%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 7.194 ms | 1.46 | 15.15 | 29% | ok | 0.9x (12%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 9.054 ms | 18.52 | 38.91 | 74% | ok | 3.5x (44%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 96.819 ms | 0.13 | 0.48 | 1% | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 18.324 ms | 0.56 | 4.57 | 9% | ok | 1.1x (13%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 170.578 ms | 0.63 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 302.385 ms | 0.35 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 107.231 ms | 1.00 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 521.036 ms | 0.41 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 445.462 ms | 0.48 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 215.386 ms | 0.99 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 624.924 ms | 0.04 | 0.00 | 0% | ok | 1.0x (12%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 21.734 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 32.559 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 344.524 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 1.383 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 106.518 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 17.210 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 141.537 ms | 0.76 | 0.02 | 0% | ok | 1.1x (13%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 4.130 ms | 0.97 | 3.89 | 7%\* | ok | 1.0x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 40.791 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 49.769 ms | 2.15 | 0.06 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 4.204 ms | 0.95 | 3.82 | 7%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 18.203 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 484.267 ms | 0.44 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.349 ms | 0.26 | 1.05 | 2%\* | ok | 1.0x (12%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 28.004 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 398.844 ms | 0.54 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.306 ms | 0.26 | 1.05 | 2%\* | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 12.684 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1034.552 ms | 0.94 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 8.025 ms | 0.93 | 3.74 | 7% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 35.478 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 252.833 ms | 3.83 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 8.132 ms | 0.92 | 3.69 | 7% | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 14.058 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 3783.848 ms | 0.51 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.605 ms | 0.24 | 0.98 | 2% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 19.611 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1872.448 ms | 1.04 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.744 ms | 0.24 | 0.98 | 2% | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 377.900 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 557.108 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 239.036 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 178.291 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 202.583 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 152.876 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 646.812 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 301.465 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 186.273 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 231.626 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 216.281 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 239.182 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 304.408 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.224 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 256^2 grid | 13.582 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 256^2 grid | 125.426 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 256^2 grid | 10.401 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 256^2 grid | 9.732 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 256^2 grid | 0.146 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 3.594 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 7.151 ms | 1.47 | 16.41 | 31% | ok | 1.0x (12%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 7.824 ms | 1.34 | 15.00 | 28% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 9.241 ms | 1.13 | 12.70 | 24% | ok | 1.0x (12%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 91.019 ms | 1.71 | 12.84 | 24% | ok | 1.1x (14%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 189.332 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 189.007 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 821.857 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 39.919 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=33554432 | 57.562 ms | 1.17 | 4.66 | 9% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 37.635 ms | 57.06 | 0.45 | 11% | ok | 7.8x (98%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 7.368 ms | 9.11 | 9.68 | 18% | ok | 1.0x (13%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 16.256 ms | 4.13 | 16.51 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.988 ms | 5.17 | 10.33 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 6.978 ms | 9.62 | 9.62 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 6.934 ms | 9.68 | 9.68 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 9.776 ms | 219.67 | 1.29 | n/a | ok | 7.8x (98%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 39.080 ms | 54.95 | 0.21 | n/a | ok | 7.8x (98%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 35.069 ms | 61.24 | 0.18 | n/a | ok | 7.8x (98%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 35.070 ms | 61.23 | 0.18 | n/a | ok | 7.8x (98%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 35.296 ms | 60.84 | 0.18 | n/a | ok | 7.8x (98%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

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

## ryzen-9-8945hs -- blas, 1 thread, class large

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 78.4 GFLOP/s, f32 126.6 GFLOP/s |
| ... hardware, AVX-512F | f64 78.4, f32 126.6 GFLOP/s |
| ... this build, scalar | f64 1.9, f32 1.9 GFLOP/s |
| bandwidth | read 36.9, update 52.0, triad 37.8 GB/s (ceiling 52.0) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 20% | 96% | 0 |
| blas2 | 10 | 70% | 77% | 0 |
| blas3 | 10 | 74% | 91% | 0 |
| factor | 12 | 13% | 28% | 0 |
| eigen | 8 | 13% | 43% | 0 |
| elementwise | 5 | 6% | 10% | 0 |
| sparse | 5 | 22% | 50% | 0 |
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
| blas1 | dot | f64 | n=33554432 | 47.218 ms | 1.42 | 11.37 | 22% | ok |
| blas1 | dot | f32 | n=33554432 | 49.018 ms | 1.37 | 5.48 | 11% | ok |
| blas1 | axpy | f64 | n=33554432 | 59.909 ms | 1.12 | 13.44 | 26% | ok |
| blas1 | axpy | f32 | n=33554432 | 59.660 ms | 1.12 | 6.75 | 13% | ok |
| blas1 | scale | f64 | n=33554432 | 10.765 ms | 3.12 | 49.87 | 96% | ok |
| blas1 | two_norm | f64 | n=33554432 | 49.045 ms | 1.37 | 5.47 | 11% | ok |
| blas1 | one_norm | f64 | n=33554432 | 25.680 ms | 1.31 | 10.45 | 20% | ok |
| blas1 | infinity_norm | f64 | n=33554432 | 25.712 ms | 1.31 | 10.44 | 20% | ok |
| blas1 | sum | f64 | n=33554432 | 25.663 ms | 1.31 | 10.46 | 20% | ok |
| blas1 | dot_k3 | f64 | n=33554432 | 996.202 ms | 0.74 | 0.54 | 1% | ok |
| blas2 | gemv | f64 | n=8192 | 14.879 ms | 9.02 | 36.09 | 69% | ok |
| blas2 | gemv | f32 | n=8192 | 7.546 ms | 17.79 | 35.58 | 68% | ok |
| blas2 | gemv_colmajor | f64 | n=8192 | 16.059 ms | 8.36 | 33.44 | 64% | ok |
| blas2 | gemv_colmajor | f32 | n=8192 | 6.734 ms | 19.93 | 39.87 | 77% | ok |
| blas2 | symv | f64 | n=8192 | 7.674 ms | 17.49 | 35.00 | 67% | ok |
| blas2 | trmv | f64 | n=8192 | 7.346 ms | 9.14 | 36.56 | 70% | ok |
| blas2 | trmv | f32 | n=8192 | 4.095 ms | 16.39 | 32.79 | 63% | ok |
| blas2 | trsv | f64 | n=8192 | 7.032 ms | 9.54 | 38.19 | 73% | ok |
| blas2 | trsv | f32 | n=8192 | 3.566 ms | 18.82 | 37.65 | 72% | ok |
| blas2 | ger | f64 | n=8192 | 29.107 ms | 4.61 | 36.89 | 71% | ok |
| blas3 | gemm | f64 | 1024x1024x1024 | 36.248 ms | 59.24 | 0.69 | 76% | ok |
| blas3 | gemm | f32 | 1024x1024x1024 | 18.690 ms | 114.90 | 0.67 | 91% | ok |
| blas3 | gemmt | f64 | 1024x1024x1024 | 23.984 ms | 44.77 | 1.05 | 57% | ok |
| blas3 | symm | f64 | 1024x1024x1024 | 37.174 ms | 57.77 | 0.68 | 74% | ok |
| blas3 | syrk | f64 | 1024x1024x1024 | 24.379 ms | 44.04 | 0.69 | 56% | ok |
| blas3 | syr2k | f64 | 1024x1024x1024 | 42.688 ms | 50.31 | 0.59 | 64% | ok |
| blas3 | trmm | f64 | 1024x1024x1024 | 18.226 ms | 58.91 | 1.15 | 75% | ok |
| blas3 | trsm | f64 | 1024x1024x1024 | 23.254 ms | 46.18 | 0.90 | 59% | ok |
| blas3 | trsm | f32 | 1024x1024x1024 | 11.377 ms | 94.38 | 0.92 | 75% | ok |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 34.860 ms | 0.48 | 3.85 | 7% | ok |
| factor | lu | f64 | 1024x1024 | 32.164 ms | 22.26 | 0.52 | 28% | ok |
| factor | lu | f32 | 1024x1024 | 25.143 ms | 28.47 | 0.33 | 22% | ok |
| factor | cholesky | f64 | 1024x1024 | 27.110 ms | 13.20 | 0.62 | 17% | ok |
| factor | cholesky | f32 | 1024x1024 | 20.947 ms | 17.09 | 0.40 | 13% | ok |
| factor | ldlt | f64 | 1024x1024 | 127.872 ms | 2.80 | 0.13 | 4% | ok |
| factor | ldlt_bk | f64 | 1024x1024 | 91.949 ms | 3.89 | 0.18 | 5% | ok |
| factor | qr | f64 | 1024x1024 | 69.734 ms | 20.53 | 0.24 | 26% | ok |
| factor | lq | f64 | 1024x1024 | 657.763 ms | 2.18 | 0.03 | 3% | ok |
| factor | hessenberg | f64 | 1024x1024 | 6925.015 ms | 0.52 | 0.00 | 1% | ok |
| factor | householder | f64 | 1024x1024 | 17.568 ms | 0.24 | 0.96 | 2%\* | ok |
| factor | inv | f64 | 1024x1024 | 568.188 ms | 3.78 | 0.03 | 5% | ok |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 36.989 ms | 19.35 | 0.45 | 25% | ok |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 13.176 ms | 13.58 | 0.32 | 17% | ok |
| eigen | eigen_symmetric | f64 | 512x512 | 3696.633 ms | 0.33 | 0.00 | 0% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 18106.514 ms | 0.07 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 512x512 | 298.294 ms | 4.50 | 0.01 | 6% | ok |
| eigen | schur | f64 | 512x512 | 337.197 ms | 9.95 | 0.01 | 13% | ok |
| eigen | svd | f64 | 512x512 | 86.778 ms | 34.03 | 0.05 | 43% | ok |
| eigen | expm | f64 | 512x512 | 172.943 ms | 11.64 | 0.02 | 15% | ok |
| eigen | sylvester | f64 | 512x512 | 1553.904 ms | 5.18 | 0.00 | 7% | ok |
| elementwise | exp | f64 | n=33554432 | 133.379 ms | 0.25 | 4.03 | 8% | ok |
| elementwise | log | f64 | n=33554432 | 215.842 ms | 0.16 | 2.49 | 5% | ok |
| elementwise | sin | f64 | n=33554432 | 192.122 ms | 0.17 | 2.79 | 5% | ok |
| elementwise | sqrt | f64 | n=33554432 | 105.048 ms | 0.32 | 5.11 | 10% | ok |
| elementwise | erf | f64 | n=33554432 | 169.972 ms | 0.20 | 3.16 | 6% | ok |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 4.168 ms | 2.51 | 26.15 | 50% | ok |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 6.457 ms | 1.62 | 16.88 | 32% | ok |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 30.703 ms | 5.46 | 11.47 | 22% | ok |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 100.251 ms | 0.13 | 0.46 | 1% | ok |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 19.583 ms | 0.52 | 4.28 | 8% | ok |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 171.115 ms | 0.63 | 0.02 | 1% | ok |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 303.339 ms | 0.35 | 0.01 | 0% | ok |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 104.551 ms | 1.02 | 0.03 | 1% | ok |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 522.022 ms | 0.41 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 454.568 ms | 0.47 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 215.712 ms | 0.99 | 0.01 | 1% | ok |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 623.124 ms | 0.04 | 0.00 | 0% | ok |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 21.754 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 32.423 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 341.277 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 1.368 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 108.392 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 17.826 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 142.644 ms | 0.75 | 0.02 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 4.093 ms | 0.98 | 3.92 | 8%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 41.034 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 49.843 ms | 2.15 | 0.06 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 4.148 ms | 0.97 | 3.87 | 7%\* | ok |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 18.763 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 488.982 ms | 0.44 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.314 ms | 0.26 | 1.05 | 2%\* | ok |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 28.107 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 409.527 ms | 0.52 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.404 ms | 0.26 | 1.04 | 2%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 12.709 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1130.666 ms | 0.86 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 7.934 ms | 0.95 | 3.78 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 35.129 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 253.163 ms | 3.83 | 0.01 | 5% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 7.857 ms | 0.95 | 3.82 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 14.115 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 3645.126 ms | 0.53 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.511 ms | 0.25 | 0.98 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 19.387 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1950.247 ms | 0.99 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.360 ms | 0.25 | 0.99 | 2% | ok |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 388.392 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 574.978 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 240.858 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 189.210 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 214.234 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 167.103 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 616.813 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 306.172 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 201.467 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 239.536 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 216.702 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 232.920 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 289.753 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.214 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 256^2 grid | 13.478 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 256^2 grid | 124.763 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 256^2 grid | 10.244 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 256^2 grid | 9.633 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 256^2 grid | 0.155 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 7.346 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 7.058 ms | 1.48 | 16.63 | 32% | ok |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 7.693 ms | 1.36 | 15.26 | 29% | ok |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 9.115 ms | 1.15 | 12.88 | 25% | ok |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 100.645 ms | 1.55 | 11.61 | 22% | ok |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 187.958 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 189.142 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 820.716 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 41.788 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=33554432 | 49.020 ms | 1.37 | 5.48 | 11% | ok |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 298.905 ms | 7.18 | 0.06 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 7.408 ms | 9.06 | 9.63 | 19% | ok |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 16.251 ms | 4.13 | 16.52 | n/a | ok |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 13.351 ms | 5.03 | 10.05 | n/a | ok |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 8.540 ms | 7.86 | 7.86 | n/a | ok |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 8.540 ms | 7.86 | 7.86 | n/a | ok |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 78.772 ms | 27.26 | 0.16 | n/a | ok |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 305.520 ms | 7.03 | 0.03 | n/a | ok |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 274.692 ms | 7.82 | 0.02 | n/a | ok |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 274.764 ms | 7.82 | 0.02 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 275.789 ms | 7.79 | 0.02 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

## ryzen-9-8945hs -- blas, 8 threads, class large

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 556.1 GFLOP/s, f32 1011.6 GFLOP/s |
| ... hardware, AVX-512F | f64 556.1, f32 1011.6 GFLOP/s |
| ... this build, scalar | f64 14.6, f32 14.6 GFLOP/s |
| bandwidth | read 52.7, update 53.1, triad 40.9 GB/s (ceiling 53.1) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 82% | 110% | 0 | 51% |
| blas2 | 10 | 100% | 104% | 0 | 18% |
| blas3 | 10 | 58% | 85% | 0 | 72% |
| factor | 12 | 2% | 13% | 0 | 13% |
| eigen | 8 | 1% | 6% | 0 | 13% |
| elementwise | 5 | 6% | 10% | 0 | 13% |
| sparse | 5 | 32% | 105% | 0 | 13% |
| direct | 19 | 0% | 7% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 7% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 29% | 31% | 0 | 13% |
| mg | 1 | 24% | 24% | 0 | 14% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 10% | 18% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 97% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 9.154 ms | 7.33 | 58.65 | 110% | ok | 5.2x (64%) |
| blas1 | dot | f32 | n=33554432 | 6.148 ms | 10.92 | 43.66 | 82% | ok | 8.0x (100%) |
| blas1 | axpy | f64 | n=33554432 | 14.703 ms | 4.56 | 54.77 | 103% | ok | 4.1x (51%) |
| blas1 | axpy | f32 | n=33554432 | 7.596 ms | 8.83 | 53.01 | 100% | ok | 7.9x (98%) |
| blas1 | scale | f64 | n=33554432 | 9.390 ms | 3.57 | 57.17 | 108% | ok | 1.1x (14%) |
| blas1 | two_norm | f64 | n=33554432 | 6.148 ms | 10.92 | 43.66 | 82% | ok | 8.0x (100%) |
| blas1 | one_norm | f64 | n=33554432 | 25.700 ms | 1.31 | 10.44 | 20% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=33554432 | 25.647 ms | 1.31 | 10.47 | 20% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=33554432 | 25.597 ms | 1.31 | 10.49 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=33554432 | 995.021 ms | 0.74 | 0.54 | 1% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=8192 | 10.163 ms | 13.21 | 52.84 | 100% | ok | 1.5x (18%) |
| blas2 | gemv | f32 | n=8192 | 5.119 ms | 26.22 | 52.45 | 99% | ok | 1.5x (18%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 9.720 ms | 13.81 | 55.24 | 104% | ok | 1.7x (21%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 4.973 ms | 26.99 | 53.99 | 102% | ok | 1.4x (17%) |
| blas2 | symv | f64 | n=8192 | 5.037 ms | 26.65 | 53.32 | 100% | ok | 1.5x (19%) |
| blas2 | trmv | f64 | n=8192 | 4.964 ms | 13.52 | 54.10 | 102% | ok | 1.5x (18%) |
| blas2 | trmv | f32 | n=8192 | 2.591 ms | 25.90 | 51.83 | 98% | ok | 1.6x (20%) |
| blas2 | trsv | f64 | n=8192 | 6.671 ms | 10.06 | 40.26 | 76% | ok | 1.1x (13%) |
| blas2 | trsv | f32 | n=8192 | 3.885 ms | 17.27 | 34.56 | 65% | ok | 0.9x (11%) |
| blas2 | ger | f64 | n=8192 | 19.540 ms | 6.87 | 54.96 | 104% | ok | 1.5x (19%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 4.869 ms | 441.04 | 5.17 | 79% | ok | 7.4x (93%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 2.500 ms | 858.96 | 5.03 | 85% | ok | 7.5x (93%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 7.216 ms | 148.81 | 3.49 | 27% | ok | 3.3x (42%) |
| blas3 | symm | f64 | 1024x1024x1024 | 5.038 ms | 426.23 | 4.99 | 77% | ok | 7.4x (92%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 8.616 ms | 124.62 | 1.95 | 22% | ok | 2.8x (35%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 12.727 ms | 168.73 | 1.98 | 30% | ok | 3.4x (42%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 3.328 ms | 322.68 | 6.30 | 58% | ok | 5.5x (68%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 4.042 ms | 265.65 | 5.19 | 48% | ok | 5.8x (72%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 1.701 ms | 631.20 | 6.16 | 62% | ok | 6.7x (84%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 34.350 ms | 0.49 | 3.91 | 7% | ok | 1.0x (13%) |
| factor | lu | f64 | 1024x1024 | 29.110 ms | 24.59 | 0.58 | 4% | ok | 1.1x (14%) |
| factor | lu | f32 | 1024x1024 | 24.994 ms | 28.64 | 0.34 | 3% | ok | 1.0x (13%) |
| factor | cholesky | f64 | 1024x1024 | 26.154 ms | 13.68 | 0.64 | 2% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 1024x1024 | 21.669 ms | 16.52 | 0.39 | 2% | ok | 1.0x (12%) |
| factor | ldlt | f64 | 1024x1024 | 127.978 ms | 2.80 | 0.13 | 1% | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 1024x1024 | 92.847 ms | 3.85 | 0.18 | 1% | ok | 1.0x (12%) |
| factor | qr | f64 | 1024x1024 | 56.880 ms | 25.17 | 0.29 | 5% | ok | 1.2x (15%) |
| factor | lq | f64 | 1024x1024 | 135.916 ms | 10.53 | 0.12 | 2% | ok | 4.8x (60%) |
| factor | hessenberg | f64 | 1024x1024 | 1385.629 ms | 2.58 | 0.01 | 0% | ok | 5.0x (62%) |
| factor | householder | f64 | 1024x1024 | 2.354 ms | 1.78 | 7.13 | 13%\* | ok | 7.5x (93%) |
| factor | inv | f64 | 1024x1024 | 578.179 ms | 3.71 | 0.03 | 1% | ok | 1.0x (12%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 44.504 ms | 16.08 | 0.38 | 3% | ok | 0.8x (10%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 22.794 ms | 7.85 | 0.18 | 1% | ok | 0.6x (7%) |
| eigen | eigen_symmetric | f64 | 512x512 | 3275.852 ms | 0.37 | 0.00 | 0% | ok | 1.1x (14%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 18065.556 ms | 0.07 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 512x512 | 311.652 ms | 4.31 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | schur | f64 | 512x512 | 368.352 ms | 9.11 | 0.01 | 2% | ok | 0.9x (11%) |
| eigen | svd | f64 | 512x512 | 84.663 ms | 34.88 | 0.05 | 6% | ok | 1.0x (13%) |
| eigen | expm | f64 | 512x512 | 148.898 ms | 13.52 | 0.03 | 2% | ok | 1.2x (15%) |
| eigen | sylvester | f64 | 512x512 | 1578.997 ms | 5.10 | 0.00 | 1% | ok | 1.0x (12%) |
| elementwise | exp | f64 | n=33554432 | 133.486 ms | 0.25 | 4.02 | 8% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=33554432 | 212.330 ms | 0.16 | 2.53 | 5% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=33554432 | 191.755 ms | 0.17 | 2.80 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=33554432 | 105.031 ms | 0.32 | 5.11 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=33554432 | 169.890 ms | 0.20 | 3.16 | 6% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 1.964 ms | 5.33 | 55.48 | 105% | ok | 2.1x (27%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 6.506 ms | 1.61 | 16.75 | 32% | ok | 1.0x (12%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 8.963 ms | 18.70 | 39.30 | 74% | ok | 3.4x (43%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 97.702 ms | 0.13 | 0.47 | 1% | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 18.301 ms | 0.56 | 4.58 | 9% | ok | 1.1x (13%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 172.512 ms | 0.62 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 301.842 ms | 0.35 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 107.750 ms | 0.99 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 519.407 ms | 0.41 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 453.276 ms | 0.47 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 217.428 ms | 0.98 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 627.905 ms | 0.04 | 0.00 | 0% | ok | 1.0x (12%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 22.533 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 32.543 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 343.664 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 1.376 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 108.343 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 17.473 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 142.297 ms | 0.75 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 4.304 ms | 0.93 | 3.73 | 7%\* | ok | 1.0x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 41.270 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 49.819 ms | 2.15 | 0.06 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 4.244 ms | 0.95 | 3.78 | 7%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 18.542 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 487.956 ms | 0.44 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.322 ms | 0.26 | 1.05 | 2%\* | ok | 1.0x (12%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 27.986 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 398.826 ms | 0.54 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.335 ms | 0.26 | 1.05 | 2%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 12.805 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1058.097 ms | 0.92 | 0.00 | 0% | ok | 1.1x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 7.836 ms | 0.96 | 3.83 | 7% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 35.350 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 252.883 ms | 3.83 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 7.937 ms | 0.94 | 3.78 | 7% | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 14.077 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 3646.052 ms | 0.53 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.698 ms | 0.24 | 0.98 | 2% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 19.787 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1939.819 ms | 1.00 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.524 ms | 0.25 | 0.98 | 2% | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 360.498 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 522.315 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 235.177 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 171.202 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 194.714 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 149.967 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 595.118 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 293.833 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 181.838 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 219.529 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 201.205 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 224.594 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 281.965 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.212 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 256^2 grid | 13.914 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 256^2 grid | 126.734 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 256^2 grid | 10.652 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 256^2 grid | 9.911 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 256^2 grid | 0.150 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 7.232 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 7.039 ms | 1.49 | 16.68 | 31% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 7.725 ms | 1.36 | 15.19 | 29% | ok | 1.0x (12%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 9.083 ms | 1.15 | 12.92 | 24% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 90.685 ms | 1.72 | 12.89 | 24% | ok | 1.1x (14%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 190.547 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 191.492 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 820.581 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 38.796 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| mixed | dot | f32>f64 | n=33554432 | 49.044 ms | 1.37 | 5.47 | 10% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 38.079 ms | 56.40 | 0.44 | 10% | ok | 7.8x (98%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 7.413 ms | 9.05 | 9.62 | 18% | ok | 1.0x (12%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 16.268 ms | 4.13 | 16.50 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.992 ms | 5.17 | 10.33 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 8.539 ms | 7.86 | 7.86 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 8.523 ms | 7.87 | 7.87 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 10.173 ms | 211.09 | 1.24 | n/a | ok | 7.7x (97%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 39.025 ms | 55.03 | 0.21 | n/a | ok | 7.8x (98%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 35.057 ms | 61.26 | 0.18 | n/a | ok | 7.8x (98%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 35.045 ms | 61.28 | 0.18 | n/a | ok | 7.8x (98%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 35.278 ms | 60.87 | 0.18 | n/a | ok | 7.8x (98%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

