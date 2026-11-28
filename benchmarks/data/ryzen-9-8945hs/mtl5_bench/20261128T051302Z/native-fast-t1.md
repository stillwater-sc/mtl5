## ryzen-9-8945hs -- native-fast, 1 thread, class large

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 63.3 GFLOP/s, f32 126.6 GFLOP/s |
| ... hardware, AVX-512F | f64 63.3, f32 126.6 GFLOP/s |
| ... this build, SSE2 | f64 40.7, f32 81.5 GFLOP/s |
| bandwidth | read 45.6, update 51.9, triad 37.8 GB/s (ceiling 51.9) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 105% | 110% | 0 |
| blas2 | 10 | 67% | 74% | 0 |
| blas3 | 10 | 27% | 32% | 0 |
| factor | 12 | 3% | 6% | 0 |
| eigen | 8 | 2% | 11% | 0 |
| elementwise | 5 | 7% | 10% | 0 |
| sparse | 5 | 21% | 53% | 0 |
| direct | 19 | 1% | 8% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 7% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 29% | 35% | 0 |
| mg | 1 | 23% | 23% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 57% | 59% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=33554432 | 9.873 ms | 6.80 | 54.38 | 105% | ok |
| blas1 | dot | f32 | n=33554432 | 4.985 ms | 13.46 | 53.85 | 104% | ok |
| blas1 | axpy | f64 | n=33554432 | 14.609 ms | 4.59 | 55.12 | 106% | ok |
| blas1 | axpy | f32 | n=33554432 | 7.321 ms | 9.17 | 55.00 | 106% | ok |
| blas1 | scale | f64 | n=33554432 | 9.812 ms | 3.42 | 54.72 | 105% | ok |
| blas1 | two_norm | f64 | n=33554432 | 4.704 ms | 14.27 | 57.06 | 110% | ok |
| blas1 | one_norm | f64 | n=33554432 | 25.609 ms | 1.31 | 10.48 | 20% | ok |
| blas1 | infinity_norm | f64 | n=33554432 | 25.743 ms | 1.30 | 10.43 | 20% | ok |
| blas1 | sum | f64 | n=33554432 | 25.603 ms | 1.31 | 10.48 | 20% | ok |
| blas1 | dot_k3 | f64 | n=33554432 | 994.787 ms | 0.74 | 0.54 | 1% | ok |
| blas2 | gemv | f64 | n=8192 | 14.456 ms | 9.28 | 37.15 | 72% | ok |
| blas2 | gemv | f32 | n=8192 | 6.942 ms | 19.34 | 38.68 | 74% | ok |
| blas2 | gemv_colmajor | f64 | n=8192 | 15.524 ms | 8.65 | 34.59 | 67% | ok |
| blas2 | gemv_colmajor | f32 | n=8192 | 8.071 ms | 16.63 | 33.27 | 64% | ok |
| blas2 | symv | f64 | n=8192 | 15.564 ms | 8.62 | 17.26 | 33% | ok |
| blas2 | trmv | f64 | n=8192 | 7.408 ms | 9.06 | 36.26 | 70% | ok |
| blas2 | trmv | f32 | n=8192 | 4.137 ms | 16.22 | 32.46 | 63% | ok |
| blas2 | trsv | f64 | n=8192 | 7.423 ms | 9.04 | 36.18 | 70% | ok |
| blas2 | trsv | f32 | n=8192 | 4.011 ms | 16.73 | 33.48 | 64% | ok |
| blas2 | ger | f64 | n=8192 | 65.230 ms | 2.06 | 16.46 | 32% | ok |
| blas3 | gemm | f64 | 1024x1024x1024 | 107.327 ms | 20.01 | 0.23 | 32% | ok |
| blas3 | gemm | f32 | 1024x1024x1024 | 55.328 ms | 38.81 | 0.23 | 31% | ok |
| blas3 | gemmt | f64 | 1024x1024x1024 | 64.803 ms | 16.57 | 0.39 | 26% | ok |
| blas3 | symm | f64 | 1024x1024x1024 | 113.251 ms | 18.96 | 0.22 | 30% | ok |
| blas3 | syrk | f64 | 1024x1024x1024 | 63.065 ms | 17.03 | 0.27 | 27% | ok |
| blas3 | syr2k | f64 | 1024x1024x1024 | 123.364 ms | 17.41 | 0.20 | 27% | ok |
| blas3 | trmm | f64 | 1024x1024x1024 | 82.670 ms | 12.99 | 0.25 | 21% | ok |
| blas3 | trsm | f64 | 1024x1024x1024 | 83.128 ms | 12.92 | 0.25 | 20% | ok |
| blas3 | trsm | f32 | 1024x1024x1024 | 49.261 ms | 21.80 | 0.21 | 17% | ok |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 34.989 ms | 0.48 | 3.84 | 7% | ok |
| factor | lu | f64 | 1024x1024 | 316.382 ms | 2.26 | 0.05 | 4% | ok |
| factor | lu | f32 | 1024x1024 | 309.647 ms | 2.31 | 0.03 | 2% | ok |
| factor | cholesky | f64 | 1024x1024 | 125.710 ms | 2.85 | 0.13 | 4% | ok |
| factor | cholesky | f32 | 1024x1024 | 124.902 ms | 2.87 | 0.07 | 2% | ok |
| factor | ldlt | f64 | 1024x1024 | 126.096 ms | 2.84 | 0.13 | 4% | ok |
| factor | ldlt_bk | f64 | 1024x1024 | 92.434 ms | 3.87 | 0.18 | 6% | ok |
| factor | qr | f64 | 1024x1024 | 5912.673 ms | 0.24 | 0.00 | 0% | ok |
| factor | lq | f64 | 1024x1024 | 656.107 ms | 2.18 | 0.03 | 3% | ok |
| factor | hessenberg | f64 | 1024x1024 | 6894.227 ms | 0.52 | 0.00 | 1% | ok |
| factor | householder | f64 | 1024x1024 | 17.497 ms | 0.24 | 0.96 | 2%\* | ok |
| factor | inv | f64 | 1024x1024 | 852.508 ms | 2.52 | 0.02 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 323.579 ms | 2.21 | 0.05 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 944.332 ms | 0.19 | 0.00 | 0% | ok |
| eigen | eigen_symmetric | f64 | 512x512 | 3721.782 ms | 0.32 | 0.00 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 18229.194 ms | 0.07 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 512x512 | 1641.692 ms | 0.82 | 0.00 | 1% | ok |
| eigen | schur | f64 | 512x512 | 3038.953 ms | 1.10 | 0.00 | 2% | ok |
| eigen | svd | f64 | 512x512 | 1374.185 ms | 2.15 | 0.00 | 3% | ok |
| eigen | expm | f64 | 512x512 | 280.324 ms | 7.18 | 0.01 | 11% | ok |
| eigen | sylvester | f64 | 512x512 | 6752.941 ms | 1.19 | 0.00 | 2% | ok |
| elementwise | exp | f64 | n=33554432 | 137.228 ms | 0.24 | 3.91 | 8% | ok |
| elementwise | log | f64 | n=33554432 | 206.277 ms | 0.16 | 2.60 | 5% | ok |
| elementwise | sin | f64 | n=33554432 | 198.095 ms | 0.17 | 2.71 | 5% | ok |
| elementwise | sqrt | f64 | n=33554432 | 104.804 ms | 0.32 | 5.12 | 10% | ok |
| elementwise | erf | f64 | n=33554432 | 155.330 ms | 0.22 | 3.46 | 7% | ok |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 3.982 ms | 2.63 | 27.37 | 53% | ok |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 6.540 ms | 1.60 | 16.66 | 32% | ok |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 31.683 ms | 5.29 | 11.12 | 21% | ok |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 99.060 ms | 0.13 | 0.47 | 1% | ok |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 19.573 ms | 0.52 | 4.28 | 8% | ok |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 172.243 ms | 0.62 | 0.02 | 1% | ok |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 277.962 ms | 0.38 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 100.622 ms | 1.06 | 0.03 | 2% | ok |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 539.872 ms | 0.40 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 448.142 ms | 0.48 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 214.704 ms | 1.00 | 0.01 | 2% | ok |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 615.492 ms | 0.04 | 0.00 | 0% | ok |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 21.737 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 32.411 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 345.163 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 1.349 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 106.419 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 17.338 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 143.175 ms | 0.75 | 0.02 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 4.132 ms | 0.97 | 3.89 | 7%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 40.893 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 49.788 ms | 2.15 | 0.06 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 4.123 ms | 0.97 | 3.89 | 8%\* | ok |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 18.417 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 483.628 ms | 0.44 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.295 ms | 0.26 | 1.05 | 2%\* | ok |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 27.931 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 399.865 ms | 0.54 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.326 ms | 0.26 | 1.05 | 2%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 12.611 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1120.603 ms | 0.86 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 7.851 ms | 0.96 | 3.82 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 35.457 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 251.759 ms | 3.85 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 8.022 ms | 0.93 | 3.74 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 14.000 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 3580.943 ms | 0.54 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.454 ms | 0.25 | 0.98 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 19.247 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1879.154 ms | 1.03 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.431 ms | 0.25 | 0.99 | 2% | ok |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 326.123 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 456.093 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 219.413 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 155.997 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 178.806 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 152.060 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 416.151 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 264.785 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 178.609 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 175.707 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 156.398 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 168.432 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 201.729 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.225 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 256^2 grid | 13.497 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 256^2 grid | 126.894 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 256^2 grid | 10.295 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 256^2 grid | 9.694 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 256^2 grid | 0.146 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 3.578 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 6.456 ms | 1.62 | 18.18 | 35% | ok |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 7.867 ms | 1.33 | 14.92 | 29% | ok |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 9.169 ms | 1.14 | 12.80 | 25% | ok |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 99.210 ms | 1.57 | 11.78 | 23% | ok |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 187.151 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 187.870 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 824.997 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 32.522 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=33554432 | 8.712 ms | 7.70 | 30.81 | 59% | ok |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 151.377 ms | 14.19 | 0.11 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 2.426 ms | 27.66 | 29.39 | 57% | ok |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 5.649 ms | 11.88 | 47.52 | n/a | ok |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 2.462 ms | 27.26 | 54.52 | n/a | ok |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 1.790 ms | 37.48 | 37.48 | n/a | ok |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 1.634 ms | 41.08 | 41.08 | n/a | ok |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 148.185 ms | 14.49 | 0.08 | n/a | ok |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 173.516 ms | 12.38 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 177.314 ms | 12.11 | 0.04 | n/a | ok |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 35.430 ms | 60.61 | 0.18 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 35.248 ms | 60.93 | 0.18 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

