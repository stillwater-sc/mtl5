## i7-12700k -- native-fast, 8 threads, class large

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 600.8 GFLOP/s, f32 1201.8 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.8, f32 1201.7 GFLOP/s |
| ... this build, AVX2 | f64 600.8, f32 1201.8 GFLOP/s |
| bandwidth | read 81.9, update 78.9, triad 76.9 GB/s (ceiling 81.9) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 89% | 108% | 0 | 22% |
| blas2 | 10 | 46% | 108% | 0 | 13% |
| blas3 | 10 | 36% | 80% | 0 | 47% |
| factor | 12 | 2% | 23% | 0 | 61% |
| eigen | 8 | 0% | 5% | 0 | 15% |
| elementwise | 5 | 4% | 5% | 0 | 13% |
| sparse | 5 | 32% | 113% | 0 | 14% |
| direct | 19 | 0% | 8% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 7% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 14% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 33% | 0 | 13% |
| mg | 1 | 23% | 23% | 0 | 14% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 41% | 54% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 88% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 6.380 ms | 10.52 | 84.15 | 103% | ok | 2.4x (30%) |
| blas1 | dot | f32 | n=33554432 | 3.040 ms | 22.07 | 88.30 | 108% | ok | 2.5x (31%) |
| blas1 | axpy | f64 | n=33554432 | 11.049 ms | 6.07 | 72.88 | 89% | ok | 1.8x (22%) |
| blas1 | axpy | f32 | n=33554432 | 5.455 ms | 12.30 | 73.82 | 90% | ok | 1.8x (22%) |
| blas1 | scale | f64 | n=33554432 | 7.408 ms | 4.53 | 72.47 | 88% | ok | 1.9x (24%) |
| blas1 | two_norm | f64 | n=33554432 | 3.053 ms | 21.98 | 87.93 | 107% | ok | 3.1x (39%) |
| blas1 | one_norm | f64 | n=33554432 | 16.338 ms | 2.05 | 16.43 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=33554432 | 27.801 ms | 1.21 | 9.66 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=33554432 | 16.084 ms | 2.09 | 16.69 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=33554432 | 294.866 ms | 2.50 | 1.82 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=8192 | 6.359 ms | 21.11 | 84.45 | 103% | ok | 2.2x (27%) |
| blas2 | gemv | f32 | n=8192 | 3.040 ms | 44.16 | 88.34 | 108% | ok | 2.3x (28%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 6.416 ms | 20.92 | 83.69 | 102% | ok | 2.2x (28%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 3.128 ms | 42.90 | 85.83 | 105% | ok | 2.3x (28%) |
| blas2 | symv | f64 | n=8192 | 12.124 ms | 11.07 | 22.15 | 27% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=8192 | 7.106 ms | 9.44 | 37.79 | 46% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=8192 | 3.606 ms | 18.61 | 37.24 | 45% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=8192 | 7.114 ms | 9.43 | 37.75 | 46% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=8192 | 3.639 ms | 18.44 | 36.90 | 45% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=8192 | 28.466 ms | 4.72 | 37.72 | 46% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 4.457 ms | 481.79 | 5.65 | 80% | ok | 6.8x (85%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 2.225 ms | 965.09 | 5.65 | 80% | ok | 6.6x (83%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 4.993 ms | 215.03 | 5.04 | 36% | ok | 3.8x (47%) |
| blas3 | symm | f64 | 1024x1024x1024 | 6.603 ms | 325.25 | 3.81 | 54% | ok | 5.0x (62%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 5.106 ms | 210.29 | 3.29 | 35% | ok | 3.4x (43%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 9.600 ms | 223.68 | 2.62 | 37% | ok | 3.7x (47%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 9.327 ms | 115.13 | 2.25 | 19% | ok | 2.3x (29%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 9.617 ms | 111.65 | 2.18 | 19% | ok | 2.3x (29%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 4.928 ms | 217.87 | 2.13 | 18% | ok | 2.1x (27%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 43.345 ms | 0.39 | 3.10 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 1024x1024 | 16.424 ms | 43.58 | 1.02 | 7% | ok | 6.1x (77%) |
| factor | lu | f32 | 1024x1024 | 11.543 ms | 62.01 | 0.73 | 5% | ok | 4.9x (61%) |
| factor | cholesky | f64 | 1024x1024 | 30.976 ms | 11.55 | 0.54 | 2% | ok | 2.2x (28%) |
| factor | cholesky | f32 | 1024x1024 | 29.392 ms | 12.18 | 0.29 | 1% | ok | 2.1x (26%) |
| factor | ldlt | f64 | 1024x1024 | 66.375 ms | 5.39 | 0.25 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 1024x1024 | 38.704 ms | 9.25 | 0.43 | 2% | ok | 1.1x (13%) |
| factor | qr | f64 | 1024x1024 | 433.847 ms | 3.30 | 0.04 | 1% | ok | 5.2x (65%) |
| factor | lq | f64 | 1024x1024 | 40.662 ms | 35.21 | 0.41 | 6% | ok | 5.2x (65%) |
| factor | hessenberg | f64 | 1024x1024 | 498.497 ms | 7.18 | 0.03 | 1% | ok | 5.1x (64%) |
| factor | householder | f64 | 1024x1024 | 0.893 ms | 4.70 | 18.79 | 23%\* | ok | 7.6x (95%) |
| factor | inv | f64 | 1024x1024 | 356.455 ms | 6.02 | 0.05 | 1% | ok | 1.2x (16%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 17.985 ms | 39.80 | 0.93 | 7% | ok | 3.4x (43%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 156.722 ms | 1.14 | 0.03 | 0% | ok | 1.7x (22%) |
| eigen | eigen_symmetric | f64 | 512x512 | 790.947 ms | 1.53 | 0.01 | 0% | ok | 1.2x (15%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 11838.384 ms | 0.10 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 512x512 | 324.653 ms | 4.13 | 0.01 | 1% | ok | 1.4x (17%) |
| eigen | schur | f64 | 512x512 | 1120.761 ms | 2.99 | 0.00 | 0% | ok | 1.2x (15%) |
| eigen | svd | f64 | 512x512 | 1153.151 ms | 2.56 | 0.00 | 0% | ok | 0.9x (12%) |
| eigen | expm | f64 | 512x512 | 72.063 ms | 27.94 | 0.06 | 5% | ok | 1.4x (18%) |
| eigen | sylvester | f64 | 512x512 | 2324.628 ms | 3.46 | 0.00 | 1% | ok | 1.2x (15%) |
| elementwise | exp | f64 | n=33554432 | 161.856 ms | 0.21 | 3.32 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=33554432 | 186.318 ms | 0.18 | 2.88 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=33554432 | 243.760 ms | 0.14 | 2.20 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=33554432 | 121.413 ms | 0.28 | 4.42 | 5% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=33554432 | 814.947 ms | 0.04 | 0.66 | 1% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 1.176 ms | 8.91 | 92.68 | 113% | ok | 3.0x (38%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 4.215 ms | 2.49 | 25.86 | 32% | ok | 1.0x (12%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 6.843 ms | 24.50 | 51.47 | 63% | ok | 2.5x (31%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 42.065 ms | 0.31 | 1.10 | 1% | ok | 0.7x (9%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 13.206 ms | 0.77 | 6.35 | 8% | ok | 1.1x (14%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 102.654 ms | 1.04 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 193.110 ms | 0.55 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 53.188 ms | 2.01 | 0.06 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 297.757 ms | 0.72 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 326.500 ms | 0.66 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 125.324 ms | 1.71 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 374.840 ms | 0.06 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 16.472 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 23.223 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 262.464 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 0.916 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 90.583 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 8.392 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 94.790 ms | 1.13 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 2.601 ms | 1.54 | 6.17 | 8%\* | ok | 0.9x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 17.938 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 32.319 ms | 3.31 | 0.09 | 1% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 2.658 ms | 1.51 | 6.04 | 7%\* | ok | 0.9x (12%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 8.098 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 273.105 ms | 0.78 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.750 ms | 0.46 | 1.84 | 2%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 13.066 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 301.279 ms | 0.71 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.780 ms | 0.46 | 1.83 | 2%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 8.789 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 584.498 ms | 1.66 | 0.00 | 0% | ok | 1.1x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 5.072 ms | 1.48 | 5.91 | 7% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 18.989 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 205.238 ms | 4.72 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 5.459 ms | 1.37 | 5.49 | 7% | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 8.765 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 1762.700 ms | 1.10 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.556 ms | 0.38 | 1.53 | 2% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 12.906 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 2217.056 ms | 0.87 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.562 ms | 0.38 | 1.53 | 2% | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 213.888 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 195.408 ms | 0.00 | 0.00 | n/a | ok | 1.3x (16%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 133.595 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 96.044 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 101.908 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 92.749 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 193.616 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 166.461 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 101.903 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 103.542 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 93.194 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 96.710 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 114.692 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.132 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | ilu_0 | f64 | 256^2 grid | 6.198 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 256^2 grid | 50.461 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 256^2 grid | 4.274 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 256^2 grid | 4.007 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 256^2 grid | 0.113 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 2.813 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 4.350 ms | 2.41 | 26.98 | 33% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 4.929 ms | 2.13 | 23.81 | 29% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 5.728 ms | 1.83 | 20.49 | 25% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 61.580 ms | 2.53 | 18.98 | 23% | ok | 1.1x (14%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 288.921 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 290.717 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 489.930 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 18.831 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=33554432 | 8.143 ms | 8.24 | 32.97 | 40% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 6.646 ms | 323.12 | 2.52 | 54% | ok | 7.1x (89%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 2.114 ms | 31.74 | 33.72 | 41% | ok | 1.0x (13%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 7.898 ms | 8.50 | 33.99 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 3.830 ms | 17.52 | 35.05 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 2.089 ms | 32.12 | 32.12 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 2.097 ms | 32.00 | 32.00 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 4.411 ms | 486.86 | 2.85 | n/a | ok | 7.1x (89%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 6.166 ms | 348.26 | 1.36 | n/a | ok | 7.1x (89%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 6.180 ms | 347.50 | 1.02 | n/a | ok | 7.1x (89%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 2.333 ms | 920.62 | 2.70 | n/a | ok | 7.0x (88%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 2.084 ms | 1030.46 | 3.02 | n/a | ok | 7.0x (88%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

