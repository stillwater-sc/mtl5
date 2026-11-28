## i7-12700k -- blas, 8 threads, class large

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 600.6 GFLOP/s, f32 1201.4 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.6, f32 1201.4 GFLOP/s |
| ... this build, scalar | f64 10.0, f32 9.9 GFLOP/s |
| bandwidth | read 82.4, update 77.5, triad 75.5 GB/s (ceiling 82.4) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 88% | 106% | 0 | 29% |
| blas2 | 10 | 102% | 106% | 0 | 28% |
| blas3 | 10 | 75% | 92% | 0 | 82% |
| factor | 12 | 8% | 23% | 0 | 22% |
| eigen | 8 | 4% | 15% | 0 | 15% |
| elementwise | 5 | 3% | 5% | 0 | 13% |
| sparse | 5 | 32% | 114% | 0 | 14% |
| direct | 31 | 1% | 13% | 0 | 14% |
| ordering | 5 | n/a | n/a | 0 | 14% |
| direct3d | 21 | 7% | 31% | 0 | 14% |
| krylov | 13 | n/a | n/a | 0 | 14% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 33% | 0 | 13% |
| mg | 1 | 23% | 23% | 0 | 14% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 33% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 90% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 6.306 ms | 10.64 | 85.13 | 103% | ok | 3.7x (46%) |
| blas1 | dot | f32 | n=33554432 | 3.086 ms | 21.74 | 86.97 | 106% | ok | 5.6x (70%) |
| blas1 | axpy | f64 | n=33554432 | 11.151 ms | 6.02 | 72.22 | 88% | ok | 2.3x (29%) |
| blas1 | axpy | f32 | n=33554432 | 5.567 ms | 12.05 | 72.33 | 88% | ok | 3.4x (43%) |
| blas1 | scale | f64 | n=33554432 | 7.460 ms | 4.50 | 71.97 | 87% | ok | 1.9x (24%) |
| blas1 | two_norm | f64 | n=33554432 | 3.273 ms | 20.50 | 82.02 | 99% | ok | 6.1x (77%) |
| blas1 | one_norm | f64 | n=33554432 | 16.314 ms | 2.06 | 16.45 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=33554432 | 27.872 ms | 1.20 | 9.63 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=33554432 | 16.159 ms | 2.08 | 16.61 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=33554432 | 316.006 ms | 2.34 | 1.70 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=8192 | 6.556 ms | 20.47 | 81.92 | 99% | ok | 2.2x (27%) |
| blas2 | gemv | f32 | n=8192 | 3.196 ms | 41.99 | 84.00 | 102% | ok | 2.2x (28%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 6.404 ms | 20.96 | 83.85 | 102% | ok | 2.3x (29%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 3.138 ms | 42.78 | 85.57 | 104% | ok | 2.3x (28%) |
| blas2 | symv | f64 | n=8192 | 3.186 ms | 42.12 | 84.29 | 102% | ok | 2.4x (30%) |
| blas2 | trmv | f64 | n=8192 | 3.177 ms | 21.13 | 84.55 | 103% | ok | 2.4x (30%) |
| blas2 | trmv | f32 | n=8192 | 1.537 ms | 43.67 | 87.37 | 106% | ok | 2.6x (32%) |
| blas2 | trsv | f64 | n=8192 | 7.190 ms | 9.33 | 37.35 | 45% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=8192 | 3.571 ms | 18.79 | 37.61 | 46% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=8192 | 15.173 ms | 8.85 | 70.78 | 86% | ok | 1.8x (23%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 4.057 ms | 529.37 | 6.20 | 88% | ok | 7.2x (90%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 1.951 ms | 1100.85 | 6.45 | 92% | ok | 7.3x (92%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 3.629 ms | 295.86 | 6.93 | 49% | ok | 4.9x (61%) |
| blas3 | symm | f64 | 1024x1024x1024 | 4.078 ms | 526.55 | 6.17 | 88% | ok | 7.2x (90%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 4.290 ms | 250.30 | 3.91 | 42% | ok | 4.3x (54%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 6.713 ms | 319.91 | 3.75 | 53% | ok | 5.4x (67%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 2.398 ms | 447.82 | 8.75 | 75% | ok | 6.5x (81%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 2.386 ms | 450.06 | 8.79 | 75% | ok | 6.6x (82%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 1.088 ms | 986.72 | 9.64 | 82% | ok | 6.8x (85%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 44.200 ms | 0.38 | 3.04 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 1024x1024 | 8.514 ms | 84.07 | 1.97 | 14% | ok | 2.2x (27%) |
| factor | lu | f32 | 1024x1024 | 6.991 ms | 102.39 | 1.20 | 9% | ok | 1.8x (22%) |
| factor | cholesky | f64 | 1024x1024 | 7.727 ms | 46.32 | 2.17 | 8% | ok | 1.5x (19%) |
| factor | cholesky | f32 | 1024x1024 | 6.205 ms | 57.68 | 1.35 | 5% | ok | 1.3x (16%) |
| factor | ldlt | f64 | 1024x1024 | 67.050 ms | 5.34 | 0.25 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 1024x1024 | 39.531 ms | 9.05 | 0.42 | 2% | ok | 1.1x (14%) |
| factor | qr | f64 | 1024x1024 | 20.223 ms | 70.79 | 0.83 | 12% | ok | 1.8x (22%) |
| factor | lq | f64 | 1024x1024 | 45.853 ms | 31.22 | 0.37 | 5% | ok | 4.8x (60%) |
| factor | hessenberg | f64 | 1024x1024 | 513.014 ms | 6.98 | 0.03 | 1% | ok | 5.2x (65%) |
| factor | householder | f64 | 1024x1024 | 0.898 ms | 4.68 | 18.69 | 23%\* | ok | 7.7x (96%) |
| factor | inv | f64 | 1024x1024 | 335.909 ms | 6.39 | 0.05 | 1% | ok | 1.1x (13%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 13.487 ms | 53.08 | 1.24 | 9% | ok | 1.6x (20%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 6.709 ms | 26.67 | 0.63 | 4% | ok | 1.4x (17%) |
| eigen | eigen_symmetric | f64 | 512x512 | 808.955 ms | 1.49 | 0.01 | 0% | ok | 1.2x (15%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 12547.254 ms | 0.10 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 512x512 | 115.830 ms | 11.59 | 0.04 | 2% | ok | 1.0x (13%) |
| eigen | schur | f64 | 512x512 | 128.305 ms | 26.15 | 0.03 | 4% | ok | 1.1x (14%) |
| eigen | svd | f64 | 512x512 | 32.682 ms | 90.35 | 0.13 | 15% | ok | 1.4x (17%) |
| eigen | expm | f64 | 512x512 | 68.228 ms | 29.51 | 0.06 | 5% | ok | 1.4x (17%) |
| eigen | sylvester | f64 | 512x512 | 422.132 ms | 19.08 | 0.01 | 3% | ok | 1.1x (13%) |
| elementwise | exp | f64 | n=33554432 | 161.257 ms | 0.21 | 3.33 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=33554432 | 188.217 ms | 0.18 | 2.85 | 3% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=33554432 | 242.842 ms | 0.14 | 2.21 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=33554432 | 120.099 ms | 0.28 | 4.47 | 5% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=33554432 | 807.370 ms | 0.04 | 0.66 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 1.164 ms | 9.00 | 93.65 | 114% | ok | 3.2x (39%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 4.150 ms | 2.52 | 26.26 | 32% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 6.840 ms | 24.51 | 51.50 | 62% | ok | 2.8x (35%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 29.262 ms | 0.45 | 1.57 | 2% | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 13.253 ms | 0.77 | 6.32 | 8% | ok | 1.1x (14%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 100.691 ms | 1.06 | 0.03 | 0% | ok | 1.1x (13%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 184.568 ms | 0.58 | 0.02 | 0% | ok | 1.1x (13%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 49.503 ms | 2.16 | 0.06 | 0% | ok | 1.1x (14%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 310.186 ms | 0.69 | 0.01 | 0% | ok | 1.1x (13%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 287.356 ms | 0.74 | 0.01 | 0% | ok | 1.2x (15%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 134.989 ms | 1.59 | 0.02 | 0% | ok | 1.1x (14%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 366.144 ms | 0.06 | 0.00 | 0% | ok | 1.1x (14%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 16.301 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 22.962 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 259.743 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 0.888 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 89.591 ms | 0.00 | 0.00 | n/a | ok | 1.2x (14%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 8.312 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 89.421 ms | 1.20 | 0.03 | 0% | ok | 1.1x (14%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 2.587 ms | 1.55 | 6.21 | 8%\* | ok | 1.4x (18%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 17.863 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 30.150 ms | 3.55 | 0.10 | 1% | ok | 1.1x (13%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 2.505 ms | 1.60 | 6.41 | 8%\* | ok | 1.0x (13%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 7.880 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 289.682 ms | 0.74 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.704 ms | 0.46 | 1.84 | 2%\* | ok | 1.1x (13%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 12.836 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 268.758 ms | 0.80 | 0.01 | 0% | ok | 1.1x (13%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.746 ms | 0.46 | 1.84 | 2%\* | ok | 1.1x (13%) |
| direct | ref_cholmod | f64 | 192^2 grid, nnz 183552 | 24.949 ms | 4.29 | 0.12 | 1% | ok | 1.0x (13%) |
| direct | ref_cholmod.analyze | f64 | 192^2 grid, nnz 183552 | 6.668 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | ref_cholmod.factor | f64 | 192^2 grid, nnz 183552 | 15.826 ms | 6.76 | 0.19 | 1% | ok | 1.0x (12%) |
| direct | ref_cholmod.solve | f64 | 192^2 grid, nnz 183552 | 1.462 ms | 2.75 | 10.98 | 13%\* | ok | 1.2x (15%) |
| direct | ref_pardiso_spd | f64 | 192^2 grid, nnz 183552 | 11.708 ms | 9.14 | 0.25 | 2% | ok | 1.8x (22%) |
| direct | ref_pardiso_spd.analyze | f64 | 192^2 grid, nnz 183552 | 6.499 ms | 0.00 | 0.00 | n/a | ok | 1.3x (16%) |
| direct | ref_pardiso_spd.factor | f64 | 192^2 grid, nnz 183552 | 2.752 ms | 38.87 | 1.07 | 6% | ok | 3.4x (42%) |
| direct | ref_pardiso_spd.solve | f64 | 192^2 grid, nnz 183552 | 2.142 ms | 1.87 | 7.50 | 9%\* | ok | 1.0x (12%) |
| direct | ref_pardiso_lu | f64 | 192^2 grid, nnz 183552 | 32.696 ms | 6.54 | 0.09 | 1% | ok | 1.5x (18%) |
| direct | ref_pardiso_lu.analyze | f64 | 192^2 grid, nnz 183552 | 24.665 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct | ref_pardiso_lu.factor | f64 | 192^2 grid, nnz 183552 | 4.737 ms | 45.17 | 0.62 | 8% | ok | 3.8x (47%) |
| direct | ref_pardiso_lu.solve | f64 | 192^2 grid, nnz 183552 | 2.845 ms | 1.41 | 5.64 | 7%\* | ok | 1.6x (20%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 8.678 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 595.191 ms | 1.63 | 0.00 | 0% | ok | 1.7x (22%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 5.002 ms | 1.50 | 6.00 | 7% | ok | 1.3x (16%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 18.748 ms | 0.00 | 0.00 | n/a | ok | 1.2x (14%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 178.169 ms | 5.44 | 0.01 | 1% | ok | 1.1x (14%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 4.984 ms | 1.50 | 6.02 | 7% | ok | 1.5x (18%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 8.608 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 1878.950 ms | 1.03 | 0.00 | 0% | ok | 1.1x (14%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.628 ms | 0.38 | 1.53 | 2% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 12.714 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1353.463 ms | 1.43 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.592 ms | 0.38 | 1.53 | 2% | ok | 1.0x (12%) |
| direct3d | ref_cholmod.analyze | f64 | 24^3 grid, nnz 93312 | 6.462 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | ref_cholmod.factor | f64 | 24^3 grid, nnz 93312 | 18.439 ms | 52.56 | 0.08 | 9% | ok | 1.6x (20%) |
| direct3d | ref_cholmod.solve | f64 | 24^3 grid, nnz 93312 | 1.166 ms | 6.43 | 25.73 | 31% | ok | 1.0x (12%) |
| direct3d | ref_pardiso_spd.analyze | f64 | 24^3 grid, nnz 93312 | 12.607 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| direct3d | ref_pardiso_spd.factor | f64 | 24^3 grid, nnz 93312 | 5.928 ms | 163.50 | 0.25 | 27% | ok | 5.0x (62%) |
| direct3d | ref_pardiso_spd.solve | f64 | 24^3 grid, nnz 93312 | 2.152 ms | 3.48 | 13.94 | 17% | ok | 1.0x (12%) |
| direct3d | ref_pardiso_lu.analyze | f64 | 24^3 grid, nnz 93312 | 14.366 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| direct3d | ref_pardiso_lu.factor | f64 | 24^3 grid, nnz 93312 | 12.148 ms | 159.56 | 0.12 | 27% | ok | 4.9x (61%) |
| direct3d | ref_pardiso_lu.solve | f64 | 24^3 grid, nnz 93312 | 3.055 ms | 2.45 | 9.82 | 12% | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 237.072 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 225.284 ms | 0.00 | 0.00 | n/a | ok | 1.3x (16%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 148.778 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 106.325 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 113.024 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 98.419 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 271.639 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 184.908 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 109.118 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 123.476 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 110.414 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 115.511 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 154.259 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.159 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 256^2 grid | 5.982 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 256^2 grid | 49.316 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 256^2 grid | 4.280 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 256^2 grid | 4.012 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 256^2 grid | 0.105 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 4.524 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 4.371 ms | 2.40 | 26.85 | 33% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 4.940 ms | 2.12 | 23.76 | 29% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 6.264 ms | 1.67 | 18.74 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 61.840 ms | 2.52 | 18.90 | 23% | ok | 1.1x (14%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 289.041 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 291.080 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 483.527 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 20.350 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=33554432 | 19.272 ms | 3.48 | 13.93 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 41.035 ms | 52.33 | 0.41 | 9% | ok | 7.6x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 2.624 ms | 25.57 | 27.17 | 33% | ok | 1.0x (13%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 9.360 ms | 7.17 | 28.68 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 4.618 ms | 14.53 | 29.07 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 6.852 ms | 9.79 | 9.79 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 6.859 ms | 9.78 | 9.78 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 19.454 ms | 110.39 | 0.65 | n/a | ok | 7.2x (90%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 21.854 ms | 98.27 | 0.38 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 10.738 ms | 199.99 | 0.59 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 10.741 ms | 199.93 | 0.59 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 28.128 ms | 76.35 | 0.22 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

