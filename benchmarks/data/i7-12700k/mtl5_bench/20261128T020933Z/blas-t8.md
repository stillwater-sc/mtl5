## i7-12700k -- blas, 8 threads, class large

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 600.7 GFLOP/s, f32 1201.5 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.7, f32 1201.5 GFLOP/s |
| ... this build, scalar | f64 10.0, f32 9.9 GFLOP/s |
| bandwidth | read 81.7, update 77.2, triad 75.4 GB/s (ceiling 81.7) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 88% | 107% | 0 | 29% |
| blas2 | 10 | 103% | 107% | 0 | 28% |
| blas3 | 10 | 75% | 92% | 0 | 82% |
| factor | 12 | 8% | 23% | 0 | 22% |
| eigen | 8 | 4% | 15% | 0 | 15% |
| elementwise | 5 | 3% | 5% | 0 | 13% |
| sparse | 5 | 32% | 115% | 0 | 14% |
| direct | 31 | 1% | 13% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 21 | 7% | 31% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 14% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 33% | 0 | 13% |
| mg | 1 | 23% | 23% | 0 | 14% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 33% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 90% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 6.288 ms | 10.67 | 85.39 | 105% | ok | 3.7x (46%) |
| blas1 | dot | f32 | n=33554432 | 3.074 ms | 21.83 | 87.32 | 107% | ok | 5.8x (72%) |
| blas1 | axpy | f64 | n=33554432 | 11.154 ms | 6.02 | 72.20 | 88% | ok | 2.3x (29%) |
| blas1 | axpy | f32 | n=33554432 | 5.602 ms | 11.98 | 71.88 | 88% | ok | 3.4x (42%) |
| blas1 | scale | f64 | n=33554432 | 7.460 ms | 4.50 | 71.96 | 88% | ok | 1.9x (24%) |
| blas1 | two_norm | f64 | n=33554432 | 3.279 ms | 20.47 | 81.87 | 100% | ok | 6.2x (77%) |
| blas1 | one_norm | f64 | n=33554432 | 16.324 ms | 2.06 | 16.44 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=33554432 | 27.898 ms | 1.20 | 9.62 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=33554432 | 16.160 ms | 2.08 | 16.61 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=33554432 | 318.015 ms | 2.32 | 1.69 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=8192 | 6.556 ms | 20.47 | 81.91 | 100% | ok | 2.2x (27%) |
| blas2 | gemv | f32 | n=8192 | 3.205 ms | 41.88 | 83.77 | 103% | ok | 2.2x (28%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 6.409 ms | 20.94 | 83.78 | 103% | ok | 2.3x (29%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 3.137 ms | 42.79 | 85.60 | 105% | ok | 2.3x (28%) |
| blas2 | symv | f64 | n=8192 | 3.190 ms | 42.07 | 84.18 | 103% | ok | 2.4x (30%) |
| blas2 | trmv | f64 | n=8192 | 3.171 ms | 21.16 | 84.69 | 104% | ok | 2.4x (30%) |
| blas2 | trmv | f32 | n=8192 | 1.542 ms | 43.52 | 87.09 | 107% | ok | 2.6x (32%) |
| blas2 | trsv | f64 | n=8192 | 7.224 ms | 9.29 | 37.18 | 46% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=8192 | 3.567 ms | 18.81 | 37.65 | 46% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=8192 | 15.166 ms | 8.85 | 70.81 | 87% | ok | 1.8x (23%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 4.059 ms | 529.10 | 6.20 | 88% | ok | 7.2x (90%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 1.945 ms | 1104.15 | 6.47 | 92% | ok | 7.4x (92%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 3.648 ms | 294.35 | 6.90 | 49% | ok | 4.9x (61%) |
| blas3 | symm | f64 | 1024x1024x1024 | 4.070 ms | 527.62 | 6.18 | 88% | ok | 7.2x (90%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 4.283 ms | 250.69 | 3.92 | 42% | ok | 4.3x (54%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 6.732 ms | 318.99 | 3.74 | 53% | ok | 5.3x (67%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 2.374 ms | 452.26 | 8.83 | 75% | ok | 6.5x (82%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 2.382 ms | 450.83 | 8.81 | 75% | ok | 6.6x (82%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 1.087 ms | 987.39 | 9.64 | 82% | ok | 6.8x (85%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 43.657 ms | 0.38 | 3.07 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 1024x1024 | 8.319 ms | 86.05 | 2.02 | 14% | ok | 2.2x (28%) |
| factor | lu | f32 | 1024x1024 | 6.962 ms | 102.82 | 1.20 | 9% | ok | 1.8x (22%) |
| factor | cholesky | f64 | 1024x1024 | 7.718 ms | 46.37 | 2.17 | 8% | ok | 1.5x (18%) |
| factor | cholesky | f32 | 1024x1024 | 6.149 ms | 58.21 | 1.36 | 5% | ok | 1.3x (16%) |
| factor | ldlt | f64 | 1024x1024 | 66.767 ms | 5.36 | 0.25 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 1024x1024 | 39.235 ms | 9.12 | 0.43 | 2% | ok | 1.1x (14%) |
| factor | qr | f64 | 1024x1024 | 20.147 ms | 71.06 | 0.83 | 12% | ok | 1.8x (22%) |
| factor | lq | f64 | 1024x1024 | 45.467 ms | 31.49 | 0.37 | 5% | ok | 4.8x (60%) |
| factor | hessenberg | f64 | 1024x1024 | 511.182 ms | 7.00 | 0.03 | 1% | ok | 5.1x (64%) |
| factor | householder | f64 | 1024x1024 | 0.894 ms | 4.70 | 18.77 | 23%\* | ok | 7.6x (96%) |
| factor | inv | f64 | 1024x1024 | 334.851 ms | 6.41 | 0.05 | 1% | ok | 1.0x (13%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 13.440 ms | 53.26 | 1.25 | 9% | ok | 1.6x (20%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 6.708 ms | 26.68 | 0.63 | 4% | ok | 1.4x (17%) |
| eigen | eigen_symmetric | f64 | 512x512 | 815.986 ms | 1.48 | 0.01 | 0% | ok | 1.2x (15%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 12501.523 ms | 0.10 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 512x512 | 118.675 ms | 11.31 | 0.04 | 2% | ok | 1.0x (12%) |
| eigen | schur | f64 | 512x512 | 128.612 ms | 26.09 | 0.03 | 4% | ok | 1.1x (14%) |
| eigen | svd | f64 | 512x512 | 33.685 ms | 87.66 | 0.12 | 15% | ok | 1.3x (16%) |
| eigen | expm | f64 | 512x512 | 68.088 ms | 29.57 | 0.06 | 5% | ok | 1.4x (17%) |
| eigen | sylvester | f64 | 512x512 | 420.506 ms | 19.15 | 0.01 | 3% | ok | 1.1x (13%) |
| elementwise | exp | f64 | n=33554432 | 160.953 ms | 0.21 | 3.34 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=33554432 | 188.456 ms | 0.18 | 2.85 | 3% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=33554432 | 242.340 ms | 0.14 | 2.22 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=33554432 | 120.632 ms | 0.28 | 4.45 | 5% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=33554432 | 810.645 ms | 0.04 | 0.66 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 1.160 ms | 9.03 | 93.94 | 115% | ok | 3.1x (38%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 4.187 ms | 2.50 | 26.03 | 32% | ok | 1.0x (12%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 6.842 ms | 24.50 | 51.48 | 63% | ok | 2.8x (35%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 29.368 ms | 0.45 | 1.57 | 2% | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 13.231 ms | 0.77 | 6.33 | 8% | ok | 1.1x (14%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 101.098 ms | 1.06 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 184.471 ms | 0.58 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 49.371 ms | 2.17 | 0.06 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 316.236 ms | 0.68 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 299.610 ms | 0.71 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 135.085 ms | 1.58 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 360.035 ms | 0.06 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 16.204 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 22.791 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 259.801 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 0.882 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 91.321 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 8.263 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 92.430 ms | 1.16 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 2.581 ms | 1.56 | 6.22 | 8%\* | ok | 0.9x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 17.788 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 28.730 ms | 3.72 | 0.10 | 1% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 2.488 ms | 1.61 | 6.45 | 8%\* | ok | 0.9x (12%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 7.918 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 289.717 ms | 0.74 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.751 ms | 0.46 | 1.83 | 2%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 12.855 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 268.064 ms | 0.80 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.737 ms | 0.46 | 1.84 | 2%\* | ok | 1.0x (13%) |
| direct | ref_cholmod | f64 | 192^2 grid, nnz 183552 | 24.980 ms | 4.28 | 0.12 | 1% | ok | 1.0x (12%) |
| direct | ref_cholmod.analyze | f64 | 192^2 grid, nnz 183552 | 6.777 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | ref_cholmod.factor | f64 | 192^2 grid, nnz 183552 | 15.775 ms | 6.78 | 0.19 | 1% | ok | 1.0x (12%) |
| direct | ref_cholmod.solve | f64 | 192^2 grid, nnz 183552 | 1.458 ms | 2.75 | 11.01 | 13%\* | ok | 1.0x (13%) |
| direct | ref_pardiso_spd | f64 | 192^2 grid, nnz 183552 | 11.623 ms | 9.21 | 0.25 | 2% | ok | 1.6x (20%) |
| direct | ref_pardiso_spd.analyze | f64 | 192^2 grid, nnz 183552 | 6.489 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| direct | ref_pardiso_spd.factor | f64 | 192^2 grid, nnz 183552 | 2.824 ms | 37.89 | 1.04 | 6% | ok | 3.1x (39%) |
| direct | ref_pardiso_spd.solve | f64 | 192^2 grid, nnz 183552 | 2.152 ms | 1.87 | 7.46 | 9%\* | ok | 1.0x (12%) |
| direct | ref_pardiso_lu | f64 | 192^2 grid, nnz 183552 | 31.450 ms | 6.80 | 0.09 | 1% | ok | 1.4x (18%) |
| direct | ref_pardiso_lu.analyze | f64 | 192^2 grid, nnz 183552 | 24.871 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | ref_pardiso_lu.factor | f64 | 192^2 grid, nnz 183552 | 4.573 ms | 46.79 | 0.64 | 8% | ok | 3.2x (40%) |
| direct | ref_pardiso_lu.solve | f64 | 192^2 grid, nnz 183552 | 2.841 ms | 1.41 | 5.65 | 7%\* | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 8.683 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 594.423 ms | 1.63 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 5.005 ms | 1.50 | 5.99 | 7% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 18.753 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 177.654 ms | 5.46 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 4.991 ms | 1.50 | 6.01 | 7% | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 8.580 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 1895.462 ms | 1.02 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.629 ms | 0.38 | 1.53 | 2% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 12.679 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1358.744 ms | 1.43 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.611 ms | 0.38 | 1.53 | 2% | ok | 1.0x (12%) |
| direct3d | ref_cholmod.analyze | f64 | 24^3 grid, nnz 93312 | 6.476 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | ref_cholmod.factor | f64 | 24^3 grid, nnz 93312 | 17.934 ms | 54.04 | 0.08 | 9% | ok | 1.6x (20%) |
| direct3d | ref_cholmod.solve | f64 | 24^3 grid, nnz 93312 | 1.175 ms | 6.38 | 25.53 | 31% | ok | 1.0x (12%) |
| direct3d | ref_pardiso_spd.analyze | f64 | 24^3 grid, nnz 93312 | 12.636 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| direct3d | ref_pardiso_spd.factor | f64 | 24^3 grid, nnz 93312 | 5.929 ms | 163.47 | 0.25 | 27% | ok | 4.9x (62%) |
| direct3d | ref_pardiso_spd.solve | f64 | 24^3 grid, nnz 93312 | 2.166 ms | 3.46 | 13.85 | 17% | ok | 1.0x (12%) |
| direct3d | ref_pardiso_lu.analyze | f64 | 24^3 grid, nnz 93312 | 14.421 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| direct3d | ref_pardiso_lu.factor | f64 | 24^3 grid, nnz 93312 | 12.188 ms | 159.04 | 0.12 | 26% | ok | 4.8x (60%) |
| direct3d | ref_pardiso_lu.solve | f64 | 24^3 grid, nnz 93312 | 3.051 ms | 2.46 | 9.83 | 12% | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 237.004 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 226.537 ms | 0.00 | 0.00 | n/a | ok | 1.3x (16%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 149.043 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 106.315 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 112.813 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 98.212 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 271.614 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 184.905 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 109.319 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 123.436 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 110.715 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 116.269 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 153.586 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.155 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | ilu_0 | f64 | 256^2 grid | 5.964 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | ilut | f64 | 256^2 grid | 49.138 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 256^2 grid | 4.277 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 256^2 grid | 4.008 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 256^2 grid | 0.107 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 4.539 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 4.380 ms | 2.39 | 26.80 | 33% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 4.921 ms | 2.13 | 23.85 | 29% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 6.246 ms | 1.68 | 18.79 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 61.797 ms | 2.52 | 18.91 | 23% | ok | 1.1x (14%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 290.798 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 291.536 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 491.592 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 20.332 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=33554432 | 19.277 ms | 3.48 | 13.93 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 41.032 ms | 52.34 | 0.41 | 9% | ok | 7.6x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 2.623 ms | 25.58 | 27.18 | 33% | ok | 1.0x (13%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 9.369 ms | 7.16 | 28.65 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 4.627 ms | 14.50 | 29.01 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 6.853 ms | 9.79 | 9.79 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 6.856 ms | 9.79 | 9.79 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 19.222 ms | 111.72 | 0.65 | n/a | ok | 7.2x (90%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 21.876 ms | 98.17 | 0.38 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 10.738 ms | 200.00 | 0.59 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 10.739 ms | 199.97 | 0.59 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 28.112 ms | 76.39 | 0.22 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

