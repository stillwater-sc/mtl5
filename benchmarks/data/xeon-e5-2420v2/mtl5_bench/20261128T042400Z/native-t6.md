## xeon-e5-2420v2 -- native, 6 threads, class large

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 59.0 GFLOP/s, f32 118.0 GFLOP/s |
| ... hardware, SSE2 | f64 59.0, f32 118.0 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.7 GFLOP/s |
| bandwidth | read 11.4, update 10.7, triad 8.5 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 53% | 107% | 0 | 85% |
| blas2 | 10 | 78% | 108% | 0 | 16% |
| blas3 | 10 | 7% | 40% | 0 | 17% |
| factor | 12 | 6% | 40% | 0 | 53% |
| eigen | 8 | 2% | 12% | 0 | 20% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 68% | 97% | 0 | 17% |
| direct | 19 | 1% | 15% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 3% | 14% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 69% | 76% | 0 | 17% |
| mg | 1 | 59% | 59% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 17% | 64% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 76% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 131.339 ms | 0.51 | 4.09 | 36% | ok | 5.6x (93%) |
| blas1 | dot | f32 | n=33554432 | 21.979 ms | 3.05 | 12.21 | 107% | ok | 5.1x (85%) |
| blas1 | axpy | f64 | n=33554432 | 152.110 ms | 0.44 | 5.29 | 46% | ok | 5.6x (93%) |
| blas1 | axpy | f32 | n=33554432 | 67.037 ms | 1.00 | 6.01 | 53% | ok | 5.5x (91%) |
| blas1 | scale | f64 | n=33554432 | 53.903 ms | 0.62 | 9.96 | 87% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=33554432 | 132.236 ms | 0.51 | 2.03 | 18% | ok | 5.6x (93%) |
| blas1 | one_norm | f64 | n=33554432 | 39.016 ms | 0.86 | 6.88 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=33554432 | 52.308 ms | 0.64 | 5.13 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=33554432 | 39.110 ms | 0.86 | 6.86 | 60% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=33554432 | 1114.926 ms | 0.66 | 0.48 | 4% | ok | 1.0x (16%) |
| blas2 | gemv | f64 | n=8192 | 44.127 ms | 3.04 | 12.17 | 106% | ok | 1.3x (22%) |
| blas2 | gemv | f32 | n=8192 | 22.156 ms | 6.06 | 12.12 | 106% | ok | 1.5x (25%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 46.237 ms | 2.90 | 11.61 | 102% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 21.844 ms | 6.14 | 12.29 | 108% | ok | 1.2x (21%) |
| blas2 | symv | f64 | n=8192 | 59.962 ms | 2.24 | 4.48 | 39% | ok | 0.9x (15%) |
| blas2 | trmv | f64 | n=8192 | 31.004 ms | 2.16 | 8.66 | 76% | ok | 1.0x (16%) |
| blas2 | trmv | f32 | n=8192 | 21.192 ms | 3.17 | 6.34 | 55% | ok | 0.9x (14%) |
| blas2 | trsv | f64 | n=8192 | 30.244 ms | 2.22 | 8.88 | 78% | ok | 1.0x (16%) |
| blas2 | trsv | f32 | n=8192 | 19.347 ms | 3.47 | 6.94 | 61% | ok | 0.9x (15%) |
| blas2 | ger | f64 | n=8192 | 121.637 ms | 1.10 | 8.83 | 77% | ok | 1.0x (16%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 91.836 ms | 23.38 | 0.27 | 40% | ok | 5.6x (93%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 51.184 ms | 41.96 | 0.25 | 36% | ok | 5.7x (95%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 275.061 ms | 3.90 | 0.09 | 7% | ok | 1.0x (17%) |
| blas3 | symm | f64 | 1024x1024x1024 | 524.059 ms | 4.10 | 0.05 | 7% | ok | 1.0x (17%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 287.800 ms | 3.73 | 0.06 | 6% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 576.935 ms | 3.72 | 0.04 | 6% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 256.369 ms | 4.19 | 0.08 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 258.302 ms | 4.16 | 0.08 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 127.454 ms | 8.42 | 0.08 | 7% | ok | 1.0x (17%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 131.413 ms | 0.13 | 1.02 | 9% | ok | 1.0x (17%) |
| factor | lu | f64 | 1024x1024 | 70.374 ms | 10.17 | 0.24 | 17% | ok | 3.5x (58%) |
| factor | lu | f32 | 1024x1024 | 41.005 ms | 17.46 | 0.20 | 15% | ok | 3.2x (53%) |
| factor | cholesky | f64 | 1024x1024 | 94.932 ms | 3.77 | 0.18 | 6% | ok | 2.2x (36%) |
| factor | cholesky | f32 | 1024x1024 | 92.406 ms | 3.87 | 0.09 | 3% | ok | 2.2x (36%) |
| factor | ldlt | f64 | 1024x1024 | 213.927 ms | 1.67 | 0.08 | 3% | ok | 1.0x (16%) |
| factor | ldlt_bk | f64 | 1024x1024 | 105.566 ms | 3.39 | 0.16 | 6% | ok | 1.0x (17%) |
| factor | qr | f64 | 1024x1024 | 1625.886 ms | 0.88 | 0.01 | 1% | ok | 4.2x (70%) |
| factor | lq | f64 | 1024x1024 | 149.453 ms | 9.58 | 0.11 | 16% | ok | 3.9x (65%) |
| factor | hessenberg | f64 | 1024x1024 | 1868.339 ms | 1.92 | 0.01 | 3% | ok | 4.1x (69%) |
| factor | householder | f64 | 1024x1024 | 3.709 ms | 1.13 | 4.52 | 40% | ok | 5.5x (92%) |
| factor | inv | f64 | 1024x1024 | 999.110 ms | 2.15 | 0.02 | 4% | ok | 1.2x (20%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 55.810 ms | 12.83 | 0.30 | 22% | ok | 2.6x (44%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 415.554 ms | 0.43 | 0.01 | 1% | ok | 1.5x (25%) |
| eigen | eigen_symmetric | f64 | 512x512 | 1849.656 ms | 0.65 | 0.00 | 1% | ok | 1.1x (19%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 14618.110 ms | 0.08 | 0.00 | 0% | ok | 1.0x (16%) |
| eigen | eigenvalue | f64 | 512x512 | 867.405 ms | 1.55 | 0.00 | 3% | ok | 1.2x (21%) |
| eigen | schur | f64 | 512x512 | 2609.530 ms | 1.29 | 0.00 | 2% | ok | 1.1x (19%) |
| eigen | svd | f64 | 512x512 | 2691.159 ms | 1.10 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 512x512 | 279.563 ms | 7.20 | 0.02 | 12% | ok | 2.6x (44%) |
| eigen | sylvester | f64 | 512x512 | 5914.523 ms | 1.36 | 0.00 | 2% | ok | 1.2x (20%) |
| elementwise | exp | f64 | n=33554432 | 603.338 ms | 0.06 | 0.89 | 8% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=33554432 | 562.534 ms | 0.06 | 0.95 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=33554432 | 682.965 ms | 0.05 | 0.79 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=33554432 | 362.306 ms | 0.09 | 1.48 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=33554432 | 9909.443 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 9.851 ms | 1.06 | 11.06 | 97% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 12.717 ms | 0.82 | 8.57 | 75% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 45.532 ms | 3.68 | 7.74 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 101.113 ms | 0.13 | 0.46 | 4% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 37.528 ms | 0.27 | 2.23 | 20% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 305.572 ms | 0.35 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 554.638 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 156.001 ms | 0.69 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 893.036 ms | 0.24 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 886.451 ms | 0.24 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 379.959 ms | 0.56 | 0.01 | 1% | ok | 1.0x (16%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 1122.399 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 43.879 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 63.125 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 581.725 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 2.393 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 253.894 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 26.485 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 271.118 ms | 0.39 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 9.098 ms | 0.44 | 1.76 | 15% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 54.559 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 92.644 ms | 1.15 | 0.03 | 2% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 9.150 ms | 0.44 | 1.75 | 15% | ok | 1.0x (17%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 27.446 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 828.009 ms | 0.26 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.306 ms | 0.13 | 0.53 | 5% | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 40.951 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 802.785 ms | 0.27 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.552 ms | 0.13 | 0.53 | 5% | ok | 1.0x (16%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 23.665 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1951.215 ms | 0.50 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 19.378 ms | 0.39 | 1.55 | 14% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 61.987 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 606.383 ms | 1.60 | 0.00 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 19.431 ms | 0.39 | 1.54 | 14% | ok | 1.0x (17%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 24.041 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 5763.888 ms | 0.34 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 87.394 ms | 0.09 | 0.34 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 33.856 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 4954.223 ms | 0.39 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 88.076 ms | 0.09 | 0.34 | 3% | ok | 1.0x (16%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 1745.140 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 2985.520 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 768.787 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 888.924 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 920.712 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 576.145 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 4875.976 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 1221.857 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 718.851 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 1390.252 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 1294.590 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 1494.922 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 2047.635 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.676 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 256^2 grid | 18.100 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 256^2 grid | 148.465 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 256^2 grid | 12.140 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | ildl | f64 | 256^2 grid | 11.411 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 256^2 grid | 0.370 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 8.982 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 14.932 ms | 0.70 | 7.86 | 69% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 13.512 ms | 0.78 | 8.69 | 76% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 16.023 ms | 0.65 | 7.33 | 64% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 173.540 ms | 0.90 | 6.73 | 59% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 563.830 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 576.668 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 1419.158 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 172.257 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=33554432 | 754.142 ms | 0.09 | 0.36 | 3% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 219.074 ms | 9.80 | 0.08 | 17% | ok | 4.1x (69%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 9.732 ms | 6.90 | 7.33 | 64% | ok | 1.0x (17%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 26.321 ms | 2.55 | 10.20 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.737 ms | 5.27 | 10.54 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 20.379 ms | 3.29 | 3.29 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 20.464 ms | 3.28 | 3.28 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 100.213 ms | 21.43 | 0.13 | n/a | ok | 4.6x (76%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 65.393 ms | 32.84 | 0.13 | n/a | ok | 5.5x (92%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 33.821 ms | 63.50 | 0.19 | n/a | ok | 5.6x (93%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 33.740 ms | 63.65 | 0.19 | n/a | ok | 5.6x (93%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 103.507 ms | 20.75 | 0.06 | n/a | ok | 5.5x (91%) |

